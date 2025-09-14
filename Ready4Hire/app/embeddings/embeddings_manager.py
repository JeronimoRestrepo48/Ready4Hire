import numpy as np
import umap
import hdbscan
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Ready4Hire - Gestor de Embeddings y Selección Avanzada
======================================================

¿Qué es?
---------
Este módulo es el "cerebro semántico" del sistema. Gestiona la representación vectorial (embeddings) de preguntas y respuestas, permitiendo búsquedas inteligentes, clustering temático y selección personalizada de preguntas.

¿Para quién es?
---------------
- Técnicos: pueden entender y modificar la lógica de embeddings, clustering (UMAP/HDBSCAN), penalización por repetición y ranking profundo (RankNet).
- No técnicos: el sistema selecciona automáticamente las preguntas más relevantes y variadas, asegurando una experiencia de entrevista realista y adaptativa.

¿Cómo funciona?
----------------
1. Embeddings: convierte preguntas y respuestas en vectores numéricos usando modelos de lenguaje (SentenceTransformers).
2. Clustering: agrupa preguntas por temas usando UMAP y HDBSCAN, para diversificar y cubrir todos los tópicos.
3. Selección avanzada: penaliza preguntas repetidas, incentiva variedad y usa softmax para diversificación.
4. RankNet: si hay un modelo entrenado, reordena las preguntas usando deep learning para máxima personalización.
5. Control de recursos: limita hilos y uso de CPU para evitar sobrecarga en servidores modestos.

Componentes principales:
-----------------------
- EmbeddingsManager: clase central, expone métodos para seleccionar, filtrar y rankear preguntas.
- advanced_question_selector: lógica avanzada de selección, combinando IA, clustering y penalización.
- RankNet: red neuronal profunda para ranking personalizado (opcional, si está entrenada).

Ejemplo de uso rápido:
----------------------
    emb_mgr = EmbeddingsManager()
    preguntas = emb_mgr.advanced_question_selector('contexto del usuario', history=[], top_k=5, technical=True)

Autor: JeronimoRestrepo48
Licencia: MIT
"""
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass
try:
    import numpy
    numpy.seterr(all='ignore')
except ImportError:
    pass
"""
Ready4Hire - Gestor de Embeddings
---------------------------------
Este módulo gestiona la generación, actualización y búsqueda semántica de embeddings para preguntas, respuestas y recursos en entrevistas IA.
Utiliza SentenceTransformers para comparar similitud semántica y seleccionar preguntas/referencias relevantes.
Incluye control de recursos y refresco dinámico para aprendizaje continuo.
Autor: JeronimoRestrepo48
Licencia: MIT
"""
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import json

class EmbeddingsManager:
    """
    EmbeddingsManager: gestor de representaciones semánticas y selector avanzado de preguntas.

    Responsabilidades:
    - Mantener y actualizar embeddings para preguntas técnicas y de soft skills.
    - Proveer búsquedas semánticas, selección diversificada y explicaciones de selección.
    - Soportar hooks de ranking avanzado (RankNet) y restricción opcional a un subconjunto (custom_pool).

    API principal:
    - advanced_question_selector(user_context, history, top_k, technical, custom_pool)
      devuelve una lista de preguntas seleccionadas según contexto y políticas internas.
    """

    def advanced_question_selector(self, user_context, history=None, top_k=5, technical=True, custom_pool=None):
        """
        Selección avanzada de preguntas usando:
        - Embeddings (SentenceTransformer)
        - Reducción de dimensionalidad (UMAP)
        - Clustering (HDBSCAN)
        - Penalización por repetición
        - Softmax para diversificación
        - Hook para modelo de ranking profundo (RankNet)
        - custom_pool (optional): lista de preguntas (subset of data) para restringir la selección
        """
        data = self.tech_data if technical else self.soft_data
        embeddings = self.tech_embeddings if technical else self.soft_embeddings
        # 1. Embedding del contexto del usuario
        context_emb = self.model.encode([self._normalize(user_context)], convert_to_tensor=True)
        # 2. Reducción de dimensionalidad (UMAP)
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=10, random_state=42)
        emb_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
        emb_umap = reducer.fit_transform(emb_np)
        # Ensure emb_umap is a NumPy array
        if not isinstance(emb_umap, np.ndarray):
            emb_umap = np.array(emb_umap)
        # 3. Clustering (HDBSCAN)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clusterer.fit_predict(emb_umap)
        # 4. Similitud semántica
        sim_scores = util.pytorch_cos_sim(context_emb, embeddings)[0].cpu().numpy()
        # 5. Penalización por repetición
        penal = np.zeros(len(data))
        if history:
            seen = set([h.get('question') or h.get('scenario') for h in history if 'question' in h or 'scenario' in h])
            for i, q in enumerate(data):
                qtxt = q.get('question') or q.get('scenario')
                if qtxt in seen:
                    penal[i] = -0.5  # Penaliza preguntas ya vistas
        # 6. Bonus por cluster poco cubierto
        if history:
            hist_clusters = [cluster_labels[i] for i, q in enumerate(data) if (q.get('question') or q.get('scenario')) in [h.get('question') or h.get('scenario') for h in history]]
            rare_clusters = set(np.unique(cluster_labels)) - set(hist_clusters)
            for i, cl in enumerate(cluster_labels):
                if cl in rare_clusters:
                    penal[i] += 0.2  # Incentiva variedad temática
        # 7. Score final y softmax
        scores = sim_scores + penal
        probs = np.exp(scores) / np.sum(np.exp(scores))
        # 8. Top-k diversificado y maximización de cobertura temática
        # Selecciona preguntas de clusters distintos primero. Si se proporciona custom_pool,
        # restringir la selección únicamente a índices que pertenecen a custom_pool.
        allowed_idxs = None
        if custom_pool is not None:
            try:
                allowed_idxs = set(i for i, q in enumerate(data) if q in custom_pool)
                if not allowed_idxs:
                    allowed_idxs = None
            except Exception:
                allowed_idxs = None

        cluster_to_idx = {}
        for idx in np.argsort(probs)[::-1]:
            if allowed_idxs is not None and idx not in allowed_idxs:
                continue
            cl = cluster_labels[idx]
            if cl not in cluster_to_idx:
                cluster_to_idx[cl] = idx
            if len(cluster_to_idx) >= top_k:
                break
        # Si no hay suficientes clusters distintos, completa con los siguientes mejores dentro del allowed set
        extra_needed = top_k - len(cluster_to_idx)
        extra = [i for i in np.argsort(probs)[::-1] if i not in cluster_to_idx.values() and (allowed_idxs is None or i in allowed_idxs)][:extra_needed]
        final_idx = list(cluster_to_idx.values()) + extra
        selected = [data[i] for i in final_idx]
        # 9. (Opcional) Modelo de ranking profundo
        # Si tienes un modelo RankNet entrenado, puedes reordenar selected aquí
        # Ejemplo de hook:
        # if hasattr(self, 'ranknet'):
        #     selected = self.ranknet_rank(selected, user_context, history)
        return selected

    class RankNet(nn.Module):
        """Modelo de ranking profundo para preguntas (ejemplo, puedes entrenar/fine-tune)"""
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    def ranknet_rank(self, questions, user_context, history):
        # Hook para reordenar preguntas usando RankNet (debes entrenar el modelo y cargar pesos)
        # Aquí solo es un ejemplo de integración
        # features = ... # Extrae features relevantes de preguntas y contexto
        # scores = self.ranknet(torch.tensor(features, dtype=torch.float32)).detach().numpy().flatten()
        # idx = np.argsort(scores)[::-1]
        # return [questions[i] for i in idx]
        return questions
    def filter_questions_by_role(self, role: str, top_k: int = 10, technical=True):
        """
        Devuelve las preguntas más relevantes al rol usando embeddings y clustering.
        Si technical=True, filtra preguntas técnicas; si False, blandas.
        """
        if not role:
            # Si no hay rol, devolver aleatorio
            return self.tech_data[:top_k] if technical else self.soft_data[:top_k]
        query = self._normalize(role)
        query_emb = self.model.encode([query], convert_to_tensor=True)
        if technical:
            hits = util.semantic_search(query_emb, self.tech_embeddings, top_k=top_k)[0]
            return [self.tech_data[int(hit['corpus_id'])] for hit in hits]
        else:
            hits = util.semantic_search(query_emb, self.soft_embeddings, top_k=top_k)[0]
            return [self.soft_data[int(hit['corpus_id'])] for hit in hits]
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Inicializa el gestor de embeddings:
        - model_name: modelo de SentenceTransformers a usar.
        - Carga datasets técnicos y blandos.
        - Calcula embeddings iniciales para preguntas y buenas respuestas.
        """
        self.model = SentenceTransformer(model_name)
        self.tech_data = self._load_jsonl(Path(__file__).parent / '../datasets/tech_questions.jsonl')
        self.soft_data = self._load_jsonl(Path(__file__).parent / '../datasets/soft_skills.jsonl')
        self._update_embeddings()

        # Cargar RankNet entrenado si existe
        import torch
        import os
        ranknet_path = Path(__file__).parent / '../datasets/ranknet_model.pt'
        if ranknet_path.exists():
            input_dim = 384 * 2  # MiniLM-L6-v2
            self.ranknet = self.RankNet(input_dim)
            self.ranknet.load_state_dict(torch.load(ranknet_path, map_location='cpu'))
            self.ranknet.eval()
        else:
            self.ranknet = None

    def _normalize(self, text: str) -> str:
        """
        Normaliza texto para comparación semántica (minúsculas, sin espacios extra).
        """
        return text.lower().strip()

    def _update_embeddings(self):
        """
        Calcula y actualiza los embeddings de:
        - Preguntas técnicas (incluye nivel y rol)
        - Preguntas blandas (escenario, nivel, rol)
        - Buenas respuestas (campos answer/expected)
        Permite búsquedas y validaciones semánticas robustas.
        """
        # Embeddings técnicos: pregunta + nivel + rol
        tech_texts = [
            self._normalize(f"{q.get('question','')} {q.get('level','')} {q.get('role','')}")
            for q in self.tech_data
        ]
        self.tech_embeddings = self.model.encode(tech_texts, convert_to_tensor=True)
        # Embeddings soft: escenario + nivel + rol
        soft_texts = [
            self._normalize(f"{q.get('scenario','')} {q.get('level','')} {q.get('role','')}")
            for q in self.soft_data
        ]
        self.soft_embeddings = self.model.encode(soft_texts, convert_to_tensor=True)
        # Embeddings de buenas respuestas: usar answers y expected de tech y soft
        good_texts = []
        self.good_answers_index = []  # ("tech"/"soft", idx, tipo)
        for i, q in enumerate(self.tech_data):
            if q.get('answer'):
                good_texts.append(self._normalize(q['answer']))
                self.good_answers_index.append(("tech", i, "answer"))
            if q.get('expected'):
                good_texts.append(self._normalize(q['expected']))
                self.good_answers_index.append(("tech", i, "expected"))
        for i, q in enumerate(self.soft_data):
            if q.get('answer'):
                good_texts.append(self._normalize(q['answer']))
                self.good_answers_index.append(("soft", i, "answer"))
            if q.get('expected'):
                good_texts.append(self._normalize(q['expected']))
                self.good_answers_index.append(("soft", i, "expected"))
        self.good_embeddings = self.model.encode(good_texts, convert_to_tensor=True) if good_texts else None

    def _load_jsonl(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def _embed_questions(self, questions):
        return self.model.encode(questions, convert_to_tensor=True)

    def find_most_similar_tech(self, user_input: str, top_k: int = 1):
        query = self._normalize(user_input)
        query_emb = self.model.encode([query], convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.tech_embeddings, top_k=top_k)[0]
        results = [self.tech_data[int(hit['corpus_id'])] for hit in hits]
        # Buscar también en buenas respuestas (answers/expected de tech y soft)
        if self.good_embeddings is not None:
            good_hits = util.semantic_search(query_emb, self.good_embeddings, top_k=1)[0]
            if good_hits and good_hits[0]['score'] > 0.7:
                src, idx, typ = self.good_answers_index[int(good_hits[0]['corpus_id'])]
                if src == "tech":
                    results.append(self.tech_data[idx])
                elif src == "soft":
                    results.append(self.soft_data[idx])
        return results

    def find_most_similar_soft(self, user_input: str, top_k: int = 1):
        query = self._normalize(user_input)
        query_emb = self.model.encode([query], convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.soft_embeddings, top_k=top_k)[0]
        return [self.soft_data[int(hit['corpus_id'])] for hit in hits]

    def refresh(self):
        """Permite recargar y re-embeddear los datos para aprendizaje continuo."""
        self.tech_data = self._load_jsonl(Path(__file__).parent / '../datasets/tech_questions.jsonl')
        self.soft_data = self._load_jsonl(Path(__file__).parent / '../datasets/soft_skills.jsonl')
        self._update_embeddings()

    def explain_selection(self, user_context, history=None, top_k=5, technical=True):
        """
        Devuelve las preguntas seleccionadas junto con una explicación de por qué fueron elegidas:
        - Similitud semántica
        - Penalización por repetición
        - Bonus por cluster
        - Cluster asignado
        """
        data = self.tech_data if technical else self.soft_data
        embeddings = self.tech_embeddings if technical else self.soft_embeddings
        context_emb = self.model.encode([self._normalize(user_context)], convert_to_tensor=True)
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=10, random_state=42)
        emb_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
        emb_umap = reducer.fit_transform(emb_np)
        if not isinstance(emb_umap, np.ndarray):
            emb_umap = np.array(emb_umap)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clusterer.fit_predict(emb_umap)
        sim_scores = util.pytorch_cos_sim(context_emb, embeddings)[0].cpu().numpy()
        penal = np.zeros(len(data))
        explanation = []
        if history:
            seen = set([h.get('question') or h.get('scenario') for h in history if 'question' in h or 'scenario' in h])
            for i, q in enumerate(data):
                qtxt = q.get('question') or q.get('scenario')
                if qtxt in seen:
                    penal[i] = -0.5
        if history:
            hist_clusters = [cluster_labels[i] for i, q in enumerate(data) if (q.get('question') or q.get('scenario')) in [h.get('question') or h.get('scenario') for h in history]]
            rare_clusters = set(np.unique(cluster_labels)) - set(hist_clusters)
            for i, cl in enumerate(cluster_labels):
                if cl in rare_clusters:
                    penal[i] += 0.2
        scores = sim_scores + penal
        probs = np.exp(scores) / np.sum(np.exp(scores))
        top_idx = np.argsort(probs)[-top_k:][::-1]
        for i in top_idx:
            q = data[i]
            qtxt = q.get('question') or q.get('scenario')
            explanation.append({
                'question': qtxt,
                'sim_score': float(sim_scores[i]),
                'penalty': float(penal[i]),
                'cluster': int(cluster_labels[i]),
                'final_score': float(scores[i]),
                'probability': float(probs[i]),
                'explanation': f"Similitud: {sim_scores[i]:.2f}, Penalización: {penal[i]:.2f}, Cluster: {cluster_labels[i]}, Score final: {scores[i]:.2f}, Probabilidad: {probs[i]:.2f}"
            })
        return explanation
