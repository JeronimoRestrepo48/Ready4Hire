import os
# Limitar hilos de numpy/OpenBLAS/MKL y PyTorch para evitar sobrecarga de CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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
