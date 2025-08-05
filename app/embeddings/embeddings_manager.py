# embeddings_manager.py
"""
Módulo para gestión de embeddings y búsqueda semántica en entrevistas.
Utiliza SentenceTransformers para generar y comparar embeddings de preguntas y respuestas.
"""
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import json

class EmbeddingsManager:
    def update_from_feedback(self, history):
        """
        Aprende de las respuestas buenas del historial del usuario y actualiza los embeddings.
        Si el usuario respondió correctamente a una pregunta, la agrega a good_answers.jsonl y actualiza embeddings.
        """
        from pathlib import Path
        import json
        good_path = Path(__file__).parent / '../datasets/good_answers.jsonl'
        new_entries = []
        for h in history:
            if 'agent' in h and 'user' in h:
                entry = {"question": h['agent'], "answer": h['user']}
                new_entries.append(entry)
        if new_entries:
            with open(good_path, 'a', encoding='utf-8') as f:
                for entry in new_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            # Recargar good_data y embeddings
            self.good_data = self._load_jsonl(good_path)
            self._update_embeddings()
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.tech_data = self._load_jsonl(Path(__file__).parent / '../datasets/tech_questions.jsonl')
        self.soft_data = self._load_jsonl(Path(__file__).parent / '../datasets/soft_skills.jsonl')
        self.good_data = self._load_jsonl(Path(__file__).parent / '../datasets/good_answers.jsonl') if (Path(__file__).parent / '../datasets/good_answers.jsonl').exists() else []
        self._update_embeddings()

    def _normalize(self, text: str) -> str:
        return text.lower().strip()

    def _update_embeddings(self):
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
        # Embeddings de buenas respuestas
        if self.good_data:
            good_texts = [self._normalize(f"{g.get('question','')} {g.get('role','')}") for g in self.good_data]
            self.good_embeddings = self.model.encode(good_texts, convert_to_tensor=True)
        else:
            self.good_embeddings = None

    def _load_jsonl(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def _embed_questions(self, questions):
        return self.model.encode(questions, convert_to_tensor=True)

    def find_most_similar_tech(self, user_input: str, top_k: int = 1):
        query = self._normalize(user_input)
        query_emb = self.model.encode([query], convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.tech_embeddings, top_k=top_k)[0]
        # También buscar en buenas respuestas si existen
        results = [self.tech_data[int(hit['corpus_id'])] for hit in hits]
        if self.good_embeddings is not None:
            good_hits = util.semantic_search(query_emb, self.good_embeddings, top_k=1)[0]
            if good_hits and good_hits[0]['score'] > 0.7:
                results.append(self.good_data[int(good_hits[0]['corpus_id'])])
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
        self.good_data = self._load_jsonl(Path(__file__).parent / '../datasets/good_answers.jsonl') if (Path(__file__).parent / '../datasets/good_answers.jsonl').exists() else []
        self._update_embeddings()
