"""
RAG (Retrieval-Augmented Generation) Service
Mejora las respuestas del LLM con contexto relevante recuperado
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Contexto recuperado del knowledge base"""

    text: str
    source: str
    relevance_score: float
    metadata: Dict


@dataclass
class KnowledgeDocument:
    """Documento en el knowledge base"""

    id: str
    text: str
    role: Optional[str]
    topic: Optional[str]
    difficulty: Optional[str]
    metadata: Dict


class RAGService:
    """
    Servicio de Retrieval-Augmented Generation.

    Funcionalidad:
    1. Embeddings de documentos con Sentence Transformers
    2. Vector store con FAISS para búsqueda rápida
    3. Recuperación de top-k documentos relevantes
    4. Enriquecimiento de prompts con contexto
    5. Cache de embeddings para performance

    Benefits:
    - Respuestas más precisas y contextuales
    - Menor alucinación del LLM
    - Conocimiento actualizable sin reentrenar
    - Explicaciones con fuentes citables
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        knowledge_base_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()

        self.knowledge_base: List[KnowledgeDocument] = []
        self.index: Optional[faiss.Index] = None

        self.knowledge_base_path = knowledge_base_path
        self.index_path = index_path

        # Load existing index if available
        if index_path and index_path.exists():
            self._load_index()

    def add_documents(self, documents: List[KnowledgeDocument]) -> None:
        """
        Añade documentos al knowledge base.

        Args:
            documents: Lista de documentos a añadir
        """
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to RAG knowledge base...")

        # Generar embeddings
        texts = [doc.text for doc in documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Crear o actualizar index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)

        # Añadir embeddings al index
        self.index.add(embeddings.astype("float32"))

        # Guardar documentos
        self.knowledge_base.extend(documents)

        logger.info(f"Successfully added {len(documents)} documents. Total: {len(self.knowledge_base)}")

    def retrieve_context(
        self, query: str, top_k: int = 3, role_filter: Optional[str] = None, topic_filter: Optional[str] = None
    ) -> List[RetrievedContext]:
        """
        Recupera contexto relevante para una query.

        Args:
            query: Query del usuario
            top_k: Número de documentos a recuperar
            role_filter: Filtrar por rol (opcional)
            topic_filter: Filtrar por topic (opcional)

        Returns:
            Lista de contextos recuperados ordenados por relevancia
        """
        if self.index is None or len(self.knowledge_base) == 0:
            logger.warning("RAG knowledge base is empty, no context retrieved")
            return []

        # Generar embedding de la query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)

        # Buscar en el index (recuperar más de lo necesario para filtrar)
        search_k = min(top_k * 3, len(self.knowledge_base))
        distances, indices = self.index.search(query_embedding.astype("float32"), search_k)

        # Recuperar documentos
        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.knowledge_base):
                continue

            doc = self.knowledge_base[idx]

            # Aplicar filtros
            if role_filter and doc.role and doc.role.lower() != role_filter.lower():
                continue
            if topic_filter and doc.topic and doc.topic.lower() != topic_filter.lower():
                continue

            # Convertir distancia L2 a similarity score (0-1)
            similarity = 1 / (1 + dist)

            retrieved.append(
                RetrievedContext(
                    text=doc.text,
                    source=doc.metadata.get("source", "knowledge_base"),
                    relevance_score=float(similarity),
                    metadata=doc.metadata,
                )
            )

            if len(retrieved) >= top_k:
                break

        return retrieved

    def augment_prompt(
        self,
        base_prompt: str,
        query: str,
        role: Optional[str] = None,
        topic: Optional[str] = None,
        max_context_length: int = 1000,
    ) -> str:
        """
        Enriquece un prompt con contexto recuperado.

        Args:
            base_prompt: Prompt base
            query: Query del usuario
            role: Rol para filtrar contexto (opcional)
            topic: Topic para filtrar contexto (opcional)
            max_context_length: Longitud máxima del contexto

        Returns:
            Prompt enriquecido con contexto relevante
        """
        # Recuperar contexto
        contexts = self.retrieve_context(query=query, top_k=3, role_filter=role, topic_filter=topic)

        if not contexts:
            return base_prompt

        # Construir contexto añadido
        context_parts = []
        total_length = 0

        for ctx in contexts:
            text_with_source = f"[Fuente: {ctx.source}] {ctx.text}"

            if total_length + len(text_with_source) > max_context_length:
                break

            context_parts.append(text_with_source)
            total_length += len(text_with_source)

        if not context_parts:
            return base_prompt

        # Combinar contexto con prompt
        augmented_prompt = f"""CONTEXTO RELEVANTE RECUPERADO:
{chr(10).join(context_parts)}

---

{base_prompt}

INSTRUCCIÓN: Usa el contexto recuperado para enriquecer tu evaluación cuando sea relevante, pero no lo menciones explícitamente."""

        return augmented_prompt

    def save_index(self, path: Optional[Path] = None) -> None:
        """Guarda el índice y knowledge base en disco"""
        save_path = path or self.index_path
        if not save_path:
            logger.warning("No save path specified for RAG index")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar FAISS index
        if self.index:
            faiss.write_index(self.index, str(save_path.with_suffix(".faiss")))

        # Guardar knowledge base
        with open(save_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.knowledge_base, f)

        logger.info(f"RAG index saved to {save_path}")

    def _load_index(self) -> None:
        """Carga el índice desde disco"""
        if not self.index_path or not self.index_path.exists():
            return

        try:
            # Cargar FAISS index
            faiss_path = self.index_path.with_suffix(".faiss")
            if faiss_path.exists():
                self.index = faiss.read_index(str(faiss_path))

            # Cargar knowledge base
            pkl_path = self.index_path.with_suffix(".pkl")
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    self.knowledge_base = pickle.load(f)

            logger.info(f"RAG index loaded: {len(self.knowledge_base)} documents")

        except Exception as e:
            logger.error(f"Error loading RAG index: {e}")
            self.index = None
            self.knowledge_base = []


def build_knowledge_base_from_dataset(
    dataset_path: Path, include_best_practices: bool = True
) -> List[KnowledgeDocument]:
    """
    Construye knowledge base desde el dataset de preguntas.

    Args:
        dataset_path: Path al dataset JSONL
        include_best_practices: Si incluir best practices adicionales

    Returns:
        Lista de documentos para el knowledge base
    """
    import json

    documents = []

    # Leer dataset
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # Crear documento con contexto de la pregunta
                doc_text = f"""
Pregunta: {data.get('question', '')}
Topic: {data.get('topic', '')}
Dificultad: {data.get('difficulty', '')}
Conceptos esperados: {', '.join(data.get('expected_concepts', []))}
"""

                # Añadir hints si existen
                if "hints" in data:
                    hints = data["hints"]
                    doc_text += f"\nPistas: {hints.get('level_3', '')}"

                doc = KnowledgeDocument(
                    id=data.get("id", ""),
                    text=doc_text.strip(),
                    role=data.get("role"),
                    topic=data.get("topic"),
                    difficulty=data.get("difficulty"),
                    metadata={
                        "source": "interview_dataset",
                        "category": data.get("category", ""),
                        "keywords": data.get("keywords", []),
                    },
                )

                documents.append(doc)

    # Añadir best practices (mock - en producción desde DB o archivos)
    if include_best_practices:
        best_practices = [
            KnowledgeDocument(
                id="bp_solid",
                text="""SOLID Principles: Single Responsibility (una clase, una razón para cambiar), 
Open/Closed (abierto a extensión, cerrado a modificación), Liskov Substitution (subtipos deben 
ser reemplazables), Interface Segregation (interfaces específicas mejor que generales), 
Dependency Inversion (depender de abstracciones).""",
                role="Software Engineer",
                topic="Design Patterns",
                difficulty="mid",
                metadata={"source": "best_practices", "category": "principles"},
            ),
            KnowledgeDocument(
                id="bp_testing",
                text="""Testing Best Practices: Usar el patrón AAA (Arrange-Act-Assert), escribir tests 
independientes y reproducibles, cubrir edge cases, usar mocks para dependencias externas, 
mantener tests rápidos y simples.""",
                role="Software Engineer",
                topic="Testing",
                difficulty="mid",
                metadata={"source": "best_practices", "category": "testing"},
            ),
        ]

        documents.extend(best_practices)

    return documents


# Factory
_rag_service = None


def get_rag_service(initialize: bool = True) -> RAGService:
    """Obtiene instancia singleton del RAG service"""
    global _rag_service

    if _rag_service is None:
        from app.config import settings

        index_path = Path(settings.DATA_PATH) / "rag_index"
        _rag_service = RAGService(index_path=index_path)

        # Inicializar con dataset si es la primera vez
        if initialize and len(_rag_service.knowledge_base) == 0:
            logger.info("Initializing RAG knowledge base...")

            # Construir desde datasets
            dataset_path = Path(settings.DATA_PATH) / "datasets" / "massive_questions_dataset.jsonl"
            if dataset_path.exists():
                docs = build_knowledge_base_from_dataset(dataset_path)
                _rag_service.add_documents(docs)
                _rag_service.save_index()

    return _rag_service
