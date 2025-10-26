"""
RAG (Retrieval-Augmented Generation) API Routes
Endpoints para gestión de knowledge base y búsqueda semántica
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging

from app.infrastructure.embeddings.rag_service import (
    get_rag_service,
    build_knowledge_base_from_dataset,
    KnowledgeDocument,
)
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/rag", tags=["RAG"])


# ============================================================================
# DTOs
# ============================================================================


class SearchRequest(BaseModel):
    """Request para búsqueda semántica"""

    query: str = Field(..., description="Query de búsqueda")
    top_k: int = Field(3, ge=1, le=10, description="Número de resultados")
    role_filter: Optional[str] = Field(None, description="Filtrar por rol")
    topic_filter: Optional[str] = Field(None, description="Filtrar por topic")


class SearchResult(BaseModel):
    """Resultado de búsqueda"""

    text: str
    source: str
    relevance_score: float
    metadata: Dict


class SearchResponse(BaseModel):
    """Respuesta de búsqueda"""

    results: List[SearchResult]
    query: str
    total_found: int


class AddDocumentRequest(BaseModel):
    """Request para añadir documento"""

    text: str = Field(..., description="Texto del documento")
    role: Optional[str] = Field(None, description="Rol asociado")
    topic: Optional[str] = Field(None, description="Topic asociado")
    difficulty: Optional[str] = Field(None, description="Dificultad")
    metadata: Dict = Field(default_factory=dict)


class KnowledgeBaseStats(BaseModel):
    """Estadísticas del knowledge base"""

    total_documents: int
    indexed: bool
    roles_covered: List[str]
    topics_covered: List[str]


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
    """
    Búsqueda semántica en el knowledge base.

    Usa embeddings para encontrar contexto relevante.
    """
    try:
        rag = get_rag_service(initialize=True)

        contexts = rag.retrieve_context(
            query=request.query, top_k=request.top_k, role_filter=request.role_filter, topic_filter=request.topic_filter
        )

        results = [
            SearchResult(text=ctx.text, source=ctx.source, relevance_score=ctx.relevance_score, metadata=ctx.metadata)
            for ctx in contexts
        ]

        return SearchResponse(results=results, query=request.query, total_found=len(results))

    except Exception as e:
        logger.error(f"Error in RAG search: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Search failed: {str(e)}")


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def add_document(request: AddDocumentRequest):
    """
    Añade un documento al knowledge base.

    El documento será indexado y disponible para búsqueda.
    """
    try:
        rag = get_rag_service(initialize=True)

        # Crear documento
        doc = KnowledgeDocument(
            id=f"custom_{len(rag.knowledge_base)}",
            text=request.text,
            role=request.role,
            topic=request.topic,
            difficulty=request.difficulty,
            metadata=request.metadata,
        )

        # Añadir al knowledge base
        rag.add_documents([doc])

        # Guardar índice actualizado
        rag.save_index()

        logger.info(f"✅ Document added: {doc.id}")

        return {
            "message": "Document added successfully",
            "document_id": doc.id,
            "total_documents": len(rag.knowledge_base),
        }

    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to add document: {str(e)}"
        )


@router.get("/stats", response_model=KnowledgeBaseStats)
async def get_knowledge_base_stats():
    """
    Obtiene estadísticas del knowledge base.
    """
    try:
        rag = get_rag_service(initialize=True)

        # Obtener roles y topics únicos
        roles = set()
        topics = set()

        for doc in rag.knowledge_base:
            if doc.role:
                roles.add(doc.role)
            if doc.topic:
                topics.add(doc.topic)

        return KnowledgeBaseStats(
            total_documents=len(rag.knowledge_base),
            indexed=rag.index is not None,
            roles_covered=sorted(list(roles)),
            topics_covered=sorted(list(topics)),
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get stats: {str(e)}")


@router.post("/rebuild-index", status_code=status.HTTP_200_OK)
async def rebuild_index():
    """
    Reconstruye el índice del knowledge base desde el dataset.

    ⚠️ Operación costosa - usar solo cuando sea necesario.
    """
    try:
        from app.config import settings

        rag = get_rag_service(initialize=False)

        # Construir desde dataset
        dataset_path = Path(settings.DATA_PATH) / "datasets" / "massive_questions_dataset.jsonl"

        if not dataset_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset file not found")

        logger.info(f"🔄 Rebuilding index from {dataset_path}")

        docs = build_knowledge_base_from_dataset(dataset_path)
        rag.add_documents(docs)
        rag.save_index()

        logger.info(f"✅ Index rebuilt: {len(docs)} documents")

        return {"message": "Index rebuilt successfully", "documents_indexed": len(docs)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to rebuild index: {str(e)}"
        )


@router.delete("/index", status_code=status.HTTP_200_OK)
async def clear_index():
    """
    Limpia el índice del knowledge base.

    ⚠️ Esta operación es destructiva.
    """
    try:
        rag = get_rag_service(initialize=False)

        rag.knowledge_base = []
        rag.index = None

        logger.warning("⚠️ RAG index cleared")

        return {"message": "Index cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to clear index: {str(e)}"
        )
