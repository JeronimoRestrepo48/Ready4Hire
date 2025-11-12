"""
Qdrant Vector Database Client para búsqueda semántica ultra-rápida.
Indexa y busca preguntas usando embeddings vectoriales.
"""

import logging
from typing import List, Dict, Any, Optional
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Qdrant not available: {e}")
    QdrantClient = None
    Distance = VectorParams = PointStruct = None
    Filter = FieldCondition = MatchValue = SearchRequest = None
    QDRANT_AVAILABLE = False
from sentence_transformers import SentenceTransformer
import hashlib

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Cliente para Qdrant Vector Database.
    Proporciona búsqueda semántica ultra-rápida de preguntas.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Inicializa el cliente de Qdrant.
        
        Args:
            host: Host de Qdrant
            port: Puerto de Qdrant
            embedding_model: Modelo de embeddings de SentenceTransformers
        """
        self.client = QdrantClient(host=host, port=port)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_size = 384  # Tamaño para all-MiniLM-L6-v2
        
        # Nombres de colecciones
        self.collections = {
            "technical_questions": "technical_questions",
            "soft_skills_questions": "soft_skills_questions",
            "user_profiles": "user_profiles"
        }
        
        self._init_collections()
        logger.info(f"✅ Qdrant client initialized: {host}:{port}")
    
    def _init_collections(self) -> None:
        """Inicializa las colecciones necesarias"""
        try:
            # Collection para preguntas técnicas
            if not self.client.collection_exists(self.collections["technical_questions"]):
                self.client.create_collection(
                    collection_name=self.collections["technical_questions"],
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Created technical_questions collection")
            
            # Collection para preguntas de soft skills
            if not self.client.collection_exists(self.collections["soft_skills_questions"]):
                self.client.create_collection(
                    collection_name=self.collections["soft_skills_questions"],
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Created soft_skills_questions collection")
            
            # Collection para perfiles de usuario
            if not self.client.collection_exists(self.collections["user_profiles"]):
                self.client.create_collection(
                    collection_name=self.collections["user_profiles"],
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Created user_profiles collection")
                
        except Exception as e:
            logger.error(f"❌ Error initializing collections: {e}")
            raise
    
    async def index_questions(
        self,
        questions: List[Dict[str, Any]],
        category: str = "technical"
    ) -> int:
        """
        Indexa una lista de preguntas en Qdrant.
        
        Args:
            questions: Lista de preguntas con estructura:
                {
                    "id": str,
                    "text": str,
                    "role": str,
                    "difficulty": str,
                    "category": str,
                    "topic": str,
                    "expected_concepts": list,
                    "keywords": list
                }
            category: Categoría de preguntas (technical o soft_skills)
            
        Returns:
            Número de preguntas indexadas
        """
        try:
            collection_name = (
                self.collections["technical_questions"]
                if category == "technical"
                else self.collections["soft_skills_questions"]
            )
            
            logger.info(f"🔄 Indexing {len(questions)} {category} questions...")
            
            # Generar embeddings en batch
            texts = [q["text"] for q in questions]
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Crear points para Qdrant
            points = []
            for i, question in enumerate(questions):
                point_id = self._generate_point_id(question["id"])
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload={
                        "question_id": question["id"],
                        "text": question["text"],
                        "role": question.get("role", ""),
                        "difficulty": question.get("difficulty", "mid"),
                        "category": question.get("category", category),
                        "topic": question.get("topic", ""),
                        "expected_concepts": question.get("expected_concepts", []),
                        "keywords": question.get("keywords", [])
                    }
                ))
            
            # Upsert en Qdrant (batch)
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"✅ Indexed {len(points)} questions")
            return len(points)
            
        except Exception as e:
            logger.error(f"❌ Error indexing questions: {e}")
            raise
    
    async def search_similar_questions(
        self,
        query_text: str,
        role: Optional[str] = None,
        difficulty: Optional[str] = None,
        category: str = "technical",
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Busca preguntas similares usando búsqueda vectorial.
        
        Args:
            query_text: Texto de consulta (puede ser contexto del usuario, respuesta anterior, etc)
            role: Filtrar por rol específico
            difficulty: Filtrar por dificultad
            category: Categoría de preguntas
            limit: Número máximo de resultados
            score_threshold: Score mínimo de similitud (0-1)
            
        Returns:
            Lista de preguntas similares con scores
        """
        try:
            collection_name = (
                self.collections["technical_questions"]
                if category == "technical"
                else self.collections["soft_skills_questions"]
            )
            
            # Generar embedding del query
            query_embedding = self.embedding_model.encode(
                query_text,
                convert_to_numpy=True
            )
            
            # Construir filtros
            filter_conditions = []
            if role:
                filter_conditions.append(
                    FieldCondition(
                        key="role",
                        match=MatchValue(value=role)
                    )
                )
            if difficulty:
                filter_conditions.append(
                    FieldCondition(
                        key="difficulty",
                        match=MatchValue(value=difficulty)
                    )
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Búsqueda en Qdrant
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Formatear resultados
            results = []
            for hit in search_results:
                results.append({
                    "question_id": hit.payload["question_id"],
                    "text": hit.payload["text"],
                    "role": hit.payload["role"],
                    "difficulty": hit.payload["difficulty"],
                    "topic": hit.payload["topic"],
                    "expected_concepts": hit.payload["expected_concepts"],
                    "similarity_score": hit.score
                })
            
            logger.info(f"✅ Found {len(results)} similar questions (threshold={score_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching questions: {e}")
            return []
    
    async def find_questions_by_user_context(
        self,
        user_skills: List[str],
        user_interests: List[str],
        target_role: str,
        difficulty: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Encuentra preguntas personalizadas basadas en el contexto del usuario.
        
        Args:
            user_skills: Skills actuales del usuario
            user_interests: Intereses del usuario
            target_role: Rol objetivo
            difficulty: Nivel de dificultad
            limit: Número de preguntas
            
        Returns:
            Lista de preguntas personalizadas
        """
        try:
            # Crear query text combinando contexto
            context_text = f"""
            User skills: {', '.join(user_skills)}
            Interests: {', '.join(user_interests)}
            Target role: {target_role}
            """
            
            # Buscar preguntas similares
            questions = await self.search_similar_questions(
                query_text=context_text,
                role=target_role,
                difficulty=difficulty,
                category="technical",
                limit=limit,
                score_threshold=0.6
            )
            
            logger.info(f"✅ Found {len(questions)} personalized questions")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Error finding personalized questions: {e}")
            return []
    
    def _generate_point_id(self, question_id: str) -> int:
        """
        Genera un ID numérico único para Qdrant a partir del question_id.
        
        Args:
            question_id: ID de la pregunta (string)
            
        Returns:
            ID numérico para Qdrant
        """
        # Usar hash MD5 y convertir a int
        hash_object = hashlib.md5(question_id.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        # Limitar a 64 bits para Qdrant
        return hash_int % (2**63 - 1)
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de una colección.
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Dict con estadísticas
        """
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {}
    
    async def delete_question(self, question_id: str, category: str = "technical") -> bool:
        """
        Elimina una pregunta del índice.
        
        Args:
            question_id: ID de la pregunta
            category: Categoría
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            collection_name = (
                self.collections["technical_questions"]
                if category == "technical"
                else self.collections["soft_skills_questions"]
            )
            
            point_id = self._generate_point_id(question_id)
            self.client.delete(
                collection_name=collection_name,
                points_selector=[point_id]
            )
            
            logger.info(f"✅ Deleted question {question_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting question: {e}")
            return False


# Factory para obtener instancia global
_qdrant_instance: Optional[QdrantVectorStore] = None


def get_qdrant_client(
    host: str = "localhost",
    port: int = 6333
) -> QdrantVectorStore:
    """
    Factory para obtener instancia del cliente Qdrant.
    Usa singleton pattern para reutilizar conexión.
    """
    global _qdrant_instance
    
    if _qdrant_instance is None:
        _qdrant_instance = QdrantVectorStore(host=host, port=port)
        logger.info("⚡ Qdrant client initialized")
    
    return _qdrant_instance

