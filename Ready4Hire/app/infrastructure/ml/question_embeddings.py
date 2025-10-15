"""
Question Embeddings and Ranking Service
Servicio de embeddings y ranking de preguntas usando SentenceTransformers y RankNet
"""
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Control de recursos
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Verificar dependencias
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_CLUSTERING_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ sentence-transformers not available")

try:
    import umap
    import hdbscan
    _CLUSTERING_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ umap-learn or hdbscan not available")


class RankNet(nn.Module):
    """
    Red neuronal para ranking de preguntas.
    
    Arquitectura:
    - Input: Embeddings concatenados (contexto + pregunta)
    - Hidden: 64 -> 32 -> 1 (score)
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class QuestionEmbeddingsService:
    """
    Servicio de embeddings y ranking de preguntas.
    
    Features:
    - Embeddings con SentenceTransformers
    - Búsqueda semántica de preguntas similares
    - Ranking con RankNet (opcional)
    - Clustering temático (UMAP + HDBSCAN)
    - Penalización por repetición
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        ranknet_model_path: Optional[str] = None
    ):
        """
        Inicializa el servicio de embeddings.
        
        Args:
            model_name: Nombre del modelo SentenceTransformers
            ranknet_model_path: Ruta al modelo RankNet entrenado (opcional)
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.ranknet_model: Optional[RankNet] = None
        self.ranknet_model_path = ranknet_model_path
        self._initialized = False
    
    def _ensure_initialized(self):
        """Asegura que el modelo esté cargado"""
        if self._initialized:
            return
        
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers no está disponible. "
                "Instala con: pip install sentence-transformers"
            )
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Modelo {self.model_name} cargado")
            
            # Cargar RankNet si está disponible
            if self.ranknet_model_path and Path(self.ranknet_model_path).exists():
                self._load_ranknet()
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def _load_ranknet(self):
        """Carga el modelo RankNet entrenado"""
        try:
            # Cargar un ejemplo para determinar input_dim
            # En producción, esto debería estar en metadata
            input_dim = 768  # Default para all-MiniLM-L6-v2
            
            self.ranknet_model = RankNet(input_dim)
            assert self.ranknet_model_path is not None
            self.ranknet_model.load_state_dict(
                torch.load(self.ranknet_model_path, map_location='cpu')
            )
            self.ranknet_model.eval()
            
            logger.info(f"✅ RankNet cargado desde {self.ranknet_model_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar RankNet: {str(e)}")
            self.ranknet_model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos
        
        Returns:
            Array de embeddings (shape: [n_texts, embedding_dim])
        """
        self._ensure_initialized()
        assert self.model is not None, "Model should be initialized"
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"❌ Error generando embeddings: {str(e)}")
            raise
    
    def find_similar(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        text_field: str = 'question'
    ) -> List[Tuple[Dict, float]]:
        """
        Encuentra preguntas similares usando búsqueda semántica.
        
        Args:
            query: Texto de consulta (contexto del usuario)
            candidates: Lista de preguntas candidatas
            top_k: Número de resultados
            text_field: Campo del dict con el texto a comparar
        
        Returns:
            Lista de tuplas (pregunta, similarity_score)
        """
        self._ensure_initialized()
        
        # Generar embedding del query
        query_emb = self.encode([query])[0]
        
        # Generar embeddings de candidatos
        candidate_texts = [c.get(text_field, '') for c in candidates]
        candidate_embs = self.encode(candidate_texts)
        
        # Calcular similitud coseno
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_emb.reshape(1, -1), candidate_embs)[0]
        
        # Ordenar por similitud
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (candidates[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results
    
    def rank_with_ranknet(
        self,
        user_context: str,
        candidates: List[Dict],
        text_field: str = 'question'
    ) -> List[Tuple[Dict, float]]:
        """
        Rankea preguntas usando RankNet.
        
        Args:
            user_context: Contexto del usuario
            candidates: Lista de preguntas candidatas
            text_field: Campo con el texto
        
        Returns:
            Lista de tuplas (pregunta, score) ordenadas por score
        """
        if not self.ranknet_model:
            logger.warning("⚠️ RankNet no disponible, usando similitud básica")
            return self.find_similar(user_context, candidates, len(candidates), text_field)
        
        self._ensure_initialized()
        
        try:
            # Embeddings
            context_emb = self.encode([user_context])[0]
            candidate_texts = [c.get(text_field, '') for c in candidates]
            candidate_embs = self.encode(candidate_texts)
            
            # Concatenar contexto + pregunta
            features = np.array([
                np.concatenate([context_emb, q_emb])
                for q_emb in candidate_embs
            ])
            
            # Predecir scores
            with torch.no_grad():
                scores = self.ranknet_model(
                    torch.tensor(features, dtype=torch.float32)
                ).numpy().flatten()
            
            # Ordenar por score
            top_indices = np.argsort(scores)[::-1]
            
            results = [
                (candidates[i], float(scores[i]))
                for i in top_indices
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en RankNet: {str(e)}")
            # Fallback a similitud básica
            return self.find_similar(user_context, candidates, len(candidates), text_field)
    
    def cluster_questions(
        self,
        questions: List[Dict],
        text_field: str = 'question',
        min_cluster_size: int = 3
    ) -> Dict[int, List[Dict]]:
        """
        Agrupa preguntas por temas usando UMAP + HDBSCAN.
        
        Args:
            questions: Lista de preguntas
            text_field: Campo con el texto
            min_cluster_size: Tamaño mínimo de cluster
        
        Returns:
            Diccionario {cluster_id: [preguntas]}
        """
        if not _CLUSTERING_AVAILABLE:
            logger.warning("⚠️ Clustering no disponible")
            return {0: questions}  # Un solo cluster
        
        self._ensure_initialized()
        
        try:
            import umap
            import hdbscan
            
            # Generar embeddings
            texts = [q.get(text_field, '') for q in questions]
            embeddings = self.encode(texts)
            
            # Reducir dimensionalidad con UMAP
            reducer = umap.UMAP(n_components=10, random_state=42)
            reduced_embs = np.asarray(reducer.fit_transform(embeddings))
            
            # Clustering con HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean'
            )
            cluster_labels = clusterer.fit_predict(reduced_embs)
            
            # Agrupar por cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(questions[i])
            
            logger.info(f"✅ {len(clusters)} clusters encontrados")
            
            return clusters
            
        except Exception as e:
            logger.error(f"❌ Error en clustering: {str(e)}")
            return {0: questions}
    
    def is_available(self) -> bool:
        """Verifica si el servicio está disponible"""
        try:
            self._ensure_initialized()
            return True
        except Exception:
            return False


# Instancia global (singleton)
_embeddings_service = None


def get_embeddings_service(
    model_name: str = 'all-MiniLM-L6-v2',
    ranknet_model_path: Optional[str] = None
) -> QuestionEmbeddingsService:
    """Obtiene la instancia global del servicio de embeddings"""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = QuestionEmbeddingsService(model_name, ranknet_model_path)
    return _embeddings_service
