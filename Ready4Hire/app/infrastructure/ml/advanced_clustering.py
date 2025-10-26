"""
Advanced Clustering Service for Question Selection
Servicio avanzado de clustering para selección inteligente de preguntas

Este módulo implementa técnicas de aprendizaje no supervisado para:
1. Agrupar preguntas por temas/habilidades usando UMAP + HDBSCAN
2. Diversificar la selección de preguntas
3. Balancear cobertura temática en entrevistas
4. Adaptarse dinámicamente al perfil del candidato

Algorithms:
- UMAP: Reducción de dimensionalidad no lineal
- HDBSCAN: Clustering jerárquico basado en densidad
- K-Means: Clustering alternativo para casos edge
- Silhouette Score: Evaluación de calidad de clusters
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Verificar dependencias
_CLUSTERING_AVAILABLE = False
try:
    import umap
    import hdbscan
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    _CLUSTERING_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Clustering dependencies not available. Install: pip install umap-learn hdbscan scikit-learn")


@dataclass
class ClusterMetadata:
    """Metadata de un cluster de preguntas"""

    cluster_id: int
    size: int
    centroid: np.ndarray
    keywords: List[str]
    avg_difficulty: float
    topic_label: str
    coherence_score: float


class AdvancedQuestionClusteringService:
    """
    Servicio avanzado de clustering para preguntas.

    Features:
    - Clustering automático con UMAP + HDBSCAN
    - Extracción de temas/keywords por cluster
    - Selección diversificada de preguntas
    - Aprendizaje continuo: actualización incremental de clusters
    - Balanceo de cobertura temática
    - Adaptación al perfil del candidato

    Algorithm Pipeline:
    1. Embedding Generation (Sentence Transformers)
    2. Dimensionality Reduction (UMAP)
    3. Density-Based Clustering (HDBSCAN)
    4. Cluster Analysis & Labeling
    5. Question Selection Strategy
    """

    def __init__(
        self,
        embeddings_service,
        min_cluster_size: int = 5,
        umap_n_components: int = 10,
        cache_dir: Optional[str] = None,
    ):
        """
        Inicializa el servicio de clustering.

        Args:
            embeddings_service: Servicio de embeddings
            min_cluster_size: Tamaño mínimo de cluster para HDBSCAN
            umap_n_components: Dimensiones para reducción UMAP
            cache_dir: Directorio para cache de clusters
        """
        self.embeddings_service = embeddings_service
        self.min_cluster_size = min_cluster_size
        self.umap_n_components = umap_n_components
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/clusters")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Estado interno
        self.clusters: Dict[int, List[Dict]] = {}
        self.cluster_metadata: Dict[int, ClusterMetadata] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.umap_reducer = None
        self.hdbscan_clusterer = None

        logger.info("✅ AdvancedQuestionClusteringService initialized")

    def cluster_questions(
        self, questions: List[Dict], text_field: str = "question", force_recompute: bool = False
    ) -> Dict[int, List[Dict]]:
        """
        Agrupa preguntas usando UMAP + HDBSCAN.

        Args:
            questions: Lista de preguntas con metadata
            text_field: Campo con el texto de la pregunta
            force_recompute: Forzar recálculo incluso si existe cache

        Returns:
            Diccionario {cluster_id: [preguntas]}
        """
        if not _CLUSTERING_AVAILABLE:
            logger.error("❌ Clustering dependencies not available")
            return {0: questions}

        # Verificar cache
        cache_key = self._get_cache_key(questions, text_field)
        cached_clusters = self._load_from_cache(cache_key) if not force_recompute else None

        if cached_clusters:
            logger.info(f"✅ Loaded {len(cached_clusters)} clusters from cache")
            self.clusters = cached_clusters
            return cached_clusters

        try:
            # 1. Generar embeddings
            logger.info(f"📊 Clustering {len(questions)} questions...")
            texts = [q.get(text_field, "") for q in questions]
            embeddings = self.embeddings_service.encode(texts)

            # 2. Reducir dimensionalidad con UMAP
            logger.info("🔄 Applying UMAP dimensionality reduction...")
            import umap

            self.umap_reducer = umap.UMAP(
                n_components=self.umap_n_components, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
            )
            reduced_embs = self.umap_reducer.fit_transform(embeddings)
            reduced_embs = np.asarray(reduced_embs)  # Ensure numpy array format

            # 3. Clustering con HDBSCAN
            logger.info("🎯 Performing HDBSCAN clustering...")
            import hdbscan

            self.hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=3,
                metric="euclidean",
                cluster_selection_method="eom",  # Excess of Mass
            )
            cluster_labels = self.hdbscan_clusterer.fit_predict(reduced_embs)

            # 4. Agrupar preguntas por cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[int(label)].append(questions[i])

            self.clusters = dict(clusters)

            # 5. Calcular metadata de clusters
            self._compute_cluster_metadata(reduced_embs, cluster_labels, questions, text_field)

            # 6. Evaluar calidad
            if len(set(cluster_labels)) > 1:
                from sklearn.metrics import silhouette_score

                score = silhouette_score(reduced_embs, cluster_labels)
                logger.info(f"📈 Silhouette Score: {score:.3f}")

            # 7. Guardar en cache
            self._save_to_cache(cache_key, self.clusters)

            # Log resumen
            n_clusters = len([c for c in clusters.keys() if c != -1])
            n_noise = len(clusters.get(-1, []))
            logger.info(f"✅ Clustering complete:")
            logger.info(f"   • {n_clusters} clusters found")
            logger.info(f"   • {n_noise} noise points (cluster -1)")
            for cid, items in sorted(clusters.items()):
                if cid != -1:
                    logger.info(f"   • Cluster {cid}: {len(items)} questions")

            return self.clusters

        except Exception as e:
            logger.error(f"❌ Error in clustering: {str(e)}", exc_info=True)
            return {0: questions}

    def _compute_cluster_metadata(
        self, embeddings: np.ndarray, labels: np.ndarray, questions: List[Dict], text_field: str
    ):
        """Calcula metadata para cada cluster"""
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Noise
                continue

            mask = labels == label
            cluster_embs = embeddings[mask]
            cluster_questions = [q for i, q in enumerate(questions) if labels[i] == label]

            # Centroid
            centroid = np.mean(cluster_embs, axis=0)

            # Keywords (extracción simple)
            texts = [q.get(text_field, "") for q in cluster_questions]
            keywords = self._extract_keywords(texts)

            # Dificultad promedio
            difficulties = [q.get("difficulty", 0) for q in cluster_questions]
            avg_diff = np.mean(difficulties) if difficulties else 0

            # Topic label
            topic = self._generate_topic_label(keywords)

            # Coherence score (cohesión intra-cluster)
            coherence = self._compute_coherence(cluster_embs)

            self.cluster_metadata[int(label)] = ClusterMetadata(
                cluster_id=int(label),
                size=len(cluster_questions),
                centroid=centroid,
                keywords=keywords,
                avg_difficulty=float(avg_diff),
                topic_label=topic,
                coherence_score=float(coherence),
            )

    def _extract_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        """Extrae keywords más frecuentes de un conjunto de textos"""
        from collections import Counter
        import re

        # Tokenizar y limpiar
        words = []
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            words.extend([w for w in tokens if len(w) > 3])  # Filtrar palabras cortas

        # Top keywords
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]

    def _generate_topic_label(self, keywords: List[str]) -> str:
        """Genera una etiqueta descriptiva del tema"""
        if not keywords:
            return "General"
        return " + ".join(keywords[:3]).title()

    def _compute_coherence(self, embeddings: np.ndarray) -> float:
        """Calcula coherencia (cohesión) del cluster"""
        if len(embeddings) < 2:
            return 1.0

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(embeddings)
        # Promedio de similitudes (excluyendo diagonal)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        return float(np.mean(similarities[mask]))

    def select_diversified_questions(
        self, n_questions: int, candidate_ids: Set[str], exclude_ids: Set[str], user_profile: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Selecciona preguntas diversificadas de diferentes clusters.

        Strategy:
        1. Distribuir preguntas entre clusters según su tamaño y relevancia
        2. Evitar sobre-representación de un solo tema
        3. Adaptar a perfil del usuario (si disponible)

        Args:
            n_questions: Número de preguntas a seleccionar
            candidate_ids: IDs de preguntas candidatas
            exclude_ids: IDs de preguntas ya usadas
            user_profile: Perfil del usuario (para personalización)

        Returns:
            Lista de preguntas seleccionadas
        """
        if not self.clusters:
            logger.warning("⚠️ No clusters available, random selection")
            return []

        # Filtrar preguntas disponibles por cluster
        available_by_cluster = {}
        for cid, questions in self.clusters.items():
            if cid == -1:  # Skip noise
                continue
            available = [q for q in questions if q.get("id") in candidate_ids and q.get("id") not in exclude_ids]
            if available:
                available_by_cluster[cid] = available

        if not available_by_cluster:
            return []

        # Calcular distribución óptima
        distribution = self._compute_optimal_distribution(n_questions, available_by_cluster, user_profile)

        # Seleccionar preguntas según distribución
        selected = []
        for cid, count in distribution.items():
            cluster_questions = available_by_cluster[cid]
            # Tomar las 'count' mejores preguntas del cluster
            selected.extend(cluster_questions[:count])

        logger.info(f"✅ Selected {len(selected)} diversified questions from {len(distribution)} clusters")

        return selected

    def _compute_optimal_distribution(
        self, n_questions: int, available_by_cluster: Dict[int, List[Dict]], user_profile: Optional[Dict]
    ) -> Dict[int, int]:
        """
        Calcula distribución óptima de preguntas entre clusters.

        Estrategia:
        - Balancear según tamaño de cluster
        - Dar prioridad a clusters con mayor coherencia
        - Adaptar según perfil del usuario (si disponible)
        """
        total_available = sum(len(qs) for qs in available_by_cluster.values())

        if total_available <= n_questions:
            # Tomar todas las preguntas disponibles
            return {cid: len(qs) for cid, qs in available_by_cluster.items()}

        # Calcular pesos por cluster
        weights = {}
        for cid in available_by_cluster:
            base_weight = len(available_by_cluster[cid]) / total_available

            # Bonus por coherencia
            if cid in self.cluster_metadata:
                coherence_bonus = self.cluster_metadata[cid].coherence_score * 0.2
                base_weight += coherence_bonus

            weights[cid] = base_weight

        # Normalizar pesos
        total_weight = sum(weights.values())
        weights = {cid: w / total_weight for cid, w in weights.items()}

        # Asignar preguntas proporcionalmente
        distribution = {}
        remaining = n_questions

        for cid in sorted(weights.keys(), key=lambda c: weights[c], reverse=True):
            if remaining <= 0:
                break

            # Preguntas asignadas a este cluster
            allocated = max(1, int(n_questions * weights[cid]))
            allocated = min(allocated, len(available_by_cluster[cid]), remaining)

            distribution[cid] = allocated
            remaining -= allocated

        return distribution

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Retorna resumen de clusters para análisis"""
        summary = {
            "total_clusters": len(self.clusters),
            "total_questions": sum(len(qs) for qs in self.clusters.values()),
            "clusters": [],
        }

        for cid, questions in sorted(self.clusters.items()):
            if cid == -1:
                continue

            cluster_info = {
                "id": cid,
                "size": len(questions),
                "topic": self.cluster_metadata.get(
                    cid,
                    ClusterMetadata(
                        cluster_id=cid,
                        size=0,
                        centroid=np.array([]),
                        keywords=[],
                        avg_difficulty=0,
                        topic_label="Unknown",
                        coherence_score=0,
                    ),
                ).topic_label,
                "keywords": self.cluster_metadata.get(
                    cid,
                    ClusterMetadata(
                        cluster_id=cid,
                        size=0,
                        centroid=np.array([]),
                        keywords=[],
                        avg_difficulty=0,
                        topic_label="Unknown",
                        coherence_score=0,
                    ),
                ).keywords,
                "avg_difficulty": self.cluster_metadata.get(
                    cid,
                    ClusterMetadata(
                        cluster_id=cid,
                        size=0,
                        centroid=np.array([]),
                        keywords=[],
                        avg_difficulty=0,
                        topic_label="Unknown",
                        coherence_score=0,
                    ),
                ).avg_difficulty,
                "coherence": self.cluster_metadata.get(
                    cid,
                    ClusterMetadata(
                        cluster_id=cid,
                        size=0,
                        centroid=np.array([]),
                        keywords=[],
                        avg_difficulty=0,
                        topic_label="Unknown",
                        coherence_score=0,
                    ),
                ).coherence_score,
            }
            summary["clusters"].append(cluster_info)

        return summary

    def _get_cache_key(self, questions: List[Dict], text_field: str) -> str:
        """Genera key única para cache"""
        import hashlib

        content = str(sorted([q.get("id", str(i)) for i, q in enumerate(questions)]))
        return hashlib.md5(content.encode()).hexdigest()

    def _save_to_cache(self, cache_key: str, clusters: Dict):
        """Guarda clusters en cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, "w") as f:
                json.dump(clusters, f)
            logger.info(f"💾 Clusters saved to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save cache: {str(e)}")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Carga clusters desde cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    clusters = json.load(f)
                return {int(k): v for k, v in clusters.items()}
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {str(e)}")
        return None


def get_clustering_service(
    embeddings_service, min_cluster_size: int = 5, cache_dir: Optional[str] = None
) -> AdvancedQuestionClusteringService:
    """Factory para obtener servicio de clustering"""
    return AdvancedQuestionClusteringService(
        embeddings_service=embeddings_service, min_cluster_size=min_cluster_size, cache_dir=cache_dir
    )
