"""
ML Adapter Layer - Interfaz simplificada para módulos ML

Este adapter provee una interfaz unificada y robusta para:
- Advanced Clustering
- Continuous Learning
- Question Embeddings

Características principales:
1. Inicialización lazy de ML modules
2. Error handling robusto con fallbacks
3. Logging detallado para debugging
4. Interfaz simple y limpia
5. Degradación graceful si ML falla

Author: Jeronimo Restrepo Angel
Date: 2025-10-15
Version: 1.0
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from app.domain.entities.question import Question

logger = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """Configuración para ML Adapter."""

    enable_clustering: bool = True
    enable_continuous_learning: bool = True
    enable_embeddings: bool = True
    clustering_min_size: int = 5
    umap_components: int = 10
    mab_exploration_rate: float = 0.1
    fallback_on_error: bool = True


class MLAdapter:
    """
    Adapter para módulos ML avanzados.

    Este adapter encapsula la complejidad de los módulos ML
    y provee una interfaz simple para el resto del sistema.

    Example:
        ```python
        adapter = MLAdapter(embeddings_service=emb_service)

        # Seleccionar pregunta con ML
        question = adapter.select_question_ml(
            candidates=questions,
            previous_ids=prev_ids,
            strategy='balanced'
        )

        # Actualizar performance
        adapter.update_performance(
            question_id='q123',
            score=7.5,
            time_taken=60
        )
        ```
    """

    def __init__(self, embeddings_service=None, config: Optional[MLConfig] = None):
        """
        Inicializa el adapter ML.

        Args:
            embeddings_service: Servicio de embeddings (opcional)
            config: Configuración ML (usa defaults si None)
        """
        self.config = config or MLConfig()
        self.embeddings_service = embeddings_service

        # ML modules (lazy initialization)
        self._clustering_service = None
        self._learning_system = None

        # Status flags
        self._clustering_available = False
        self._learning_available = False
        self._embeddings_available = embeddings_service is not None

        logger.info(
            f"MLAdapter inicializado: "
            f"clustering={self.config.enable_clustering}, "
            f"learning={self.config.enable_continuous_learning}, "
            f"embeddings={self._embeddings_available}"
        )

    def select_question_ml(
        self,
        candidates: List[Question],
        previous_ids: List[str],
        strategy: str = "balanced",
        interview_context: Optional[Dict] = None,
    ) -> Optional[Question]:
        """
        Selecciona pregunta usando ML avanzado.

        Aplica (en orden):
        1. Clustering para diversificación de tópicos
        2. MAB para exploración/explotación
        3. Semantic diversity si disponible
        4. Random como último recurso

        Args:
            candidates: Lista de preguntas candidatas
            previous_ids: IDs de preguntas ya hechas
            strategy: 'balanced', 'exploit', 'explore', 'adaptive'
            interview_context: Contexto de la entrevista (opcional)

        Returns:
            Pregunta seleccionada o None si no hay candidatas
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        try:
            # Estrategia 1: Clustering (si habilitado y disponible)
            if self.config.enable_clustering:
                selected = self._select_with_clustering(candidates=candidates, previous_ids=previous_ids)
                if selected:
                    logger.info(f"✅ ML: Selección por clustering: {selected.id}")
                    return selected

            # Estrategia 2: MAB (si habilitado y disponible)
            if self.config.enable_continuous_learning:
                selected = self._select_with_mab(candidates=candidates, strategy=strategy)
                if selected:
                    logger.info(f"✅ ML: Selección por MAB ({strategy}): {selected.id}")
                    return selected

            # Estrategia 3: Semantic diversity (si embeddings disponible)
            if self._embeddings_available:
                selected = self._select_by_diversity(candidates=candidates, previous_ids=previous_ids)
                if selected:
                    logger.info(f"✅ ML: Selección por diversidad: {selected.id}")
                    return selected

            # Fallback: Random
            import random

            selected = random.choice(candidates)
            logger.debug(f"⚠️ ML: Fallback a selección random: {selected.id}")
            return selected

        except Exception as e:
            logger.error(f"❌ Error en select_question_ml: {e}", exc_info=True)

            if self.config.fallback_on_error and candidates:
                import random

                return random.choice(candidates)

            return None

    def _select_with_clustering(self, candidates: List[Question], previous_ids: List[str]) -> Optional[Question]:
        """Selección usando clustering."""
        try:
            # Lazy init clustering service
            if not self._clustering_service:
                self._init_clustering_service()

            if not self._clustering_available or self._clustering_service is None:
                return None

            # Preparar datos para clustering
            questions_data = [
                {"id": q.id, "question": q.text, "difficulty": q.difficulty, "category": q.category} for q in candidates
            ]

            # Cluster preguntas
            clusters = self._clustering_service.cluster_questions(questions=questions_data, text_field="question")

            if not clusters or len(clusters) == 0:
                logger.debug("No se encontraron clusters válidos")
                return None

            # Seleccionar de cluster menos visitado
            # Simplificación: por ahora seleccionar del cluster más grande
            largest_cluster_id = max(clusters.keys(), key=lambda k: len(clusters[k]))
            cluster_questions = clusters[largest_cluster_id]

            if cluster_questions:
                # Seleccionar primera pregunta del cluster
                selected_data = cluster_questions[0]
                selected_id = selected_data.get("id")

                # Encontrar Question object
                for q in candidates:
                    if q.id == selected_id:
                        return q

            return None

        except Exception as e:
            logger.error(f"Error en clustering selection: {e}", exc_info=True)
            return None

    def _select_with_mab(self, candidates: List[Question], strategy: str) -> Optional[Question]:
        """Selección usando Multi-Armed Bandits."""
        try:
            # Lazy init learning system
            if not self._learning_system:
                self._init_learning_system()

            if not self._learning_available:
                return None

            # Por ahora, implementación simplificada
            # TODO: Implementar selección MAB completa
            logger.debug("MAB selection not fully implemented yet")
            return None

        except Exception as e:
            logger.error(f"Error en MAB selection: {e}", exc_info=True)
            return None

    def _select_by_diversity(self, candidates: List[Question], previous_ids: List[str]) -> Optional[Question]:
        """Selección por diversidad semántica."""
        try:
            if not self.embeddings_service or not previous_ids:
                return None

            # TODO: Implementar semantic diversity selection
            # Requiere método get_embedding en embeddings_service
            logger.debug("Semantic diversity not fully implemented yet")
            return None

        except Exception as e:
            logger.error(f"Error en diversity selection: {e}", exc_info=True)
            return None

    def update_performance(
        self,
        question_id: str,
        score: float,
        time_taken: int,
        interview_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Actualiza performance de una pregunta (feedback loop).

        Esto permite que el sistema aprenda de entrevistas reales.

        Args:
            question_id: ID de la pregunta
            score: Score obtenido (0-10)
            time_taken: Tiempo en segundos
            interview_id: ID de entrevista (opcional)
            metadata: Metadata adicional (opcional)
        """
        try:
            if not self.config.enable_continuous_learning:
                return

            # Lazy init learning system
            if not self._learning_system:
                self._init_learning_system()

            if not self._learning_available:
                logger.debug("Learning system not available, skipping update")
                return

            # TODO: Implementar actualización de performance
            # Requiere método update_question_performance en learning_system
            logger.debug(f"Performance update: q={question_id}, " f"score={score}, time={time_taken}s")

        except Exception as e:
            logger.error(f"Error updating performance: {e}", exc_info=True)

    def _init_clustering_service(self) -> None:
        """Inicializa clustering service (lazy)."""
        try:
            if not self.config.enable_clustering:
                return

            # Verificar si dependencias están disponibles
            try:
                import umap
                import hdbscan
            except ImportError:
                logger.warning("⚠️ Clustering dependencies not available. " "Install: pip install umap-learn hdbscan")
                self._clustering_available = False
                return

            # Verificar si embeddings service está disponible
            if not self.embeddings_service:
                logger.warning("⚠️ Embeddings service not available for clustering")
                self._clustering_available = False
                return

            # Importar y crear clustering service
            from app.infrastructure.ml.advanced_clustering import AdvancedQuestionClusteringService

            self._clustering_service = AdvancedQuestionClusteringService(
                embeddings_service=self.embeddings_service,
                min_cluster_size=self.config.clustering_min_size,
                umap_n_components=self.config.umap_components,
            )

            self._clustering_available = True
            logger.info("✅ Clustering service initialized")

        except Exception as e:
            logger.error(f"❌ Error initializing clustering: {e}", exc_info=True)
            self._clustering_available = False

    def _init_learning_system(self) -> None:
        """Inicializa continuous learning system (lazy)."""
        try:
            if not self.config.enable_continuous_learning:
                return

            # Importar y crear learning system
            from app.infrastructure.ml.continuous_learning import ContinuousLearningSystem

            self._learning_system = ContinuousLearningSystem(
                storage_dir=".cache/learning", exploration_rate=self.config.mab_exploration_rate
            )

            self._learning_available = True
            logger.info("✅ Continuous learning system initialized")

        except Exception as e:
            logger.error(f"❌ Error initializing learning system: {e}", exc_info=True)
            self._learning_available = False

    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene status del adapter.

        Returns:
            Dict con información de status de cada componente
        """
        return {
            "adapter": "MLAdapter v1.0",
            "config": {
                "clustering_enabled": self.config.enable_clustering,
                "learning_enabled": self.config.enable_continuous_learning,
                "embeddings_enabled": self.config.enable_embeddings,
                "fallback_on_error": self.config.fallback_on_error,
            },
            "availability": {
                "clustering": self._clustering_available,
                "learning": self._learning_available,
                "embeddings": self._embeddings_available,
            },
            "services": {
                "clustering_service": self._clustering_service is not None,
                "learning_system": self._learning_system is not None,
                "embeddings_service": self.embeddings_service is not None,
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del adapter y servicios ML.

        Returns:
            Dict con métricas de uso y performance
        """
        metrics = {"status": self.get_status()}

        # Agregar métricas de learning system si disponible
        if self._learning_system and self._learning_available:
            try:
                # TODO: Implementar get_metrics en learning_system
                metrics["learning"] = {"available": True}
            except Exception as e:
                logger.error(f"Error getting learning metrics: {e}")

        return metrics
