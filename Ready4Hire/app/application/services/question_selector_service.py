"""
Enhanced Question Selector Service con ML Avanzado.

Este servicio integra:
- Advanced Clustering (UMAP + HDBSCAN)
- Continuous Learning (Multi-Armed Bandits)
- Question Embeddings (SentenceTransformers + RankNet)

Mejoras sobre la versión anterior:
1. Selección basada en clusters para mejor diversidad
2. Aprendizaje continuo con feedback de entrevistas
3. Exploración/explotación balanceada (MAB)
4. Tracking de performance por pregunta

Author: Jeronimo Restrepo Angel
Date: 2025-10-15
Version: 2.0
"""
from typing import List, Dict, Any, Optional, Tuple
import random
import logging
from dataclasses import dataclass

from app.domain.entities.interview import Interview
from app.domain.entities.question import Question
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.repositories.question_repository import QuestionRepository
from app.infrastructure.ml.question_embeddings import QuestionEmbeddingsService
from app.infrastructure.ml.advanced_clustering import AdvancedQuestionClusteringService
from app.infrastructure.ml.continuous_learning import ContinuousLearningSystem, InterviewFeedback

logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuración para el selector de preguntas."""
    use_clustering: bool = True
    use_continuous_learning: bool = True
    use_embeddings: bool = True
    exploration_strategy: str = 'balanced'  # 'balanced', 'exploit', 'explore', 'adaptive'
    min_cluster_diversity: float = 0.3
    max_questions_per_cluster: int = 2
    fallback_to_simple: bool = True


class EnhancedQuestionSelectorService:
    """
    Servicio avanzado de selección de preguntas con ML.
    
    Features:
    - Clustering automático de preguntas por tópico
    - Multi-armed bandits para exploración/explotación
    - Tracking de performance y aprendizaje continuo
    - Degradación graceful si ML falla
    
    Example:
        ```python
        selector = EnhancedQuestionSelectorService(
            question_repo=repo,
            embeddings_service=embeddings,
            clustering_service=clustering,
            learning_system=learning
        )
        
        question = await selector.select_next_question(
            interview=interview,
            previous_question_ids=prev_ids,
            last_answer_score=7.5
        )
        ```
    """
    
    def __init__(
        self,
        question_repository: QuestionRepository,
        embeddings_service: Optional[QuestionEmbeddingsService] = None,
        clustering_service: Optional[AdvancedQuestionClusteringService] = None,
        learning_system: Optional[ContinuousLearningSystem] = None,
        config: Optional[SelectionConfig] = None
    ):
        """
        Inicializa el selector enhanced.
        
        Args:
            question_repository: Repositorio de preguntas
            embeddings_service: Servicio de embeddings y ranking
            clustering_service: Servicio de clustering avanzado
            learning_system: Sistema de aprendizaje continuo
            config: Configuración del selector
        """
        self.question_repo = question_repository
        self.embeddings_service = embeddings_service
        self.clustering_service = clustering_service
        self.learning_system = learning_system
        self.config = config or SelectionConfig()
        
        # Cache
        self._question_cache: Dict[str, List[Question]] = {}
        self._cluster_cache: Dict[str, Any] = {}
        
        logger.info(
            f"EnhancedQuestionSelectorService inicializado: "
            f"clustering={self.config.use_clustering}, "
            f"learning={self.config.use_continuous_learning}, "
            f"strategy={self.config.exploration_strategy}"
        )
    
    async def select_next_question(
        self,
        interview: Interview,
        previous_question_ids: List[str],
        last_answer_score: Optional[float] = None
    ) -> Optional[Question]:
        """
        Selecciona la siguiente pregunta usando ML avanzado.
        
        Flujo:
        1. Determinar dificultad objetivo (adaptativo)
        2. Obtener candidatas filtradas
        3. Aplicar clustering para diversificación
        4. Usar MAB para exploración/explotación
        5. Seleccionar mejor pregunta
        
        Args:
            interview: Entrevista actual
            previous_question_ids: IDs de preguntas ya hechas
            last_answer_score: Score de última respuesta (0-10)
        
        Returns:
            Siguiente pregunta o None si no hay más
        """
        candidates = []
        try:
            # Paso 1: Determinar dificultad objetivo
            target_difficulty = self._determine_target_difficulty(
                interview=interview,
                last_answer_score=last_answer_score
            )
            
            logger.debug(
                f"Target difficulty para {interview.id}: {target_difficulty.value}"
            )
            
            # Paso 2: Obtener candidatas
            candidates = await self._get_candidate_questions(
                role=interview.role,
                category=interview.interview_type,
                difficulty=target_difficulty,
                previous_ids=previous_question_ids
            )
            
            if not candidates:
                # Fallback: Cualquier pregunta no hecha
                candidates = await self._get_fallback_questions(
                    role=interview.role,
                    category=interview.interview_type,
                    previous_ids=previous_question_ids
                )
            
            if not candidates:
                logger.warning(f"No hay más preguntas para {interview.id}")
                return None
            
            # Paso 3 & 4: Selección inteligente con ML
            selected = await self._select_best_question_ml(
                candidates=candidates,
                interview=interview,
                previous_ids=previous_question_ids
            )
            
            logger.info(
                f"Pregunta seleccionada para {interview.id}: {selected.id} "
                f"(difficulty={selected.difficulty}, topic={selected.topic})"
            )
            
            return selected
            
        except Exception as e:
            logger.error(f"Error en select_next_question: {e}", exc_info=True)
            
            if self.config.fallback_to_simple and candidates:
                # Fallback graceful
                logger.warning("Usando fallback a selección simple")
                return random.choice(candidates)
            
            return None
    
    async def _select_best_question_ml(
        self,
        candidates: List[Question],
        interview: Interview,
        previous_ids: List[str]
    ) -> Question:
        """
        Selección inteligente usando ML.
        
        Estrategias aplicadas:
        1. Clustering para diversificación de tópicos
        2. MAB para exploración/explotación
        3. Performance tracking para optimización
        """
        if len(candidates) == 1:
            return candidates[0]
        
        # Estrategia 1: Clustering (si habilitado)
        if self.config.use_clustering and self.clustering_service:
            selected = await self._select_with_clustering(
                candidates=candidates,
                previous_ids=previous_ids
            )
            if selected:
                return selected
        
        # Estrategia 2: Continuous Learning (si habilitado)
        if self.config.use_continuous_learning and self.learning_system:
            selected = self._select_with_mab(
                candidates=candidates,
                interview=interview
            )
            if selected:
                return selected
        
        # Estrategia 3: Semantic Diversity (si embeddings disponible)
        if self.config.use_embeddings and self.embeddings_service:
            selected = await self._select_by_semantic_diversity(
                candidates=candidates,
                previous_ids=previous_ids
            )
            if selected:
                return selected
        
        # Fallback: Random
        logger.debug("Usando selección random (fallback)")
        return random.choice(candidates)
    
    async def _select_with_clustering(
        self,
        candidates: List[Question],
        previous_ids: List[str]
    ) -> Optional[Question]:
        """
        Selecciona pregunta usando clustering para diversificación.
        
        Objetivo: Maximizar cobertura de tópicos diferentes.
        """
        try:
            if not self.clustering_service:
                return None
            
            # Cluster candidatas - convertir a formato dict requerido
            questions_dict = [{'text': q.text, 'id': q.id} for q in candidates]
            clusters_by_id = self.clustering_service.cluster_questions(questions_dict)
            
            if not clusters_by_id:
                logger.debug("No se encontraron clusters válidos")
                return None
            
            # Identificar clusters ya visitados
            visited_clusters = self._get_visited_clusters(
                previous_ids=previous_ids,
                all_questions=candidates
            )
            
            # Encontrar cluster menos visitado con preguntas disponibles
            min_visits = float('inf')
            target_cluster = None
            
            for cluster_id, cluster_questions in clusters_by_id.items():
                if cluster_id == -1:  # Noise cluster
                    continue
                
                visits = visited_clusters.get(cluster_id, 0)
                if visits < min_visits and len(cluster_questions) > 0:
                    min_visits = visits
                    target_cluster = cluster_id
            
            if target_cluster is None:
                logger.debug("No se encontró cluster objetivo")
                return None
            
            # Obtener preguntas del cluster objetivo
            cluster_questions_dict = clusters_by_id[target_cluster]
            
            # Usar diversified selection dentro del cluster
            selected_questions = self.clustering_service.select_diversified_questions(
                n_questions=1,
                candidate_ids=set(q['id'] for q in cluster_questions_dict),
                exclude_ids=set(previous_ids),
                user_profile=None
            )
            
            if selected_questions and len(selected_questions) > 0:
                # Encontrar el objeto Question correspondiente
                selected_id = selected_questions[0]['id']
                for question in candidates:
                    if question.id == selected_id:
                        logger.info(
                            f"Seleccionada pregunta de cluster {target_cluster} "
                            f"(visits={min_visits})"
                        )
                        return question
            
            return None
            
        except Exception as e:
            logger.error(f"Error en clustering selection: {e}", exc_info=True)
            return None
    
    def _select_with_mab(
        self,
        candidates: List[Question],
        interview: Interview
    ) -> Optional[Question]:
        """
        Selecciona pregunta usando Multi-Armed Bandits.
        
        Balancea exploración (preguntas nuevas) vs explotación (preguntas probadas).
        """
        try:
            if not self.learning_system:
                return None
            
            # Obtener IDs de candidatas
            candidate_ids = [q.id for q in candidates]
            
            # Usar MAB para rankear preguntas
            rankings = self.learning_system.get_question_rankings(
                question_ids=candidate_ids,
                strategy=self.config.exploration_strategy
            )
            
            if not rankings:
                return None
            
            # Seleccionar la pregunta con mejor score
            selected_id = rankings[0][0]  # Primera tupla (id, score)
            
            # Encontrar pregunta seleccionada
            for question in candidates:
                if question.id == selected_id:
                    logger.info(
                        f"MAB seleccionó {selected_id} "
                        f"(strategy={self.config.exploration_strategy}, score={rankings[0][1]:.3f})"
                    )
                    return question
            
            return None
            
        except Exception as e:
            logger.error(f"Error en MAB selection: {e}", exc_info=True)
            return None
    
    async def _select_by_semantic_diversity(
        self,
        candidates: List[Question],
        previous_ids: List[str]
    ) -> Optional[Question]:
        """
        Selecciona pregunta maximizando diversidad semántica.
        
        Usa embeddings para encontrar pregunta más diferente a las previas.
        """
        try:
            if not self.embeddings_service or not previous_ids:
                return None
            
            import numpy as np
            
            # Obtener embeddings de preguntas previas (últimas 3)
            previous_embeddings = []
            for qid in previous_ids[-3:]:
                emb = await self._get_question_embedding(qid)
                if emb is not None:
                    previous_embeddings.append(emb)
            
            if not previous_embeddings:
                return None
            
            # Calcular diversidad para cada candidata
            best_question = None
            max_diversity = -1
            
            for candidate in candidates:
                candidate_emb = await self._get_question_embedding(candidate.id)
                if candidate_emb is None:
                    continue
                
                # Distancia promedio a preguntas previas (cosine distance)
                distances = []
                for prev_emb in previous_embeddings:
                    similarity = np.dot(candidate_emb, prev_emb) / (
                        np.linalg.norm(candidate_emb) * np.linalg.norm(prev_emb)
                    )
                    distance = 1 - similarity
                    distances.append(distance)
                
                avg_diversity = np.mean(distances)
                
                if avg_diversity > max_diversity:
                    max_diversity = avg_diversity
                    best_question = candidate
            
            if best_question:
                logger.info(
                    f"Semantic diversity seleccionó {best_question.id} "
                    f"(diversity={max_diversity:.3f})"
                )
            
            return best_question
            
        except Exception as e:
            logger.error(f"Error en semantic diversity: {e}", exc_info=True)
            return None
    
    async def _get_question_embedding(self, question_id: str) -> Optional[Any]:
        """Obtiene embedding de una pregunta (con caché)."""
        try:
            if not self.embeddings_service:
                return None
            
            # TODO: Implementar get_embedding en QuestionEmbeddingsService
            # Por ahora retornar None
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo embedding: {e}")
            return None
    
    def _get_visited_clusters(
        self,
        previous_ids: List[str],
        all_questions: List[Question]
    ) -> Dict[int, int]:
        """
        Calcula cuántas veces se visitó cada cluster.
        
        Returns:
            Dict con cluster_id -> count
        """
        # TODO: Implementar tracking de clusters visitados
        # Por ahora retornar dict vacío
        return {}
    
    def _determine_target_difficulty(
        self,
        interview: Interview,
        last_answer_score: Optional[float]
    ) -> SkillLevel:
        """
        Determina dificultad objetivo adaptativa.
        
        Lógica:
        - Score >= 7.5: Subir dificultad
        - Score >= 5.0: Mantener dificultad
        - Score < 5.0: Bajar dificultad
        """
        current_difficulty = getattr(interview, 'skill_level', SkillLevel.JUNIOR)
        
        if last_answer_score is None:
            return current_difficulty
        
        if last_answer_score >= 7.5:
            # Excelente -> Subir
            if current_difficulty == SkillLevel.JUNIOR:
                return SkillLevel.MID
            elif current_difficulty == SkillLevel.MID:
                return SkillLevel.SENIOR
            else:
                return SkillLevel.SENIOR
        
        elif last_answer_score >= 5.0:
            # Aceptable -> Mantener
            return current_difficulty
        
        else:
            # Bajo -> Bajar
            if current_difficulty == SkillLevel.SENIOR:
                return SkillLevel.MID
            elif current_difficulty == SkillLevel.MID:
                return SkillLevel.JUNIOR
            else:
                return SkillLevel.JUNIOR
    
    async def _get_candidate_questions(
        self,
        role: str,
        category: Optional[str],
        difficulty: SkillLevel,
        previous_ids: List[str]
    ) -> List[Question]:
        """Obtiene candidatas filtradas por criterios."""
        all_questions = await self.question_repo.find_by_role(role)
        
        candidates = [
            q for q in all_questions
            if (category is None or q.category == category)
            and q.difficulty == difficulty
            and q.id not in previous_ids
        ]
        
        return candidates
    
    async def _get_fallback_questions(
        self,
        role: str,
        category: Optional[str],
        previous_ids: List[str]
    ) -> List[Question]:
        """Obtiene preguntas de respaldo sin filtro de dificultad."""
        all_questions = await self.question_repo.find_by_role(role)
        
        candidates = [
            q for q in all_questions
            if (category is None or q.category == category)
            and q.id not in previous_ids
        ]
        
        return candidates
    
    async def update_question_performance(
        self,
        interview: Interview,
        question_id: str,
        answer_score: float,
        time_taken: int
    ) -> None:
        """
        Actualiza performance de una pregunta (feedback loop).
        
        Esto alimenta el sistema de continuous learning.
        
        Args:
            interview: Entrevista completada
            question_id: ID de la pregunta evaluada
            answer_score: Score obtenido (0-10)
            time_taken: Tiempo de respuesta en segundos
        """
        try:
            if not self.config.use_continuous_learning or not self.learning_system:
                return
            
            # Crear feedback
            from datetime import datetime
            feedback = InterviewFeedback(
                interview_id=interview.id,
                user_id=interview.user_id,
                role=interview.role,
                questions_asked=[question_id],
                scores=[answer_score],
                response_times=[time_taken],
                final_evaluation=answer_score,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'difficulty': interview.skill_level.value,
                    'was_correct': answer_score >= 6.0,
                    'candidate_level': interview.skill_level.value
                }
            )
            
            # Actualizar sistema
            self.learning_system.record_interview_feedback(feedback)
            
            logger.info(
                f"Performance actualizada para {question_id}: "
                f"score={answer_score}, time={time_taken}s"
            )
            
        except Exception as e:
            logger.error(f"Error actualizando performance: {e}", exc_info=True)
    
    async def get_selection_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del selector.
        
        Returns:
            Dict con estadísticas de selección
        """
        metrics = {
            'config': {
                'use_clustering': self.config.use_clustering,
                'use_continuous_learning': self.config.use_continuous_learning,
                'use_embeddings': self.config.use_embeddings,
                'exploration_strategy': self.config.exploration_strategy
            },
            'cache': {
                'questions_cached': len(self._question_cache),
                'clusters_cached': len(self._cluster_cache)
            }
        }
        
        # Agregar métricas de learning system si disponible
        if self.learning_system:
            try:
                # Usar analyze_question_pool en lugar de get_metrics
                metrics['learning'] = self.learning_system.analyze_question_pool()
            except Exception as e:
                logger.debug(f"Could not get learning metrics: {e}")
        
        return metrics
