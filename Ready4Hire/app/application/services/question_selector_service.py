"""
Servicio de selección adaptativa de preguntas.
Usa embeddings y rendimiento histórico para seleccionar la mejor siguiente pregunta.
"""
from typing import List, Dict, Any, Optional, Tuple
import random
from app.domain.entities.interview import Interview
from app.domain.entities.question import Question
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.repositories.question_repository import QuestionRepository
# EmbeddingsManager import removed - module deprecated
# TODO: Create adapter for app.infrastructure.ml.question_embeddings.QuestionEmbeddingsService


class QuestionSelectorService:
    """Servicio para seleccionar preguntas adaptativas durante la entrevista."""
    
    def __init__(
        self,
        question_repository: QuestionRepository,
        embeddings_manager: Optional[Any] = None  # Type changed from EmbeddingsManager (deprecated)
    ):
        """
        Inicializa el selector de preguntas.
        
        Args:
            question_repository: Repositorio de preguntas
            embeddings_manager: Manager de embeddings para búsqueda semántica
        """
        self.question_repo = question_repository
        self.embeddings_manager = embeddings_manager
        self._question_cache: Dict[str, List[Question]] = {}
    
    async def select_next_question(
        self,
        interview: Interview,
        previous_question_ids: List[str],
        last_answer_score: Optional[float] = None
    ) -> Optional[Question]:
        """
        Selecciona la siguiente pregunta basándose en el contexto de la entrevista.
        
        Args:
            interview: Entrevista actual
            previous_question_ids: IDs de preguntas ya realizadas
            last_answer_score: Puntuación de la última respuesta (para ajustar dificultad)
        
        Returns:
            Siguiente pregunta o None si no hay más preguntas
        """
        # Determinar dificultad objetivo
        target_difficulty = self._determine_target_difficulty(
            interview=interview,
            last_answer_score=last_answer_score
        )
        
        # Obtener preguntas candidatas
        candidate_questions = await self._get_candidate_questions(
            role=interview.role,
            category=None,
            difficulty=target_difficulty,
            previous_ids=previous_question_ids
        )
        
        if not candidate_questions:
            # Fallback: Cualquier pregunta no realizada
            candidate_questions = await self._get_fallback_questions(
                role=interview.role,
                category=None,
                previous_ids=previous_question_ids
            )
        
        if not candidate_questions:
            return None
        
        # Seleccionar la mejor pregunta
        selected_question = self._select_best_question(
            candidates=candidate_questions,
            interview=interview,
            previous_ids=previous_question_ids
        )
        
        return selected_question
    
    def _determine_target_difficulty(
        self,
        interview: Interview,
        last_answer_score: Optional[float]
    ) -> SkillLevel:
        """
        Determina la dificultad objetivo para la siguiente pregunta.
        
        Estrategia:
        - Si la última puntuación es alta (>7): Aumentar dificultad
        - Si la última puntuación es baja (<5): Mantener o reducir dificultad
        - Si es la primera pregunta: Usar dificultad actual de la entrevista
        """
        current_difficulty = getattr(interview, 'difficulty', SkillLevel.JUNIOR)
        
        if last_answer_score is None:
            # Primera pregunta
            return current_difficulty
        
        # Ajuste dinámico basado en rendimiento
        if last_answer_score >= 7.5:
            # Rendimiento excelente -> Subir dificultad
            if current_difficulty == SkillLevel.JUNIOR:
                return SkillLevel.MID
            elif current_difficulty == SkillLevel.MID:
                return SkillLevel.SENIOR
            else:
                return SkillLevel.SENIOR
        
        elif last_answer_score >= 5.0:
            # Rendimiento aceptable -> Mantener dificultad
            return current_difficulty
        
        else:
            # Rendimiento bajo -> Reducir dificultad
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
        """Obtiene preguntas candidatas de la dificultad objetivo."""
        # Buscar en repositorio
        all_questions = await self.question_repo.find_by_role(role)
        
        # Filtrar por categoría (si se proporciona), dificultad y no repetidas
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
    
    def _select_best_question(
        self,
        candidates: List[Question],
        interview: Interview,
        previous_ids: List[str]
    ) -> Question:
        """
        Selecciona la mejor pregunta del conjunto de candidatas.
        
        Estrategias:
        1. Si hay embeddings: Usar diversidad semántica
        2. Si no: Selección aleatoria ponderada
        """
        if not candidates:
            raise ValueError("No hay preguntas candidatas disponibles")
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Estrategia 1: Diversidad semántica (si hay embeddings)
        if self.embeddings_manager and previous_ids:
            return self._select_by_semantic_diversity(
                candidates=candidates,
                previous_ids=previous_ids
            )
        
        # Estrategia 2: Selección aleatoria
        return random.choice(candidates)
    
    def _select_by_semantic_diversity(
        self,
        candidates: List[Question],
        previous_ids: List[str]
    ) -> Question:
        """
        Selecciona la pregunta más diversa semánticamente respecto a las previas.
        
        Busca maximizar la variedad de temas cubiertos.
        """
        try:
            # Obtener embeddings de preguntas previas
            previous_embeddings = []
            for qid in previous_ids[-3:]:  # Últimas 3 preguntas
                if self.embeddings_manager is not None:
                    emb = self.embeddings_manager.get_embedding(qid)
                    if emb is not None:
                        previous_embeddings.append(emb)
            
            if not previous_embeddings:
                return random.choice(candidates)
            
            # Calcular diversidad para cada candidata
            import numpy as np
            best_question = None
            max_diversity = -1
            for candidate in candidates:
                if self.embeddings_manager is None:
                    continue
                candidate_emb = self.embeddings_manager.get_embedding(candidate.id)
                if candidate_emb is None:
                    continue
                    continue
                
                # Calcular distancia promedio a preguntas previas
                distances = [
                    1 - np.dot(candidate_emb, prev_emb) / (
                        np.linalg.norm(candidate_emb) * np.linalg.norm(prev_emb)
                    )
                    for prev_emb in previous_embeddings
                ]
                avg_diversity = sum(distances) / len(distances)
                
                if avg_diversity > max_diversity:
                    max_diversity = avg_diversity
                    best_question = candidate
            
            return best_question if best_question else random.choice(candidates)
            
        except Exception as e:
            # Fallback si hay error
            return random.choice(candidates)
    
    async def select_initial_question(
        self,
        role: str,
        category: str,
        difficulty: SkillLevel
    ) -> Question:
        """
        Selecciona la pregunta inicial para una nueva entrevista.
        
        Criterio: Pregunta representativa del nivel inicial.
        """
        questions = await self.question_repo.find_by_difficulty(difficulty.value)
        
        candidates = [
            q for q in questions
            if q.role == role and q.category == category
        ]
        
        if not candidates:
            # Fallback: Cualquier pregunta del rol y categoría
            all_questions = await self.question_repo.find_by_role(role)
            candidates = [
                q for q in all_questions
                if q.category == category
            ]
        
        if not candidates:
            raise ValueError(
                f"No hay preguntas disponibles para role={role}, category={category}"
            )
        
        # Seleccionar aleatoriamente entre las candidatas
        return random.choice(candidates)
    
    async def get_question_recommendations(
        self,
        interview: Interview,
        top_k: int = 3
    ) -> List[Tuple[Question, float]]:
        """
        Obtiene recomendaciones de preguntas con puntuación de relevancia.
        
        Útil para análisis o previsualización de próximas preguntas.
        
        Args:
            interview: Entrevista actual
            top_k: Número de recomendaciones
        
        Returns:
            Lista de tuplas (Question, relevance_score)
        """
        previous_ids = [a.question_id for a in getattr(interview, 'answers', [])]
        
        # Obtener candidatas
        candidates = await self._get_candidate_questions(
            role=interview.role,
            category=None,
            difficulty=getattr(interview, 'difficulty', SkillLevel.JUNIOR),
            previous_ids=previous_ids
        )
        
        if not candidates:
            return []
        
        # Calcular puntuación de relevancia
        recommendations = []
        for candidate in candidates[:top_k * 2]:  # Oversample
            score = self._calculate_relevance_score(
                question=candidate,
                interview=interview
            )
            recommendations.append((candidate, score))
        
        # Ordenar por puntuación y retornar top_k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def _calculate_relevance_score(
        self,
        question: Question,
        interview: Interview
    ) -> float:
        """
        Calcula puntuación de relevancia de una pregunta para la entrevista actual.
        
        Factores:
        - Alineación con dificultad actual
        - Diversidad respecto a preguntas previas
        - Cobertura de keywords no vistos
        """
        score = 0.0
        
        # Factor 1: Alineación con dificultad (peso: 0.4)
        # Determine current difficulty from interview state
        current_difficulty = getattr(interview, 'difficulty', SkillLevel.JUNIOR)
        if question.difficulty == current_difficulty:
            score += 0.4
        
        # Factor 2: Diversidad semántica (peso: 0.3)
        # TODO: Implementar cuando embeddings estén disponibles
        score += 0.15  # Placeholder
        
        # Factor 3: Keywords no cubiertos (peso: 0.3)
        covered_keywords = set()
        for answer in getattr(interview, 'answers', []):
            # Asumiendo que Answer tiene acceso a Question
            # En implementación real, necesitaríamos cargar las preguntas
            pass
        
        question_keywords = set(question.keywords)
        new_keywords_ratio = len(question_keywords - covered_keywords) / max(
            len(question_keywords), 1
        )
        score += 0.3 * new_keywords_ratio
        
        return score
