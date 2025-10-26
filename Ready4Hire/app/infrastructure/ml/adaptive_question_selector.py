"""
Selector Adaptativo de Preguntas con ML
Usa Machine Learning para seleccionar la mejor siguiente pregunta
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from pathlib import Path


@dataclass
class CandidateState:
    """Estado actual del candidato"""

    current_skill_level: str  # junior, mid, senior
    average_score: float
    recent_scores: List[float]  # Últimos 5 scores
    topics_covered: List[str]
    topics_strong: List[str]  # Topics con score >= 8.0
    topics_weak: List[str]  # Topics con score < 6.5
    total_questions: int
    hints_used: int
    response_time_avg: float  # Segundos promedio
    confidence_trend: str  # "improving", "stable", "declining"


@dataclass
class QuestionCandidate:
    """Candidato a próxima pregunta"""

    question_id: str
    question_text: str
    difficulty: str
    topic: str
    category: str
    estimated_appropriateness: float  # 0-1, mayor = más apropiada


class AdaptiveQuestionSelector:
    """
    Selector inteligente de preguntas basado en ML.

    Estrategias:
    1. Difficulty Adaptation: Ajusta dificultad según performance
    2. Topic Diversification: Cubre múltiples topics
    3. Weakness Targeting (practice mode): Enfoca en áreas débiles
    4. Strength Validation (exam mode): Valida fortalezas
    5. Time-aware: Considera tiempo restante
    6. Engagement: Mantiene motivación alta
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.neural_ranker = None

        # Pesos para scoring (ajustables)
        self.weights = {
            "difficulty_match": 0.30,
            "topic_diversity": 0.20,
            "weakness_targeting": 0.25,
            "engagement": 0.15,
            "time_remaining": 0.10,
        }

    def select_next_question(
        self,
        candidate_state: CandidateState,
        available_questions: List[Dict],
        interview_mode: str = "practice",
        time_remaining: Optional[int] = None,
    ) -> QuestionCandidate:
        """
        Selecciona la mejor siguiente pregunta adaptativa mente.

        Args:
            candidate_state: Estado actual del candidato
            available_questions: Preguntas disponibles
            interview_mode: "practice" o "exam"
            time_remaining: Segundos restantes (opcional)

        Returns:
            QuestionCandidate con la mejor pregunta seleccionada
        """
        if not available_questions:
            raise ValueError("No questions available")

        # 1. Calcular appropriateness score para cada pregunta
        scored_questions = []
        for q in available_questions:
            score = self._calculate_appropriateness(q, candidate_state, interview_mode, time_remaining)

            scored_questions.append((score, q))

        # 2. Ordenar por score
        scored_questions.sort(reverse=True, key=lambda x: x[0])

        # 3. Aplicar randomness controlado (top-k sampling)
        # Para evitar siempre la misma pregunta y mantener variedad
        top_k = min(3, len(scored_questions))
        top_candidates = scored_questions[:top_k]

        # Softmax con temperature para sampling
        scores = np.array([s[0] for s in top_candidates])
        temperature = 0.5  # Menor = más determinista
        probs = self._softmax(scores / temperature)

        # Sample basado en probabilidades
        selected_idx = np.random.choice(len(top_candidates), p=probs)
        selected_score, selected_q = top_candidates[selected_idx]

        return QuestionCandidate(
            question_id=selected_q["id"],
            question_text=selected_q["question"],
            difficulty=selected_q["difficulty"],
            topic=selected_q["topic"],
            category=selected_q["category"],
            estimated_appropriateness=selected_score,
        )

    def _calculate_appropriateness(
        self, question: Dict, state: CandidateState, mode: str, time_remaining: Optional[int]
    ) -> float:
        """Calcula qué tan apropiada es una pregunta"""

        # 1. Difficulty Match Score
        difficulty_score = self._score_difficulty_match(
            question["difficulty"], state.average_score, state.confidence_trend
        )

        # 2. Topic Diversity Score
        topic_score = self._score_topic_diversity(question["topic"], state.topics_covered, state.total_questions)

        # 3. Weakness/Strength Targeting Score
        target_score = self._score_targeting(question["topic"], state.topics_weak, state.topics_strong, mode)

        # 4. Engagement Score
        engagement_score = self._score_engagement(question, state.total_questions, state.response_time_avg)

        # 5. Time Remaining Score
        time_score = self._score_time_constraint(question["difficulty"], time_remaining, state.total_questions)

        # Combinar con pesos
        total_score = (
            self.weights["difficulty_match"] * difficulty_score
            + self.weights["topic_diversity"] * topic_score
            + self.weights["weakness_targeting"] * target_score
            + self.weights["engagement"] * engagement_score
            + self.weights["time_remaining"] * time_score
        )

        return total_score

    def _score_difficulty_match(self, question_difficulty: str, avg_score: float, trend: str) -> float:
        """Score basado en match de dificultad"""

        # Mapear dificultad a número
        diff_map = {"junior": 1, "mid": 2, "senior": 3}
        q_level = diff_map.get(question_difficulty, 2)

        # Determinar dificultad ideal basada en performance
        if avg_score >= 8.5 and trend != "declining":
            ideal_level = min(3, q_level + 1)  # Aumentar dificultad
        elif avg_score >= 7.0:
            ideal_level = q_level  # Mantener dificultad
        else:
            ideal_level = max(1, q_level - 1)  # Disminuir dificultad

        # Score basado en distancia a ideal
        distance = abs(q_level - ideal_level)
        return 1.0 - (distance * 0.3)

    def _score_topic_diversity(self, question_topic: str, covered_topics: List[str], total_questions: int) -> float:
        """Score basado en diversidad de topics"""

        # Si el topic ya fue cubierto, penalizar
        if question_topic in covered_topics:
            # Penalizar menos si ya hay muchas preguntas
            penalty = 0.5 if total_questions < 5 else 0.7
            return penalty

        return 1.0  # Topic nuevo = máximo score

    def _score_targeting(
        self, question_topic: str, weak_topics: List[str], strong_topics: List[str], mode: str
    ) -> float:
        """Score basado en targeting de debilidades/fortalezas"""

        if mode == "practice":
            # En práctica, priorizar áreas débiles
            if question_topic in weak_topics:
                return 1.0  # Máximo score para debilidades
            elif question_topic in strong_topics:
                return 0.4  # Bajo score para fortalezas
            return 0.7  # Score medio para topics neutros

        else:  # exam mode
            # En examen, balance entre débil y fuerte
            if question_topic in weak_topics:
                return 0.7  # Incluir algunas debilidades
            elif question_topic in strong_topics:
                return 0.8  # Priorizar validación de fortalezas
            return 0.9  # Máximo para neutros (exploración)

    def _score_engagement(self, question: Dict, total_questions: int, avg_response_time: float) -> float:
        """Score basado en mantener engagement"""

        # Después de varias preguntas difíciles, incluir una más fácil
        # para mantener motivación (si detectamos tiempos largos)

        if total_questions >= 5 and avg_response_time > 180:  # >3min promedio
            # Candidato está luchando, dar una pregunta más fácil
            if question["difficulty"] == "junior":
                return 1.0
            elif question["difficulty"] == "mid":
                return 0.6
            else:
                return 0.3

        return 0.8  # Score neutral por defecto

    def _score_time_constraint(
        self, question_difficulty: str, time_remaining: Optional[int], total_questions: int
    ) -> float:
        """Score basado en tiempo restante"""

        if time_remaining is None:
            return 1.0  # Sin límite de tiempo

        # Estimar tiempo por pregunta según dificultad
        time_estimates = {"junior": 120, "mid": 180, "senior": 300}  # 2 min  # 3 min  # 5 min

        estimated_time = time_estimates.get(question_difficulty, 180)

        # Si queda poco tiempo, evitar preguntas largas
        if time_remaining < estimated_time * 1.5:
            # Preferir preguntas más cortas
            if question_difficulty == "junior":
                return 1.0
            elif question_difficulty == "mid":
                return 0.5
            else:
                return 0.2

        return 1.0  # Tiempo suficiente

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax para convertir scores a probabilidades"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()

    def get_recommended_difficulty(self, state: CandidateState) -> str:
        """Recomienda dificultad basada en performance"""
        if state.average_score >= 8.5 and state.confidence_trend == "improving":
            return "senior"
        elif state.average_score >= 7.5:
            return "mid"
        elif state.average_score >= 6.0:
            return "mid" if state.total_questions > 3 else "junior"
        else:
            return "junior"


# Factory
_adaptive_selector = None


def get_adaptive_selector() -> AdaptiveQuestionSelector:
    """Obtiene instancia singleton del selector adaptativo"""
    global _adaptive_selector
    if _adaptive_selector is None:
        _adaptive_selector = AdaptiveQuestionSelector()
    return _adaptive_selector
