"""
Continuous Learning System for Question Selection
Sistema de aprendizaje continuo para mejorar la selección de preguntas

Este módulo implementa:
1. Recolección de feedback de entrevistas
2. Actualización incremental de modelos
3. A/B testing de estrategias de selección
4. Análisis de rendimiento de preguntas
5. Re-ranking automático basado en efectividad

Techniques:
- Online Learning: Actualización incremental sin reentrenamiento completo
- Reinforcement Learning: Reward basado en calidad de respuestas
- Bandit Algorithms: Exploración vs Explotación
- Performance Analytics: Métricas de efectividad por pregunta
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QuestionPerformance:
    """Métricas de rendimiento de una pregunta"""

    question_id: str
    times_asked: int
    avg_score: float
    avg_response_time: float
    difficulty_actual: float  # Dificultad percibida vs declarada
    discrimination_power: float  # Qué tan bien diferencia candidatos
    last_updated: str

    def to_dict(self) -> Dict:
        """
        Convierte la instancia a diccionario para serialización.

        Returns:
            Dict: Representación serializable del objeto
        """
        return asdict(self)


@dataclass
class InterviewFeedback:
    """Feedback de una entrevista completa"""

    interview_id: str
    user_id: str
    role: str
    questions_asked: List[str]
    scores: List[float]
    response_times: List[float]
    final_evaluation: float
    timestamp: str
    metadata: Dict[str, Any]


class ContinuousLearningSystem:
    """
    Sistema de aprendizaje continuo para optimización de preguntas.

    Features:
    - Tracking de rendimiento por pregunta
    - Actualización incremental de rankings
    - A/B testing de estrategias
    - Análisis de dificultad real vs declarada
    - Re-balanceo automático de pool de preguntas
    - Multi-armed bandit para exploración

    Learning Strategy:
    1. Collect: Recopilar feedback de cada entrevista
    2. Analyze: Analizar patrones y tendencias
    3. Update: Actualizar métricas y rankings
    4. Adapt: Ajustar estrategia de selección
    """

    def __init__(
        self,
        storage_dir: str = ".cache/learning",
        window_size: int = 100,  # Ventana para métricas móviles
        exploration_rate: float = 0.1,  # Epsilon para epsilon-greedy
    ):
        """
        Inicializa el sistema de aprendizaje continuo.

        Args:
            storage_dir: Directorio para persistir datos
            window_size: Tamaño de ventana para promedios móviles
            exploration_rate: Tasa de exploración para nuevas preguntas
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.exploration_rate = exploration_rate

        # Estado interno
        self.question_performance: Dict[str, QuestionPerformance] = {}
        self.feedback_history: deque = deque(maxlen=1000)
        self.strategy_metrics: Dict[str, Dict] = defaultdict(
            lambda: {"selections": 0, "avg_satisfaction": 0.0, "avg_completion_rate": 0.0}
        )

        # Load existing data
        self._load_state()

        logger.info(f"✅ ContinuousLearningSystem initialized")
        logger.info(f"   • Tracking {len(self.question_performance)} questions")
        logger.info(f"   • {len(self.feedback_history)} feedback entries")

    def record_interview_feedback(self, feedback: InterviewFeedback):
        """
        Registra feedback de una entrevista completada.

        Args:
            feedback: Feedback estructurado de la entrevista
        """
        try:
            self.feedback_history.append(feedback)

            # Actualizar métricas por pregunta
            for i, q_id in enumerate(feedback.questions_asked):
                score = feedback.scores[i] if i < len(feedback.scores) else 0
                response_time = feedback.response_times[i] if i < len(feedback.response_times) else 0

                self._update_question_performance(q_id, score, response_time)

            # Persistir
            self._save_state()

            logger.info(f"📝 Feedback recorded for interview {feedback.interview_id}")

        except Exception as e:
            logger.error(f"❌ Error recording feedback: {str(e)}", exc_info=True)

    def _update_question_performance(self, question_id: str, score: float, response_time: float):
        """Actualiza métricas de una pregunta con nuevo feedback"""
        if question_id not in self.question_performance:
            # Inicializar si es nueva
            self.question_performance[question_id] = QuestionPerformance(
                question_id=question_id,
                times_asked=0,
                avg_score=0.0,
                avg_response_time=0.0,
                difficulty_actual=0.0,
                discrimination_power=0.0,
                last_updated=datetime.now().isoformat(),
            )

        perf = self.question_performance[question_id]

        # Actualización incremental (exponential moving average)
        alpha = 0.1  # Factor de aprendizaje
        perf.times_asked += 1
        perf.avg_score = (1 - alpha) * perf.avg_score + alpha * score
        perf.avg_response_time = (1 - alpha) * perf.avg_response_time + alpha * response_time

        # Dificultad actual (score bajo = difícil)
        perf.difficulty_actual = 1.0 - perf.avg_score / 10.0

        perf.last_updated = datetime.now().isoformat()

    def get_question_rankings(self, question_ids: List[str], strategy: str = "balanced") -> List[Tuple[str, float]]:
        """
        Retorna preguntas rankeadas según métricas de rendimiento.

        Strategies:
        - 'balanced': Balancea exploración y explotación
        - 'exploit': Siempre usar las mejores preguntas
        - 'explore': Preferir preguntas menos usadas
        - 'adaptive': Adaptar según contexto

        Args:
            question_ids: IDs de preguntas candidatas
            strategy: Estrategia de ranking

        Returns:
            Lista de (question_id, score) ordenada
        """
        rankings = []

        for q_id in question_ids:
            perf = self.question_performance.get(q_id)

            if perf is None:
                # Pregunta nueva: dar score alto para exploración
                score = 0.8 + np.random.random() * 0.2
            else:
                # Calcular score según estrategia
                if strategy == "exploit":
                    score = self._compute_exploit_score(perf)
                elif strategy == "explore":
                    score = self._compute_explore_score(perf)
                elif strategy == "adaptive":
                    score = self._compute_adaptive_score(perf)
                else:  # balanced
                    score = self._compute_balanced_score(perf)

            rankings.append((q_id, score))

        # Ordenar por score descendente
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def _compute_exploit_score(self, perf: QuestionPerformance) -> float:
        """Score para explotación: usar preguntas probadamente efectivas"""
        # Combinar avg_score y discrimination_power
        score = 0.7 * perf.avg_score / 10.0 + 0.3 * perf.discrimination_power

        # Penalizar preguntas muy usadas (evitar desgaste)
        if perf.times_asked > 50:
            penalty = min(0.2, (perf.times_asked - 50) / 1000)
            score -= penalty

        return max(0, min(1, score))

    def _compute_explore_score(self, perf: QuestionPerformance) -> float:
        """Score para exploración: preferir preguntas menos usadas"""
        # Invertir times_asked
        novelty_score = 1.0 / (1.0 + perf.times_asked / 10.0)

        # Pequeño peso a calidad para no usar preguntas malas
        quality_score = perf.avg_score / 10.0

        score = 0.8 * novelty_score + 0.2 * quality_score

        return max(0, min(1, score))

    def _compute_balanced_score(self, perf: QuestionPerformance) -> float:
        """Score balanceado: epsilon-greedy entre exploit y explore"""
        if np.random.random() < self.exploration_rate:
            return self._compute_explore_score(perf)
        else:
            return self._compute_exploit_score(perf)

    def _compute_adaptive_score(self, perf: QuestionPerformance) -> float:
        """Score adaptativo: ajustar según contexto actual"""
        # Reducir exploración si hay suficiente data
        total_samples = sum(p.times_asked for p in self.question_performance.values())

        if total_samples < 100:
            # Fase inicial: explorar más
            return self._compute_explore_score(perf)
        elif total_samples > 500:
            # Fase madura: explotar
            return self._compute_exploit_score(perf)
        else:
            # Fase intermedia: balancear
            return self._compute_balanced_score(perf)

    def analyze_question_pool(self) -> Dict[str, Any]:
        """
        Analiza el pool de preguntas y retorna insights.

        Returns:
            Diccionario con análisis y recomendaciones
        """
        if not self.question_performance:
            return {"status": "no_data"}

        performances = list(self.question_performance.values())

        analysis = {
            "total_questions": len(performances),
            "total_interviews": len(self.feedback_history),
            "avg_questions_per_interview": (
                np.mean([len(f.questions_asked) for f in self.feedback_history]) if self.feedback_history else 0
            ),
            # Métricas globales
            "avg_score_all_questions": np.mean([p.avg_score for p in performances]),
            "avg_response_time": np.mean([p.avg_response_time for p in performances]),
            # Distribución de uso
            "most_used_questions": sorted(
                [(p.question_id, p.times_asked) for p in performances], key=lambda x: x[1], reverse=True
            )[:10],
            "least_used_questions": sorted([(p.question_id, p.times_asked) for p in performances], key=lambda x: x[1])[
                :10
            ],
            # Calidad
            "top_performing_questions": sorted(
                [(p.question_id, p.avg_score) for p in performances], key=lambda x: x[1], reverse=True
            )[:10],
            "underperforming_questions": sorted(
                [(p.question_id, p.avg_score) for p in performances], key=lambda x: x[1]
            )[:10],
            # Recomendaciones
            "recommendations": self._generate_recommendations(performances),
        }

        return analysis

    def _generate_recommendations(self, performances: List[QuestionPerformance]) -> List[str]:
        """Genera recomendaciones basadas en análisis"""
        recs = []

        # Preguntas sobre-usadas
        overused = [p for p in performances if p.times_asked > 100]
        if overused:
            recs.append(f"⚠️ {len(overused)} questions are overused. Consider rotating.")

        # Preguntas nunca usadas o muy poco
        underused = [p for p in performances if p.times_asked < 5]
        if len(underused) > len(performances) * 0.3:
            recs.append(f"💡 {len(underused)} questions are underused. Increase exploration rate.")

        # Preguntas con bajo rendimiento
        poor_performance = [p for p in performances if p.avg_score < 3.0 and p.times_asked > 10]
        if poor_performance:
            recs.append(f"❌ {len(poor_performance)} questions consistently get low scores. Review or remove.")

        # Preguntas con alta varianza (buenos discriminadores)
        # TODO: Implementar cálculo de varianza

        if not recs:
            recs.append("✅ Question pool looks healthy!")

        return recs

    def get_optimal_question_sequence(
        self, candidate_profile: Dict, n_questions: int, available_questions: List[str]
    ) -> List[str]:
        """
        Genera secuencia óptima de preguntas para un candidato.

        Strategy:
        1. Empezar con pregunta de calibración (dificultad media)
        2. Ajustar dificultad según rendimiento
        3. Cubrir diferentes clusters temáticos
        4. Terminar con pregunta distintiva

        Args:
            candidate_profile: Perfil del candidato
            n_questions: Número de preguntas a seleccionar
            available_questions: Pool de preguntas disponibles

        Returns:
            Secuencia ordenada de question IDs
        """
        # Rankear preguntas
        rankings = self.get_question_rankings(available_questions, strategy="adaptive")

        # Seleccionar top N
        selected_ids = [q_id for q_id, _ in rankings[:n_questions]]

        logger.info(f"✅ Generated optimal sequence of {len(selected_ids)} questions")

        return selected_ids

    def _save_state(self):
        """Persiste estado del sistema"""
        try:
            # Guardar performance de preguntas
            perf_file = self.storage_dir / "question_performance.json"
            with open(perf_file, "w") as f:
                data = {q_id: perf.to_dict() for q_id, perf in self.question_performance.items()}
                json.dump(data, f, indent=2)

            # Guardar feedback reciente
            feedback_file = self.storage_dir / "feedback_history.json"
            with open(feedback_file, "w") as f:
                data = [asdict(f) for f in list(self.feedback_history)]
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Error saving state: {str(e)}")

    def _load_state(self):
        """Carga estado persistido"""
        try:
            # Cargar performance
            perf_file = self.storage_dir / "question_performance.json"
            if perf_file.exists():
                with open(perf_file, "r") as f:
                    data = json.load(f)
                    self.question_performance = {
                        q_id: QuestionPerformance(**perf_data) for q_id, perf_data in data.items()
                    }
                logger.info(f"📂 Loaded {len(self.question_performance)} question performances")

            # Cargar feedback
            feedback_file = self.storage_dir / "feedback_history.json"
            if feedback_file.exists():
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    self.feedback_history = deque([InterviewFeedback(**fb) for fb in data], maxlen=1000)
                logger.info(f"📂 Loaded {len(self.feedback_history)} feedback entries")

        except Exception as e:
            logger.warning(f"⚠️ Could not load previous state: {str(e)}")


def get_continuous_learning_system(
    storage_dir: str = ".cache/learning", exploration_rate: float = 0.1
) -> ContinuousLearningSystem:
    """Factory para obtener sistema de aprendizaje continuo"""
    return ContinuousLearningSystem(storage_dir=storage_dir, exploration_rate=exploration_rate)
