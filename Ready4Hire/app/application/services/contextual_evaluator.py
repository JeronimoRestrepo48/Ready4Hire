"""
Evaluador Contextual (v2.2 - Mejora #2)

Evalúa respuestas considerando el contexto completo de la entrevista:
- Respuestas anteriores del candidato
- Patrones de comportamiento (fortalezas, debilidades recurrentes)
- Progresión del candidato (mejora/empeora con el tiempo)
- Consistencia técnica entre respuestas

Mejora la precisión de evaluación al tener contexto histórico completo.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.domain.entities.interview import Interview
from app.domain.entities.answer import Answer
from app.application.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)


class InterviewHistoryAnalyzer:
    """
    Analiza el historial de una entrevista para extraer patrones y contexto.

    Identifica:
    - Conceptos recurrentemente fuertes/débiles
    - Tendencias de rendimiento (mejorando/empeorando)
    - Áreas técnicas dominadas vs áreas con gaps
    - Consistencia en respuestas relacionadas
    """

    def __init__(self):
        """Inicializa el analizador de historial."""
        pass

    def analyze(self, interview: Interview) -> Dict[str, Any]:
        """
        Analiza el historial completo de la entrevista.

        Args:
            interview: Entrevista con historial de respuestas

        Returns:
            Dict con análisis contextual:
            {
                "total_answers": int,
                "average_score": float,
                "performance_trend": "improving" | "declining" | "stable",
                "strong_concepts": List[str],
                "weak_concepts": List[str],
                "category_performance": Dict[str, float],
                "consistency_score": float,  # 0-1
                "has_sufficient_context": bool
            }
        """
        answers = interview.answers_history

        if not answers:
            return self._empty_context()

        # Análisis básico
        total_answers = len(answers)
        scores = [a.score for a in answers if a.score]
        average_score = sum(scores) / len(scores) if scores else 0.0

        # Tendencia de rendimiento (últimos 3 vs primeros 3)
        performance_trend = self._calculate_performance_trend(scores)

        # Conceptos fuertes y débiles
        strong_concepts, weak_concepts = self._identify_concept_patterns(answers)

        # Performance por categoría
        category_performance = self._calculate_category_performance(interview.questions_history, answers)

        # Consistencia (qué tan parecidos son los scores entre preguntas similares)
        consistency_score = self._calculate_consistency(answers)

        return {
            "total_answers": total_answers,
            "average_score": round(average_score, 2),
            "performance_trend": performance_trend,
            "strong_concepts": strong_concepts[:5],  # Top 5
            "weak_concepts": weak_concepts[:5],  # Top 5
            "category_performance": category_performance,
            "consistency_score": round(consistency_score, 2),
            "has_sufficient_context": total_answers >= 3,
        }

    def _empty_context(self) -> Dict[str, Any]:
        """Retorna contexto vacío para entrevistas sin historial."""
        return {
            "total_answers": 0,
            "average_score": 0.0,
            "performance_trend": "stable",
            "strong_concepts": [],
            "weak_concepts": [],
            "category_performance": {},
            "consistency_score": 0.0,
            "has_sufficient_context": False,
        }

    def _calculate_performance_trend(self, scores: List[float]) -> str:
        """
        Calcula la tendencia de rendimiento.
        Compara los primeros N con los últimos N scores.
        """
        if len(scores) < 4:
            return "stable"

        # Comparar primeros 3 vs últimos 3
        n = min(3, len(scores) // 2)
        first_avg = sum(scores[:n]) / n
        last_avg = sum(scores[-n:]) / n

        diff = last_avg - first_avg

        if diff > 1.0:  # Mejora significativa
            return "improving"
        elif diff < -1.0:  # Declive significativo
            return "declining"
        else:
            return "stable"

    def _identify_concept_patterns(self, answers: List[Answer]) -> tuple[List[str], List[str]]:
        """
        Identifica conceptos recurrentemente fuertes y débiles.

        Returns:
            (strong_concepts, weak_concepts)
        """
        # Contadores de conceptos por rendimiento
        concept_scores: Dict[str, List[float]] = {}

        for answer in answers:
            if not answer.evaluation_details:
                continue

            score = answer.score if answer.score else 5.0

            # Conceptos cubiertos (fuertes)
            concepts_covered = answer.evaluation_details.get("concepts_covered", [])
            for concept in concepts_covered:
                if concept not in concept_scores:
                    concept_scores[concept] = []
                concept_scores[concept].append(score)

            # Conceptos faltantes (débiles)
            missing_concepts = answer.evaluation_details.get("missing_concepts", [])
            for concept in missing_concepts:
                if concept not in concept_scores:
                    concept_scores[concept] = []
                concept_scores[concept].append(score - 3.0)  # Penalizar ausencia

        # Calcular promedios
        concept_avgs = {
            concept: sum(scores) / len(scores)
            for concept, scores in concept_scores.items()
            if len(scores) >= 2  # Mínimo 2 menciones
        }

        # Ordenar
        sorted_concepts = sorted(concept_avgs.items(), key=lambda x: x[1], reverse=True)

        # Separar fuertes (>7) y débiles (<5)
        strong = [c for c, avg in sorted_concepts if avg >= 7.0]
        weak = [c for c, avg in sorted_concepts if avg < 5.0]

        return strong, weak

    def _calculate_category_performance(self, questions: List, answers: List[Answer]) -> Dict[str, float]:
        """
        Calcula el rendimiento promedio por categoría (technical, soft_skills).
        """
        from app.domain.entities.question import Question

        category_scores: Dict[str, List[float]] = {}

        for i, answer in enumerate(answers):
            if i >= len(questions):
                continue

            question = questions[i]
            if not isinstance(question, Question):
                continue

            category = question.category
            score = answer.score if answer.score else 5.0

            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)

        # Promedios
        return {category: round(sum(scores) / len(scores), 2) for category, scores in category_scores.items() if scores}

    def _calculate_consistency(self, answers: List[Answer]) -> float:
        """
        Calcula qué tan consistente es el candidato.
        Mide la varianza de los scores (menor varianza = más consistente).
        """
        scores = [a.score for a in answers if a.score]

        if len(scores) < 2:
            return 1.0  # Perfectamente consistente si solo hay 1 respuesta

        # Calcular desviación estándar normalizada
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = variance**0.5

        # Normalizar (0 = muy inconsistente, 1 = muy consistente)
        # Asumiendo que std_dev máximo razonable es 3.0 (10 puntos de escala)
        consistency = max(0.0, 1.0 - (std_dev / 3.0))

        return consistency


class ContextualEvaluator:
    """
    Evaluador que considera el contexto completo de la entrevista.

    Usa EvaluationService base + contexto histórico para:
    - Ajustar severidad de evaluación según progresión
    - Identificar inconsistencias técnicas
    - Dar feedback más personalizado
    - Reconocer mejoras del candidato
    """

    def __init__(
        self, evaluation_service: EvaluationService, history_analyzer: Optional[InterviewHistoryAnalyzer] = None
    ):
        """
        Inicializa el evaluador contextual.

        Args:
            evaluation_service: Servicio base de evaluación
            history_analyzer: Analizador de historial (se crea uno si no se provee)
        """
        self.evaluation_service = evaluation_service
        self.history_analyzer = history_analyzer or InterviewHistoryAnalyzer()
        logger.info("ContextualEvaluator inicializado")

    def evaluate_with_context(
        self,
        interview: Interview,
        question: str,
        answer: str,
        expected_concepts: List[str],
        keywords: List[str],
        category: str,
        difficulty: str,
        role: str,
    ) -> Dict[str, Any]:
        """
        Evalúa respuesta considerando el contexto de la entrevista.

        Args:
            interview: Entrevista con historial
            question: Texto de la pregunta
            answer: Respuesta del candidato
            expected_concepts: Conceptos esperados
            keywords: Palabras clave
            category: Categoría (technical, soft_skills)
            difficulty: Dificultad (junior, mid, senior)
            role: Rol del candidato

        Returns:
            Dict con evaluación mejorada + contexto histórico
        """
        # 1. Analizar historial
        context = self.history_analyzer.analyze(interview)

        # 2. Evaluar respuesta con servicio base
        base_evaluation = self.evaluation_service.evaluate_answer(
            question=question,
            answer=answer,
            expected_concepts=expected_concepts,
            keywords=keywords,
            category=category,
            difficulty=difficulty,
            role=role,
        )

        # 3. Enriquecer con contexto
        contextual_evaluation = self._enrich_with_context(
            base_evaluation=base_evaluation, context=context, expected_concepts=expected_concepts
        )

        logger.debug(
            f"Evaluación contextual completada: "
            f"score={contextual_evaluation['score']}, "
            f"context_used={'Yes' if context['has_sufficient_context'] else 'No'}"
        )

        return contextual_evaluation

    def _enrich_with_context(
        self, base_evaluation: Dict[str, Any], context: Dict[str, Any], expected_concepts: List[str]
    ) -> Dict[str, Any]:
        """
        Enriquece la evaluación base con contexto histórico.

        Ajustes:
        - Feedback más específico basado en patrones previos
        - Reconocimiento de mejoras (si performance_trend = "improving")
        - Señalamiento de inconsistencias (si concept débil pero ahora bien)
        - Ajuste leve de score según tendencia (+0.5 si mejorando, -0.5 si declinando)
        """
        enriched = base_evaluation.copy()

        # Agregar contexto al resultado
        enriched["interview_context"] = {
            "total_previous_answers": context["total_answers"],
            "average_score": context["average_score"],
            "performance_trend": context["performance_trend"],
            "consistency_score": context["consistency_score"],
        }

        if not context["has_sufficient_context"]:
            # Sin contexto suficiente, retornar evaluación base
            enriched["context_applied"] = False
            return enriched

        enriched["context_applied"] = True

        # Ajuste de score según tendencia (sutil, ±0.5 puntos)
        score_adjustment = 0.0
        if context["performance_trend"] == "improving":
            score_adjustment = 0.5
            enriched["strengths"].insert(0, "🎯 Progresión notable: tu rendimiento está mejorando consistentemente")
        elif context["performance_trend"] == "declining":
            score_adjustment = -0.5
            enriched["improvements"].insert(
                0, "⚠️ Atención: tu rendimiento está declinando. Considera tomar un descanso o revisar conceptos básicos"
            )

        # Ajustar score (mantener entre 0-10)
        original_score = enriched["score"]
        enriched["score"] = max(0.0, min(10.0, original_score + score_adjustment))
        enriched["score_adjustment"] = score_adjustment

        # Feedback sobre conceptos recurrentes
        strong_in_history = set(context["strong_concepts"])
        weak_in_history = set(context["weak_concepts"])
        current_concepts = set(expected_concepts)

        # Si concepto históricamente débil ahora está bien cubierto
        improved_concepts = weak_in_history & set(enriched.get("concepts_covered", []))
        if improved_concepts:
            enriched["strengths"].append(
                f"✨ Mejora notable en: {', '.join(list(improved_concepts)[:2])} " f"(anteriormente eran áreas débiles)"
            )

        # Si concepto históricamente fuerte ahora falta
        regressed_concepts = strong_in_history & set(enriched.get("missing_concepts", []))
        if regressed_concepts:
            enriched["improvements"].append(
                f"🔄 Inconsistencia detectada: {', '.join(list(regressed_concepts)[:2])} "
                f"(anteriormente dominabas estos conceptos)"
            )

        # Mensaje de consistencia
        if context["consistency_score"] >= 0.8:
            enriched["strengths"].append(
                f"🎖️ Rendimiento muy consistente (consistencia: {context['consistency_score']:.0%})"
            )
        elif context["consistency_score"] < 0.5:
            enriched["improvements"].append(
                f"📊 Tu rendimiento varía mucho entre preguntas (consistencia: {context['consistency_score']:.0%}). "
                f"Intenta mantener un nivel más estable"
            )

        return enriched

    def get_interview_summary(self, interview: Interview) -> Dict[str, Any]:
        """
        Genera un resumen completo del rendimiento en la entrevista.

        Args:
            interview: Entrevista completada

        Returns:
            Dict con resumen y recomendaciones personalizadas
        """
        context = self.history_analyzer.analyze(interview)

        if not context["has_sufficient_context"]:
            return {"status": "insufficient_data", "message": "Se necesitan al menos 3 respuestas para generar resumen"}

        # Generar recomendaciones
        recommendations = []

        # Basado en conceptos débiles
        if context["weak_concepts"]:
            recommendations.append(
                {
                    "priority": "high",
                    "type": "study_topics",
                    "message": f"Reforzar conocimientos en: {', '.join(context['weak_concepts'][:3])}",
                    "concepts": context["weak_concepts"],
                }
            )

        # Basado en tendencia
        if context["performance_trend"] == "declining":
            recommendations.append(
                {
                    "priority": "medium",
                    "type": "rest",
                    "message": "Tu rendimiento está declinando. Considera tomar un descanso antes de continuar.",
                }
            )
        elif context["performance_trend"] == "improving":
            recommendations.append(
                {
                    "priority": "low",
                    "type": "encouragement",
                    "message": "¡Excelente progresión! Continúa con este ritmo.",
                }
            )

        # Basado en consistencia
        if context["consistency_score"] < 0.5:
            recommendations.append(
                {
                    "priority": "medium",
                    "type": "consistency",
                    "message": "Tu rendimiento varía mucho. Intenta mantener concentración constante durante toda la entrevista.",
                }
            )

        return {
            "status": "complete",
            "total_answers": context["total_answers"],
            "average_score": context["average_score"],
            "performance_trend": context["performance_trend"],
            "consistency_score": context["consistency_score"],
            "strong_areas": context["strong_concepts"],
            "weak_areas": context["weak_concepts"],
            "category_performance": context["category_performance"],
            "recommendations": recommendations,
        }
