"""
Generador de Reportes Gráficos Interactivos
Crea reportes visuales con estadísticas y recomendaciones
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import hashlib


@dataclass
class InterviewMetrics:
    """Métricas calculadas de la entrevista"""

    average_score: float
    total_questions: int
    correct_answers: int
    hints_used: int
    total_time_seconds: int
    performance_by_topic: Dict[str, float]
    score_trend: List[float]  # Scores por pregunta
    time_per_question: List[int]  # Segundos por pregunta
    concepts_mastered: List[str]
    concepts_weak: List[str]

    def success_rate(self) -> float:
        """Porcentaje de respuestas correctas"""
        if self.total_questions == 0:
            return 0.0
        return (self.correct_answers / self.total_questions) * 100

    def percentile_rank(self) -> int:
        """Calcula percentil basado en average_score (mock)"""
        # TODO: Implementar comparación real con otros usuarios
        if self.average_score >= 9.0:
            return 95
        elif self.average_score >= 8.5:
            return 85
        elif self.average_score >= 8.0:
            return 75
        elif self.average_score >= 7.5:
            return 65
        elif self.average_score >= 7.0:
            return 50
        elif self.average_score >= 6.5:
            return 40
        else:
            return 25


@dataclass
class RecommendedResource:
    """Recurso recomendado para aprendizaje"""

    title: str
    type: str  # "book", "course", "article", "video"
    url: Optional[str]
    reason: str


@dataclass
class InterviewReport:
    """Reporte completo de entrevista"""

    interview_id: str
    candidate_name: str
    role: str
    date: datetime
    mode: str  # practice | exam
    metrics: InterviewMetrics
    strengths: List[str]
    improvements: List[str]
    recommended_resources: List[RecommendedResource]
    shareable_url: Optional[str]
    certificate_eligible: bool
    certificate_id: Optional[str]


class ReportGenerator:
    """
    Genera reportes interactivos de entrevistas.

    Features:
    - Análisis completo de performance
    - Gráficas con Chart.js
    - Recomendaciones personalizadas
    - Comparación con peers
    - Exportación a PDF/JSON
    - Links compartibles
    """

    def __init__(self, base_url: str = "https://ready4hire.com"):
        self.base_url = base_url

    def generate_report(self, interview_data: Dict, user_data: Dict) -> InterviewReport:
        """
        Genera reporte completo de entrevista.

        Args:
            interview_data: Datos de la entrevista completada
            user_data: Datos del candidato

        Returns:
            InterviewReport completo con métricas y recomendaciones
        """
        try:
            # Validar datos de entrada
            if not interview_data.get("answers_history"):
                raise ValueError("No hay respuestas en la entrevista")
            
            # Calcular métricas
            metrics = self._calculate_metrics(interview_data)

            # Análisis de fortalezas y debilidades
            strengths, improvements = self._analyze_performance(metrics, interview_data)

            # Generar recomendaciones
            resources = self._generate_recommendations(improvements, interview_data.get("role", "Unknown"))

            # Determinar elegibilidad para certificado
            mode = interview_data.get("mode", "practice")
            certificate_eligible, cert_id = self._check_certificate_eligibility(
                mode, metrics.average_score, metrics.hints_used
            )

            # Crear URL compartible
            shareable_url = self._create_shareable_url(interview_data["id"])

            # Parsear fecha de completación
            completed_at = interview_data.get("completed_at")
            if isinstance(completed_at, str):
                # Manejar diferentes formatos de fecha ISO
                try:
                    date = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                except ValueError:
                    # Intentar formato alternativo
                    date = datetime.fromisoformat(completed_at)
            elif isinstance(completed_at, datetime):
                date = completed_at
            else:
                date = datetime.now()

            return InterviewReport(
                interview_id=interview_data["id"],
                candidate_name=user_data.get("name", "Candidato"),
                role=interview_data.get("role", "Unknown"),
                date=date,
                mode=mode,
                metrics=metrics,
                strengths=strengths,
                improvements=improvements,
                recommended_resources=resources,
                shareable_url=shareable_url,
                certificate_eligible=certificate_eligible,
                certificate_id=cert_id,
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise RuntimeError(f"Error generando reporte: {str(e)}\n{error_details}")

    def _calculate_metrics(self, interview_data: Dict) -> InterviewMetrics:
        """Calcula todas las métricas de la entrevista"""
        answers = interview_data.get("answers_history", [])
        questions = interview_data.get("questions_history", [])

        if not answers:
            return InterviewMetrics(
                average_score=0.0,
                total_questions=0,
                correct_answers=0,
                hints_used=0,
                total_time_seconds=0,
                performance_by_topic={},
                score_trend=[],
                time_per_question=[],
                concepts_mastered=[],
                concepts_weak=[],
            )

        # Calcular score promedio
        scores = []
        for a in answers:
            score = a.get("score", 0)
            # Asegurar que score sea numérico
            if isinstance(score, (int, float)):
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        average_score = sum(scores) / len(scores) if scores else 0.0

        # Respuestas correctas (score >= 6.0)
        correct_answers = sum(1 for s in scores if s >= 6.0)

        # Hints usados
        hints_used = sum(int(a.get("hints_used", 0)) for a in answers)

        # Tiempo total
        total_time = sum(int(a.get("time_taken", 0)) for a in answers)

        # Performance por topic
        performance_by_topic = self._calculate_topic_performance(answers, questions)

        # Conceptos dominados y débiles
        concepts_mastered, concepts_weak = self._analyze_concepts(answers)

        # Tiempo por pregunta
        time_per_question = [int(a.get("time_taken", 0)) for a in answers]

        return InterviewMetrics(
            average_score=round(average_score, 2),
            total_questions=len(answers),
            correct_answers=correct_answers,
            hints_used=hints_used,
            total_time_seconds=total_time,
            performance_by_topic=performance_by_topic,
            score_trend=scores,
            time_per_question=time_per_question,
            concepts_mastered=concepts_mastered,
            concepts_weak=concepts_weak,
        )

    def _calculate_topic_performance(self, answers: List[Dict], questions: List[Dict]) -> Dict[str, float]:
        """Calcula performance promedio por topic"""
        topic_scores = {}
        topic_counts = {}

        for i, answer in enumerate(answers):
            if i < len(questions):
                topic = questions[i].get("topic", "General")
                score = answer.get("score", 0)
                # Asegurar que score sea numérico
                if not isinstance(score, (int, float)):
                    score = 0.0
                score = float(score)

                if topic not in topic_scores:
                    topic_scores[topic] = 0.0
                    topic_counts[topic] = 0

                topic_scores[topic] += score
                topic_counts[topic] += 1

        # Calcular promedios
        return {topic: round(topic_scores[topic] / topic_counts[topic], 2) for topic in topic_scores}

    def _analyze_concepts(self, answers: List[Dict]) -> Tuple[List[str], List[str]]:
        """Identifica conceptos dominados y débiles"""
        concept_scores = {}

        for answer in answers:
            eval_details = answer.get("evaluation_details", {})
            if not isinstance(eval_details, dict):
                eval_details = {}
            covered = eval_details.get("concepts_covered", [])
            missing = eval_details.get("missing_concepts", [])
            score = answer.get("score", 0)
            # Asegurar que score sea numérico
            if not isinstance(score, (int, float)):
                score = 0.0
            score = float(score)

            # Conceptos cubiertos con buen score
            if score >= 7.0:
                for concept in covered:
                    if concept not in concept_scores:
                        concept_scores[concept] = []
                    concept_scores[concept].append(score)

            # Conceptos faltantes o mal explicados
            if score < 7.0:
                for concept in missing:
                    if concept not in concept_scores:
                        concept_scores[concept] = []
                    concept_scores[concept].append(score)

        # Calcular promedios
        concept_avgs = {concept: sum(scores) / len(scores) for concept, scores in concept_scores.items()}

        # Separar en mastered (>=7.5) y weak (<7.0)
        mastered = [c for c, score in concept_avgs.items() if score >= 7.5]
        weak = [c for c, score in concept_avgs.items() if score < 7.0]

        return sorted(mastered[:10]), sorted(weak[:10])  # Top 10 de cada uno

    def _analyze_performance(self, metrics: InterviewMetrics, interview_data: Dict) -> Tuple[List[str], List[str]]:
        """Identifica fortalezas y áreas de mejora"""
        strengths = []
        improvements = []

        # Analizar score promedio
        if metrics.average_score >= 8.5:
            strengths.append(f"Excelente dominio general ({metrics.average_score}/10)")
        elif metrics.average_score >= 7.0:
            strengths.append(f"Buen desempeño general ({metrics.average_score}/10)")
        else:
            improvements.append(f"Fortalecer conocimientos base (score: {metrics.average_score}/10)")

        # Analizar success rate
        success_rate = metrics.success_rate()
        if success_rate >= 80:
            strengths.append(f"Alta tasa de éxito ({success_rate:.0f}% correctas)")
        elif success_rate < 60:
            improvements.append(f"Aumentar precisión (solo {success_rate:.0f}% correctas)")

        # Analizar uso de hints
        if metrics.hints_used == 0:
            strengths.append("Autonomía total - sin necesidad de pistas")
        elif metrics.hints_used > 5:
            improvements.append("Reducir dependencia de hints")

        # Analizar performance por topic
        for topic, score in metrics.performance_by_topic.items():
            if score >= 8.0:
                strengths.append(f"Dominio en {topic} ({score}/10)")
            elif score < 6.5:
                improvements.append(f"Reforzar {topic} ({score}/10)")

        # Analizar conceptos
        if metrics.concepts_mastered:
            strengths.append(f"Domina conceptos clave: {', '.join(metrics.concepts_mastered[:3])}")

        if metrics.concepts_weak:
            improvements.append(f"Estudiar: {', '.join(metrics.concepts_weak[:3])}")

        return strengths[:5], improvements[:5]  # Top 5 de cada uno

    def _generate_recommendations(self, improvements: List[str], role: str) -> List[RecommendedResource]:
        """Genera recursos recomendados basados en áreas de mejora"""
        resources = []

        # Mapeo de temas a recursos (mock - en producción usar DB o API)
        resource_map = {
            "algoritmos": RecommendedResource(
                "Introduction to Algorithms",
                "book",
                "https://mitpress.mit.edu/9780262046305",
                "Fortalece tu base en algoritmos y complejidad",
            ),
            "design patterns": RecommendedResource(
                "Design Patterns - Gang of Four",
                "book",
                "https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612",
                "Aprende los patrones de diseño clásicos",
            ),
            "testing": RecommendedResource(
                "Testing JavaScript - Kent C. Dodds",
                "course",
                "https://testingjavascript.com",
                "Mejora tus habilidades de testing",
            ),
        }

        # Buscar recursos relevantes
        for improvement in improvements:
            for key, resource in resource_map.items():
                if key.lower() in improvement.lower():
                    resources.append(resource)

        # Si no hay recursos específicos, agregar uno genérico
        if not resources:
            resources.append(
                RecommendedResource(
                    f"{role} Best Practices",
                    "article",
                    f"https://ready4hire.com/resources/{role.lower().replace(' ', '-')}",
                    "Mejores prácticas y recursos para tu rol",
                )
            )

        return resources[:5]  # Max 5 recursos

    def _check_certificate_eligibility(
        self, mode: str, average_score: float, hints_used: int
    ) -> Tuple[bool, Optional[str]]:
        """Determina si el candidato es elegible para certificado"""
        # Condiciones para certificación:
        # 1. Score >= 7.5 (en cualquier modo)
        # 2. En modo examen: sin usar hints
        # 3. En modo práctica: permitir hints pero con score más alto (>= 8.0)

        if mode == "exam":
            # En modo examen: score >= 7.5 y sin hints
            if average_score >= 7.5 and hints_used == 0:
                cert_id = self._generate_certificate_id(average_score)
                return True, cert_id
        elif mode == "practice":
            # En modo práctica: score >= 8.0 (más estricto por permitir hints)
            if average_score >= 8.0:
                cert_id = self._generate_certificate_id(average_score)
                return True, cert_id

        return False, None

    def _generate_certificate_id(self, score: float) -> str:
        """Genera ID único para certificado"""
        timestamp = datetime.now().isoformat()
        data = f"{timestamp}_{score}".encode()
        hash_obj = hashlib.sha256(data)
        return f"R4H-{hash_obj.hexdigest()[:12].upper()}"

    def _create_shareable_url(self, interview_id: str) -> str:
        """Crea URL compartible del reporte"""
        return f"{self.base_url}/reports/{interview_id}"

    def export_to_json(self, report: InterviewReport) -> str:
        """Exporta reporte a JSON"""
        return json.dumps(
            {
                "interview_id": report.interview_id,
                "candidate": report.candidate_name,
                "role": report.role,
                "date": report.date.isoformat(),
                "mode": report.mode,
                "metrics": {
                    "average_score": report.metrics.average_score,
                    "total_questions": report.metrics.total_questions,
                    "correct_answers": report.metrics.correct_answers,
                    "success_rate": report.metrics.success_rate(),
                    "hints_used": report.metrics.hints_used,
                    "percentile": report.metrics.percentile_rank(),
                },
                "strengths": report.strengths,
                "improvements": report.improvements,
                "recommended_resources": [
                    {"title": r.title, "type": r.type, "url": r.url, "reason": r.reason}
                    for r in report.recommended_resources
                ],
                "certificate": (
                    {"eligible": report.certificate_eligible, "id": report.certificate_id}
                    if report.certificate_eligible
                    else None
                ),
            },
            indent=2,
        )


# Factory function
_report_generator = None


def get_report_generator() -> ReportGenerator:
    """Obtiene instancia singleton del generador de reportes"""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator
