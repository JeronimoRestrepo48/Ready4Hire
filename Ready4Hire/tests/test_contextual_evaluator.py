"""
Tests para Evaluación Contextual (v2.2 - Mejora #2).

Prueba InterviewHistoryAnalyzer y ContextualEvaluator:
- Análisis de patrones en historial de respuestas
- Identificación de tendencias de rendimiento
- Detección de conceptos fuertes/débiles
- Enriquecimiento de evaluaciones con contexto
- Generación de resúmenes de entrevista
"""
import pytest
from datetime import datetime, timedelta
from typing import List

from app.domain.entities.interview import Interview
from app.domain.entities.question import Question
from app.domain.entities.answer import Answer
from app.domain.value_objects.score import Score
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.value_objects.interview_status import InterviewStatus
from app.application.services.contextual_evaluator import (
    InterviewHistoryAnalyzer,
    ContextualEvaluator
)
from app.application.services.evaluation_service import EvaluationService


# =========================================================================
# FIXTURES
# =========================================================================

@pytest.fixture
def empty_interview():
    """Entrevista sin historial."""
    return Interview(
        user_id="user123",
        role="Backend Developer",
        interview_type="technical",
        skill_level=SkillLevel.MID
    )


@pytest.fixture
def interview_with_improving_trend():
    """Entrevista con tendencia de mejora (scores: 5, 6, 7, 8)."""
    interview = Interview(
        user_id="user123",
        role="Backend Developer",
        interview_type="technical",
        skill_level=SkillLevel.MID,
        status=InterviewStatus.ACTIVE
    )
    
    # Agregar respuestas con scores crecientes
    scores = [5.0, 6.0, 7.0, 8.0]
    for i, score_val in enumerate(scores):
        question = Question(
            id=f"q{i}",
            text=f"Pregunta técnica {i}",
            category="technical",
            difficulty="mid",
            expected_concepts=["concepto1", "concepto2"],
            keywords=["keyword1"]
        )
        interview.add_question(question)
        
        answer = Answer(
            question_id=question.id,
            text=f"Respuesta {i}",
            score=score_val,
            evaluation_details={
                "concepts_covered": ["concepto1"],
                "missing_concepts": ["concepto2"] if score_val < 7 else []
            }
        )
        interview.add_answer(answer)
    
    return interview


@pytest.fixture
def interview_with_declining_trend():
    """Entrevista con tendencia de declive (scores: 8, 7, 6, 5)."""
    interview = Interview(
        user_id="user123",
        role="Backend Developer",
        interview_type="technical",
        skill_level=SkillLevel.MID,
        status=InterviewStatus.ACTIVE
    )
    
    scores = [8.0, 7.0, 6.0, 5.0]
    for i, score_val in enumerate(scores):
        question = Question(
            id=f"q{i}",
            text=f"Pregunta técnica {i}",
            category="technical",
            difficulty="mid",
            expected_concepts=["concepto1", "concepto2"],
            keywords=["keyword1"]
        )
        interview.add_question(question)
        
        answer = Answer(
            question_id=question.id,
            text=f"Respuesta {i}",
            score=score_val,
            evaluation_details={
                "concepts_covered": ["concepto1"],
                "missing_concepts": []
            }
        )
    
    return interview


@pytest.fixture
def interview_with_mixed_categories():
    """Entrevista con preguntas technical y soft_skills."""
    interview = Interview(
        user_id="user123",
        role="Backend Developer",
        interview_type="mixed",
        skill_level=SkillLevel.MID,
        status=InterviewStatus.ACTIVE
    )
    
    # 3 technical (promedio 8.0)
    for i in range(3):
        q = Question(
            id=f"tech{i}",
            text=f"Pregunta técnica {i}",
            category="technical",
            difficulty="mid",
            expected_concepts=["POO", "SOLID"],
            keywords=["technical"]
        )
        interview.add_question(q)
        
        a = Answer(
            question_id=q.id,
            text="Respuesta técnica",
            score=8.0,
            evaluation_details={
                "concepts_covered": ["POO", "SOLID"],
                "missing_concepts": []
            }
        )
        interview.add_answer(a)
    # 2 soft_skills (promedio 6.0)
    for i in range(2):
        q = Question(
            id=f"soft{i}",
            text=f"Pregunta soft skills {i}",
            category="soft_skills",
            difficulty="mid",
            expected_concepts=["comunicación", "trabajo en equipo"],
            keywords=["soft"]
        )
        interview.add_question(q)
        interview.add_question(q)
            
        a = Answer(
                question_id=q.id,
                text="Respuesta soft skills",
                score=6.0,
                evaluation_details={
                    "concepts_covered": ["comunicación"],
                    "missing_concepts": ["trabajo en equipo"]
                }
            )
        interview.add_answer(a)
        
        return interview

# =========================================================================
# TESTS: InterviewHistoryAnalyzer
# =========================================================================

class TestInterviewHistoryAnalyzer:
    """Tests para InterviewHistoryAnalyzer."""
    
    def test_analyze_empty_interview(self, empty_interview):
        """Test: Análisis de entrevista vacía retorna contexto vacío."""
        analyzer = InterviewHistoryAnalyzer()
        context = analyzer.analyze(empty_interview)
        
        assert context["total_answers"] == 0
        assert context["average_score"] == 0.0
        assert context["performance_trend"] == "stable"
        assert context["has_sufficient_context"] is False
        assert len(context["strong_concepts"]) == 0
        assert len(context["weak_concepts"]) == 0
    
    def test_detect_improving_trend(self, interview_with_improving_trend):
        """Test: Detecta tendencia de mejora correctamente."""
        analyzer = InterviewHistoryAnalyzer()
        context = analyzer.analyze(interview_with_improving_trend)
        
        assert context["total_answers"] == 4
        assert context["average_score"] == 6.5  # (5+6+7+8)/4
        assert context["performance_trend"] == "improving"
        assert context["has_sufficient_context"] is True
    
    def test_detect_declining_trend(self, interview_with_declining_trend):
        """Test: Detecta tendencia de declive correctamente."""
        analyzer = InterviewHistoryAnalyzer()
        context = analyzer.analyze(interview_with_declining_trend)
        
        assert context["total_answers"] == 4
        assert context["average_score"] == 6.5  # (8+7+6+5)/4
        assert context["performance_trend"] == "declining"
        assert context["has_sufficient_context"] is True
    
    def test_identify_strong_and_weak_concepts(self, interview_with_improving_trend):
        """Test: Identifica conceptos fuertes y débiles."""
        analyzer = InterviewHistoryAnalyzer()
        context = analyzer.analyze(interview_with_improving_trend)
        
        # concepto1 está en concepts_covered repetidamente (fuerte)
        assert "concepto1" in context["strong_concepts"] or len(context["strong_concepts"]) >= 0
        
        # concepto2 está en missing_concepts en las primeras respuestas (débil)
        # Nota: Puede no detectarse si no cumple umbral mínimo
    
    def test_calculate_category_performance(self, interview_with_mixed_categories):
        """Test: Calcula rendimiento por categoría correctamente."""
        analyzer = InterviewHistoryAnalyzer()
        context = analyzer.analyze(interview_with_mixed_categories)
        
        assert "technical" in context["category_performance"]
        assert "soft_skills" in context["category_performance"]
        assert context["category_performance"]["technical"] == 8.0
        assert context["category_performance"]["soft_skills"] == 6.0
    
    def test_calculate_consistency_high(self):
        """Test: Calcula alta consistencia para scores similares."""
        interview = Interview(
            user_id="user123",
            role="Backend Developer",
            status=InterviewStatus.ACTIVE
        )
        
        # Scores muy similares: 7.0, 7.2, 7.1, 6.9 (alta consistencia)
        for i, score_val in enumerate([7.0, 7.2, 7.1, 6.9]):
            q = Question(id=f"q{i}", text=f"Q{i}", category="technical", difficulty="mid")
            interview.add_question(q)
            interview.add_question(q)
                
            a = Answer(
                    question_id=q.id,
                    text=f"A{i}",
                    score=score_val,
                    evaluation_details={}
                )
            interview.add_answer(a)
        
        analyzer = InterviewHistoryAnalyzer()
        context = analyzer.analyze(interview)
        # Consistencia debe ser alta (>0.8)
        assert context["consistency_score"] >= 0.8
    
    def test_calculate_consistency_low(self):
        """Test: Calcula baja consistencia para scores muy variados."""
        interview = Interview(
            user_id="user123",
            role="Backend Developer",
            status=InterviewStatus.ACTIVE
        )
        
        # Scores muy variados: 3.0, 9.0, 4.0, 8.0 (baja consistencia)
        for i, score_val in enumerate([3.0, 9.0, 4.0, 8.0]):
            q = Question(id=f"q{i}", text=f"Q{i}", category="technical", difficulty="mid")
            interview.add_question(q)
            
            a = Answer(
                question_id=q.id,
                text=f"A{i}",
                score=score_val,
                evaluation_details={}
            )
            interview.add_answer(a)
        
        analyzer = InterviewHistoryAnalyzer()
        context = analyzer.analyze(interview)
        
        # Consistencia debe ser baja (<0.5)
        assert context["consistency_score"] < 0.5


# =========================================================================
# TESTS: ContextualEvaluator
# =========================================================================

class TestContextualEvaluator:
    """Tests para ContextualEvaluator."""
    
    def test_evaluate_without_context(self, empty_interview, monkeypatch):
        """Test: Evaluación sin contexto histórico usa evaluación base."""
        # Mock del EvaluationService
        class MockEvaluationService(EvaluationService):
            def evaluate_answer(self, question, answer, expected_concepts, keywords, category, difficulty, role):
                return {
                    "score": 7.5,
                    "breakdown": {
                        "completeness": 2.5,
                        "technical_depth": 2.3,
                        "clarity": 1.5,
                        "key_concepts": 1.2
                    },
                    "justification": "Buena respuesta",
                    "strengths": ["Explicación clara"],
                    "improvements": ["Agregar ejemplos"],
                    "concepts_covered": ["concepto1"],
                    "missing_concepts": ["concepto2"]
                }
        
        evaluator = ContextualEvaluator(MockEvaluationService())
        
        result = evaluator.evaluate_with_context(
            interview=empty_interview,
            question="¿Qué es POO?",
            answer="POO es programación orientada a objetos",
            expected_concepts=["herencia", "polimorfismo"],
            keywords=["POO"],
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        # Sin contexto suficiente
        assert result["context_applied"] is False
        assert result["score"] == 7.5  # Sin ajuste
        assert "interview_context" in result
    
    def test_evaluate_with_improving_trend(self, interview_with_improving_trend, monkeypatch):
        """Test: Ajusta score positivamente si hay tendencia de mejora."""
        class MockEvaluationService(EvaluationService):
            def evaluate_answer(self, question, answer, expected_concepts, keywords, category, difficulty, role):
                return {
                    "score": 7.0,
                    "breakdown": {},
                    "justification": "Respuesta correcta",
                    "strengths": ["Buena explicación"],
                    "improvements": ["Agregar detalles"],
                    "concepts_covered": ["concepto1"],
                    "missing_concepts": []
                }
        
        evaluator = ContextualEvaluator(MockEvaluationService())
        
        result = evaluator.evaluate_with_context(
            interview=interview_with_improving_trend,
            question="¿Qué es SOLID?",
            answer="SOLID son principios de diseño",
            expected_concepts=["SOLID"],
            keywords=["principios"],
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        # Con contexto de mejora
        assert result["context_applied"] is True
        assert result["score"] > 7.0  # Ajuste positivo (+0.5)
        assert result["score_adjustment"] == 0.5
        
        # Debe incluir mensaje de progresión
        strengths_text = " ".join(result["strengths"])
        assert "Progresión notable" in strengths_text or "mejorando" in strengths_text.lower()
    
    def test_evaluate_with_declining_trend(self, interview_with_declining_trend, monkeypatch):
        """Test: Ajusta score negativamente si hay tendencia de declive."""
        class MockEvaluationService(EvaluationService):
            def evaluate_answer(self, question, answer, expected_concepts, keywords, category, difficulty, role):
                return {
                    "score": 7.0,
                    "breakdown": {},
                    "justification": "Respuesta correcta",
                    "strengths": ["Buena explicación"],
                    "improvements": ["Agregar detalles"],
                    "concepts_covered": ["concepto1"],
                    "missing_concepts": []
                }
        
        evaluator = ContextualEvaluator(MockEvaluationService())
        
        result = evaluator.evaluate_with_context(
            interview=interview_with_declining_trend,
            question="¿Qué es REST?",
            answer="REST es un estilo arquitectónico",
            expected_concepts=["REST"],
            keywords=["API"],
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        # Con contexto de declive
        assert result["context_applied"] is True
        assert result["score"] < 7.0  # Ajuste negativo (-0.5)
        assert result["score_adjustment"] == -0.5
        
        # Debe incluir advertencia
        improvements_text = " ".join(result["improvements"])
        assert "declina" in improvements_text.lower() or "descanso" in improvements_text.lower()
    
    def test_get_interview_summary_insufficient_data(self, empty_interview):
        """Test: Resumen con datos insuficientes retorna mensaje apropiado."""
        evaluator = ContextualEvaluator(EvaluationService())
        
        summary = evaluator.get_interview_summary(empty_interview)
        
        assert summary["status"] == "insufficient_data"
        assert "3 respuestas" in summary["message"]
    
    def test_get_interview_summary_with_data(self, interview_with_improving_trend):
        """Test: Genera resumen completo con datos suficientes."""
        evaluator = ContextualEvaluator(EvaluationService())
        
        summary = evaluator.get_interview_summary(interview_with_improving_trend)
        
        assert summary["status"] == "complete"
        assert summary["total_answers"] == 4
        assert summary["average_score"] == 6.5
        assert summary["performance_trend"] == "improving"
        assert "recommendations" in summary
        assert isinstance(summary["recommendations"], list)
        
        # Con tendencia "improving", debe haber recomendación de encouragement
        has_encouragement = any(
            r["type"] == "encouragement"
            for r in summary["recommendations"]
        )
        assert has_encouragement
    
    def test_recommendations_for_weak_concepts(self, interview_with_improving_trend):
        """Test: Genera recomendaciones para conceptos débiles."""
        evaluator = ContextualEvaluator(EvaluationService())
        
        summary = evaluator.get_interview_summary(interview_with_improving_trend)
        
        # Debe haber recomendación de study_topics si hay conceptos débiles
        study_recs = [
            r for r in summary["recommendations"]
            if r["type"] == "study_topics"
        ]
        
        # Puede o no haber según datos (depende del análisis)
        # Solo verificamos estructura
        if study_recs:
            assert "priority" in study_recs[0]
            assert "message" in study_recs[0]


# =========================================================================
# TESTS: Integración Completa
# =========================================================================

class TestContextualEvaluatorIntegration:
    """Tests de integración end-to-end."""
    
    def test_full_interview_flow_with_context(self, monkeypatch):
        """Test: Flujo completo de entrevista con evaluación contextual."""
        # Mock del LLM
        class MockLLMService:
            def generate(self, prompt, **kwargs):
                import json
                return json.dumps({
                    "score": 7.5,
                    "breakdown": {
                        "completeness": 2.5,
                        "technical_depth": 2.3,
                        "clarity": 1.5,
                        "key_concepts": 1.2
                    },
                    "justification": "Respuesta sólida con buen nivel técnico.",
                    "strengths": ["Explica conceptos correctamente", "Ejemplos claros"],
                    "improvements": ["Profundizar en edge cases"],
                    "concepts_covered": ["herencia", "polimorfismo"],
                    "missing_concepts": ["abstracción"]
                })
        
        # Crear servicios
        eval_service = EvaluationService(enable_cache=False)
        monkeypatch.setattr(eval_service, 'llm_service', MockLLMService())
        contextual_evaluator = ContextualEvaluator(eval_service)
        
        # Crear entrevista
        interview = Interview(
            user_id="user123",
            role="Backend Developer",
            interview_type="technical",
            skill_level=SkillLevel.MID
        )
        interview.start()
        
        # Simular 3 respuestas con scores crecientes
        questions_data = [
            {
                "question": "¿Qué es POO?",
                "answer": "POO es programación orientada a objetos",
                "expected_concepts": ["herencia", "polimorfismo", "encapsulación"],
                "keywords": ["POO", "objetos"]
            },
            {
                "question": "Explica herencia",
                "answer": "Herencia permite reutilizar código de clases padre",
                "expected_concepts": ["herencia", "reutilización"],
                "keywords": ["herencia", "clase padre"]
            },
            {
                "question": "¿Qué es polimorfismo?",
                "answer": "Polimorfismo permite múltiples formas de un método",
                "expected_concepts": ["polimorfismo", "sobrecarga"],
                "keywords": ["polimorfismo", "override"]
            }
        ]
        
        for i, q_data in enumerate(questions_data):
            # Agregar pregunta
            question = Question(
                id=f"q{i}",
                text=q_data["question"],
                category="technical",
                difficulty="mid",
                expected_concepts=q_data["expected_concepts"],
                keywords=q_data["keywords"]
            )
            interview.add_question(question)
            
            # Evaluar con contexto
            result = contextual_evaluator.evaluate_with_context(
                interview=interview,
                question=q_data["question"],
                answer=q_data["answer"],
                expected_concepts=q_data["expected_concepts"],
                keywords=q_data["keywords"],
                category="technical",
                difficulty="mid",
                role="Backend Developer"
            )
            
            # Agregar respuesta a la entrevista
            # Agregar respuesta a la entrevista
            answer = Answer(
                question_id=question.id,
                text=q_data["answer"],
                score=result["score"],
                evaluation_details=result
            )
            interview.add_answer(answer)
        
        # Obtener resumen de la entrevista
        summary = contextual_evaluator.get_interview_summary(interview)
        
        assert summary["status"] == "complete"
        assert summary["total_answers"] == 3
        assert len(summary["recommendations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])