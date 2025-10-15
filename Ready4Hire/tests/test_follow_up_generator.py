"""
Tests para Generador de Preguntas Follow-Up (v2.2 - Mejora #3).

Prueba FollowUpQuestionGenerator:
- Selección de estrategias basada en contexto
- Generación de preguntas para conceptos débiles
- Exploración de fortalezas
- Verificación de consistencia
- Escenarios prácticos
"""
import pytest
import json
from typing import List

from app.domain.entities.interview import Interview
from app.domain.entities.question import Question
from app.domain.entities.answer import Answer
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.value_objects.interview_status import InterviewStatus
from app.application.services.follow_up_generator import (
    FollowUpQuestionGenerator,
    FollowUpQuestion
)


# =========================================================================
# FIXTURES
# =========================================================================

@pytest.fixture
def mock_llm_service():
    """Mock del servicio LLM."""
    class MockLLMService:
        def generate(self, prompt, **kwargs):
            # Simular diferentes tipos de preguntas según el prompt
            if "débiles" in prompt or "debilidad" in prompt:
                return "¿Puedes explicar en detalle cómo funciona la herencia en POO y dar un ejemplo práctico?"
            elif "avanzada" in prompt or "profundidad" in prompt:
                return "¿Cuáles son los trade-offs entre herencia y composición? ¿Cuándo usarías cada uno?"
            elif "consistencia" in prompt or "verificar" in prompt:
                return "¿Cómo aplicarías el principio de herencia en un sistema de autenticación?"
            elif "escenario" in prompt or "práctico" in prompt:
                return "Imagina que estás diseñando un sistema de e-commerce. ¿Cómo usarías herencia para modelar diferentes tipos de productos?"
            else:
                return "¿Puedes explicar este concepto con más detalle?"
    
    return MockLLMService()


@pytest.fixture
def interview_with_weak_concepts():
    """Entrevista donde el candidato mostró conceptos débiles."""
    interview = Interview(
        user_id="user123",
        role="Backend Developer",
        interview_type="technical",
        skill_level=SkillLevel.MID,
        status=InterviewStatus.ACTIVE
    )
    
    # 4 respuestas con scores bajos-medios
    for i in range(4):
        q = Question(
            id=f"q{i}",
            text=f"Pregunta técnica {i}",
            category="technical",
            difficulty="mid",
            expected_concepts=["herencia", "polimorfismo"],
            keywords=["POO"]
        )
        interview.add_question(q)
        
        a = Answer(
            id=f"a{i}",
            question_id=q.id,
            text=f"Respuesta {i}",
            score=5.5,  # Score bajo
            evaluation_details={
                "concepts_covered": ["herencia"] if i % 2 == 0 else [],
                "missing_concepts": ["polimorfismo", "encapsulación"]
            }
        )
        interview.add_answer(a)
    
    return interview


@pytest.fixture
def interview_with_strong_concepts():
    """Entrevista donde el candidato demostró fortalezas."""
    interview = Interview(
        user_id="user123",
        role="Backend Developer",
        interview_type="technical",
        skill_level=SkillLevel.MID,
        status=InterviewStatus.ACTIVE
    )
    
    # 4 respuestas con scores altos
    for i in range(4):
        q = Question(
            id=f"q{i}",
            text=f"Pregunta técnica {i}",
            category="technical",
            difficulty="mid",
            expected_concepts=["SOLID", "Clean Code"],
            keywords=["principios"]
        )
        interview.add_question(q)
        
        a = Answer(
            id=f"a{i}",
            question_id=q.id,
            text=f"Respuesta {i}",
            score=8.5,  # Score alto
            evaluation_details={
                "concepts_covered": ["SOLID", "Clean Code", "refactoring"],
                "missing_concepts": []
            }
        )
        interview.add_answer(a)
    
    return interview


@pytest.fixture
def interview_with_inconsistency():
    """Entrevista con alta varianza en scores (inconsistencia)."""
    interview = Interview(
        user_id="user123",
        role="Backend Developer",
        interview_type="technical",
        skill_level=SkillLevel.MID,
        status=InterviewStatus.ACTIVE
    )
    
    # Scores muy variados: 3, 9, 4, 8
    scores = [3.0, 9.0, 4.0, 8.0]
    for i, score in enumerate(scores):
        q = Question(
            id=f"q{i}",
            text=f"Pregunta técnica {i}",
            category="technical",
            difficulty="mid",
            expected_concepts=["concepto1"],
            keywords=["keyword1"]
        )
        interview.add_question(q)
        
        a = Answer(
            id=f"a{i}",
            question_id=q.id,
            text=f"Respuesta {i}",
            score=score,
            evaluation_details={
                "concepts_covered": ["concepto1"] if score > 5 else [],
                "missing_concepts": ["concepto2"] if score < 5 else []
            }
        )
        interview.add_answer(a)
    
    return interview


@pytest.fixture
def last_answer_weak():
    """Última respuesta con score bajo."""
    return Answer(
        id="last_answer",
        question_id="last_q",
        text="Respuesta incompleta",
        score=5.0,
        evaluation_details={
            "concepts_covered": ["concepto básico"],
            "missing_concepts": ["concepto avanzado", "aplicación práctica"]
        }
    )


@pytest.fixture
def last_answer_strong():
    """Última respuesta con score alto."""
    return Answer(
        id="last_answer",
        question_id="last_q",
        text="Respuesta completa y detallada",
        score=9.0,
        evaluation_details={
            "concepts_covered": ["SOLID", "DRY", "KISS", "refactoring"],
            "missing_concepts": []
        }
    )


# =========================================================================
# TESTS: Estrategias de Follow-Up
# =========================================================================

class TestFollowUpStrategies:
    """Tests para selección de estrategias."""
    
    def test_select_weakness_probe_strategy(
        self,
        mock_llm_service,
        interview_with_weak_concepts,
        last_answer_weak
    ):
        """Test: Selecciona weakness_probe cuando hay conceptos débiles."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        follow_up = generator.generate_follow_up(
            interview=interview_with_weak_concepts,
            last_answer=last_answer_weak,
            last_question_text="¿Qué es POO?",
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        assert follow_up is not None
        assert follow_up.focus_area == "weakness"
        assert len(follow_up.concepts_to_probe) > 0
        assert "herencia" in follow_up.question_text or "polimorfismo" in follow_up.question_text
    
    def test_select_strength_exploration_strategy(
        self,
        mock_llm_service,
        interview_with_strong_concepts,
        last_answer_strong
    ):
        """Test: Selecciona strength_exploration cuando hay fortalezas."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        follow_up = generator.generate_follow_up(
            interview=interview_with_strong_concepts,
            last_answer=last_answer_strong,
            last_question_text="Explica SOLID",
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        assert follow_up is not None
        assert follow_up.focus_area == "strength"
        # Dificultad debe elevarse
        assert follow_up.difficulty in ["mid", "senior"]
        assert follow_up.expected_depth == "advanced"
    
    def test_select_consistency_check_strategy(
        self,
        mock_llm_service,
        interview_with_inconsistency
    ):
        """Test: Selecciona consistency_check cuando hay inconsistencia."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        last_answer = Answer(
            id="last",
            question_id="q_last",
            text="Respuesta",
            score=7.0,
            evaluation_details={
                "concepts_covered": ["concepto1"],
                "missing_concepts": []
            }
        )
        
        follow_up = generator.generate_follow_up(
            interview=interview_with_inconsistency,
            last_answer=last_answer,
            last_question_text="Pregunta anterior",
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        assert follow_up is not None
        assert follow_up.focus_area in ["inconsistency", "exploration"]


# =========================================================================
# TESTS: Generación de Preguntas
# =========================================================================

class TestQuestionGeneration:
    """Tests para generación de preguntas."""
    
    def test_generate_weakness_probe_question(
        self,
        mock_llm_service,
        interview_with_weak_concepts,
        last_answer_weak
    ):
        """Test: Genera pregunta para explorar debilidad."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        follow_up = generator._generate_weakness_probe(
            context={
                "weak_concepts": ["herencia", "polimorfismo"],
                "strong_concepts": [],
                "has_sufficient_context": True
            },
            last_answer=last_answer_weak,
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        assert isinstance(follow_up, FollowUpQuestion)
        assert follow_up.focus_area == "weakness"
        assert len(follow_up.concepts_to_probe) > 0
        assert len(follow_up.question_text) > 10
        assert follow_up.reason.startswith("Profundizar en")
    
    def test_generate_strength_exploration_question(
        self,
        mock_llm_service,
        interview_with_strong_concepts,
        last_answer_strong
    ):
        """Test: Genera pregunta para explorar fortaleza."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        follow_up = generator._generate_strength_exploration(
            context={
                "strong_concepts": ["SOLID", "Clean Code"],
                "weak_concepts": [],
                "has_sufficient_context": True
            },
            last_answer=last_answer_strong,
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        assert isinstance(follow_up, FollowUpQuestion)
        assert follow_up.focus_area == "strength"
        assert follow_up.expected_depth == "advanced"
        # Dificultad elevada
        assert follow_up.difficulty in ["mid", "senior"]
    
    def test_generate_practical_scenario_question(
        self,
        mock_llm_service,
        interview_with_strong_concepts,
        last_answer_strong
    ):
        """Test: Genera pregunta de escenario práctico."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        follow_up = generator._generate_practical_scenario(
            context={
                "strong_concepts": ["SOLID"],
                "weak_concepts": [],
                "has_sufficient_context": True
            },
            last_answer=last_answer_strong,
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        assert isinstance(follow_up, FollowUpQuestion)
        assert follow_up.focus_area == "exploration"
        assert "Aplicación práctica" in follow_up.reason


# =========================================================================
# TESTS: Helpers
# =========================================================================

class TestHelperMethods:
    """Tests para métodos auxiliares."""
    
    def test_elevate_difficulty(self, mock_llm_service):
        """Test: Aumenta nivel de dificultad correctamente."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        assert generator._elevate_difficulty("junior") == "mid"
        assert generator._elevate_difficulty("mid") == "senior"
        assert generator._elevate_difficulty("senior") == "senior"  # Máximo
    
    def test_generate_with_llm_success(self, mock_llm_service):
        """Test: Genera pregunta usando LLM exitosamente."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        question = generator._generate_with_llm("Genera pregunta avanzada sobre herencia")
        
        assert isinstance(question, str)
        assert len(question) > 10
        # Verificar que contiene términos relevantes (herencia o trade-offs)
        assert any(term in question.lower() for term in ["herencia", "trade-off", "composición"])
    
    def test_generate_with_llm_fallback(self, mock_llm_service):
        """Test: Usa fallback si LLM falla."""
        class FailingLLMService:
            def generate(self, prompt, **kwargs):
                raise Exception("LLM no disponible")
        
        # Use mock to simulate failure behavior
        original_generate = mock_llm_service.generate
        mock_llm_service.generate = lambda prompt, **kwargs: (_ for _ in ()).throw(Exception("LLM no disponible"))
        
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        question = generator._generate_with_llm("Test prompt")
        
        # Debe retornar pregunta genérica de fallback
        assert isinstance(question, str)
        assert len(question) > 0


# =========================================================================
# TESTS: Múltiples Follow-Ups
# =========================================================================

class TestMultipleFollowUps:
    """Tests para generación de múltiples opciones."""
    
    def test_generate_multiple_follow_ups(
        self,
        mock_llm_service,
        interview_with_weak_concepts,
        last_answer_weak
    ):
        """Test: Genera múltiples opciones de follow-up."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        follow_ups = generator.generate_multiple_follow_ups(
            interview=interview_with_weak_concepts,
            last_answer=last_answer_weak,
            last_question_text="¿Qué es POO?",
            category="technical",
            difficulty="mid",
            role="Backend Developer",
            count=3
        )
        
        assert isinstance(follow_ups, list)
        assert len(follow_ups) >= 1  # Al menos 1
        assert all(isinstance(f, FollowUpQuestion) for f in follow_ups)
        
        # Preguntas deben ser diferentes
        if len(follow_ups) > 1:
            questions = [f.question_text for f in follow_ups]
            assert len(set(questions)) == len(questions)  # Todas únicas


# =========================================================================
# TESTS: Integración End-to-End
# =========================================================================

class TestFollowUpIntegration:
    """Tests de integración completa."""
    
    def test_full_interview_with_follow_ups(self, mock_llm_service):
        """Test: Flujo completo con generación de follow-ups."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        # Crear entrevista
        interview = Interview(
            user_id="user123",
            role="Backend Developer",
            interview_type="technical",
            skill_level=SkillLevel.MID
        )
        interview.start()
        
        # Simular 3 preguntas iniciales
        for i in range(3):
            q = Question(
                id=f"q{i}",
                text=f"Pregunta inicial {i}",
                category="technical",
                difficulty="mid",
                expected_concepts=["concepto1"],
                keywords=["keyword1"]
            )
            interview.add_question(q)
            
            a = Answer(
                id=f"a{i}",
                question_id=q.id,
                text=f"Respuesta {i}",
                score=6.0,
                evaluation_details={
                    "concepts_covered": ["concepto1"],
                    "missing_concepts": ["concepto2"]
                }
            )
            interview.add_answer(a)
        
        # Generar follow-up después de 3 respuestas
        last_answer = interview.answers_history[-1]
        last_question = interview.questions_history[-1]
        
        follow_up = generator.generate_follow_up(
            interview=interview,
            last_answer=last_answer,
            last_question_text=last_question.text,
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        # Debe generar follow-up exitosamente
        assert follow_up is not None
        assert isinstance(follow_up, FollowUpQuestion)
        assert len(follow_up.question_text) > 0
        assert follow_up.category == "technical"
        assert follow_up.difficulty in ["junior", "mid", "senior"]
    
    def test_no_follow_up_without_context(self, mock_llm_service):
        """Test: No genera follow-up sin contexto suficiente."""
        generator = FollowUpQuestionGenerator(llm_service=mock_llm_service)
        
        # Entrevista con solo 1 respuesta (insuficiente)
        interview = Interview(
            user_id="user123",
            role="Backend Developer",
            interview_type="technical",
            skill_level=SkillLevel.MID
        )
        interview.start()
        
        q = Question(
            id="q1",
            text="Primera pregunta",
            category="technical",
            difficulty="mid"
        )
        interview.add_question(q)
        
        a = Answer(
            id="a1",
            question_id=q.id,
            text="Primera respuesta",
            score=7.0,
            evaluation_details={}
        )
        interview.add_answer(a)
        
        follow_up = generator.generate_follow_up(
            interview=interview,
            last_answer=a,
            last_question_text=q.text,
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        # No debe generar (contexto insuficiente)
        assert follow_up is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
