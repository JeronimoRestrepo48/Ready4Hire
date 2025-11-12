"""
Facade Pattern Implementation.

Proporciona interfaces simplificadas para sistemas complejos,
ocultando la complejidad interna.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class LLMFacade:
    """
    Facade para servicios LLM.
    
    Proporciona una interfaz simplificada para operaciones LLM comunes.
    """
    
    def __init__(self, llm_service: Any):
        self.llm_service = llm_service
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evalúa una respuesta de forma simple.
        
        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            expected_concepts: Conceptos esperados
            **kwargs: Parámetros adicionales
            
        Returns:
            Dict con evaluación
        """
        from app.application.services.evaluation_service import EvaluationService
        eval_service = EvaluationService(llm_service=self.llm_service)
        return eval_service.evaluate_answer(
            question=question,
            answer=answer,
            expected_concepts=expected_concepts,
            **kwargs
        )
    
    def generate_feedback(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        role: str,
        **kwargs
    ) -> str:
        """
        Genera feedback de forma simple.
        
        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            evaluation: Resultado de evaluación
            role: Rol del candidato
            **kwargs: Parámetros adicionales
            
        Returns:
            Texto de feedback
        """
        from app.application.services.feedback_service import FeedbackService
        feedback_service = FeedbackService(llm_service=self.llm_service)
        return feedback_service.generate_feedback(
            question=question,
            answer=answer,
            evaluation=evaluation,
            role=role,
            **kwargs
        )
    
    def generate_hint(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        attempt: int,
        **kwargs
    ) -> str:
        """
        Genera una pista de forma simple.
        
        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            expected_concepts: Conceptos esperados
            attempt: Número de intento
            **kwargs: Parámetros adicionales
            
        Returns:
            Texto de pista
        """
        from app.infrastructure.llm.advanced_prompts import get_prompt_engine
        prompt_engine = get_prompt_engine()
        hint_prompt = prompt_engine.get_hint_prompt(
            role=kwargs.get("role", ""),
            question=question,
            answer=answer,
            expected_concepts=expected_concepts,
            attempts=attempt,
        )
        return self.llm_service.generate(prompt=hint_prompt, max_tokens=150, **kwargs)


class MLFacade:
    """
    Facade para servicios ML.
    
    Proporciona una interfaz simplificada para operaciones ML comunes.
    """
    
    def __init__(
        self,
        embeddings_service: Any,
        emotion_detector: Any = None,
        difficulty_adjuster: Any = None,
    ):
        self.embeddings_service = embeddings_service
        self.emotion_detector = emotion_detector
        self.difficulty_adjuster = difficulty_adjuster
    
    def encode_text(self, text: str) -> List[float]:
        """Codifica texto en embedding."""
        return self.embeddings_service.encode([text])[0]
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Codifica múltiples textos en embeddings."""
        return self.embeddings_service.encode(texts)
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """Detecta emoción en texto."""
        if not self.emotion_detector:
            raise ValueError("Emotion detector no está disponible")
        return self.emotion_detector.detect(text)
    
    def adjust_difficulty(
        self,
        current_difficulty: str,
        performance: Dict[str, Any]
    ) -> str:
        """Ajusta dificultad según performance."""
        if not self.difficulty_adjuster:
            return current_difficulty
        return self.difficulty_adjuster.adjust(current_difficulty, performance)


class InfrastructureFacade:
    """
    Facade principal para toda la infraestructura.
    
    Proporciona acceso simplificado a todos los servicios
    de infraestructura.
    """
    
    def __init__(self, container: Any):
        self.container = container
        self._llm_facade = None
        self._ml_facade = None
    
    @property
    def llm(self) -> LLMFacade:
        """Retorna facade LLM."""
        if self._llm_facade is None:
            self._llm_facade = LLMFacade(self.container.llm_service)
        return self._llm_facade
    
    @property
    def ml(self) -> MLFacade:
        """Retorna facade ML."""
        if self._ml_facade is None:
            self._ml_facade = MLFacade(
                embeddings_service=self.container.embeddings_service,
                emotion_detector=self.container.emotion_detector,
                difficulty_adjuster=self.container.difficulty_adjuster,
            )
        return self._ml_facade
    
    def evaluate_interview_answer(
        self,
        question: str,
        answer: str,
        role: str,
        category: str = "technical",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evalúa una respuesta de entrevista de forma simplificada.
        
        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            role: Rol del candidato
            category: Categoría de la pregunta
            **kwargs: Parámetros adicionales
            
        Returns:
            Dict con evaluación completa
        """
        # Obtener pregunta del repositorio para expected_concepts
        # Por simplicidad, usar defaults
        expected_concepts = kwargs.pop("expected_concepts", [])
        keywords = kwargs.pop("keywords", [])
        
        # Evaluar
        evaluation = self.llm.evaluate_answer(
            question=question,
            answer=answer,
            expected_concepts=expected_concepts,
            category=category,
            role=role,
            **kwargs
        )
        
        # Generar feedback
        feedback = self.llm.generate_feedback(
            question=question,
            answer=answer,
            evaluation=evaluation,
            role=role,
            category=category,
            **kwargs
        )
        
        return {
            "evaluation": evaluation,
            "feedback": feedback,
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene estado de salud de todos los servicios."""
        return self.container.health_check()

