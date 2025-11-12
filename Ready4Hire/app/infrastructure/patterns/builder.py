"""
Builder Pattern Implementation.

Proporciona builders para construir objetos complejos
de forma paso a paso y clara.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class InterviewBuilder:
    """
    Builder para construir objetos Interview.
    
    Permite construcción fluida y clara de entrevistas.
    """
    
    def __init__(self):
        self._user_id: Optional[str] = None
        self._role: Optional[str] = None
        self._interview_type: Optional[str] = None
        self._difficulty: Optional[str] = None
        self._questions: List[Any] = []
        self._metadata: Dict[str, Any] = {}
    
    def with_user_id(self, user_id: str) -> 'InterviewBuilder':
        """Configura el user ID."""
        self._user_id = user_id
        return self
    
    def with_role(self, role: str) -> 'InterviewBuilder':
        """Configura el rol."""
        self._role = role
        return self
    
    def with_interview_type(self, interview_type: str) -> 'InterviewBuilder':
        """Configura el tipo de entrevista."""
        self._interview_type = interview_type
        return self
    
    def with_difficulty(self, difficulty: str) -> 'InterviewBuilder':
        """Configura la dificultad."""
        self._difficulty = difficulty
        return self
    
    def with_questions(self, questions: List[Any]) -> 'InterviewBuilder':
        """Configura las preguntas."""
        self._questions = questions
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'InterviewBuilder':
        """Agrega metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> Any:
        """Construye el objeto Interview."""
        from app.domain.entities.interview import Interview
        
        if not self._user_id:
            raise ValueError("user_id es requerido")
        if not self._role:
            raise ValueError("role es requerido")
        if not self._interview_type:
            self._interview_type = "technical"
        if not self._difficulty:
            self._difficulty = "mid"
        
        interview = Interview(
            user_id=self._user_id,
            role=self._role,
            interview_type=self._interview_type,
            difficulty=self._difficulty,
        )
        
        # Agregar preguntas
        for question in self._questions:
            interview.add_question(question)
        
        # Agregar metadata
        interview.metadata.update(self._metadata)
        
        return interview


class EvaluationBuilder:
    """
    Builder para construir objetos Evaluation.
    
    Permite construcción fluida de evaluaciones.
    """
    
    def __init__(self):
        self._question: Optional[str] = None
        self._answer: Optional[str] = None
        self._score: Optional[float] = None
        self._is_correct: Optional[bool] = None
        self._feedback: Optional[str] = None
        self._strengths: List[str] = []
        self._improvements: List[str] = []
        self._metadata: Dict[str, Any] = {}
    
    def with_question(self, question: str) -> 'EvaluationBuilder':
        """Configura la pregunta."""
        self._question = question
        return self
    
    def with_answer(self, answer: str) -> 'EvaluationBuilder':
        """Configura la respuesta."""
        self._answer = answer
        return self
    
    def with_score(self, score: float) -> 'EvaluationBuilder':
        """Configura el score."""
        self._score = score
        return self
    
    def with_is_correct(self, is_correct: bool) -> 'EvaluationBuilder':
        """Configura si es correcta."""
        self._is_correct = is_correct
        return self
    
    def with_feedback(self, feedback: str) -> 'EvaluationBuilder':
        """Configura el feedback."""
        self._feedback = feedback
        return self
    
    def with_strength(self, strength: str) -> 'EvaluationBuilder':
        """Agrega una fortaleza."""
        self._strengths.append(strength)
        return self
    
    def with_improvement(self, improvement: str) -> 'EvaluationBuilder':
        """Agrega una mejora."""
        self._improvements.append(improvement)
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'EvaluationBuilder':
        """Agrega metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construye el dict de evaluación."""
        if self._score is None:
            raise ValueError("score es requerido")
        if self._is_correct is None:
            self._is_correct = self._score >= 6.0
        
        evaluation = {
            "score": self._score,
            "is_correct": self._is_correct,
            "feedback": self._feedback or "",
            "strengths": self._strengths,
            "improvements": self._improvements,
            **self._metadata,
        }
        
        if self._question:
            evaluation["question"] = self._question
        if self._answer:
            evaluation["answer"] = self._answer
        
        return evaluation


class QuestionSelectorBuilder:
    """
    Builder para construir QuestionSelector con configuración compleja.
    """
    
    def __init__(self):
        self._repository = None
        self._embeddings_service = None
        self._clustering_service = None
        self._learning_system = None
        self._use_clustering = False
        self._use_embeddings = False
        self._use_learning = False
        self._exploration_strategy = "balanced"
    
    def with_repository(self, repository) -> 'QuestionSelectorBuilder':
        """Configura el repositorio."""
        self._repository = repository
        return self
    
    def with_embeddings(self, embeddings_service) -> 'QuestionSelectorBuilder':
        """Configura el servicio de embeddings."""
        self._embeddings_service = embeddings_service
        self._use_embeddings = True
        return self
    
    def with_clustering(self, clustering_service) -> 'QuestionSelectorBuilder':
        """Configura el servicio de clustering."""
        self._clustering_service = clustering_service
        self._use_clustering = True
        return self
    
    def with_learning(self, learning_system) -> 'QuestionSelectorBuilder':
        """Configura el sistema de aprendizaje."""
        self._learning_system = learning_system
        self._use_learning = True
        return self
    
    def with_exploration_strategy(self, strategy: str) -> 'QuestionSelectorBuilder':
        """Configura la estrategia de exploración."""
        self._exploration_strategy = strategy
        return self
    
    def build(self) -> Any:
        """Construye el QuestionSelector."""
        from app.application.services.question_selector_service import (
            EnhancedQuestionSelectorService,
            SelectionConfig,
        )
        
        if not self._repository:
            raise ValueError("repository es requerido")
        
        config = SelectionConfig(
            use_clustering=self._use_clustering,
            use_continuous_learning=self._use_learning,
            use_embeddings=self._use_embeddings,
            exploration_strategy=self._exploration_strategy,
            fallback_to_simple=True,
        )
        
        return EnhancedQuestionSelectorService(
            question_repository=self._repository,
            embeddings_service=self._embeddings_service,
            clustering_service=self._clustering_service,
            learning_system=self._learning_system,
            config=config,
        )

