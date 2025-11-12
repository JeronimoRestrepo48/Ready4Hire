"""
Strategy Pattern Implementation.

Proporciona estrategias intercambiables para diferentes algoritmos
de evaluación, feedback, selección de preguntas, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import logging

logger = logging.getLogger(__name__)


class EvaluationStrategy(ABC):
    """
    Estrategia base para evaluación de respuestas.
    
    Permite diferentes estrategias:
    - LLM-based (default)
    - Heuristic-based (fallback)
    - Hybrid (combinación)
    """
    
    @abstractmethod
    def evaluate(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        keywords: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evalúa una respuesta.
        
        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            expected_concepts: Conceptos esperados
            keywords: Palabras clave relevantes
            **kwargs: Parámetros adicionales
            
        Returns:
            Dict con score, feedback, etc.
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Retorna el nombre de la estrategia."""
        pass


class LLMEvaluationStrategy(EvaluationStrategy):
    """Estrategia de evaluación basada en LLM."""
    
    def __init__(self, llm_service: Any):
        self.llm_service = llm_service
    
    def evaluate(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        keywords: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Evalúa usando LLM."""
        # Esta implementación se conectará con EvaluationService
        from app.application.services.evaluation_service import EvaluationService
        eval_service = EvaluationService(llm_service=self.llm_service)
        return eval_service.evaluate_answer(
            question=question,
            answer=answer,
            expected_concepts=expected_concepts,
            keywords=keywords,
            **kwargs
        )
    
    def get_strategy_name(self) -> str:
        return "llm"


class HeuristicEvaluationStrategy(EvaluationStrategy):
    """Estrategia de evaluación heurística (fallback)."""
    
    def evaluate(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        keywords: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Evalúa usando heurísticas."""
        from app.application.services.evaluation_service import EvaluationService
        eval_service = EvaluationService(llm_service=None)
        return eval_service._heuristic_evaluation(
            answer=answer,
            expected_concepts=expected_concepts,
            keywords=keywords
        )
    
    def get_strategy_name(self) -> str:
        return "heuristic"


class FeedbackStrategy(ABC):
    """
    Estrategia base para generación de feedback.
    
    Permite diferentes estrategias:
    - Personalized (default)
    - Concise (breve)
    - Detailed (detallado)
    - Encouraging (alentador)
    """
    
    @abstractmethod
    def generate(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Genera feedback.
        
        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            evaluation: Resultado de evaluación
            **kwargs: Parámetros adicionales
            
        Returns:
            Texto de feedback
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Retorna el nombre de la estrategia."""
        pass


class PersonalizedFeedbackStrategy(FeedbackStrategy):
    """Estrategia de feedback personalizado."""
    
    def __init__(self, llm_service: Any, role: str, category: str):
        self.llm_service = llm_service
        self.role = role
        self.category = category
    
    def generate(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        **kwargs
    ) -> str:
        """Genera feedback personalizado."""
        from app.application.services.feedback_service import FeedbackService
        feedback_service = FeedbackService(llm_service=self.llm_service)
        return feedback_service.generate_feedback(
            question=question,
            answer=answer,
            evaluation=evaluation,
            role=self.role,
            category=self.category,
            **kwargs
        )
    
    def get_strategy_name(self) -> str:
        return "personalized"


class QuestionSelectionStrategy(ABC):
    """
    Estrategia base para selección de preguntas.
    
    Permite diferentes estrategias:
    - Random (aleatorio)
    - Clustering (basado en embeddings)
    - Difficulty-based (por dificultad)
    - Adaptive (adaptativo según performance)
    """
    
    @abstractmethod
    def select(
        self,
        questions: List[Any],
        count: int,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Selecciona preguntas.
        
        Args:
            questions: Lista de preguntas disponibles
            count: Cantidad de preguntas a seleccionar
            context: Contexto adicional (rol, dificultad, etc.)
            **kwargs: Parámetros adicionales
            
        Returns:
            Lista de preguntas seleccionadas
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Retorna el nombre de la estrategia."""
        pass


class RandomSelectionStrategy(QuestionSelectionStrategy):
    """Estrategia de selección aleatoria."""
    
    def select(
        self,
        questions: List[Any],
        count: int,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """Selecciona preguntas aleatoriamente."""
        import random
        return random.sample(questions, min(count, len(questions)))
    
    def get_strategy_name(self) -> str:
        return "random"


class ClusteringSelectionStrategy(QuestionSelectionStrategy):
    """Estrategia de selección basada en clustering."""
    
    def __init__(self, clustering_service: Any):
        self.clustering_service = clustering_service
    
    def select(
        self,
        questions: List[Any],
        count: int,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """Selecciona preguntas usando clustering."""
        # Implementación simplificada
        # En producción, usaría el clustering_service completo
        import random
        return random.sample(questions, min(count, len(questions)))
    
    def get_strategy_name(self) -> str:
        return "clustering"


class StrategyRegistry:
    """
    Registro centralizado de estrategias.
    
    Permite registrar y obtener estrategias de forma dinámica.
    """
    
    def __init__(self):
        self._evaluation_strategies: Dict[str, Type[EvaluationStrategy]] = {}
        self._feedback_strategies: Dict[str, Type[FeedbackStrategy]] = {}
        self._selection_strategies: Dict[str, Type[QuestionSelectionStrategy]] = {}
    
    def register_evaluation_strategy(
        self,
        name: str,
        strategy_class: Type[EvaluationStrategy]
    ):
        """Registra una estrategia de evaluación."""
        self._evaluation_strategies[name] = strategy_class
        logger.info(f"✅ Evaluation strategy '{name}' registrada")
    
    def register_feedback_strategy(
        self,
        name: str,
        strategy_class: Type[FeedbackStrategy]
    ):
        """Registra una estrategia de feedback."""
        self._feedback_strategies[name] = strategy_class
        logger.info(f"✅ Feedback strategy '{name}' registrada")
    
    def register_selection_strategy(
        self,
        name: str,
        strategy_class: Type[QuestionSelectionStrategy]
    ):
        """Registra una estrategia de selección."""
        self._selection_strategies[name] = strategy_class
        logger.info(f"✅ Selection strategy '{name}' registrada")
    
    def get_evaluation_strategy(
        self,
        name: str,
        **kwargs
    ) -> EvaluationStrategy:
        """Obtiene una estrategia de evaluación."""
        if name not in self._evaluation_strategies:
            raise ValueError(
                f"Evaluation strategy '{name}' no registrada. "
                f"Disponibles: {list(self._evaluation_strategies.keys())}"
            )
        return self._evaluation_strategies[name](**kwargs)
    
    def get_feedback_strategy(
        self,
        name: str,
        **kwargs
    ) -> FeedbackStrategy:
        """Obtiene una estrategia de feedback."""
        if name not in self._feedback_strategies:
            raise ValueError(
                f"Feedback strategy '{name}' no registrada. "
                f"Disponibles: {list(self._feedback_strategies.keys())}"
            )
        return self._feedback_strategies[name](**kwargs)
    
    def get_selection_strategy(
        self,
        name: str,
        **kwargs
    ) -> QuestionSelectionStrategy:
        """Obtiene una estrategia de selección."""
        if name not in self._selection_strategies:
            raise ValueError(
                f"Selection strategy '{name}' no registrada. "
                f"Disponibles: {list(self._selection_strategies.keys())}"
            )
        return self._selection_strategies[name](**kwargs)
    
    def list_evaluation_strategies(self) -> List[str]:
        """Lista estrategias de evaluación disponibles."""
        return list(self._evaluation_strategies.keys())
    
    def list_feedback_strategies(self) -> List[str]:
        """Lista estrategias de feedback disponibles."""
        return list(self._feedback_strategies.keys())
    
    def list_selection_strategies(self) -> List[str]:
        """Lista estrategias de selección disponibles."""
        return list(self._selection_strategies.keys())


# Singleton instance
_strategy_registry: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """Retorna instancia singleton del registro de estrategias."""
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = StrategyRegistry()
        # Registrar estrategias por defecto
        _strategy_registry.register_evaluation_strategy("llm", LLMEvaluationStrategy)
        _strategy_registry.register_evaluation_strategy("heuristic", HeuristicEvaluationStrategy)
        _strategy_registry.register_feedback_strategy("personalized", PersonalizedFeedbackStrategy)
        _strategy_registry.register_selection_strategy("random", RandomSelectionStrategy)
        _strategy_registry.register_selection_strategy("clustering", ClusteringSelectionStrategy)
    return _strategy_registry

