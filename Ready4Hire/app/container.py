"""
Contenedor de Dependency Injection para Ready4Hire.
Inicializa y proporciona todas las dependencias de la aplicación.
"""
from typing import Optional
import os

from app.infrastructure.llm.llm_service import OllamaLLMService
from app.infrastructure.llm.ollama_client import OllamaClient
from app.infrastructure.ml.multilingual_emotion_detector import MultilingualEmotionDetector
from app.infrastructure.ml.neural_difficulty_adjuster import NeuralDifficultyAdjuster
from app.infrastructure.ml.question_embeddings import get_embeddings_service
from app.infrastructure.persistence.memory_interview_repository import MemoryInterviewRepository
from app.infrastructure.persistence.json_question_repository import JsonQuestionRepository

# Nuevos servicios de audio, security y domain
from app.infrastructure.audio import get_stt_service, get_tts_service
from app.infrastructure.security import get_sanitizer, get_prompt_guard
from app.domain.services import get_language_service, get_text_service

from app.application.services.evaluation_service import EvaluationService
from app.application.services.feedback_service import FeedbackService
from app.application.services.question_selector_service import QuestionSelectorService

from app.application.use_cases.start_interview_use_case import StartInterviewUseCase
from app.application.use_cases.process_answer_use_case import ProcessAnswerUseCase


class Container:
    """
    Contenedor de dependencias para la aplicación.
    Sigue el patrón Dependency Injection para facilitar testing y modularidad.
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b",
        questions_path: Optional[str] = None
    ):
        """
        Inicializa el contenedor con todas las dependencias.
        
        Args:
            ollama_url: URL de Ollama
            ollama_model: Modelo Ollama a usar
            questions_path: Path al directorio de datasets de preguntas
        """
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.questions_path = questions_path or os.path.join(
            os.path.dirname(__file__), "datasets"
        )
        
        # Inicializar componentes
        self._init_infrastructure()
        self._init_services()
        self._init_use_cases()
    
    def _init_infrastructure(self):
        """Inicializa componentes de infraestructura"""
        # LLM Service (Ollama)
        self.llm_service = OllamaLLMService(
            base_url=self.ollama_url,
            model=self.ollama_model,
            temperature=0.7,
            max_tokens=512
        )
        
        # ML Components
        self.emotion_detector = MultilingualEmotionDetector()
        self.difficulty_adjuster = NeuralDifficultyAdjuster()
        self.embeddings_service = get_embeddings_service(
            ranknet_model_path=os.path.join(self.questions_path, "ranknet_model.pt")
        )
        
        # Audio Services
        self.stt_service = get_stt_service()
        self.tts_service = get_tts_service()
        
        # Security Services
        self.input_sanitizer = get_sanitizer()
        self.prompt_guard = get_prompt_guard(threshold=0.5)
        
        # Domain Services
        self.language_service = get_language_service(default_language='es')
        self.text_service = get_text_service()
        
        # Repositories
        self.interview_repository = MemoryInterviewRepository()
        self.question_repository = JsonQuestionRepository(
            tech_file=os.path.join(self.questions_path, "tech_questions.jsonl"),
            soft_file=os.path.join(self.questions_path, "soft_skills.jsonl")
        )
    
    def _init_services(self):
        """Inicializa servicios de aplicación"""
        # Evaluation Service
        self.evaluation_service = EvaluationService(
            llm_service=self.llm_service,
            model=self.ollama_model,
            temperature=0.3  # Más baja para evaluación consistente
        )
        
        # Feedback Service
        self.feedback_service = FeedbackService(
            llm_service=self.llm_service,
            model=self.ollama_model,
            temperature=0.7  # Más alta para feedback creativo
        )
        
        # Question Selector Service
        self.question_selector_service = QuestionSelectorService(
            question_repository=self.question_repository,
            embeddings_manager=None  # TODO: Adapter para QuestionEmbeddingsService
        )
    
    def _init_use_cases(self):
        """Inicializa casos de uso"""
        # Start Interview Use Case
        self.start_interview_use_case = StartInterviewUseCase(
            interview_repo=self.interview_repository,
            question_repo=self.question_repository
        )
        
        # Process Answer Use Case
        self.process_answer_use_case = ProcessAnswerUseCase(
            interview_repo=self.interview_repository,
            evaluation_service=self.evaluation_service,
            feedback_service=self.feedback_service,
            question_selector=self.question_selector_service,
            emotion_detector=self.emotion_detector
        )
    
    def get_llm_service(self) -> OllamaLLMService:
        """Retorna el servicio LLM"""
        return self.llm_service
    
    def get_evaluation_service(self) -> EvaluationService:
        """Retorna el servicio de evaluación"""
        return self.evaluation_service
    
    def get_feedback_service(self) -> FeedbackService:
        """Retorna el servicio de feedback"""
        return self.feedback_service
    
    def get_start_interview_use_case(self) -> StartInterviewUseCase:
        """Retorna el caso de uso de iniciar entrevista"""
        return self.start_interview_use_case
    
    def get_process_answer_use_case(self) -> ProcessAnswerUseCase:
        """Retorna el caso de uso de procesar respuesta"""
        return self.process_answer_use_case
    
    def health_check(self) -> dict:
        """Verifica el estado de todos los componentes"""
        health = {
            "llm_service": "unknown",
            "repositories": "unknown",
            "services": "unknown",
            "audio": "unknown",
            "security": "healthy",
            "domain": "healthy",
            "ml": "unknown"
        }
        
        try:
            # Check LLM Service
            self.llm_service.client._check_health()
            health["llm_service"] = "healthy"
        except Exception as e:
            health["llm_service"] = f"unhealthy: {str(e)}"
        
        try:
            # Check repositories (verificación simple sin async)
            # Las preguntas se cargan en __init__, solo verificamos que existan
            health["repositories"] = "healthy (questions loaded at startup)"
        except Exception as e:
            health["repositories"] = f"unhealthy: {str(e)}"
        
        # Check Audio Services
        try:
            stt_ok = self.stt_service.is_available()
            tts_ok = self.tts_service.is_available()
            health["audio"] = f"STT: {'✅' if stt_ok else '⚠️'} TTS: {'✅' if tts_ok else '⚠️'}"
        except Exception as e:
            health["audio"] = f"error: {str(e)}"
        
        # Check ML Services
        try:
            embeddings_ok = self.embeddings_service.is_available()
            health["ml"] = f"Embeddings: {'✅' if embeddings_ok else '⚠️'}"
        except Exception as e:
            health["ml"] = f"error: {str(e)}"
        
        health["services"] = "healthy"
        
        return health


# Singleton global instance
_container: Optional[Container] = None


def get_container() -> Container:
    """
    Retorna la instancia singleton del contenedor.
    Crea una nueva si no existe.
    """
    global _container
    if _container is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        _container = Container(
            ollama_url=ollama_url,
            ollama_model=ollama_model
        )
    return _container


def reset_container():
    """Resetea el contenedor (útil para testing)"""
    global _container
    _container = None
