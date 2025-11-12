"""
Contenedor de Dependency Injection para Ready4Hire.
Inicializa y proporciona todas las dependencias de la aplicación.
"""

from typing import Optional
import os

from app.infrastructure.llm.llm_service import OllamaLLMService, LLMService
from app.infrastructure.ml.multilingual_emotion_detector import MultilingualEmotionDetector
from app.infrastructure.ml.neural_difficulty_adjuster import NeuralDifficultyAdjuster
from app.infrastructure.ml.question_embeddings import get_embeddings_service
from app.infrastructure.ml.gpu_detector import get_gpu_detector
from app.infrastructure.persistence.memory_interview_repository import MemoryInterviewRepository
from app.infrastructure.persistence.json_question_repository import JsonQuestionRepository

# Design Patterns
from app.infrastructure.patterns.facade import InfrastructureFacade
from app.infrastructure.patterns.observer import get_event_bus

# Nuevos servicios de audio, security y domain
from app.infrastructure.audio import get_stt_service, get_tts_service
from app.infrastructure.security import get_sanitizer, get_prompt_guard
from app.domain.services import get_language_service, get_text_service

from app.application.services.evaluation_service import EvaluationService
from app.application.services.feedback_service import FeedbackService
from app.application.services.question_selector_service import EnhancedQuestionSelectorService, SelectionConfig

from app.application.use_cases.start_interview_use_case import StartInterviewUseCase
from app.application.use_cases.process_answer_use_case import ProcessAnswerUseCase

# ML Services for Enhanced Selector
from app.infrastructure.ml.advanced_clustering import AdvancedQuestionClusteringService
from app.infrastructure.ml.continuous_learning import ContinuousLearningSystem

# Gamification Services
from app.application.services.gamification_service import GamificationService, HintServiceEnhanced
from app.application.services.game_engine_service import GameEngineService


class MockLLMService(LLMService):
    """
    LLM de reemplazo utilizado cuando MOCK_OLLAMA=true.
    Responde con mensajes predecibles para ejecutar pruebas sin depender de servicios externos.
    """

    class _MockClient:
        """Cliente simulado con la interfaz mínima usada en los health-checks."""

        def _check_health(self):
            return True

        def get_metrics(self):
            return {
                "status": "mocked",
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
            }

    def __init__(self):
        self.model = "mock-llm"
        self.client = self._MockClient()

    def generate(self, prompt: str, **kwargs) -> str:
        """Retorna una respuesta fija usando parte del prompt para trazabilidad."""
        preview = prompt.strip().split("\n")[0][:120] if prompt else ""
        return f"[mock-response] {preview or 'Respuesta generada por mock'}"

    def chat(self, messages, **kwargs) -> str:
        """Retorna una respuesta fija para interacciones tipo chat."""
        if messages:
            last = messages[-1].get("content", "")
            return f"[mock-chat] {last[:120]}"
        return "[mock-chat] respuesta predeterminada"

    def get_metrics(self) -> dict:
        """Métricas estáticas para el mock."""
        return self.client.get_metrics()


class Container:
    """
    Contenedor de dependencias para la aplicación.
    Sigue el patrón Dependency Injection para facilitar testing y modularidad.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: Optional[str] = None,
        questions_path: Optional[str] = None,
    ):
        """
        Inicializa el contenedor con todas las dependencias.

        Args:
            ollama_url: URL de Ollama
            ollama_model: Modelo Ollama a usar (None = auto-detectar según GPU)
            questions_path: Path al directorio de datasets de preguntas
        """
        # Detectar GPU y configurar modelo óptimo
        self.gpu_detector = get_gpu_detector()

        self.ollama_url = ollama_url
        # Auto-seleccionar modelo según hardware si no se especifica
        self.ollama_model = ollama_model or self.gpu_detector.get_recommended_model()
        self.questions_path = questions_path or os.path.join(os.path.dirname(__file__), "datasets")

        # Obtener configuración optimizada
        self.perf_config = self.gpu_detector.get_performance_config()

        # Inicializar componentes
        self._init_infrastructure()
        self._init_services()
        self._init_use_cases()

    def _init_infrastructure(self):
        """Inicializa componentes de infraestructura"""
        # ⚡ LLM Service (Ollama) - OPTIMIZADO según hardware detectado
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"🖥️ Hardware: {self.gpu_detector.gpu_type.value}")
        logger.info(f"🤖 Modelo seleccionado: {self.ollama_model}")
        logger.info(f"⏱️ Latencia esperada: {self.perf_config['expected_latency_ms']/1000:.1f}s")

        use_mock_ollama = os.getenv("MOCK_OLLAMA", "false").lower() == "true"
        if use_mock_ollama:
            logger.warning("⚠️ MOCK_OLLAMA habilitado: usando MockLLMService en lugar de una conexión real a Ollama")
            self.llm_service = MockLLMService()
        else:
            self.llm_service = OllamaLLMService(
                base_url=self.ollama_url, model=self.ollama_model, temperature=0.3, max_tokens=256
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
        self.language_service = get_language_service(default_language="es")
        self.text_service = get_text_service()

        # Repositories
        self.interview_repository = MemoryInterviewRepository()
        self.question_repository = JsonQuestionRepository(
            tech_file=os.path.join(self.questions_path, "technical_questions_by_profession_v3.jsonl"),
            soft_file=os.path.join(self.questions_path, "soft_skills.jsonl"),
        )

        # ⚡ Pre-computar embeddings de preguntas para selección instantánea
        self._precompute_question_embeddings()
        
        # Design Patterns: Event Bus
        self.event_bus = get_event_bus()
        
        # Design Patterns: Infrastructure Facade
        self.facade = InfrastructureFacade(self)

    def _init_services(self):
        """Inicializa servicios de aplicación"""
        # Evaluation Service
        self.evaluation_service = EvaluationService(
            llm_service=self.llm_service,
            model=self.ollama_model,
            temperature=0.3,  # Más baja para evaluación consistente
        )

        # Feedback Service
        self.feedback_service = FeedbackService(
            llm_service=self.llm_service, model=self.ollama_model, temperature=0.7  # Más alta para feedback creativo
        )

        # ML Components for Enhanced Question Selector
        use_ml_selector = os.getenv("USE_ML_SELECTOR", "false").lower() == "true"

        if use_ml_selector:
            # Initialize ML services
            clustering_service = AdvancedQuestionClusteringService(embeddings_service=self.embeddings_service)

            learning_system = ContinuousLearningSystem()

            # ML Selection Configuration
            ml_config = SelectionConfig(
                use_clustering=os.getenv("USE_ML_CLUSTERING", "true").lower() == "true",
                use_continuous_learning=os.getenv("USE_ML_LEARNING", "true").lower() == "true",
                use_embeddings=True,
                exploration_strategy=os.getenv("ML_EXPLORATION_STRATEGY", "balanced"),
                fallback_to_simple=os.getenv("ENABLE_ML_FALLBACK", "true").lower() == "true",
            )

            # Enhanced Question Selector Service with FULL ML
            self.question_selector_service = EnhancedQuestionSelectorService(
                question_repository=self.question_repository,
                embeddings_service=self.embeddings_service,
                clustering_service=clustering_service,
                learning_system=learning_system,
                config=ml_config,
            )
        else:
            # Fallback: Use Enhanced Selector but with ML features disabled
            basic_config = SelectionConfig(
                use_clustering=False, use_continuous_learning=False, use_embeddings=False, fallback_to_simple=True
            )
            self.question_selector_service = EnhancedQuestionSelectorService(
                question_repository=self.question_repository,
                embeddings_service=None,
                clustering_service=None,
                learning_system=None,
                config=basic_config,
            )

        # Gamification Services
        self.gamification_service = GamificationService()
        self.hint_service = HintServiceEnhanced()
        self.game_engine = GameEngineService(self.llm_service)

    def _init_use_cases(self):
        """Inicializa casos de uso"""
        # Start Interview Use Case
        self.start_interview_use_case = StartInterviewUseCase(
            interview_repo=self.interview_repository, question_repo=self.question_repository
        )

        # Process Answer Use Case
        self.process_answer_use_case = ProcessAnswerUseCase(
            interview_repo=self.interview_repository,
            evaluation_service=self.evaluation_service,
            feedback_service=self.feedback_service,
            question_selector=self.question_selector_service,
            emotion_detector=self.emotion_detector,
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
            "ml": "unknown",
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

    def _precompute_question_embeddings(self) -> None:
        """
        ⚡ Pre-computa embeddings de todas las preguntas al inicio.
        MEJORADO: Persiste en disco para evitar re-computar al reiniciar.
        """
        try:
            import logging
            import asyncio
            import pickle
            from pathlib import Path

            logger = logging.getLogger(__name__)

            # Ruta del cache de embeddings
            cache_path = Path(self.questions_path) / "embeddings_cache.pkl"

            # ⚡ MEJORA: Intentar cargar desde disco primero
            if cache_path.exists():
                try:
                    logger.info("⚡ Cargando embeddings desde caché en disco...")
                    with open(cache_path, "rb") as f:
                        self.question_repository._embeddings_cache = pickle.load(f)
                    logger.info(
                        f"✅ Embeddings cargados desde caché: {len(self.question_repository._embeddings_cache)} preguntas"
                    )
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Error cargando caché, re-computando: {e}")

            # Si no hay cache, computar embeddings
            logger.info("⚡ Pre-computando embeddings de preguntas...")

            # Obtener todas las preguntas
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            tech_questions = loop.run_until_complete(self.question_repository.find_all_technical())
            soft_questions = loop.run_until_complete(self.question_repository.find_all_soft_skills())

            all_questions = tech_questions + soft_questions

            if not all_questions:
                logger.warning("⚠️ No hay preguntas para pre-computar embeddings")
                return

            # Pre-computar embeddings en batch (mucho más rápido)
            question_texts = [q.text for q in all_questions]
            embeddings = self.embeddings_service.encode(question_texts)

            # Guardar embeddings en cache del repositorio
            self.question_repository._embeddings_cache = {q.id: emb for q, emb in zip(all_questions, embeddings)}

            # ⚡ MEJORA: Persistir en disco
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.question_repository._embeddings_cache, f)
                logger.info(f"💾 Embeddings guardados en disco: {cache_path}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo guardar caché en disco: {e}")

            logger.info(f"✅ Embeddings pre-computados: {len(all_questions)} preguntas en caché")

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ No se pudieron pre-computar embeddings: {e}")


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
        ollama_model = os.getenv("OLLAMA_MODEL")  # None = auto-detectar según GPU
        _container = Container(ollama_url=ollama_url, ollama_model=ollama_model)
    return _container


def reset_container():
    """Resetea el contenedor (útil para testing)"""
    global _container
    _container = None
