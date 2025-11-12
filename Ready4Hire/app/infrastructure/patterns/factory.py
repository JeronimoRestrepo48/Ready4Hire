"""
Factory Pattern Implementation.

Proporciona factories para crear servicios de forma consistente
y desacoplada, facilitando testing y cambio de implementaciones.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Protocol
import logging

logger = logging.getLogger(__name__)


class LLMServiceProtocol(Protocol):
    """Protocolo para servicios LLM."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto usando el modelo."""
        ...


class CacheServiceProtocol(Protocol):
    """Protocolo para servicios de cache."""
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache."""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Almacena un valor en el cache."""
        ...


class BaseFactory(ABC):
    """Factory base abstracta."""
    
    @abstractmethod
    def create(self, **kwargs) -> Any:
        """Crea una instancia del servicio."""
        pass


class LLMServiceFactory(BaseFactory):
    """
    Factory para crear servicios LLM.
    
    Soporta múltiples implementaciones:
    - Ollama (default)
    - OpenAI (future)
    - Anthropic (future)
    """
    
    _registry: Dict[str, Type[LLMServiceProtocol]] = {}
    
    @classmethod
    def register(cls, name: str, service_class: Type[LLMServiceProtocol]):
        """Registra un tipo de servicio LLM."""
        cls._registry[name] = service_class
        logger.info(f"✅ LLM Service '{name}' registrado en factory")
    
    def create(
        self,
        provider: str = "ollama",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMServiceProtocol:
        """
        Crea un servicio LLM según el provider especificado.
        
        Args:
            provider: Proveedor LLM ('ollama', 'openai', etc.)
            base_url: URL base del servicio
            model: Modelo a usar
            **kwargs: Parámetros adicionales
            
        Returns:
            Instancia del servicio LLM
        """
        if provider not in self._registry:
            raise ValueError(
                f"Provider '{provider}' no registrado. "
                f"Disponibles: {list(self._registry.keys())}"
            )
        
        service_class = self._registry[provider]
        
        # Configuración específica por provider
        if provider == "ollama":
            from app.infrastructure.llm.llm_service import OllamaLLMService
            base_url = base_url or "http://localhost:11434"
            model = model or "llama3.2:3b"
            return OllamaLLMService(
                base_url=base_url,
                model=model,
                **kwargs
            )
        else:
            return service_class(base_url=base_url, model=model, **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Retorna lista de providers disponibles."""
        return list(cls._registry.keys())


class CacheServiceFactory(BaseFactory):
    """
    Factory para crear servicios de cache.
    
    Soporta múltiples implementaciones:
    - Memory (default)
    - Redis
    - File-based
    """
    
    _registry: Dict[str, Type[CacheServiceProtocol]] = {}
    
    @classmethod
    def register(cls, name: str, service_class: Type[CacheServiceProtocol]):
        """Registra un tipo de servicio de cache."""
        cls._registry[name] = service_class
        logger.info(f"✅ Cache Service '{name}' registrado en factory")
    
    def create(
        self,
        cache_type: str = "memory",
        **kwargs
    ) -> CacheServiceProtocol:
        """
        Crea un servicio de cache según el tipo especificado.
        
        Args:
            cache_type: Tipo de cache ('memory', 'redis', 'file')
            **kwargs: Parámetros específicos del cache
            
        Returns:
            Instancia del servicio de cache
        """
        if cache_type == "memory":
            from app.infrastructure.cache.evaluation_cache import EvaluationCache
            return EvaluationCache(**kwargs)
        elif cache_type == "redis":
            from app.infrastructure.cache.redis_cache import RedisCache
            return RedisCache(**kwargs)
        else:
            if cache_type not in self._registry:
                raise ValueError(
                    f"Cache type '{cache_type}' no registrado. "
                    f"Disponibles: {list(self._registry.keys())} + ['memory', 'redis']"
                )
            service_class = self._registry[cache_type]
            return service_class(**kwargs)
    
    @classmethod
    def get_available_types(cls) -> list:
        """Retorna lista de tipos de cache disponibles."""
        return ["memory", "redis"] + list(cls._registry.keys())


class MLServiceFactory(BaseFactory):
    """
    Factory para crear servicios ML.
    
    Soporta múltiples implementaciones:
    - Embeddings (sentence-transformers)
    - Emotion Detector
    - Difficulty Adjuster
    - Clustering Service
    """
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, service_class: Type):
        """Registra un tipo de servicio ML."""
        cls._registry[name] = service_class
        logger.info(f"✅ ML Service '{name}' registrado en factory")
    
    def create(
        self,
        service_type: str,
        **kwargs
    ) -> Any:
        """
        Crea un servicio ML según el tipo especificado.
        
        Args:
            service_type: Tipo de servicio ('embeddings', 'emotion', 'difficulty', 'clustering')
            **kwargs: Parámetros específicos del servicio
            
        Returns:
            Instancia del servicio ML
        """
        if service_type == "embeddings":
            from app.infrastructure.ml.question_embeddings import get_embeddings_service
            return get_embeddings_service(**kwargs)
        elif service_type == "emotion":
            from app.infrastructure.ml.multilingual_emotion_detector import MultilingualEmotionDetector
            return MultilingualEmotionDetector(**kwargs)
        elif service_type == "difficulty":
            from app.infrastructure.ml.neural_difficulty_adjuster import NeuralDifficultyAdjuster
            return NeuralDifficultyAdjuster(**kwargs)
        elif service_type == "clustering":
            from app.infrastructure.ml.advanced_clustering import AdvancedQuestionClusteringService
            embeddings_service = kwargs.pop("embeddings_service", None)
            if embeddings_service is None:
                embeddings_service = self.create("embeddings")
            return AdvancedQuestionClusteringService(
                embeddings_service=embeddings_service,
                **kwargs
            )
        else:
            if service_type not in self._registry:
                raise ValueError(
                    f"ML service type '{service_type}' no registrado. "
                    f"Disponibles: {list(self._registry.keys())} + ['embeddings', 'emotion', 'difficulty', 'clustering']"
                )
            service_class = self._registry[service_type]
            return service_class(**kwargs)
    
    @classmethod
    def get_available_types(cls) -> list:
        """Retorna lista de tipos de servicios ML disponibles."""
        return ["embeddings", "emotion", "difficulty", "clustering"] + list(cls._registry.keys())


class RepositoryFactory(BaseFactory):
    """
    Factory para crear repositorios.
    
    Soporta múltiples implementaciones:
    - Memory (para testing)
    - JSON (para datos estáticos)
    - PostgreSQL (para producción)
    """
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, repository_class: Type):
        """Registra un tipo de repositorio."""
        cls._registry[name] = repository_class
        logger.info(f"✅ Repository '{name}' registrado en factory")
    
    def create(
        self,
        repository_type: str,
        **kwargs
    ) -> Any:
        """
        Crea un repositorio según el tipo especificado.
        
        Args:
            repository_type: Tipo de repositorio ('memory', 'json', 'postgres')
            **kwargs: Parámetros específicos del repositorio
            
        Returns:
            Instancia del repositorio
        """
        if repository_type == "memory":
            from app.infrastructure.persistence.memory_interview_repository import MemoryInterviewRepository
            return MemoryInterviewRepository(**kwargs)
        elif repository_type == "json":
            from app.infrastructure.persistence.json_question_repository import JsonQuestionRepository
            return JsonQuestionRepository(**kwargs)
        elif repository_type == "postgres":
            # Future: PostgreSQL repository
            raise NotImplementedError("PostgreSQL repository not yet implemented")
        else:
            if repository_type not in self._registry:
                raise ValueError(
                    f"Repository type '{repository_type}' no registrado. "
                    f"Disponibles: {list(self._registry.keys())} + ['memory', 'json']"
                )
            repository_class = self._registry[repository_type]
            return repository_class(**kwargs)
    
    @classmethod
    def get_available_types(cls) -> list:
        """Retorna lista de tipos de repositorios disponibles."""
        return ["memory", "json"] + list(cls._registry.keys())


# Singleton instances
_llm_factory: Optional[LLMServiceFactory] = None
_cache_factory: Optional[CacheServiceFactory] = None
_ml_factory: Optional[MLServiceFactory] = None
_repository_factory: Optional[RepositoryFactory] = None


def get_llm_factory() -> LLMServiceFactory:
    """Retorna instancia singleton del LLM factory."""
    global _llm_factory
    if _llm_factory is None:
        _llm_factory = LLMServiceFactory()
        # Registrar providers por defecto
        from app.infrastructure.llm.llm_service import OllamaLLMService
        LLMServiceFactory.register("ollama", OllamaLLMService)
    return _llm_factory


def get_cache_factory() -> CacheServiceFactory:
    """Retorna instancia singleton del Cache factory."""
    global _cache_factory
    if _cache_factory is None:
        _cache_factory = CacheServiceFactory()
    return _cache_factory


def get_ml_factory() -> MLServiceFactory:
    """Retorna instancia singleton del ML factory."""
    global _ml_factory
    if _ml_factory is None:
        _ml_factory = MLServiceFactory()
    return _ml_factory


def get_repository_factory() -> RepositoryFactory:
    """Retorna instancia singleton del Repository factory."""
    global _repository_factory
    if _repository_factory is None:
        _repository_factory = RepositoryFactory()
    return _repository_factory

