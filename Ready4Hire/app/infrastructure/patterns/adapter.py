"""
Adapter Pattern Implementation.

Proporciona adaptadores para integrar servicios externos
de forma consistente y desacoplada.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class ServiceAdapter(ABC):
    """
    Adaptador base para servicios externos.
    
    Proporciona una interfaz común para diferentes proveedores.
    """
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el servicio está disponible."""
        pass
    
    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """Obtiene información del servicio."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del servicio."""
        pass


class LLMAdapter(ServiceAdapter):
    """
    Adaptador para servicios LLM.
    
    Adapta diferentes proveedores LLM (Ollama, OpenAI, etc.)
    a una interfaz común.
    """
    
    def __init__(self, provider: str = "ollama", **kwargs):
        self.provider = provider
        self.config = kwargs
        self._service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Inicializa el servicio LLM según el provider."""
        from app.infrastructure.patterns.factory import get_llm_factory
        factory = get_llm_factory()
        self._service = factory.create(provider=self.provider, **self.config)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto usando el LLM."""
        return self._service.generate(prompt=prompt, **kwargs)
    
    def chat(self, messages: list, **kwargs) -> str:
        """Genera respuesta en modo chat."""
        return self._service.chat(messages=messages, **kwargs)
    
    def is_available(self) -> bool:
        """Verifica si el servicio está disponible."""
        try:
            # Health check simple
            self._service.generate("test", max_tokens=1)
            return True
        except Exception:
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Obtiene información del servicio."""
        return {
            "provider": self.provider,
            "config": self.config,
            "available": self.is_available(),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del servicio."""
        try:
            available = self.is_available()
            return {
                "status": "healthy" if available else "unhealthy",
                "provider": self.provider,
                "available": available,
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": self.provider,
                "error": str(e),
            }


class CacheAdapter(ServiceAdapter):
    """
    Adaptador para servicios de cache.
    
    Adapta diferentes proveedores de cache (Memory, Redis, etc.)
    a una interfaz común.
    """
    
    def __init__(self, cache_type: str = "memory", **kwargs):
        self.cache_type = cache_type
        self.config = kwargs
        self._cache = None
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Inicializa el servicio de cache según el tipo."""
        from app.infrastructure.patterns.factory import get_cache_factory
        factory = get_cache_factory()
        self._cache = factory.create(cache_type=self.cache_type, **self.config)
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Almacena un valor en el cache."""
        self._cache.set(key, value, ttl=ttl)
    
    def delete(self, key: str) -> bool:
        """Elimina un valor del cache."""
        if hasattr(self._cache, 'delete'):
            return self._cache.delete(key)
        return False
    
    def is_available(self) -> bool:
        """Verifica si el servicio está disponible."""
        try:
            # Test simple
            self.set("__test__", "test")
            self.get("__test__")
            self.delete("__test__")
            return True
        except Exception:
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Obtiene información del servicio."""
        return {
            "cache_type": self.cache_type,
            "config": self.config,
            "available": self.is_available(),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del servicio."""
        try:
            available = self.is_available()
            return {
                "status": "healthy" if available else "unhealthy",
                "cache_type": self.cache_type,
                "available": available,
            }
        except Exception as e:
            return {
                "status": "error",
                "cache_type": self.cache_type,
                "error": str(e),
            }

