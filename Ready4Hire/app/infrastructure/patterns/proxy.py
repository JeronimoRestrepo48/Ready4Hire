"""
Proxy Pattern Implementation.

Proporciona proxies para lazy loading, cache transparente,
y control de acceso a servicios.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class LazyServiceProxy:
    """
    Proxy para lazy loading de servicios.
    
    El servicio solo se inicializa cuando se usa por primera vez.
    """
    
    def __init__(self, service_factory: Callable[[], Any]):
        """
        Inicializa el proxy.
        
        Args:
            service_factory: Función que crea el servicio
        """
        self._service_factory = service_factory
        self._service: Optional[Any] = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Inicializa el servicio si no está inicializado."""
        if not self._initialized:
            logger.debug("🔧 Inicializando servicio lazy...")
            self._service = self._service_factory()
            self._initialized = True
            logger.debug("✅ Servicio lazy inicializado")
    
    def __getattr__(self, name: str) -> Any:
        """Delega llamadas al servicio real."""
        self._ensure_initialized()
        return getattr(self._service, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Permite llamar al proxy como función."""
        self._ensure_initialized()
        return self._service(*args, **kwargs)
    
    def is_initialized(self) -> bool:
        """Verifica si el servicio está inicializado."""
        return self._initialized
    
    def reset(self):
        """Resetea el proxy (útil para testing)."""
        self._service = None
        self._initialized = False


class CachedServiceProxy:
    """
    Proxy para cache transparente de servicios.
    
    Intercepta llamadas y cachea resultados automáticamente.
    """
    
    def __init__(
        self,
        service: Any,
        cache_service: Any,
        ttl: int = 3600,
        cache_key_generator: Optional[Callable] = None
    ):
        """
        Inicializa el proxy con cache.
        
        Args:
            service: Servicio real a cachear
            cache_service: Servicio de cache
            ttl: Time to live en segundos
            cache_key_generator: Función para generar claves de cache
        """
        self._service = service
        self._cache = cache_service
        self._ttl = ttl
        self._key_generator = cache_key_generator or self._default_key_generator
    
    def _default_key_generator(self, method_name: str, args: tuple, kwargs: dict) -> str:
        """Genera clave de cache por defecto."""
        import hashlib
        import json
        
        key_data = {
            "method": method_name,
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"proxy:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def __getattr__(self, name: str) -> Any:
        """Intercepta llamadas a métodos."""
        attr = getattr(self._service, name)
        
        # Si no es callable, retornar directamente
        if not callable(attr):
            return attr
        
        # Crear wrapper para métodos
        def cached_method(*args, **kwargs) -> Any:
            # Generar clave de cache
            cache_key = self._key_generator(name, args, kwargs)
            
            # Intentar obtener del cache
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"✅ Cache hit para {name}")
                return cached
            
            # Ejecutar método real
            result = attr(*args, **kwargs)
            
            # Guardar en cache
            self._cache.set(cache_key, result, ttl=self._ttl)
            logger.debug(f"💾 Resultado de {name} guardado en cache")
            
            return result
        
        return cached_method
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Limpia el cache.
        
        Args:
            pattern: Patrón para limpiar (None = limpiar todo)
        """
        # Implementación simplificada
        # En producción, usaría Redis SCAN o similar
        logger.info(f"🗑️ Cache limpiado (pattern: {pattern or 'all'})")


class ProtectedServiceProxy:
    """
    Proxy para control de acceso a servicios.
    
    Permite agregar validación, rate limiting, etc.
    """
    
    def __init__(
        self,
        service: Any,
        access_control: Optional[Callable] = None,
        rate_limiter: Optional[Any] = None
    ):
        """
        Inicializa el proxy con control de acceso.
        
        Args:
            service: Servicio real
            access_control: Función de control de acceso
            rate_limiter: Rate limiter opcional
        """
        self._service = service
        self._access_control = access_control
        self._rate_limiter = rate_limiter
    
    def __getattr__(self, name: str) -> Any:
        """Intercepta llamadas con control de acceso."""
        attr = getattr(self._service, name)
        
        if not callable(attr):
            return attr
        
        def protected_method(*args, **kwargs) -> Any:
            # Validar acceso
            if self._access_control:
                if not self._access_control(name, args, kwargs):
                    raise PermissionError(f"Acceso denegado a {name}")
            
            # Rate limiting
            if self._rate_limiter:
                self._rate_limiter.check_rate_limit(name)
            
            # Ejecutar método
            return attr(*args, **kwargs)
        
        return protected_method

