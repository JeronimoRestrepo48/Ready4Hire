"""
Decorator Pattern Implementation.

Proporciona decoradores para agregar funcionalidades
a servicios existentes sin modificar su código.
"""

from functools import wraps
from typing import Callable, Any, Dict, Optional, TypeVar, Protocol, Type
import time
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceProtocol(Protocol):
    """Protocolo para servicios decorables."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto."""
        ...


def CachedService(cache_service: Any, ttl: int = 3600):
    """
    Decorador para agregar cache a un servicio.
    
    Args:
        cache_service: Servicio de cache
        ttl: Time to live en segundos
    """
    def decorator(service_class: Type[T]) -> Type[T]:
        class CachedServiceWrapper:
            def __init__(self, *args, **kwargs):
                self._service = service_class(*args, **kwargs)
                self._cache = cache_service
                self._ttl = ttl
            
            def generate(self, prompt: str, **kwargs) -> str:
                # Generar clave de cache
                cache_key = f"service:{service_class.__name__}:{hash(prompt)}:{hash(str(kwargs))}"
                
                # Intentar obtener del cache
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.debug(f"✅ Cache hit para {service_class.__name__}")
                    return cached
                
                # Generar usando el servicio real
                result = self._service.generate(prompt, **kwargs)
                
                # Guardar en cache
                self._cache.set(cache_key, result, ttl=self._ttl)
                logger.debug(f"💾 Resultado guardado en cache para {service_class.__name__}")
                
                return result
            
            def __getattr__(self, name):
                # Delegar otros métodos al servicio real
                return getattr(self._service, name)
        
        return CachedServiceWrapper
    return decorator


def LoggedService(log_level: str = "INFO"):
    """
    Decorador para agregar logging a un servicio.
    
    Args:
        log_level: Nivel de logging
    """
    def decorator(service_class: Type[T]) -> Type[T]:
        class LoggedServiceWrapper:
            def __init__(self, *args, **kwargs):
                self._service = service_class(*args, **kwargs)
                self._logger = logging.getLogger(service_class.__name__)
                self._log_level = getattr(logging, log_level.upper())
            
            def generate(self, prompt: str, **kwargs) -> str:
                self._logger.log(
                    self._log_level,
                    f"Generando con {service_class.__name__}: prompt_length={len(prompt)}"
                )
                start_time = time.time()
                
                try:
                    result = self._service.generate(prompt, **kwargs)
                    elapsed = time.time() - start_time
                    self._logger.log(
                        self._log_level,
                        f"✅ {service_class.__name__} completado en {elapsed:.2f}s"
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self._logger.error(
                        f"❌ Error en {service_class.__name__} después de {elapsed:.2f}s: {e}"
                    )
                    raise
            
            def __getattr__(self, name):
                return getattr(self._service, name)
        
        return LoggedServiceWrapper
    return decorator


def RetryService(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorador para agregar retry logic a un servicio.
    
    Args:
        max_retries: Número máximo de reintentos
        backoff_factor: Factor de backoff exponencial
    """
    def decorator(service_class: Type[T]) -> Type[T]:
        class RetryServiceWrapper:
            def __init__(self, *args, **kwargs):
                self._service = service_class(*args, **kwargs)
                self._max_retries = max_retries
                self._backoff_factor = backoff_factor
            
            def generate(self, prompt: str, **kwargs) -> str:
                last_exception = None
                
                for attempt in range(self._max_retries):
                    try:
                        return self._service.generate(prompt, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < self._max_retries - 1:
                            wait_time = self._backoff_factor * (2 ** attempt)
                            logger.warning(
                                f"⚠️ Intento {attempt + 1}/{self._max_retries} falló, "
                                f"reintentando en {wait_time:.1f}s: {e}"
                            )
                            time.sleep(wait_time)
                        else:
                            logger.error(f"❌ Todos los reintentos fallaron para {service_class.__name__}")
                
                raise last_exception
            
            def __getattr__(self, name):
                return getattr(self._service, name)
        
        return RetryServiceWrapper
    return decorator


def MetricsService(metrics_service: Any):
    """
    Decorador para agregar métricas a un servicio.
    
    Args:
        metrics_service: Servicio de métricas
    """
    def decorator(service_class: Type[T]) -> Type[T]:
        class MetricsServiceWrapper:
            def __init__(self, *args, **kwargs):
                self._service = service_class(*args, **kwargs)
                self._metrics = metrics_service
                self._service_name = service_class.__name__
            
            def generate(self, prompt: str, **kwargs) -> str:
                start_time = time.time()
                
                try:
                    result = self._service.generate(prompt, **kwargs)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Registrar métricas de éxito
                    self._metrics.inc_counter(f"{self._service_name}_requests_total")
                    self._metrics.observe_histogram(f"{self._service_name}_duration_ms", elapsed_ms)
                    
                    return result
                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Registrar métricas de error
                    self._metrics.inc_counter(f"{self._service_name}_errors_total")
                    self._metrics.observe_histogram(f"{self._service_name}_error_duration_ms", elapsed_ms)
                    
                    raise
            
            def __getattr__(self, name):
                return getattr(self._service, name)
        
        return MetricsServiceWrapper
    return decorator


# Función decoradora para métodos
def log_method_call(func: Callable) -> Callable:
    """Decorador para logging de llamadas a métodos."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"🔍 Llamando {func.__name__} con args={len(args)}, kwargs={list(kwargs.keys())}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"✅ {func.__name__} completado en {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {func.__name__} falló después de {elapsed:.2f}s: {e}")
            raise
    return wrapper


def retry_on_failure(max_retries: int = 3, exceptions: tuple = (Exception,)):
    """Decorador para retry en métodos."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = 1.0 * (2 ** attempt)
                        logger.warning(f"⚠️ Intento {attempt + 1}/{max_retries} falló, reintentando: {e}")
                        time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator

