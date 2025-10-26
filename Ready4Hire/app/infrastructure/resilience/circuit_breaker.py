"""
Circuit Breaker y Retry Logic para resiliencia del sistema.
Implementa el patrón Circuit Breaker para servicios externos (Ollama, PostgreSQL, etc).
"""

import logging
import asyncio
from typing import Any, Callable, Optional, TypeVar, ParamSpec
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class CircuitState(Enum):
    """Estados del Circuit Breaker"""
    CLOSED = "closed"      # Funcionando normalmente
    OPEN = "open"          # Circuit abierto, rechazando requests
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó


class CircuitBreaker:
    """
    Circuit Breaker para proteger servicios externos.
    
    Estados:
    - CLOSED: Funcionamiento normal
    - OPEN: Demasiados errores, rechaza requests
    - HALF_OPEN: Probando recuperación
    
    Configuración:
    - failure_threshold: Número de errores antes de abrir circuit
    - timeout: Tiempo antes de intentar recuperación
    - success_threshold: Éxitos en HALF_OPEN antes de cerrar circuit
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.utcnow()
        
        logger.info(f"🔌 Circuit Breaker initialized: {name} (threshold={failure_threshold}, timeout={timeout}s)")
    
    async def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Ejecuta función protegida por circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            CircuitOpenError: Si el circuit está abierto
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                logger.warning(f"⚠️ Circuit OPEN for {self.name}, rejecting call")
                raise CircuitOpenError(f"Circuit breaker is OPEN for {self.name}")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Verifica si debe intentar recuperación"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout
    
    def _on_success(self) -> None:
        """Callback cuando la llamada es exitosa"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"✅ Success in HALF_OPEN: {self.name} ({self.success_count}/{self.success_threshold})")
            
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count en estado CLOSED
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Callback cuando la llamada falla"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        logger.error(f"❌ Failure in {self.state.value}: {self.name} ({self.failure_count}/{self.failure_threshold})")
        
        if self.state == CircuitState.HALF_OPEN:
            # Un solo fallo en HALF_OPEN abre el circuit de nuevo
            self._transition_to_open()
        
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transiciona a estado OPEN"""
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.utcnow()
        logger.warning(f"🔴 Circuit OPEN: {self.name}")
    
    def _transition_to_half_open(self) -> None:
        """Transiciona a estado HALF_OPEN"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self.last_state_change = datetime.utcnow()
        logger.info(f"🟡 Circuit HALF_OPEN: {self.name}")
    
    def _transition_to_closed(self) -> None:
        """Transiciona a estado CLOSED"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.utcnow()
        logger.info(f"🟢 Circuit CLOSED: {self.name}")
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas del circuit breaker"""
        time_in_state = (datetime.utcnow() - self.last_state_change).total_seconds()
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_in_current_state_seconds": round(time_in_state, 2),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class CircuitOpenError(Exception):
    """Exception lanzada cuando el circuit está abierto"""
    pass


# Global registry de circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    success_threshold: int = 2
) -> CircuitBreaker:
    """
    Factory para obtener circuit breaker.
    Usa singleton pattern por nombre.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout,
            success_threshold=success_threshold
        )
    
    return _circuit_breakers[name]


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator para retry con exponential backoff.
    
    Args:
        max_attempts: Número máximo de intentos
        min_wait: Espera mínima entre reintentos (segundos)
        max_wait: Espera máxima entre reintentos (segundos)
        exponential_base: Base para backoff exponencial
        exceptions: Tupla de excepciones para reintentar
        
    Example:
        @with_retry(max_attempts=3, min_wait=2, max_wait=10)
        async def call_external_api():
            ...
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=min_wait,
                min=min_wait,
                max=max_wait,
                exp_base=exponential_base
            ),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO)
        )
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=min_wait,
                min=min_wait,
                max=max_wait,
                exp_base=exponential_base
            ),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO)
        )
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    success_threshold: int = 2
):
    """
    Decorator para proteger función con circuit breaker.
    
    Args:
        name: Nombre del circuit breaker
        failure_threshold: Fallos antes de abrir circuit
        timeout: Segundos antes de intentar recuperación
        success_threshold: Éxitos para cerrar circuit
        
    Example:
        @with_circuit_breaker("ollama_service", failure_threshold=3, timeout=30)
        async def call_ollama():
            ...
    """
    def decorator(func):
        breaker = get_circuit_breaker(
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout,
            success_threshold=success_threshold
        )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_retry_and_circuit_breaker(
    circuit_name: str,
    max_attempts: int = 3,
    circuit_failure_threshold: int = 5,
    circuit_timeout: float = 60.0
):
    """
    Combina retry + circuit breaker para máxima resiliencia.
    
    Args:
        circuit_name: Nombre del circuit breaker
        max_attempts: Intentos de retry
        circuit_failure_threshold: Fallos para abrir circuit
        circuit_timeout: Timeout del circuit
        
    Example:
        @with_retry_and_circuit_breaker(
            circuit_name="llm_service",
            max_attempts=3,
            circuit_failure_threshold=5
        )
        async def evaluate_answer(answer: str):
            ...
    """
    def decorator(func):
        # Primero aplicar circuit breaker
        func_with_breaker = with_circuit_breaker(
            name=circuit_name,
            failure_threshold=circuit_failure_threshold,
            timeout=circuit_timeout
        )(func)
        
        # Luego aplicar retry
        func_with_retry = with_retry(
            max_attempts=max_attempts,
            exceptions=(Exception,)
        )(func_with_breaker)
        
        return func_with_retry
    
    return decorator


def get_all_circuit_breakers_stats() -> list[dict]:
    """Obtiene estadísticas de todos los circuit breakers"""
    return [cb.get_stats() for cb in _circuit_breakers.values()]

