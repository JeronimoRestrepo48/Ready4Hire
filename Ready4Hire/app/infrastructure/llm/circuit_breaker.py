"""
Circuit Breaker Pattern para Ollama.
Previene cascading failures y mejora resiliencia del sistema.

Estados:
- CLOSED: Sistema funciona normal
- OPEN: Demasiados fallos, no se hacen requests
- HALF_OPEN: Probando recuperación
"""

import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Error lanzado cuando el circuit está OPEN."""

    pass


class CircuitBreaker:
    """
    Circuit Breaker para servicios externos (Ollama).

    Características:
    - Abre circuito tras N fallos consecutivos
    - Se cierra automáticamente tras timeout
    - Estado HALF_OPEN para probar recuperación
    - Thread-safe
    - Métricas de fallos
    """

    def __init__(
        self,
        failure_threshold: int = 5,  # Fallos consecutivos antes de abrir
        recovery_timeout: int = 60,  # Segundos antes de probar recuperación
        expected_exception: type = Exception,  # Excepciones que cuentan como fallo
        name: str = "circuit_breaker",
        success_threshold: int = 2,  # NUEVO: Éxitos necesarios en HALF_OPEN para cerrar
        health_check_callback: Optional[Callable] = None,  # NUEVO: Callback para health check
    ):
        """
        Inicializa el circuit breaker.

        Args:
            failure_threshold: Número de fallos antes de abrir circuito
            recovery_timeout: Segundos antes de intentar recuperación
            expected_exception: Tipo de excepción que cuenta como fallo
            name: Nombre del circuit breaker para logs
            success_threshold: Número de éxitos en HALF_OPEN antes de cerrar (default: 2)
            health_check_callback: Función opcional para health check cuando está OPEN
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self.success_threshold = success_threshold
        self.health_check_callback = health_check_callback

        # Estado
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0  # NUEVO: Contador de éxitos en HALF_OPEN
        self._last_failure_time: Optional[datetime] = None
        self._last_health_check: Optional[datetime] = None
        self._lock = Lock()

        # Métricas
        self.total_successes = 0
        self.total_failures = 0
        self.total_rejected = 0
        self.health_check_attempts = 0
        self.health_check_successes = 0

        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s, "
            f"success_threshold={success_threshold}"
        )

    @property
    def state(self) -> CircuitState:
        """Obtiene el estado actual del circuit breaker."""
        with self._lock:
            # Si está OPEN, verificar si es momento de intentar recuperación
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    # NUEVO: Intentar health check antes de pasar a HALF_OPEN
                    if self.health_check_callback:
                        try:
                            if self._perform_health_check():
                                self._state = CircuitState.HALF_OPEN
                                logger.info(f"🟡 Circuit '{self.name}' -> HALF_OPEN (health check passed, attempting recovery)")
                            else:
                                # Health check falló, resetear timer
                                self._last_failure_time = datetime.now()
                                logger.warning(f"⚠️ Circuit '{self.name}' health check failed, staying OPEN")
                        except Exception as e:
                            logger.warning(f"⚠️ Circuit '{self.name}' health check error: {e}, staying OPEN")
                            self._last_failure_time = datetime.now()
                    else:
                        # Sin health check, pasar directamente a HALF_OPEN
                        self._state = CircuitState.HALF_OPEN
                        logger.info(f"🟡 Circuit '{self.name}' -> HALF_OPEN (attempting recovery)")

            return self._state
    
    def _perform_health_check(self) -> bool:
        """Realiza un health check del servicio."""
        if not self.health_check_callback:
            return False
        
        try:
            self.health_check_attempts += 1
            result = self.health_check_callback()
            if result:
                self.health_check_successes += 1
            return result
        except Exception as e:
            logger.debug(f"Health check exception: {e}")
            return False

    def _should_attempt_reset(self) -> bool:
        """Verifica si es momento de intentar resetear el circuit."""
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta una función protegida por el circuit breaker.

        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados

        Returns:
            Resultado de la función

        Raises:
            CircuitBreakerError: Si el circuito está OPEN
            Exception: Si la función falla
        """
        current_state = self.state

        # Si está OPEN, rechazar request
        if current_state == CircuitState.OPEN:
            self.total_rejected += 1
            raise CircuitBreakerError(
                f"Circuit '{self.name}' is OPEN. " f"Service unavailable. Try again in {self.recovery_timeout}s"
            )

        try:
            # Intentar ejecutar función
            result = func(*args, **kwargs)

            # Éxito
            self._on_success()
            return result

        except self.expected_exception as e:
            # Fallo esperado
            self._on_failure()
            raise

        except Exception as e:
            # Fallo inesperado (no abre circuito)
            logger.error(f"Unexpected error in circuit '{self.name}': {e}")
            raise

    def _on_success(self):
        """Callback en éxito."""
        with self._lock:
            self.total_successes += 1
            self._failure_count = 0

            # Si estaba HALF_OPEN, acumular éxitos
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                # NUEVO: Requerir múltiples éxitos antes de cerrar
                if self._success_count >= self.success_threshold:
                    success_count_before_reset = self._success_count
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info(f"✅ Circuit '{self.name}' -> CLOSED (recovered after {success_count_before_reset} successes)")
                else:
                    logger.debug(f"Circuit '{self.name}' HALF_OPEN: {self._success_count}/{self.success_threshold} successes")

    def _on_failure(self):
        """Callback en fallo."""
        with self._lock:
            self.total_failures += 1
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            # NUEVO: Si está en HALF_OPEN, un solo fallo lo vuelve a abrir
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(f"🔴 Circuit '{self.name}' -> OPEN (failure in HALF_OPEN state)")
            # Si alcanzamos threshold, abrir circuito
            elif self._failure_count >= self.failure_threshold:
                previous_state = self._state
                self._state = CircuitState.OPEN

                if previous_state != CircuitState.OPEN:
                    logger.error(f"🔴 Circuit '{self.name}' -> OPEN " f"({self._failure_count} consecutive failures)")
            else:
                logger.warning(f"⚠️ Circuit '{self.name}' failure " f"({self._failure_count}/{self.failure_threshold})")

    def reset(self):
        """Resetea manualmente el circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")

    def get_stats(self) -> dict:
        """Obtiene estadísticas del circuit breaker."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "success_count": self._success_count,  # NUEVO
            "success_threshold": self.success_threshold,  # NUEVO
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejected": self.total_rejected,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "health_check_attempts": self.health_check_attempts,  # NUEVO
            "health_check_successes": self.health_check_successes,  # NUEVO
        }


# ============================================================================
# Decorator para funciones
# ============================================================================


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    name: Optional[str] = None,
):
    """
    Decorator para proteger funciones con circuit breaker.

    Usage:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def call_external_api():
            ...
    """

    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name,
        )

        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        # Exponer circuit breaker para inspección
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator
