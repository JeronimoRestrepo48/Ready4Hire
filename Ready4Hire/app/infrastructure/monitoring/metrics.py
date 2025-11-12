"""
Prometheus Metrics for Ready4Hire.
Exporta métricas clave del sistema para monitoreo y alertas.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Sistema de métricas compatible con Prometheus.

    Métricas exportadas:
    - Contadores: Requests, errors, evaluations
    - Histogramas: Latencia de LLM, tiempo de respuesta
    - Gauges: Entrevistas activas, circuit breaker state
    """

    def __init__(self, enabled: bool = True):
        """
        Inicializa el sistema de métricas.

        Args:
            enabled: Si False, las métricas son no-op
        """
        self.enabled = enabled
        self._lock = Lock()

        # Contadores (monótonamente crecientes)
        self.counters = defaultdict(int)

        # Gauges (valores que suben/bajan)
        self.gauges = defaultdict(float)

        # Histogramas (listas de valores)
        self.histograms = defaultdict(list)

        # Inicializar métricas básicas
        self._init_metrics()

        if enabled:
            logger.info("✅ PrometheusMetrics initialized")

    def _init_metrics(self):
        """Inicializa métricas con valores por defecto."""
        # Contadores
        self.counters.update(
            {
                "http_requests_total": 0,
                "http_errors_total": 0,
                "llm_requests_total": 0,
                "llm_errors_total": 0,
                "evaluations_total": 0,
                "evaluation_fallbacks_total": 0,
                "evaluation_retries_total": 0,
                "cache_hits_total": 0,
                "cache_misses_total": 0,
                "circuit_breaker_opens_total": 0,
                "prompt_injections_blocked_total": 0,
                "hints_generated_total": 0,
                "attempts_total": 0,
            }
        )

        # Gauges
        self.gauges.update(
            {
                "active_interviews": 0,
                "circuit_breaker_state": 0,  # 0=closed, 1=half_open, 2=open
                "cache_hit_rate": 0.0,
                "llm_avg_latency_ms": 0.0,
                "avg_evaluation_score": 0.0,  # NUEVO
                "avg_attempts_per_question": 0.0,  # NUEVO
                "fallback_rate": 0.0,  # NUEVO
            }
        )

    def inc_counter(self, name: str, value: int = 1):
        """Incrementa un contador."""
        if not self.enabled:
            return

        with self._lock:
            self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Establece el valor de un gauge."""
        if not self.enabled:
            return

        with self._lock:
            self.gauges[name] = value

    def observe_histogram(self, name: str, value: float):
        """Agrega una observación a un histograma."""
        if not self.enabled:
            return

        with self._lock:
            self.histograms[name].append(value)

            # Mantener solo últimos 1000 valores para evitar memory leak
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]

    def record_http_request(self, method: str, path: str, status_code: int, latency_ms: float):
        """Registra una request HTTP."""
        if not self.enabled:
            return

        self.inc_counter("http_requests_total")

        if status_code >= 400:
            self.inc_counter("http_errors_total")

        self.observe_histogram(f"http_request_duration_ms_{method.lower()}", latency_ms)

    def record_llm_request(self, success: bool, latency_ms: float, model: str):
        """Registra una request al LLM."""
        if not self.enabled:
            return

        self.inc_counter("llm_requests_total")

        if not success:
            self.inc_counter("llm_errors_total")

        self.observe_histogram("llm_request_duration_ms", latency_ms)

        # Actualizar latencia promedio
        with self._lock:
            values = self.histograms.get("llm_request_duration_ms", [])
            if values:
                self.gauges["llm_avg_latency_ms"] = sum(values) / len(values)

    def record_evaluation(self, success: bool, latency_ms: float, cached: bool = False, score: Optional[float] = None):
        """
        Registra una evaluación.
        
        NUEVO: Soporte para score y métricas mejoradas.
        
        Args:
            success: Si la evaluación fue exitosa
            latency_ms: Latencia en milisegundos
            cached: Si vino del caché
            score: Score de la evaluación (opcional)
        """
        if not self.enabled:
            return

        self.inc_counter("evaluations_total")

        if cached:
            self.inc_counter("cache_hits_total")
        else:
            self.inc_counter("cache_misses_total")

        if not success:
            self.inc_counter("evaluation_fallbacks_total")

        self.observe_histogram("evaluation_duration_ms", latency_ms)
        
        # NUEVO: Registrar score si está disponible
        if score is not None:
            self.observe_histogram("evaluation_scores", score)
            # Actualizar promedio de scores
            with self._lock:
                scores = self.histograms.get("evaluation_scores", [])
                if scores:
                    self.gauges["avg_evaluation_score"] = sum(scores) / len(scores)

        # Actualizar cache hit rate y fallback rate
        with self._lock:
            hits = self.counters.get("cache_hits_total", 0)
            misses = self.counters.get("cache_misses_total", 0)
            total = hits + misses
            if total > 0:
                self.gauges["cache_hit_rate"] = (hits / total) * 100
            
            fallbacks = self.counters.get("evaluation_fallbacks_total", 0)
            evaluations = self.counters.get("evaluations_total", 0)
            if evaluations > 0:
                self.gauges["fallback_rate"] = (fallbacks / evaluations) * 100

    def record_circuit_breaker_open(self, name: str):
        """Registra apertura de circuit breaker."""
        if not self.enabled:
            return

        self.inc_counter("circuit_breaker_opens_total")
        self.set_gauge("circuit_breaker_state", 2)  # OPEN
        logger.warning(f"📊 Circuit breaker '{name}' opened (metric recorded)")

    def record_circuit_breaker_close(self, name: str):
        """Registra cierre de circuit breaker."""
        if not self.enabled:
            return

        self.set_gauge("circuit_breaker_state", 0)  # CLOSED
        logger.info(f"📊 Circuit breaker '{name}' closed (metric recorded)")

    def record_prompt_injection_blocked(self):
        """Registra prompt injection bloqueado."""
        if not self.enabled:
            return

        self.inc_counter("prompt_injections_blocked_total")
        logger.warning("📊 Prompt injection blocked (metric recorded)")

    def inc_active_interviews(self):
        """Incrementa contador de entrevistas activas."""
        if not self.enabled:
            return

        with self._lock:
            self.gauges["active_interviews"] += 1

    def dec_active_interviews(self):
        """Decrementa contador de entrevistas activas."""
        if not self.enabled:
            return

        with self._lock:
            self.gauges["active_interviews"] = max(0, self.gauges["active_interviews"] - 1)

    def get_metrics_text(self) -> str:
        """
        Exporta métricas en formato Prometheus.

        Returns:
            String en formato Prometheus text exposition
        """
        lines = []

        # Header
        lines.append("# HELP ready4hire_metrics Ready4Hire Application Metrics")
        lines.append("# TYPE ready4hire_metrics untyped")
        lines.append("")

        # Contadores
        lines.append("# HELP ready4hire_counters Counter metrics")
        lines.append("# TYPE ready4hire_counters counter")
        with self._lock:
            for name, value in sorted(self.counters.items()):
                lines.append(f"ready4hire_{name} {value}")

        lines.append("")

        # Gauges
        lines.append("# HELP ready4hire_gauges Gauge metrics")
        lines.append("# TYPE ready4hire_gauges gauge")
        with self._lock:
            for name, value in sorted(self.gauges.items()):
                lines.append(f"ready4hire_{name} {value:.2f}")

        lines.append("")

        # Histogramas (simplificado: solo percentiles)
        lines.append("# HELP ready4hire_histograms Histogram metrics (p50, p95, p99)")
        lines.append("# TYPE ready4hire_histograms histogram")
        with self._lock:
            for name, values in sorted(self.histograms.items()):
                if values:
                    sorted_values = sorted(values)
                    count = len(sorted_values)

                    p50 = sorted_values[int(count * 0.50)]
                    p95 = sorted_values[int(count * 0.95)]
                    p99 = sorted_values[int(count * 0.99)]

                    lines.append(f"ready4hire_{name}_p50 {p50:.2f}")
                    lines.append(f"ready4hire_{name}_p95 {p95:.2f}")
                    lines.append(f"ready4hire_{name}_p99 {p99:.2f}")
                    lines.append(f"ready4hire_{name}_count {count}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """
        Obtiene estadísticas en formato dict.

        Returns:
            Dict con métricas
        """
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "p50": sorted(values)[int(len(values) * 0.50)] if values else 0,
                        "p95": sorted(values)[int(len(values) * 0.95)] if values else 0,
                        "p99": sorted(values)[int(len(values) * 0.99)] if values else 0,
                    }
                    for name, values in self.histograms.items()
                },
            }

    def reset(self):
        """Resetea todas las métricas (solo para testing)."""
        with self._lock:
            self._init_metrics()
            self.histograms.clear()


# ============================================================================
# Instancia global (singleton)
# ============================================================================

_metrics: Optional[PrometheusMetrics] = None


def get_metrics(enabled: bool = True) -> PrometheusMetrics:
    """Obtiene la instancia global de métricas."""
    global _metrics
    if _metrics is None:
        _metrics = PrometheusMetrics(enabled=enabled)
    return _metrics


# ============================================================================
# Decorators para medir latencia
# ============================================================================


def measure_latency(metric_name: str):
    """
    Decorator para medir latencia de funciones.

    Usage:
        @measure_latency('my_function_duration_ms')
        def my_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency_ms = (time.time() - start) * 1000
                get_metrics().observe_histogram(metric_name, latency_ms)

        return wrapper

    return decorator
