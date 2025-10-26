"""
Prometheus Metrics Exporter for Ready4Hire
Exposes application metrics in Prometheus format
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.core import CollectorRegistry
import time
from typing import Dict, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# ============================================================================
# HTTP Metrics
# ============================================================================

http_requests_total = Counter(
    "ready4hire_http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"], registry=registry
)

http_request_duration_seconds = Histogram(
    "ready4hire_http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"], registry=registry
)

# ============================================================================
# Interview Metrics
# ============================================================================

interviews_total = Counter(
    "ready4hire_interviews_total", "Total interviews started", ["difficulty", "category"], registry=registry
)

interviews_completed = Counter(
    "ready4hire_interviews_completed", "Total interviews completed", ["difficulty", "category"], registry=registry
)

interview_duration_seconds = Histogram(
    "ready4hire_interview_duration_seconds", "Interview duration in seconds", ["difficulty"], registry=registry
)

questions_answered = Counter(
    "ready4hire_questions_answered_total", "Total questions answered", ["category", "difficulty"], registry=registry
)

# ============================================================================
# LLM Metrics
# ============================================================================

llm_requests_total = Counter(
    "ready4hire_llm_requests_total", "Total LLM requests", ["model", "status"], registry=registry
)

llm_request_duration_seconds = Histogram(
    "ready4hire_llm_request_duration_seconds",
    "LLM request duration",
    ["model"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=registry,
)

llm_tokens_used = Counter(
    "ready4hire_llm_tokens_used_total", "Total tokens used by LLM", ["model", "type"], registry=registry
)

# ============================================================================
# System Metrics
# ============================================================================

active_interviews = Gauge("ready4hire_active_interviews", "Number of currently active interviews", registry=registry)

cache_hits = Counter("ready4hire_cache_hits_total", "Total cache hits", ["cache_type"], registry=registry)

cache_misses = Counter("ready4hire_cache_misses_total", "Total cache misses", ["cache_type"], registry=registry)

# ============================================================================
# Error Metrics
# ============================================================================

errors_total = Counter("ready4hire_errors_total", "Total errors", ["type", "severity"], registry=registry)

# ============================================================================
# Decorator for tracking request metrics
# ============================================================================


def track_request_metrics(endpoint: str):
    """Decorator to track request metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                errors_total.labels(type=type(e).__name__, severity="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(method="POST", endpoint=endpoint).observe(duration)

        return wrapper

    return decorator


# ============================================================================
# Helper Functions
# ============================================================================


def record_interview_start(difficulty: str, category: str):
    """Record interview start"""
    interviews_total.labels(difficulty=difficulty, category=category).inc()
    active_interviews.inc()


def record_interview_complete(difficulty: str, category: str, duration: float):
    """Record interview completion"""
    interviews_completed.labels(difficulty=difficulty, category=category).inc()
    interview_duration_seconds.labels(difficulty=difficulty).observe(duration)
    active_interviews.dec()


def record_question_answered(category: str, difficulty: str):
    """Record question answered"""
    questions_answered.labels(category=category, difficulty=difficulty).inc()


def record_llm_request(model: str, duration: float, status: str = "success", tokens: int = 0):
    """Record LLM request"""
    llm_requests_total.labels(model=model, status=status).inc()
    llm_request_duration_seconds.labels(model=model).observe(duration)
    if tokens > 0:
        llm_tokens_used.labels(model=model, type="total").inc(tokens)


def record_cache_hit(cache_type: str):
    """Record cache hit"""
    cache_hits.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """Record cache miss"""
    cache_misses.labels(cache_type=cache_type).inc()


def get_metrics() -> bytes:
    """Get metrics in Prometheus format"""
    return generate_latest(registry)


def get_metrics_dict() -> Dict[str, Any]:
    """Get metrics as dictionary"""
    metrics = {}

    for metric in registry.collect():
        for sample in metric.samples:
            metrics[sample.name] = {"value": sample.value, "labels": sample.labels, "type": metric.type}

    return metrics
