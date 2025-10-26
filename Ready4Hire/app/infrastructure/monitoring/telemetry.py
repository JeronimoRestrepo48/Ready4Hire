"""
OpenTelemetry Instrumentation para observabilidad completa.
Métricas, trazas y logs integrados con Prometheus y Grafana.
"""

import logging
import time
from typing import Optional, Dict, Any
from functools import wraps
from contextlib import asynccontextmanager

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from prometheus_client import start_http_server

logger = logging.getLogger(__name__)


class TelemetryService:
    """
    Servicio centralizado de telemetría para Ready4Hire.
    Proporciona métricas, trazas y observabilidad completa.
    """
    
    def __init__(
        self,
        service_name: str = "ready4hire",
        service_version: str = "2.1.0",
        prometheus_port: int = 8000
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.prometheus_port = prometheus_port
        
        # Resource identification
        self.resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": "production"
        })
        
        # Inicializar providers
        self._init_tracing()
        self._init_metrics()
        
        # Tracer y Meter
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Métricas custom
        self._create_custom_metrics()
        
        logger.info(f"✅ Telemetry initialized: {service_name} v{service_version}")
    
    def _init_tracing(self) -> None:
        """Inicializa OpenTelemetry Tracing"""
        provider = TracerProvider(resource=self.resource)
        
        # OTLP Exporter (para enviar a Grafana Tempo, Jaeger, etc)
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://localhost:4317",
            insecure=True
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        trace.set_tracer_provider(provider)
        logger.info("✅ Tracing configured")
    
    def _init_metrics(self) -> None:
        """Inicializa OpenTelemetry Metrics con Prometheus"""
        # Prometheus exporter
        reader = PrometheusMetricReader()
        provider = MeterProvider(resource=self.resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        
        # Start Prometheus HTTP server
        try:
            start_http_server(port=self.prometheus_port)
            logger.info(f"✅ Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            logger.warning(f"⚠️ Could not start Prometheus server: {e}")
    
    def _create_custom_metrics(self) -> None:
        """Crea métricas custom para Ready4Hire"""
        
        # ─────────────────────────────────────────────────────────
        # INTERVIEW METRICS
        # ─────────────────────────────────────────────────────────
        self.interviews_started = self.meter.create_counter(
            name="interviews_started_total",
            description="Total number of interviews started",
            unit="1"
        )
        
        self.interviews_completed = self.meter.create_counter(
            name="interviews_completed_total",
            description="Total number of interviews completed",
            unit="1"
        )
        
        self.interview_duration = self.meter.create_histogram(
            name="interview_duration_seconds",
            description="Duration of interviews in seconds",
            unit="s"
        )
        
        # ─────────────────────────────────────────────────────────
        # EVALUATION METRICS
        # ─────────────────────────────────────────────────────────
        self.evaluations_total = self.meter.create_counter(
            name="evaluations_total",
            description="Total number of answer evaluations",
            unit="1"
        )
        
        self.evaluation_duration = self.meter.create_histogram(
            name="evaluation_duration_seconds",
            description="Time taken to evaluate an answer",
            unit="s"
        )
        
        self.evaluation_score = self.meter.create_histogram(
            name="evaluation_score",
            description="Distribution of evaluation scores",
            unit="1"
        )
        
        self.llm_tokens_used = self.meter.create_counter(
            name="llm_tokens_used_total",
            description="Total LLM tokens consumed",
            unit="1"
        )
        
        # ─────────────────────────────────────────────────────────
        # CACHE METRICS
        # ─────────────────────────────────────────────────────────
        self.cache_hits = self.meter.create_counter(
            name="cache_hits_total",
            description="Total cache hits",
            unit="1"
        )
        
        self.cache_misses = self.meter.create_counter(
            name="cache_misses_total",
            description="Total cache misses",
            unit="1"
        )
        
        # ─────────────────────────────────────────────────────────
        # WEBSOCKET METRICS
        # ─────────────────────────────────────────────────────────
        self.websocket_connections = self.meter.create_up_down_counter(
            name="websocket_connections_active",
            description="Number of active WebSocket connections",
            unit="1"
        )
        
        self.websocket_messages = self.meter.create_counter(
            name="websocket_messages_total",
            description="Total WebSocket messages sent",
            unit="1"
        )
        
        # ─────────────────────────────────────────────────────────
        # CELERY METRICS
        # ─────────────────────────────────────────────────────────
        self.celery_tasks_started = self.meter.create_counter(
            name="celery_tasks_started_total",
            description="Total Celery tasks started",
            unit="1"
        )
        
        self.celery_tasks_completed = self.meter.create_counter(
            name="celery_tasks_completed_total",
            description="Total Celery tasks completed",
            unit="1"
        )
        
        self.celery_task_duration = self.meter.create_histogram(
            name="celery_task_duration_seconds",
            description="Celery task execution time",
            unit="s"
        )
        
        # ─────────────────────────────────────────────────────────
        # VECTOR DB METRICS
        # ─────────────────────────────────────────────────────────
        self.vector_search_duration = self.meter.create_histogram(
            name="vector_search_duration_seconds",
            description="Vector similarity search duration",
            unit="s"
        )
        
        self.vector_search_results = self.meter.create_histogram(
            name="vector_search_results_count",
            description="Number of results from vector search",
            unit="1"
        )
        
        logger.info("✅ Custom metrics created")
    
    def instrument_fastapi(self, app) -> None:
        """Instrumenta una aplicación FastAPI"""
        FastAPIInstrumentor.instrument_app(app)
        logger.info("✅ FastAPI instrumented")
    
    def instrument_redis(self) -> None:
        """Instrumenta Redis"""
        RedisInstrumentor().instrument()
        logger.info("✅ Redis instrumented")
    
    def instrument_sqlalchemy(self, engine) -> None:
        """Instrumenta SQLAlchemy"""
        SQLAlchemyInstrumentor().instrument(engine=engine)
        logger.info("✅ SQLAlchemy instrumented")
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager para trazar operaciones.
        
        Usage:
            async with telemetry.trace_operation("evaluate_answer", {"user_id": "123"}):
                result = await evaluate_answer()
        """
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            start_time = time.time()
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
    
    def track_interview_started(self, user_id: str, role: str, difficulty: str) -> None:
        """Track cuando se inicia una entrevista"""
        self.interviews_started.add(
            1,
            {"user_id": user_id, "role": role, "difficulty": difficulty}
        )
    
    def track_interview_completed(
        self,
        user_id: str,
        role: str,
        duration_seconds: float,
        score: float
    ) -> None:
        """Track cuando se completa una entrevista"""
        self.interviews_completed.add(1, {"user_id": user_id, "role": role})
        self.interview_duration.record(duration_seconds, {"role": role})
    
    def track_evaluation(
        self,
        duration_seconds: float,
        score: float,
        category: str,
        tokens_used: int
    ) -> None:
        """Track evaluación de respuesta"""
        self.evaluations_total.add(1, {"category": category})
        self.evaluation_duration.record(duration_seconds, {"category": category})
        self.evaluation_score.record(score, {"category": category})
        self.llm_tokens_used.add(tokens_used, {"category": category})
    
    def track_cache(self, hit: bool, cache_type: str) -> None:
        """Track cache hit/miss"""
        if hit:
            self.cache_hits.add(1, {"cache_type": cache_type})
        else:
            self.cache_misses.add(1, {"cache_type": cache_type})
    
    def track_websocket_connection(self, action: str) -> None:
        """Track WebSocket connections"""
        if action == "connect":
            self.websocket_connections.add(1)
        elif action == "disconnect":
            self.websocket_connections.add(-1)
    
    def track_websocket_message(self, message_type: str) -> None:
        """Track WebSocket messages"""
        self.websocket_messages.add(1, {"type": message_type})
    
    def track_celery_task(
        self,
        task_name: str,
        status: str,
        duration_seconds: Optional[float] = None
    ) -> None:
        """Track Celery tasks"""
        if status == "started":
            self.celery_tasks_started.add(1, {"task_name": task_name})
        elif status == "completed":
            self.celery_tasks_completed.add(1, {"task_name": task_name, "status": "success"})
            if duration_seconds:
                self.celery_task_duration.record(duration_seconds, {"task_name": task_name})
        elif status == "failed":
            self.celery_tasks_completed.add(1, {"task_name": task_name, "status": "failed"})
    
    def track_vector_search(self, duration_seconds: float, results_count: int) -> None:
        """Track vector similarity search"""
        self.vector_search_duration.record(duration_seconds)
        self.vector_search_results.record(results_count)


# Decorator para tracing automático
def trace_async(operation_name: Optional[str] = None):
    """
    Decorator para tracing automático de funciones async.
    
    Usage:
        @trace_async("evaluate_answer")
        async def evaluate_answer(answer: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            telemetry = get_telemetry_service()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            async with telemetry.trace_operation(op_name):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global instance
_telemetry_instance: Optional[TelemetryService] = None


def get_telemetry_service() -> TelemetryService:
    """Factory para obtener instancia global de telemetría"""
    global _telemetry_instance
    
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryService()
    
    return _telemetry_instance


def init_telemetry(app) -> TelemetryService:
    """
    Inicializa telemetría y retorna instancia.
    
    Usage en main.py:
        telemetry = init_telemetry(app)
    """
    telemetry = get_telemetry_service()
    telemetry.instrument_fastapi(app)
    telemetry.instrument_redis()
    
    return telemetry

