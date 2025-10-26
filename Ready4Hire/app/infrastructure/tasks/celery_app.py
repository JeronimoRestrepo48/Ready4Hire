"""
Celery App para procesamiento asíncrono de tareas.
Permite ejecutar evaluaciones, análisis y operaciones pesadas en background.
"""

import logging
from celery import Celery
from kombu import Exchange, Queue
import os

logger = logging.getLogger(__name__)

# Configuración de Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Crear app de Celery
celery_app = Celery(
    "ready4hire",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "app.infrastructure.tasks.evaluation_tasks",
        "app.infrastructure.tasks.ml_tasks",
        "app.infrastructure.tasks.notification_tasks"
    ]
)

# Configuración
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutos max por task
    task_soft_time_limit=240,  # Warning a los 4 min
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # Resultados expiran en 1 hora
    task_acks_late=True,  # ACK después de completar
    worker_disable_rate_limits=False,
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
)

# Definir colas con prioridades
celery_app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default", priority=0),
    Queue("high_priority", Exchange("high_priority"), routing_key="high", priority=10),
    Queue("low_priority", Exchange("low_priority"), routing_key="low", priority=0),
    Queue("ml_tasks", Exchange("ml_tasks"), routing_key="ml", priority=5),
    Queue("evaluations", Exchange("evaluations"), routing_key="eval", priority=7),
)

# Rutas para diferentes tipos de tareas
celery_app.conf.task_routes = {
    "app.infrastructure.tasks.evaluation_tasks.*": {"queue": "evaluations"},
    "app.infrastructure.tasks.ml_tasks.*": {"queue": "ml_tasks"},
    "app.infrastructure.tasks.notification_tasks.*": {"queue": "high_priority"},
}

logger.info(f"✅ Celery app configured: broker={CELERY_BROKER_URL}")


@celery_app.task(bind=True)
def debug_task(self):
    """Task de debug para verificar que Celery funciona"""
    logger.info(f"Request: {self.request!r}")
    return {"status": "ok", "task_id": self.request.id}

