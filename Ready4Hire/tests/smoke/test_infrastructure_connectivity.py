"""
Smoke tests opcionales para validar conectividad con servicios externos reales.

Activa estos tests exportando la variable de entorno RUN_INFRA_SMOKE=true
antes de ejecutar pytest. Por ejemplo:

    RUN_INFRA_SMOKE=true pytest tests/smoke/test_infrastructure_connectivity.py

Cada test usa variables de entorno específicas para apuntar a la instancia
real del servicio. Se proveen valores por defecto pensados para un entorno
local productivo.
"""

import os
from urllib.parse import urlparse

import pytest

RUN_INFRA_SMOKE = os.getenv("RUN_INFRA_SMOKE", "false").lower() == "true"

pytestmark = pytest.mark.skipif(
    not RUN_INFRA_SMOKE,
    reason="RUN_INFRA_SMOKE no está habilitado (exporta RUN_INFRA_SMOKE=true para ejecutar estas pruebas)",
)


def test_backend_health_endpoint():
    import httpx

    base_url = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8001")
    response = httpx.get(f"{base_url}/api/v2/health", timeout=10.0)
    response.raise_for_status()
    data = response.json()
    assert data.get("overall_status") in {"healthy", "degraded"}


def test_ollama_available():
    import httpx

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    response = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    assert "models" in payload and isinstance(payload["models"], list)


def test_redis_ping():
    redis = pytest.importorskip("redis")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
    client = redis.Redis.from_url(redis_url)
    assert client.ping() is True


def test_qdrant_collections():
    qdrant_client = pytest.importorskip("qdrant_client")

    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    client = qdrant_client.QdrantClient(url=qdrant_url, timeout=5.0)
    response = client.get_collections()
    assert response.status == "ok"


def test_postgres_connection():
    psycopg = pytest.importorskip("psycopg")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL no está definido")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            row = cur.fetchone()
            assert row == (1,)


def test_celery_worker_accepts_tasks():
    from app.infrastructure.tasks.celery_app import celery_app

    inspect = celery_app.control.inspect()
    ping = inspect.ping()
    assert ping, "No se recibieron respuestas de workers Celery"


def test_prometheus_healthy():
    import httpx

    prometheus_url = os.getenv("PROMETHEUS_HEALTH_URL", "http://127.0.0.1:9090/-/healthy")
    response = httpx.get(prometheus_url, timeout=5.0)
    response.raise_for_status()
    assert response.text.strip() == "Prometheus is Healthy"


def test_sentry_dsn_format():
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        pytest.skip("SENTRY_DSN no configurado")

    parsed = urlparse(dsn)
    assert parsed.scheme in {"http", "https"}
    assert parsed.netloc, "El DSN debe contener credenciales y host"

