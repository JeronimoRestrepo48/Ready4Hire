#!/usr/bin/env python3
"""
Infraestructura - Smoke Checks
==============================

Script utilitario para validar conectividad con los servicios críticos de Ready4Hire.

Cada chequeo intenta conectarse al servicio real usando las variables de entorno
documentadas en `docs/INFRA_SMOKE_CHECKLIST.md`. El script ofrece feedback rápido
(`OK`, `SKIPPED`, `ERROR`) y mensajes de ayuda para acelerar el diagnóstico.

Uso:
    source venv/bin/activate
    python scripts/infra_smoke_checks.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import closing
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Optional
from urllib.parse import urlparse


@dataclass
class CheckResult:
    service: str
    status: str  # OK, SKIPPED, ERROR
    detail: str


def _load_optional(module_name: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


def _check_http_endpoint(name: str, url: str, timeout: float = 5.0) -> CheckResult:
    httpx = _load_optional("httpx")
    if httpx is None:
        return CheckResult(name, "SKIPPED", "httpx no está instalado")

    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return CheckResult(name, "OK", f"{url} → {response.status_code}")
    except Exception as exc:
        return CheckResult(name, "ERROR", f"{url} → {exc}")


def _check_tcp_port(name: str, host: str, port: int) -> CheckResult:
    try:
        with closing(socket.create_connection((host, port), timeout=3.0)):
            return CheckResult(name, "OK", f"{host}:{port} aceptó conexión TCP")
    except OSError as exc:
        return CheckResult(name, "ERROR", f"{host}:{port} no responde ({exc})")


def _check_backend() -> CheckResult:
    base_url = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8001")
    return _check_http_endpoint("FastAPI", f"{base_url}/api/v2/health", timeout=10.0)


def _check_ollama() -> CheckResult:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    httpx = _load_optional("httpx")
    if httpx is None:
        return CheckResult("Ollama", "SKIPPED", "httpx no está instalado")

    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = [model.get("name") for model in response.json().get("models", [])]
        if not models:
            return CheckResult("Ollama", "ERROR", "No se encontraron modelos registrados")
        return CheckResult("Ollama", "OK", f"Modelos disponibles: {', '.join(models[:5])}")
    except Exception as exc:
        return CheckResult("Ollama", "ERROR", str(exc))


def _check_redis() -> CheckResult:
    redis = _load_optional("redis")
    if redis is None:
        return CheckResult("Redis", "SKIPPED", "Paquete redis no instalado")

    url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
    try:
        client = redis.Redis.from_url(url)
        client.ping()
        return CheckResult("Redis", "OK", f"PING exitoso ({url})")
    except Exception as exc:
        return CheckResult("Redis", "ERROR", f"{url} → {exc}")


def _check_qdrant() -> CheckResult:
    qdrant_client = _load_optional("qdrant_client")
    if qdrant_client is None:
        return CheckResult("Qdrant", "SKIPPED", "Paquete qdrant-client no instalado")

    url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    try:
        client = qdrant_client.QdrantClient(url=url, timeout=5.0)
        collections = client.get_collections()
        names = [col.name for col in collections.collections]
        return CheckResult("Qdrant", "OK", f"Colecciones: {', '.join(names) or 'sin colecciones'}")
    except Exception as exc:
        return CheckResult("Qdrant", "ERROR", f"{url} → {exc}")


def _check_postgres() -> CheckResult:
    psycopg = _load_optional("psycopg")
    if psycopg is None:
        return CheckResult("PostgreSQL", "SKIPPED", "Paquete psycopg no instalado")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return CheckResult("PostgreSQL", "SKIPPED", "DATABASE_URL no definido")

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
        return CheckResult("PostgreSQL", "OK", version)
    except Exception as exc:
        return CheckResult("PostgreSQL", "ERROR", str(exc))


def _check_celery() -> CheckResult:
    try:
        from app.infrastructure.tasks.celery_app import celery_app
    except Exception as exc:
        return CheckResult("Celery", "ERROR", f"No se pudo importar celery_app ({exc})")

    try:
        ping = celery_app.control.inspect().ping()
        if ping:
            nodes = ", ".join(ping.keys())
            return CheckResult("Celery", "OK", f"Workers activos: {nodes}")
        return CheckResult("Celery", "ERROR", "No se recibieron respuestas de workers")
    except Exception as exc:
        return CheckResult("Celery", "ERROR", str(exc))


def _check_prometheus() -> CheckResult:
    url = os.getenv("PROMETHEUS_HEALTH_URL", "http://127.0.0.1:9090/-/healthy")
    return _check_http_endpoint("Prometheus", url)


def _check_sentry() -> CheckResult:
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return CheckResult("Sentry", "SKIPPED", "SENTRY_DSN no configurado")

    sentry_sdk = _load_optional("sentry_sdk")
    if sentry_sdk is None:
        return CheckResult("Sentry", "SKIPPED", "Paquete sentry-sdk no instalado")

    try:
        sentry_sdk.init(dsn, traces_sample_rate=0.0)
        sentry_sdk.capture_message("ready4hire infra smoke check")
        return CheckResult("Sentry", "OK", "Evento de prueba enviado")
    except Exception as exc:
        return CheckResult("Sentry", "ERROR", str(exc))


def _check_blazor() -> CheckResult:
    url = os.getenv("BLAZOR_BASE_URL", "http://127.0.0.1:5000/healthz")
    return _check_http_endpoint("Blazor", url)


def _collect_checks() -> list[Callable[[], CheckResult]]:
    return [
        _check_backend,
        _check_ollama,
        _check_redis,
        _check_qdrant,
        _check_postgres,
        _check_celery,
        _check_prometheus,
        _check_sentry,
        _check_blazor,
    ]


def main() -> int:
    print("Ready4Hire · Infraestructura · Smoke Checks\n")
    print("Usando variables de entorno:")
    for key in [
        "BACKEND_BASE_URL",
        "OLLAMA_BASE_URL",
        "REDIS_URL",
        "QDRANT_URL",
        "DATABASE_URL",
        "PROMETHEUS_HEALTH_URL",
        "SENTRY_DSN",
        "BLAZOR_BASE_URL",
    ]:
        value = os.getenv(key, "<no definido>")
        print(f"  - {key} = {value}")
    print("")

    results: list[CheckResult] = []
    for check in _collect_checks():
        result = check()
        results.append(result)
        status = result.status.ljust(7)
        print(f"[{status}] {result.service}: {result.detail}")

    errors = [r for r in results if r.status == "ERROR"]
    skipped = [r for r in results if r.status == "SKIPPED"]

    print("\nResumen:")
    print(f"  OK      : {len(results) - len(errors) - len(skipped)}")
    print(f"  SKIPPED : {len(skipped)}")
    print(f"  ERROR   : {len(errors)}")

    if errors:
        print("\nRevisar servicios con estado ERROR antes de continuar con el despliegue.")
        return 1

    print("\nTodos los servicios críticos respondieron correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

