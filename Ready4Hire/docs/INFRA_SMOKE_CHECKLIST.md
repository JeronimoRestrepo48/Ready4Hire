# ✅ Checklist de Infraestructura & Smoke Tests

Este documento describe los pasos mínimos para validar que todos los servicios críticos de Ready4Hire están operativos con instancias reales antes de un despliegue a producción.

> **Nota:** Todos los servicios deben apuntar al backend FastAPI en `http://127.0.0.1:8001` (o el host/puerto definidos en producción). Ajusta las URLs según tu entorno.

---

## 1. Variables de entorno recomendadas

```bash
export BACKEND_BASE_URL="http://127.0.0.1:8001"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export REDIS_URL="redis://localhost:6379/1"
export QDRANT_URL="http://127.0.0.1:6333"
export DATABASE_URL="postgresql+psycopg://ready4hire:ready4hire@localhost:5432/ready4hire"
export PROMETHEUS_HEALTH_URL="http://127.0.0.1:9090/-/healthy"
export SENTRY_DSN="https://<public>:<secret>@o.sentry.io/<project>"
export RUN_INFRA_SMOKE=true  # Para habilitar las pruebas opcionales
```

---

## 2. Orden sugerido de arranque

1. **PostgreSQL** – inicializar base de datos y aplicar migraciones (`dotnet ef database update` para Blazor, `alembic upgrade head` para FastAPI si corresponde).
2. **Redis** – requerido por Celery y cache.
3. **Qdrant** – vector DB para RAG.
4. **Ollama** – modelos LLM locales.
5. **Backend FastAPI** – ejecutar en `8001` con `MOCK_OLLAMA=false`.
6. **Celery worker** – procesado asíncrono.
7. **Prometheus / Grafana / Alertmanager** – monitoreo.
8. **Sentry** – observabilidad.
9. **Frontend Blazor** y **Mobile/API Gateway**.

Ejemplo con Docker Compose:

```bash
docker compose -f docker-compose.yml -f docker-compose.secrets.yml up -d postgres redis qdrant ollama
docker compose up -d backend celery prometheus grafana
```

---

## 3. Validaciones manuales por servicio

### 3.1 FastAPI (Backend)
- **Comando**: `curl -s ${BACKEND_BASE_URL}/api/v2/health | jq`
- **Esperado**: `overall_status` = `healthy` o `degraded` (sin errores críticos).
- **Revisa**: Logs en `logs/audit_log.jsonl`.

### 3.2 Ollama
- **Comando**: `curl -s ${OLLAMA_BASE_URL}/api/tags | jq '.models[].name'`
- **Esperado**: Lista de modelos disponibles (`llama3.1`, `nomic-embed-text`, etc.).
- **Extra**: `ollama run llama3.1 "Di hola"` para smoke interactivo.

### 3.3 Redis
- **Comando**: `redis-cli -u ${REDIS_URL} ping`
- **Esperado**: `PONG`.
- **Extra**: `redis-cli -u ${REDIS_URL} keys '*'` (verificar namespace `ready4hire:*`).

### 3.4 Qdrant
- **Comando**: `curl -s ${QDRANT_URL}/collections | jq`
- **Esperado**: `status` = `ok` y colecciones `ready4hire_interviews`, etc.
- **Carga inicial (si vacío)**:
  ```bash
  python Ready4Hire/app/scripts/3_deployment/initialize_qdrant.py
  ```

### 3.5 PostgreSQL
- **Comando**: `psql ${DATABASE_URL} -c "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"`
- **Esperado**: Tablas `AspNetUsers`, `InterviewSessions`, `GamificationStats`, etc.
- **Backups**: `pg_dump ${DATABASE_URL} > backups/ready4hire_$(date +%F).sql`.

### 3.6 Celery Worker
- **Arranque**:
  ```bash
  cd Ready4Hire/Ready4Hire
  source venv/bin/activate
  export MOCK_OLLAMA=false
  celery -A app.infrastructure.tasks.celery_app worker -l info -Q evaluations,default
  ```
- **Comando de verificación**:
  ```python
  python -c "from app.infrastructure.tasks.celery_app import celery_app; print(celery_app.control.inspect().ping())"
  ```
- **Esperado**: diccionario con `{'celery@host': {'ok': 'pong'}}`.

### 3.7 Prometheus / Grafana / Alertmanager
- **Prometheus**: `curl -s ${PROMETHEUS_HEALTH_URL}` → `Prometheus is Healthy`.
- **Métricas FastAPI**: `curl -s ${BACKEND_BASE_URL}/metrics | head`.
- **Grafana**: ingresar a `http://localhost:3000` y confirmar dashboards `Ready4Hire - Backend`, `Celery Workers`.
- **Alertmanager**: `curl -s http://localhost:9093/-/healthy`.

### 3.8 Sentry
- **Comando**:
  ```python
  python - <<'PY'
  import os, sentry_sdk
  sentry_sdk.init(os.environ["SENTRY_DSN"])
  sentry_sdk.capture_message("ready4hire smoke test")
  print("Evento enviado. Verifica en el dashboard de Sentry.")
  PY
  ```
- **Esperado**: evento visible en el proyecto correspondiente.

### 3.9 Frontend Blazor (App Server)
- **Arranque**:
  ```bash
  cd Ready4Hire/WebApp
  dotnet run --urls "http://0.0.0.0:5000"
  ```
- **Smoke**:
  ```bash
  curl -s -I http://127.0.0.1:5000/healthz
  ```
- **Esperado**: HTTP 200 y título `Ready4Hire - Sistema Inteligente...`.

---

## 4. Smoke tests automatizados

### 4.1 Pytest opcional
```bash
cd Ready4Hire/Ready4Hire
RUN_INFRA_SMOKE=true pytest tests/smoke/test_infrastructure_connectivity.py
```
- **Requisitos**: módulos `redis`, `psycopg`, `qdrant-client`, `httpx`.
- **Salida**: todos los casos `PASSED` (sin saltos).

### 4.2 Script `infra_smoke_checks`
```bash
cd Ready4Hire/Ready4Hire
source venv/bin/activate
python scripts/infra_smoke_checks.py
```
- **Resultado**: resumen con `OK`/`SKIPPED`/`ERROR` por servicio.
- **Usa**: las mismas variables de entorno declaradas al inicio.

---

## 5. Checklist post-validación

- [ ] Backups recientes (Postgres + Qdrant).
- [ ] Alertas configuradas (Prometheus / Alertmanager / Sentry).
- [ ] Celery workers activos con colas `default`, `evaluations`.
- [ ] Scripts de limpieza (cron para Redis y logs) habilitados.
- [ ] Dashboard de Grafana actualizado con métricas >15 minutos.
- [ ] Evento de Sentry recibido (para validar DSN).
- [ ] Playwright E2E ejecutado apuntando al backend (`API_BASE_URL` = `http://localhost:8001`).
- [ ] Documentar resultados en `docs/AUDIO_INTEGRATION_VERIFICATION.md` y `docs/FLOW_VERIFICATION_REPORT.md`.

---

## 6. Problemas comunes & soluciones rápidas

| Servicio | Síntoma | Acción |
|----------|---------|--------|
| Ollama | `connection refused` | `ollama serve` no está activo o firewall bloquea 11434. |
| Redis | `NOAUTH Authentication required` | Revisar `REDIS_URL` (usuario/pass) o activar `requirepass` en config. |
| Qdrant | `status: error` | Ejecutar `docker logs qdrant` y verificar ruta de almacenamiento (`/qdrant/storage`). |
| Celery | Ping vacío | Worker no conectado. Revisa variables de entorno `CELERY_BROKER_URL`, `RESULT_BACKEND`. |
| Prometheus | HTTP 503 | Archivo YAML inválido. Ejecuta `promtool check config monitoring/prometheus.yml`. |
| Sentry | Evento no llega | DSN incorrecto o firewall saliente bloqueado. |
| FastAPI | `/health` degradado | Revisa `logs/audit_log.jsonl`, dependencias en `app.container.Container.health_check()`. |

---

## 7. Referencias rápidas

- `Ready4Hire/docs/CONFIGURATION.md`
- `Ready4Hire/docs/TROUBLESHOOTING.md`
- `monitoring/prometheus.yml`
- `docker-compose.yml` + `docker-compose.secrets.yml`
- `scripts/infra_smoke_checks.py`

---

**Última actualización:** 2025-11-11  
**Responsable:** Equipo de Plataforma Ready4Hire

