# 🚀 Runbook de Despliegue a Producción - Ready4Hire

Este runbook consolida los pasos operativos para promover una versión de Ready4Hire a producción, incluyendo validaciones previas, ejecución del despliegue y verificación posterior.

---

## 1. Preparativos previos

1. **Revisión de cambios**
   - Confirmar que los PRs críticos están mergeados.
   - Validar versiones de backend (`Ready4Hire/Ready4Hire`) y frontend (`Ready4Hire/WebApp`).

2. **Variables de entorno**
   - Alinear `.env` y `appsettings.Production.json` con los valores actualizados (ver `docs/CONFIGURATION.md`).
   - Confirmar `BACKEND_BASE_URL=http://127.0.0.1:8001` en pruebas y producción.

3. **Dependencias externas**  
   Asegurar instancias reales listas:
   - Ollama (`OLLAMA_BASE_URL`)
   - Redis (`REDIS_URL`)
   - Qdrant (`QDRANT_URL`)
   - PostgreSQL (`DATABASE_URL`)
   - Celery Broker/Result Backend
   - Prometheus/Grafana/Alertmanager
   - Sentry (DSN activo)

4. **Backups**
   - `pg_dump` de la base de datos de producción.
   - Exportar colecciones Qdrant (`qdrant export`).
   - Snapshot de volúmenes Docker si aplica.

5. **Tests obligatorios**
   ```bash
   # Backend unit/integration
   cd Ready4Hire/Ready4Hire
   source venv/bin/activate
   pytest

   # Heavy services smoke
   pytest tests/smoke/test_heavy_services.py

   # Infraestructura real (con servicios levantados)
   RUN_INFRA_SMOKE=true pytest tests/smoke/test_infrastructure_connectivity.py

   # E2E Playwright apuntando a backend en 8001
   cd ../e2e-tests
   npm install
   npx playwright test
   ```

6. **Cheques manuales**
   - Ejecutar `python scripts/infra_smoke_checks.py`.
   - Revisar dashboards Grafana >15 min (sin alertas activas).
   - Validar logs recientes en Sentry y Prometheus.

---

## 2. Despliegue backend (FastAPI)

### 2.1 Docker Compose (entorno estándar)
```bash
cd /opt/ready4hire
git pull origin main
docker compose -f docker-compose.yml -f docker-compose.secrets.yml pull
docker compose -f docker-compose.yml -f docker-compose.secrets.yml up -d backend celery worker
```

### 2.2 Verificación rápida
```bash
curl -s ${BACKEND_BASE_URL}/api/v2/health | jq
docker compose ps backend
docker compose logs backend --tail=100
```

### 2.3 Migraciones (si aplica)
```bash
source venv/bin/activate
alembic upgrade head  # Backend Python
```

---

## 3. Despliegue frontend (Blazor Server)

### 3.1 Publicar build
```bash
cd Ready4Hire/WebApp
dotnet publish -c Release -o /var/www/ready4hire
```

### 3.2 Reiniciar servicio
```bash
sudo systemctl restart ready4hire-blazor.service
sudo systemctl status ready4hire-blazor.service
```

### 3.3 Smoke
```bash
curl -s -I http://127.0.0.1:5000/healthz
```

---

## 4. Post-despliegue inmediato

1. **Smoke Tests**
   - `python Ready4Hire/scripts/infra_smoke_checks.py`
   - `npx playwright test --reporter=list --grep @smoke`

2. **Monitoreo**
   - Verificar tableros Grafana (`Ready4Hire - Backend`, `Celery Queues`).
   - Confirmar que Prometheus muestra nuevas series en <60s.
   - Revisar Alertmanager (0 alertas firing).

3. **Observabilidad**
   - Enviar evento de prueba a Sentry (`python Ready4Hire/scripts/infra_smoke_checks.py` lo realiza automáticamente si DSN definido).
   - Verificar logs de nginx y aplicación (`/var/log/nginx/ready4hire_access.log`).

4. **Validaciones funcionales**
   - Login en Blazor con cuenta real.
   - Start Interview → Validar flujo completo con Ollama real.
   - Generación de certificado (`/api/v2/certificates/...`).
   - Operaciones de gamificación (puntos/leaderboard).

---

## 5. Monitoreo durante la primera hora

- Revisar cada 15 minutos:
  - Latencia P95 backend `< 500ms`.
  - Tasa de errores `< 0.5%`.
  - Colas de Celery sin backlog acumulado.
  - Recursos del servidor (`htop`, `docker stats`).

- Alertas a vigilar:
  - `ready4hire_backend_down`
  - `ready4hire_celery_queue_backlog`
  - `ready4hire_prometheus_scrape_failed`

---

## 6. Rollback

1. **Docker Compose**
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.secrets.yml down
   docker compose -f docker-compose.yml -f docker-compose.secrets.yml up -d backend@previous celery@previous
   ```

2. **Restaurar DB/Qdrant**
   ```bash
   psql ${DATABASE_URL} < backups/ready4hire_<fecha>.sql
   qdrant tools import --input backups/qdrant_<fecha>.tar
   ```

3. **Frontend**
   ```bash
   sudo systemctl restart ready4hire-blazor@previous.service
   ```

Documentar causas y acciones en `docs/INCIDENT_LOG.md`.

---

## 7. Checklist final

- [ ] Despliegue completado sin errores.
- [ ] Smoke tests (manuales + automáticos) en verde.
- [ ] Alertas sin incidencias.
- [ ] Evento de Sentry recibido.
- [ ] Dashboard de negocio actualizado (DAU/MAU, entrevistas).
- [ ] Comunicaciones enviadas al equipo (Slack/Email).

---

**Última actualización:** 2025-11-11  
**Responsable:** Equipo DevOps Ready4Hire

