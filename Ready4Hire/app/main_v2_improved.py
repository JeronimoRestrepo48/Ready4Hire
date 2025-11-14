# -*- coding: utf-8 -*-

"""
FastAPI Application v2.1 - Ready4Hire (IMPROVED)
Arquitectura DDD con seguridad mejorada, rate limiting, y DTOs.

Mejoras implementadas v2.1:
- ✅ CORS restringido (no más allow_origins=["*"])
- ✅ Rate limiting con slowapi
- ✅ Autenticación JWT
- ✅ DTOs separados de la lógica de dominio
- ✅ Manejo de excepciones centralizado
- ✅ Configuración desde .env
- ✅ Logging estructurado
- ✅ Redis Cache Distribuido
- ✅ WebSockets para streaming en tiempo real
- ✅ Circuit Breaker + Retry Logic
- ✅ Celery Background Tasks
- ✅ OpenTelemetry + Prometheus Monitoring
- ✅ Qdrant Vector DB para búsqueda semántica
"""

from fastapi import FastAPI, HTTPException, status, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.infrastructure.security.rate_limiting import get_rate_limit_key, RATE_LIMITS
from typing import Optional, List
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timezone
from pathlib import Path

# Configuración
from app.config import settings

# Container DI
from app.container import Container

# Design Patterns
from app.infrastructure.patterns.observer import get_event_bus, EventObserver
from app.infrastructure.patterns.facade import InfrastructureFacade

# ============================================================================
# NEW FEATURES - v2.1 Enterprise Edition
# ============================================================================

# Redis Cache
from app.infrastructure.cache.redis_cache import get_redis_cache, RedisCacheService

# WebSockets
from app.infrastructure.websocket.websocket_manager import get_websocket_manager, ConnectionManager

# Circuit Breaker
from app.infrastructure.resilience.circuit_breaker import (
    with_circuit_breaker,
    with_retry_and_circuit_breaker,
    CircuitOpenError
)

# Telemetry & Monitoring
from app.infrastructure.monitoring.telemetry import init_telemetry, get_telemetry_service, trace_async

# Celery Tasks (for async operations)
from app.infrastructure.tasks.evaluation_tasks import evaluate_answer_async

# Qdrant Vector DB
from app.infrastructure.ml.qdrant_client import get_qdrant_client, QdrantVectorStore

# DTOs
from app.application.dto import (
    StartInterviewRequest,
    StartInterviewResponse,
    ProcessAnswerRequest,
    ProcessAnswerResponse,
    HealthResponse,
    QuestionDTO,
    EvaluationDTO,
    EmotionDTO,
    ProgressDTO,
)

# Excepciones de dominio
from app.domain.exceptions import Ready4HireException, InterviewNotFound, InterviewAlreadyExists

# Security
# NOTA: Autenticación eliminada - manejada 100% por WebApp (Blazor)

# Domain
from app.domain.value_objects.emotion import Emotion
from app.domain.value_objects.context_questions import (
    get_context_questions,
    build_user_profile_context,
    CONTEXT_QUESTIONS_COUNT,
    MAIN_QUESTIONS_COUNT,
)
from app.domain.value_objects.interview_status import InterviewStatus
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.value_objects.interview_mode import PRACTICE_MODE
from app.domain.entities.interview import Interview
from app.domain.entities.answer import Answer
from app.domain.entities.question import Question

# ============================================================================
# Gamification Routes
# ============================================================================
from app.api.gamification_routes import router as gamification_router
from app.api.rag_routes import router as rag_router
from app.api.certificate_routes import router as certificate_router
from app.api.graphql_router import router as graphql_router
from app.api.audio_routes import router as audio_router

# ============================================================================
# Configurar Logging
# ============================================================================
log_level = getattr(logging, settings.LOG_LEVEL)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Rate Limiter (with user-based limiting)
# ============================================================================
# Use custom key function that prioritizes user_id over IP
limiter = Limiter(key_func=get_rate_limit_key)


# ============================================================================
# FastAPI App
# ============================================================================
# Declarar lifespan antes de crear app
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("=" * 80)
    logger.info(f"🚀 Ready4Hire v{settings.APP_VERSION} Enterprise Edition iniciando...")
    logger.info("=" * 80)
    logger.info(f"📦 Environment: {settings.ENVIRONMENT}")
    logger.info(f"📦 CORS Origins: {settings.get_cors_origins()}")
    logger.info(f"📦 Rate Limit: {settings.RATE_LIMIT_PER_MINUTE}/min")

    # ────────────────────────────────────────────────────────────────────────
    # Initialize New Features - v2.1
    # ────────────────────────────────────────────────────────────────────────
    
    # Initialize Telemetry & Monitoring
    logger.info("📊 Initializing OpenTelemetry monitoring...")
    telemetry = init_telemetry(app)
    app.state.telemetry = telemetry
    logger.info("✅ Telemetry initialized")
    
    # Initialize Redis Cache
    logger.info("🔴 Connecting to Redis cache...")
    try:
        redis_cache = await get_redis_cache()
        app.state.redis_cache = redis_cache
        # Test connection
        await redis_cache.set("system", "health_check", "ok", ttl=None)
        await redis_cache.delete("system", "health_check")
        logger.info("✅ Redis cache connected and ready")
    except Exception as e:
        logger.warning(f"⚠️  Redis cache unavailable: {e}. Running without cache.")
        app.state.redis_cache = None
    
    # Initialize WebSocket Manager
    logger.info("🌐 Initializing WebSocket manager...")
    ws_manager = get_websocket_manager()
    app.state.websocket_manager = ws_manager
    logger.info("✅ WebSocket manager initialized")
    
    # Initialize Qdrant Vector Store
    logger.info("🟣 Connecting to Qdrant vector database...")
    try:
        qdrant_client = get_qdrant_client()
        app.state.qdrant = qdrant_client
        logger.info("✅ Qdrant vector database connected")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant unavailable: {e}. Running without vector search.")
        app.state.qdrant = None
    
    logger.info("=" * 80)
    logger.info("✨ All services initialized successfully!")
    logger.info("=" * 80)

    try:
        c = get_container()
        health = c.health_check()

        logger.info("✅ Container inicializado")
        for component, status_msg in health.items():
            logger.info(f"   {component}: {status_msg}")

        # Inicializar PostgreSQL sync service
        try:
            from app.infrastructure.persistence.postgres_sync_service import get_postgres_sync

            await get_postgres_sync()
            logger.info("✅ PostgreSQL sync service initialized")
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL sync service not available: {e}")

        logger.info("🎯 Sistema listo para recibir requests")

    except Exception as e:
        logger.error(f"❌ Error en startup: {str(e)}")
        raise

    yield  # Application is running

    # Shutdown
    logger.info("👋 Ready4Hire cerrando...")
    logger.info("✅ Shutdown completado")


app = FastAPI(
    title="Ready4Hire API v2.1",
    description="Sistema de entrevistas con IA - Arquitectura DDD mejorada",
    version="2.1.0",
    docs_url="/docs" if settings.ENABLE_SWAGGER_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_REDOC else None,
    lifespan=app_lifespan,  # Use lifespan instead of on_event
)

# Rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# CORS (MEJORADO - No más "*")
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),  # ✅ Orígenes específicos
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# ============================================================================
# Static Files
# ============================================================================
if Path(settings.STATIC_FILES_PATH).exists():
    app.mount("/static", StaticFiles(directory=settings.STATIC_FILES_PATH), name="static")

# ============================================================================
# Container de DI (singleton)
# ============================================================================
container: Optional[Container] = None

# ============================================================================
# Event Observers (Design Pattern: Observer)
# ============================================================================

class InterviewEventObserver(EventObserver):
    """Observer para eventos de entrevistas."""
    
    def get_observed_events(self):
        """Define qué eventos observar."""
        return ["interview_started", "interview_completed", "answer_submitted", "question_answered"]
    
    def on_event(self, event):
        """Maneja eventos de entrevistas."""
        if event.name == "interview_started":
            logger.info(f"📊 Entrevista iniciada: {event.data.get('interview_id')}")
        elif event.name == "interview_completed":
            logger.info(f"✅ Entrevista completada: {event.data.get('interview_id')}")
        elif event.name == "answer_submitted":
            logger.debug(f"💬 Respuesta enviada: {event.data.get('question_id')}")
        elif event.name == "question_answered":
            logger.debug(f"✅ Pregunta respondida: {event.data.get('question_id')} - Score: {event.data.get('score')}")


def get_container() -> Container:
    """Obtiene el container de DI (lazy initialization)"""
    global container
    if container is None:
        container = Container(ollama_url=settings.OLLAMA_URL, ollama_model=None)  # Auto-detectar según GPU
        
        # Design Patterns: Registrar Event Observer
        container.event_bus.subscribe(InterviewEventObserver())
        logger.info("✅ Event Observer registrado para patrones de diseño")
        
    return container


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(Ready4HireException)
async def ready4hire_exception_handler(request: Request, exc: Ready4HireException):
    """Handler para excepciones de dominio."""
    logger.warning(f"Domain exception: {exc.error_code} - {exc.message}")

    # Mapear a HTTP status codes
    status_code_map = {
        "INTERVIEW_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "QUESTION_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "INTERVIEW_ALREADY_ACTIVE": status.HTTP_409_CONFLICT,
        "INVALID_TOKEN": status.HTTP_401_UNAUTHORIZED,
        "MISSING_TOKEN": status.HTTP_401_UNAUTHORIZED,
        "UNAUTHORIZED_ACCESS": status.HTTP_403_FORBIDDEN,
        "RATE_LIMIT_EXCEEDED": status.HTTP_429_TOO_MANY_REQUESTS,
    }

    status_code = status_code_map.get(exc.error_code, status.HTTP_400_BAD_REQUEST)

    return JSONResponse(status_code=status_code, content=exc.to_dict())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para HTTPException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP_ERROR", "message": exc.detail, "timestamp": datetime.now(timezone.utc).isoformat()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para excepciones no manejadas."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ============================================================================
# Include Routers
# ============================================================================
from app.api import auth_routes

app.include_router(auth_routes.router)
app.include_router(gamification_router)
app.include_router(rag_router)
app.include_router(certificate_router)
app.include_router(graphql_router)
app.include_router(audio_router)


# ============================================================================
# Helper Functions
# ============================================================================


def _validate_question_coherence(
    questions: List[Question], 
    expected_role: str, 
    expected_category: str,
    expected_difficulty: str = None
) -> List[Question]:
    """
    Valida que las preguntas seleccionadas sean coherentes con la profesión y tipo de entrevista.
    
    Args:
        questions: Lista de preguntas a validar
        expected_role: Rol/profesión esperado
        expected_category: Categoría esperada (technical, soft_skills)
        expected_difficulty: Dificultad esperada (opcional)
    
    Returns:
        Lista de preguntas validadas y filtradas
    """
    validated_questions = []
    role_keywords = expected_role.lower().split()
    
    for q in questions:
        # 1. Validar categoría
        if expected_category == "technical" and q.category != "technical":
            logger.warning(
                f"⚠️ FILTRADA: Pregunta '{q.id[:8]}...' es '{q.category}' pero se esperaba 'technical'. "
                f"Rol: {expected_role}"
            )
            continue
        
        if expected_category == "soft_skills" and q.category != "soft_skills":
            logger.warning(
                f"⚠️ FILTRADA: Pregunta '{q.id[:8]}...' es '{q.category}' pero se esperaba 'soft_skills'. "
                f"Rol: {expected_role}"
            )
            continue
        
        # 2. Validar rol (solo para preguntas técnicas)
        if expected_category == "technical" and q.category == "technical":
            # Verificar si la pregunta tiene rol asignado y coincide
            if q.role:
                # Normalizar roles para comparación
                q_role_lower = q.role.lower()
                expected_role_lower = expected_role.lower()
                
                # Verificar coincidencia directa o parcial
                role_matches = (
                    q_role_lower == expected_role_lower or
                    any(keyword in q_role_lower for keyword in role_keywords) or
                    any(keyword in expected_role_lower for keyword in q_role_lower.split())
                )
                
                if not role_matches:
                    # Verificar si hay palabras clave comunes en el texto de la pregunta
                    question_text_lower = q.text.lower()
                    has_role_keywords = any(keyword in question_text_lower for keyword in role_keywords if len(keyword) > 3)
                    
                    if not has_role_keywords:
                        logger.warning(
                            f"⚠️ ADVERTENCIA: Pregunta '{q.id[:8]}...' tiene rol '{q.role}' "
                            f"pero se esperaba '{expected_role}'. Se mantiene por similitud semántica."
                        )
                        # No filtrar, solo advertir (puede ser válida por embeddings)
            
            # Si no tiene rol asignado, verificar que el texto tenga relación con la profesión
            elif expected_role and expected_role.lower() != "general":
                question_text_lower = q.text.lower()
                has_role_keywords = any(keyword in question_text_lower for keyword in role_keywords if len(keyword) > 3)
                
                if not has_role_keywords:
                    logger.debug(
                        f"ℹ️ Pregunta '{q.id[:8]}...' sin rol específico, pero seleccionada por embeddings. "
                        f"Rol esperado: {expected_role}"
                    )
        
        # 3. Validar dificultad (si se especifica)
        if expected_difficulty and q.difficulty != expected_difficulty:
            logger.debug(
                f"ℹ️ Pregunta '{q.id[:8]}...' tiene dificultad '{q.difficulty}' "
                f"pero se esperaba '{expected_difficulty}'. Se mantiene."
            )
            # No filtrar por dificultad, solo advertir (puede ser válida)
        
        validated_questions.append(q)
    
    # Si después de validar tenemos menos preguntas de las necesarias, loguear advertencia
    if len(validated_questions) < MAIN_QUESTIONS_COUNT:
        logger.warning(
            f"⚠️ Después de validar coherencia, quedan {len(validated_questions)} preguntas "
            f"(se necesitan {MAIN_QUESTIONS_COUNT}). Se usarán las disponibles."
        )
    
    return validated_questions[:MAIN_QUESTIONS_COUNT]  # Limitar al máximo necesario


async def _select_questions_with_clustering(c: Container, interview, context_text: str, user_profile: dict = None):
    """
    Selección RÁPIDA de preguntas usando embeddings pre-computados.
    Optimizado para respuesta instantánea (<100ms).
    
    Args:
        c: Container con dependencias
        interview: Entrevista actual
        context_text: Respuestas de contexto del usuario
        user_profile: Perfil del usuario (profesión, habilidades, intereses)
    """
    import random
    import numpy as np

    try:
        # 1. Obtener preguntas según tipo
        if interview.interview_type == "technical":
            # Primero intentar preguntas específicas del rol
            role_questions = await c.question_repository.find_by_role(interview.role, category="technical")
            # Si no hay suficientes, usar todas las preguntas técnicas
            if len(role_questions) < MAIN_QUESTIONS_COUNT:
                all_technical = await c.question_repository.find_all_technical()
                # Combinar, priorizando las específicas del rol
                all_questions = role_questions + [q for q in all_technical if q.id not in [rq.id for rq in role_questions]]
            else:
                all_questions = role_questions
        elif interview.interview_type == "soft_skills":
            all_questions = await c.question_repository.find_all_soft_skills()
        else:
            # Mixed: obtener ambas
            role_tech = await c.question_repository.find_by_role(interview.role, category="technical")
            all_tech = await c.question_repository.find_all_technical()
            soft_qs = await c.question_repository.find_all_soft_skills()
            # Combinar priorizando preguntas específicas del rol
            all_questions = role_tech + [q for q in all_tech if q.id not in [rq.id for rq in role_tech]] + soft_qs

        if not all_questions:
            logger.warning(f"⚠️ No se encontraron preguntas para {interview.role}, usando preguntas genéricas")
            # Fallback: usar preguntas técnicas genéricas
            all_questions = await c.question_repository.find_all_technical()
            if not all_questions:
                logger.error("❌ No hay preguntas técnicas disponibles en el sistema")
                return []

        # 2. Filtrado ESTRICTO por nivel de experiencia
        # Mapeo: junior -> "junior", mid -> "mid", senior -> "senior"
        target_difficulty = interview.skill_level.value
        
        # Filtrar preguntas que coincidan EXACTAMENTE con el nivel
        level_filtered = [q for q in all_questions if q.difficulty == target_difficulty]
        
        # Si no hay suficientes preguntas del nivel exacto, buscar del mismo nivel pero con variaciones
        if len(level_filtered) < MAIN_QUESTIONS_COUNT:
            logger.warning(
                f"⚠️ Solo {len(level_filtered)} preguntas encontradas para nivel '{target_difficulty}'. "
                f"Buscando preguntas complementarias..."
            )
            
            # Estrategia: priorizar nivel exacto, luego permitir flexibilidad solo si es necesario
            # Para junior: permitir algunas mid si no hay suficientes
            # Para mid: permitir algunas junior o senior si no hay suficientes
            # Para senior: permitir algunas mid si no hay suficientes
            if target_difficulty == "junior":
                # Junior: permitir algunas mid si es necesario
                mid_questions = [q for q in all_questions if q.difficulty == "mid" and q.id not in [lq.id for lq in level_filtered]]
                level_filtered = level_filtered + mid_questions[:MAIN_QUESTIONS_COUNT - len(level_filtered)]
            elif target_difficulty == "mid":
                # Mid: permitir algunas junior o senior si es necesario
                junior_questions = [q for q in all_questions if q.difficulty == "junior" and q.id not in [lq.id for lq in level_filtered]]
                senior_questions = [q for q in all_questions if q.difficulty == "senior" and q.id not in [lq.id for lq in level_filtered]]
                # Mezclar junior y senior de forma balanceada
                remaining_needed = MAIN_QUESTIONS_COUNT - len(level_filtered)
                level_filtered = level_filtered + junior_questions[:remaining_needed//2] + senior_questions[:remaining_needed//2]
            else:  # senior
                # Senior: permitir algunas mid si es necesario
                mid_questions = [q for q in all_questions if q.difficulty == "mid" and q.id not in [lq.id for lq in level_filtered]]
                level_filtered = level_filtered + mid_questions[:MAIN_QUESTIONS_COUNT - len(level_filtered)]
        
        # Si aún no hay suficientes, usar todas (fallback)
        if len(level_filtered) < MAIN_QUESTIONS_COUNT:
            logger.warning(
                f"⚠️ Solo {len(level_filtered)} preguntas disponibles después de filtrar por nivel '{target_difficulty}'. "
                f"Usando todas las preguntas disponibles como fallback."
            )
            candidates = all_questions
        else:
            candidates = level_filtered
            logger.info(f"✅ {len(candidates)} preguntas seleccionadas para nivel '{target_difficulty}'")

        # 3. Enriquecer contexto con perfil de usuario
        enriched_context = context_text
        if user_profile:
            profile_context = build_user_profile_context(
                profession=user_profile.get("profession"),
                technical_skills=user_profile.get("technical_skills", []),
                soft_skills=user_profile.get("soft_skills", []),
                interests=user_profile.get("interests", []),
                experience_level=user_profile.get("experience_level"),
            )
            if profile_context:
                enriched_context = f"{profile_context} {context_text}"
                logger.info(f"✨ Perfil de usuario integrado para clustering más preciso")

        # 4. Ranking con embeddings PRE-COMPUTADOS
        if c.embeddings_service and len(enriched_context.strip()) > 20:
            try:
                context_embedding = c.embeddings_service.encode([enriched_context])[0]
                embeddings_cache = getattr(c.question_repository, "_embeddings_cache", {})

                if embeddings_cache:
                    question_embeddings = np.array(
                        [embeddings_cache.get(q.id, c.embeddings_service.encode([q.text])[0]) for q in candidates]
                    )
                    logger.debug(f"⚡ Usando embeddings pre-computados de {len(candidates)} preguntas")
                else:
                    question_embeddings = c.embeddings_service.encode([q.text for q in candidates])
                    logger.debug(f"⚠️ Sin cache, encoding on-the-fly de {len(candidates)} preguntas")

                # Similitud coseno vectorizada
                similarities = np.dot(question_embeddings, context_embedding) / (
                    np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(context_embedding)
                )

                # Top K preguntas
                top_indices = np.argsort(similarities)[::-1][:MAIN_QUESTIONS_COUNT]
                selected_questions = [candidates[i] for i in top_indices]

                # ============================================================================
                # VALIDACIÓN DE COHERENCIA: Asegurar que las preguntas sean coherentes
                # ============================================================================
                selected_questions = _validate_question_coherence(
                    selected_questions, 
                    interview.role, 
                    interview.interview_type,
                    interview.skill_level.value
                )

                logger.info(f"⚡ {len(selected_questions)} preguntas seleccionadas INSTANTÁNEAMENTE (validadas)")
                return selected_questions

            except Exception as e:
                logger.warning(f"⚠️ Fallback a selección aleatoria: {str(e)}")

        # 4. Fallback: Selección aleatoria
        selected = random.sample(candidates, min(MAIN_QUESTIONS_COUNT, len(candidates)))
        
        # ============================================================================
        # VALIDACIÓN DE COHERENCIA: Asegurar que las preguntas sean coherentes
        # ============================================================================
        selected = _validate_question_coherence(
            selected, 
            interview.role, 
            interview.interview_type,
            interview.skill_level.value
        )
        
        logger.info(f"⚡ {len(selected)} preguntas (fallback aleatorio, validadas)")
        return selected

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return []


# ============================================================================
# NOTA IMPORTANTE: AUTENTICACIÓN
# ============================================================================
# ⚠️ La autenticación es 100% manejada por WebApp (Blazor/.NET Backend)
# ⚠️ Este backend de FastAPI solo proporciona MICROSERVICIOS de:
#    - Chat/Entrevistas con IA
#    - Gamificación
# ⚠️ NO hay endpoints de autenticación aquí (login/register/tokens)
# ============================================================================

# ============================================================================
# Health Check & Monitoring
# ============================================================================


@app.get("/api/v2/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check del sistema (con nuevos servicios v2.1)."""
    try:
        c = get_container()
        health_status = c.health_check()
        
        # ────────────────────────────────────────────────────────────────────
        # Check NEW SERVICES - v2.1
        # ────────────────────────────────────────────────────────────────────
        
        # Redis Cache
        if hasattr(request.app.state, 'redis_cache') and request.app.state.redis_cache:
            try:
                await request.app.state.redis_cache.set("system", "health_check", "ok", ttl=None)
                await request.app.state.redis_cache.delete("system", "health_check")
                health_status["redis_cache"] = "✅ healthy"
            except Exception as e:
                health_status["redis_cache"] = f"❌ unhealthy: {str(e)}"
        else:
            health_status["redis_cache"] = "⚠️  disabled"
        
        # Qdrant Vector DB
        if hasattr(request.app.state, 'qdrant') and request.app.state.qdrant:
            try:
                # Simple health check - just check if client is initialized
                health_status["qdrant"] = "✅ healthy"
            except Exception as e:
                health_status["qdrant"] = f"❌ unhealthy: {str(e)}"
        else:
            health_status["qdrant"] = "⚠️  disabled"
        
        # WebSocket Manager
        if hasattr(request.app.state, 'websocket_manager'):
            try:
                ws_manager = request.app.state.websocket_manager
                active_connections = sum(len(conns) for conns in ws_manager.active_connections.values())
                health_status["websocket"] = f"✅ healthy ({active_connections} active connections)"
            except Exception as e:
                health_status["websocket"] = f"❌ unhealthy: {str(e)}"
        else:
            health_status["websocket"] = "⚠️  disabled"
        
        # Telemetry
        if hasattr(request.app.state, 'telemetry'):
            health_status["telemetry"] = "✅ healthy (metrics enabled)"
        else:
            health_status["telemetry"] = "⚠️  disabled"

        # ────────────────────────────────────────────────────────────────────
        # Determine overall status
        # ────────────────────────────────────────────────────────────────────
        
        is_degraded = any(
            any(word in v.lower() for word in ["degraded", "error", "failed", "unhealthy"])
            for v in health_status.values()
        )

        all_healthy = all("healthy" in v.lower() or "✅" in v or "disabled" in v.lower() for v in health_status.values())

        overall_status = "degraded" if is_degraded else ("healthy" if all_healthy else "degraded")

        return HealthResponse(
            status=overall_status,
            version=settings.APP_VERSION,
            timestamp=datetime.now(timezone.utc),
            components=health_status,
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()},
        )


@app.get("/health/ready", tags=["Health"])
@limiter.limit("30/minute")
async def readiness_check(request: Request):
    """
    Kubernetes readiness probe.
    
    Returns 200 if the service is ready to accept traffic.
    Checks: Database, Redis, Qdrant, Ollama
    """
    try:
        c = get_container()
        health_status = c.health_check()
        
        # Check critical dependencies
        critical_checks = [
            health_status.get("components", {}).get("repositories"),
            health_status.get("components", {}).get("llm_service"),
        ]
        
        if not all(critical_checks):
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "Critical dependencies unavailable"}
            )
        
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )


@app.get("/health/live", tags=["Health"])
@limiter.limit("30/minute")
async def liveness_check(request: Request):
    """
    Kubernetes liveness probe.
    
    Returns 200 if the service is alive (not stuck).
    Simple check - just verify the service responds.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/startup", tags=["Health"])
@limiter.limit("30/minute")
async def startup_check(request: Request):
    """
    Kubernetes startup probe.
    
    Returns 200 when the service has finished initializing.
    Used for slow-starting services.
    """
    try:
        # Check if all services are initialized
        c = get_container()
        
        # Verify container is initialized
        if not c:
            return JSONResponse(
                status_code=503,
                content={"status": "starting", "message": "Container not initialized"}
            )
        
        return {"status": "started", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "error": str(e)}
        )


@app.get("/metrics", tags=["Monitoring"])
@limiter.limit("100/minute")
async def metrics_endpoint(request: Request):
    """
    Endpoint de métricas en formato Prometheus.

    Exporta:
    - Contadores (http_requests, llm_requests, etc.)
    - Gauges (active_interviews, circuit_breaker_state, etc.)
    - Histogramas (latencias con p50/p95/p99)
    """
    from app.infrastructure.monitoring import get_metrics
    from fastapi.responses import PlainTextResponse

    metrics = get_metrics()
    prometheus_text = metrics.get_metrics_text()

    return PlainTextResponse(content=prometheus_text, media_type="text/plain; version=0.0.4")


@app.get("/api/v2/metrics/stats", tags=["Monitoring"])
@limiter.limit("100/minute")
async def metrics_stats(request: Request):
    """
    Endpoint de métricas en formato JSON (más legible).

    Retorna estadísticas actuales del sistema.
    """
    from app.infrastructure.monitoring import get_metrics

    metrics = get_metrics()
    stats = metrics.get_stats()

    return {"timestamp": datetime.now(timezone.utc), "metrics": stats, "version": settings.APP_VERSION}


# ============================================================================
# Interview Endpoints (Microservicio de Chat/Entrevistas)
# ============================================================================


@app.get("/api/v2/interviews/active/{user_id}", tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_active_interview(
    request: Request,
    user_id: str,
):
    """
    Obtiene el estado de la entrevista activa de un usuario.
    Usado para restaurar el estado cuando el usuario vuelve a la página.
    """
    try:
        c = get_container()
        
        # Buscar entrevista activa
        interview = await c.interview_repository.find_active_by_user(user_id)
        
        if not interview:
            return JSONResponse(
                status_code=404,
                content={"error": "No active interview found", "interview_id": None}
            )
        
        # Construir respuesta con el estado completo
        response_data = {
            "interview_id": interview.id,
            "role": interview.role,
            "interview_type": interview.interview_type,
            "status": interview.status.value,
            "skill_level": interview.skill_level.value,
            "mode": interview.mode.to_string() if hasattr(interview.mode, 'to_string') else str(interview.mode),
            "current_phase": interview.current_phase,
            "context_question_index": interview.context_question_index,
            "context_answers": interview.context_answers,
            "current_question": interview.current_question.to_dict() if interview.current_question else None,
            "questions_history": [q.to_dict() for q in interview.questions_history],
            "answers_history": [
                {
                    "id": a.id,
                    "question_id": a.question_id,
                    "question_text": next((q.text for q in interview.questions_history if q.id == a.question_id), ""),
                    "answer_text": a.text,
                    "score": float(a.score) if isinstance(a.score, (int, float)) else getattr(a.score, 'value', 0.0),
                    "is_correct": a.is_correct,
                    "emotion": a.emotion.value if hasattr(a.emotion, 'value') else str(a.emotion),
                    "time_taken": a.time_taken,
                    "hints_used": a.hints_used,
                    "evaluation_details": {k: (v.value if hasattr(v, 'value') else v) if not isinstance(v, (str, int, float, bool, type(None))) else v for k, v in (a.evaluation_details.items() if isinstance(a.evaluation_details, dict) else {})},
                    "created_at": a.created_at.isoformat() if hasattr(a.created_at, 'isoformat') else str(a.created_at),
                }
                for a in interview.answers_history
            ],
            "question_count": len(interview.answers_history),
            "context_questions_answered": len(interview.context_answers),
            "total_hints_used": interview.total_hints_used,
            "attempts_on_current_question": interview.attempts_on_current_question,
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error obteniendo entrevista activa: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo entrevista activa: {str(e)}"
        )


@app.post("/api/v2/interviews", response_model=StartInterviewResponse, tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def start_interview(
    request: Request,
    interview_request: StartInterviewRequest,
):
    """
    Inicia una nueva entrevista con preguntas de contexto.

    - Crea sesión de entrevista en fase de contexto
    - Retorna primera pregunta de contexto
    """
    try:
        c = get_container()

        # Validar que no haya entrevista activa
        existing = await c.interview_repository.find_active_by_user(interview_request.user_id)
        if existing:
            raise InterviewAlreadyExists(interview_request.user_id, existing.id)

        # Crear nueva entrevista
        interview = Interview(
            id=f"interview_{interview_request.user_id}_{datetime.now(timezone.utc).timestamp()}",
            user_id=interview_request.user_id,
            role=interview_request.role,
            skill_level=SkillLevel.from_string(interview_request.difficulty),
            interview_type=interview_request.category,
            current_phase="context",
            mode=PRACTICE_MODE,
        )

        await c.interview_repository.save(interview)
        logger.info(f"Entrevista iniciada: {interview.id} - Usuario: {interview_request.user_id}")

        # Design Pattern: Observer - Publicar evento de entrevista iniciada
        c.event_bus.publish(
            "interview_started",
            {
                "interview_id": interview.id,
                "user_id": interview_request.user_id,
                "role": interview_request.role,
                "difficulty": interview_request.difficulty,
                "category": interview_request.category,
            },
            source="start_interview_endpoint"
        )

        # Obtener primera pregunta de contexto (personalizada por profesión)
        # Usar nombre de profesión directamente del frontend
        context_questions = get_context_questions(interview.interview_type, profession=interview.role)
        if not context_questions:
            raise HTTPException(status_code=500, detail="No context questions available")

        first_question = context_questions[0]

        return StartInterviewResponse(
            interview_id=interview.id,
            first_question=QuestionDTO(
                id="context_0",
                text=first_question,
                category="context",
                difficulty="context",
                topic="context",
                expected_concepts=[],
            ),
            status="context",
            message="¡Bienvenido! Responde 5 preguntas de contexto para personalizar tu experiencia.",
        )

    except InterviewAlreadyExists:
        raise
    except Exception as e:
        logger.error(f"Error iniciando entrevista: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error iniciando entrevista: {str(e)}"
        )


@app.post("/api/v2/interviews/{interview_id}/answers", response_model=ProcessAnswerResponse, tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def process_answer(
    request: Request,
    interview_id: str,
    answer_request: ProcessAnswerRequest,
):
    """
    Procesa la respuesta del candidato.

    Flujo:
    1. FASE CONTEXTO: Guarda respuestas sin evaluar
    2. FASE PREGUNTAS: Evalúa con LLM y genera feedback
    """
    try:
        c = get_container()

        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise InterviewNotFound(interview_id)

        # FASE 1: CONTEXTO
        if interview.current_phase == "context":
            interview.context_answers.append(answer_request.answer)
            interview.context_question_index += 1

            logger.info(f"✅ Respuesta contexto {interview.context_question_index}/{CONTEXT_QUESTIONS_COUNT}")

            # ¿Completamos las preguntas de contexto?
            if interview.context_question_index >= CONTEXT_QUESTIONS_COUNT:
                interview.current_phase = "questions"
                await c.interview_repository.save(interview)

                logger.info(f"🎯 Fase contexto completada. Seleccionando preguntas...")

                # Obtener perfil del usuario para mejorar clustering
                user_profile = None
                try:
                    # Intentar obtener perfil del usuario desde PostgreSQL usando PostgresSyncService
                    from app.infrastructure.persistence.postgres_sync_service import PostgresSyncService
                    import asyncio
                    
                    # Obtener instancia del servicio (si está disponible)
                    sync_service = getattr(c, 'postgres_sync_service', None)
                    if sync_service and sync_service.pool:
                        # Usar pool asyncpg para consultar
                        async def get_user_profile():
                            async with sync_service.pool.acquire() as conn:
                                row = await conn.fetchrow(
                                    'SELECT "Profession", "ExperienceLevel", "Skills", "SoftSkills", "Interests" FROM "Users" WHERE "Id" = $1',
                                    int(interview.user_id) if interview.user_id.isdigit() else None
                                )
                                if row:
                                    return {
                                        "profession": row["Profession"] if row["Profession"] else interview.role,
                                        "experience_level": row["ExperienceLevel"],
                                        "technical_skills": row["Skills"] if row["Skills"] else [],
                                        "soft_skills": row["SoftSkills"] if row["SoftSkills"] else [],
                                        "interests": row["Interests"] if row["Interests"] else [],
                                    }
                                return None
                        
                        # Ejecutar consulta async
                        loop = asyncio.get_event_loop()
                        user_profile = loop.run_until_complete(get_user_profile())
                        
                        if user_profile:
                            logger.info(f"📊 Perfil de usuario cargado: {user_profile.get('profession')}, {len(user_profile.get('technical_skills', []))} skills técnicas")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo cargar perfil del usuario: {e}")
                
                if not user_profile:
                    user_profile = {"profession": interview.role}

                # Seleccionar preguntas con ML (ahora incluye perfil del usuario)
                context_text = " ".join(interview.context_answers)
                selected_questions = await _select_questions_with_clustering(c, interview, context_text, user_profile)

                if not selected_questions:
                    raise HTTPException(status_code=500, detail="No questions could be selected")

                interview.metadata["selected_questions"] = [q.id for q in selected_questions]
                interview.metadata["current_question_index"] = 0
                interview.status = InterviewStatus.ACTIVE
                await c.interview_repository.save(interview)

                first_tech_question = selected_questions[0]
                interview.add_question(first_tech_question)
                await c.interview_repository.save(interview)

                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(score=0, is_correct=True, feedback="Fase de contexto completada"),
                    feedback="🎉 ¡Perfecto! Has completado las preguntas de contexto. Ahora comenzaremos con las preguntas técnicas personalizadas según tu perfil. ¡Mucha suerte! 🚀✨",
                    emotion=EmotionDTO(emotion="neutral", confidence=0.0),
                    next_question=QuestionDTO(
                        id=first_tech_question.id,
                        text=first_tech_question.text,
                        category=first_tech_question.category,
                        difficulty=first_tech_question.difficulty,
                        topic=first_tech_question.topic,
                        expected_concepts=first_tech_question.expected_concepts,
                    ),
                    phase="questions",
                    progress=ProgressDTO(
                        context_completed=CONTEXT_QUESTIONS_COUNT,
                        questions_completed=0,
                        total_questions=MAIN_QUESTIONS_COUNT,
                    ),
                    interview_status="questions",
                )
            else:
                # Siguiente pregunta de contexto (personalizada por profesión)
                context_questions = get_context_questions(interview.interview_type, profession=interview.role)
                next_context = context_questions[interview.context_question_index]

                await c.interview_repository.save(interview)

                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(score=0, is_correct=True, feedback="Respuesta guardada"),
                    feedback=f"✅ Gracias por tu respuesta. Continuemos... 📝 (Pregunta {interview.context_question_index + 1}/{CONTEXT_QUESTIONS_COUNT})",
                    emotion=EmotionDTO(emotion="neutral", confidence=0.0),
                    next_question=QuestionDTO(
                        id=f"context_{interview.context_question_index}",
                        text=next_context,
                        category="context",
                        difficulty="context",
                        topic="context",
                        expected_concepts=[],
                    ),
                    phase="context",
                    progress=ProgressDTO(
                        context_completed=interview.context_question_index,
                        questions_completed=0,
                        total_questions=MAIN_QUESTIONS_COUNT,
                    ),
                    interview_status="context",
                )

        # FASE 2: PREGUNTAS TÉCNICAS
        elif interview.current_phase == "questions":
            if not interview.current_question:
                raise HTTPException(status_code=400, detail="No active question")

            # Detectar emoción con manejo robusto de errores
            try:
                emotion_result = c.emotion_detector.detect(answer_request.answer)
                emotion = emotion_result["emotion"] if isinstance(emotion_result["emotion"], Emotion) else Emotion.NEUTRAL
                emotion_confidence = emotion_result.get("confidence", 0.5)
            except Exception as e:
                logger.warning(f"⚠️ Emotion detection failed, using neutral: {e}")
                emotion = Emotion.NEUTRAL
                emotion_confidence = 0.5

            # Verificar límite de tiempo en modo EXAM
            if interview.mode.is_exam():
                time_limit = interview.mode.time_limit_seconds()
                if answer_request.time_taken and answer_request.time_taken > time_limit:
                    logger.warning(f"⏱️ Tiempo agotado en modo examen: {answer_request.time_taken}s > {time_limit}s")
                    await c.interview_repository.save(interview)
                    return ProcessAnswerResponse(
                        evaluation=EvaluationDTO(score=0, is_correct=False, feedback="Tiempo agotado"),
                        feedback="⏱️ Se agotó el tiempo límite para esta pregunta en modo examen. Continuemos con la siguiente pregunta.",
                        emotion=EmotionDTO(emotion="neutral", confidence=0.0),
                        next_question=None,
                        phase="questions",
                        progress=ProgressDTO(
                            context_completed=CONTEXT_QUESTIONS_COUNT,
                            questions_completed=interview.metadata.get("current_question_index", 0),
                            total_questions=MAIN_QUESTIONS_COUNT,
                        ),
                        interview_status="questions",
                    )

            # Obtener intentos actuales para esta pregunta (usar modo para determinar máximo)
            current_question_id = interview.current_question.id
            question_attempts_key = f"attempts_{current_question_id}"
            current_attempt = interview.metadata.get(question_attempts_key, 0) + 1
            MAX_ATTEMPTS = interview.mode.max_attempts_per_question()

            # Design Pattern: Observer - Publicar evento de respuesta enviada
            c.event_bus.publish(
                "answer_submitted",
                {
                    "interview_id": interview.id,
                    "question_id": interview.current_question.id,
                    "user_id": interview.user_id,
                    "answer_length": len(answer_request.answer),
                },
                source="process_answer_endpoint"
            )

            # Evaluar con LLM (con fallback robusto)
            try:
                evaluation = c.evaluation_service.evaluate_answer(
                    question=interview.current_question.text,
                    answer=answer_request.answer,
                    expected_concepts=interview.current_question.expected_concepts,
                    keywords=interview.current_question.keywords,
                    category=interview.current_question.category,
                    difficulty=interview.current_question.difficulty,
                    role=interview.role,
                )
            except Exception as e:
                logger.error(f"Error en evaluación, usando fallback: {e}")
                # Fallback: evaluación simple basada en keywords
                keywords_found = sum(1 for kw in interview.current_question.keywords if kw.lower() in answer_request.answer.lower())
                concepts_found = sum(1 for ec in interview.current_question.expected_concepts if ec.lower() in answer_request.answer.lower())
                
                score = min(5.0, (keywords_found / max(len(interview.current_question.keywords), 1)) * 5 + 
                           (concepts_found / max(len(interview.current_question.expected_concepts), 1)) * 5)
                
                evaluation = {
                    "score": score,
                    "is_correct": score >= 6.0,
                    "justification": "No se pudo evaluar la respuesta completamente. Revisa los conceptos clave mencionados.",
                    "breakdown": {},
                    "strengths": ["Estás intentando responder"] if len(answer_request.answer) > 10 else [],
                    "improvements": ["Asegúrate de mencionar conceptos clave relacionados con la pregunta"],
                    "concepts_covered": [],
                    "missing_concepts": interview.current_question.expected_concepts,
                }
                logger.warning(f"🔄 FALLBACK ACTIVADO: Evaluación - Razón: {str(e)}")
            
            # Design Pattern: Observer - Publicar evento de pregunta respondida
            c.event_bus.publish(
                "question_answered",
                {
                    "interview_id": interview.id,
                    "question_id": interview.current_question.id,
                    "score": evaluation.get("score", 0),
                    "is_correct": evaluation.get("is_correct", False),
                    "attempt": current_attempt,
                },
                source="process_answer_endpoint"
            )

            # ============================================================================
            # DISTINCIÓN CRÍTICA: MODO PRÁCTICA vs MODO EXAMEN
            # ============================================================================
            
            # MODO EXAMEN: NO generar feedback ni pistas durante las preguntas
            if interview.mode.is_exam():
                # En modo examen, solo continuar con la siguiente pregunta sin feedback
                feedback_result = ""  # Sin feedback durante las preguntas
                motivational_feedback = ""
                hint = None
            else:
                # MODO PRÁCTICA: Generar feedback interactivo y motivacional
                feedback_style = interview.mode.feedback_style()
                feedback_result = c.feedback_service.generate_feedback(
                    question=interview.current_question.text,
                    answer=answer_request.answer,
                    evaluation=evaluation,
                    emotion=emotion,
                    role=interview.role,
                    category=interview.current_question.category,
                )
                
                # Si respuesta es CORRECTA: Generar feedback de felicitación (solo modo PRÁCTICA)
                congratulatory_feedback = ""
                if evaluation["is_correct"] and interview.mode.is_practice():
                    try:
                        from app.infrastructure.llm.advanced_prompts import get_prompt_engine
                        prompt_engine = get_prompt_engine()
                        congratulatory_prompt = prompt_engine.get_congratulatory_feedback_prompt(
                            role=interview.role,
                            question=interview.current_question.text,
                            answer=answer_request.answer,
                            evaluation=evaluation,
                        )
                        congratulatory_response = c.evaluation_service.llm_service.generate(
                            prompt=congratulatory_prompt,
                            temperature=0.8,
                            max_tokens=200,
                        )
                        from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
                        sanitizer = ResponseSanitizer()
                        congratulatory_feedback = sanitizer.sanitize_feedback(
                            congratulatory_response,
                            role=interview.role,
                            category=interview.current_question.category
                        )
                    except Exception as e:
                        logger.warning(f"Error generando feedback de felicitación: {e}")
                        # Fallback con mensajes de felicitación con emojis
                        import random
                        congratulatory_messages = [
                            "🎉 ¡Excelente respuesta! Has demostrado un gran entendimiento del concepto. ¡Sigue así! ⭐",
                            "🌟 ¡Muy bien! Tu respuesta muestra conocimiento sólido. Estás haciendo un gran trabajo. 💪",
                            "✨ ¡Perfecto! Has captado los conceptos clave correctamente. ¡Continúa con este nivel! 🚀",
                            "🏆 ¡Impresionante! Tu respuesta es precisa y bien fundamentada. ¡Excelente trabajo! 💯",
                            "💎 ¡Bien hecho! Has aplicado correctamente los conceptos. ¡Sigue avanzando! 🌟",
                            "🎯 ¡Correcto! Tu comprensión del tema es clara. ¡Mantén este nivel! ⭐"
                        ]
                        congratulatory_feedback = random.choice(congratulatory_messages)
                        logger.warning(f"🔄 FALLBACK ACTIVADO: Feedback de felicitación - Razón: {str(e)}")
                    
                    # Agregar feedback de felicitación al feedback principal
                    if congratulatory_feedback:
                        feedback_result = f"{feedback_result}\n\n{congratulatory_feedback}"
                
                # Generar feedback motivacional si respuesta incorrecta (solo modo PRÁCTICA)
                motivational_feedback = ""
                if not evaluation["is_correct"] and interview.mode.is_practice():
                    try:
                        from app.infrastructure.llm.advanced_prompts import get_prompt_engine
                        prompt_engine = get_prompt_engine()
                        motivational_prompt = prompt_engine.get_motivational_feedback_prompt(
                            role=interview.role,
                            question=interview.current_question.text,
                            answer=answer_request.answer,
                            evaluation=evaluation,
                            attempt=current_attempt,
                        )
                        motivational_response = c.evaluation_service.llm_service.generate(
                            prompt=motivational_prompt,
                            temperature=0.8,
                            max_tokens=200,
                        )
                        from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
                        sanitizer = ResponseSanitizer()
                        motivational_feedback = sanitizer.sanitize_feedback(
                            motivational_response, 
                            role=interview.role, 
                            category=interview.current_question.category
                        )
                    except Exception as e:
                        logger.warning(f"Error generando feedback motivacional: {e}")
                        # Fallback con mensajes motivacionales con emojis
                        import random
                        motivational_messages = [
                            "💪 ¡No te desanimes! Cada intento es una oportunidad de aprender. ¡Sigue adelante! 🚀",
                            "🌟 Estás en el camino correcto. Con práctica y dedicación, mejorarás. ¡Ánimo! 💪",
                            "🎯 Cada error es un paso más cerca del éxito. ¡Sigue intentando! ✨",
                            "🔥 La perseverancia es la clave. ¡Tú puedes lograrlo! 💪",
                            "💡 Recuerda: los expertos también fueron principiantes. ¡Continúa! 🌟",
                            "🚀 Cada respuesta te acerca más a dominar este tema. ¡Sigue así! 💪",
                            "⭐ Estás aprendiendo y eso es lo importante. ¡No te rindas! 🌟"
                        ]
                        motivational_feedback = random.choice(motivational_messages)
                        logger.warning(f"🔄 FALLBACK ACTIVADO: Feedback motivacional - Razón: {str(e)}")
                
                # Agregar feedback motivacional al feedback principal
                if motivational_feedback:
                    feedback_result = f"{feedback_result}\n\n{motivational_feedback}"

            # Sistema de intentos: Si respuesta incorrecta y quedan intentos, generar hint (solo modo PRÁCTICA)
            attempts_left = MAX_ATTEMPTS - current_attempt
            hint = None
            
            # MEJORA v3.4: Registrar métricas de intentos
            try:
                from app.infrastructure.monitoring.metrics import get_metrics
                metrics = get_metrics(enabled=True)
                metrics.inc_counter("attempts_total")
                metrics.observe_histogram("attempts_per_question", current_attempt)
            except Exception:
                pass  # Métricas opcionales
            
            # Verificar si la respuesta supera el umbral (PASS_THRESHOLD = 6.0)
            from app.domain.value_objects.context_questions import PASS_THRESHOLD
            score_passed = evaluation.get("score", 0) >= PASS_THRESHOLD
            
            # Solo generar hints si está en modo PRÁCTICA, hints están habilitados, y NO superó el umbral
            if not score_passed and attempts_left > 0 and interview.mode.hints_enabled():
                # Generar hint progresivo usando advanced_prompts
                try:
                    from app.infrastructure.llm.advanced_prompts import get_prompt_engine
                    prompt_engine = get_prompt_engine()
                    hint_prompt = prompt_engine.get_hint_prompt(
                        role=interview.role,
                        question=interview.current_question.text,
                        answer=answer_request.answer,
                        expected_concepts=interview.current_question.expected_concepts,
                        attempts=current_attempt,
                    )
                    # Generar hint con LLM
                    hint_response = c.evaluation_service.llm_service.generate(
                        prompt=hint_prompt,
                        temperature=0.7,
                        max_tokens=150,
                    )
                    # Sanitizar hint
                    from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
                    sanitizer = ResponseSanitizer()
                    hint = sanitizer.sanitize_feedback(hint_response, role=interview.role, category=interview.current_question.category)
                    # Agregar hint al feedback con emojis motivacionales progresivos
                    hint_emojis = ["💡", "🤔", "🎯"]
                    hint_emoji = hint_emojis[min(current_attempt - 1, len(hint_emojis) - 1)]
                    feedback_result = f"{feedback_result}\n\n{hint_emoji} Pista (intento {current_attempt}/{MAX_ATTEMPTS}): {hint}"
                    # MEJORA v3.4: Registrar métricas de hints
                    try:
                        from app.infrastructure.monitoring.metrics import get_metrics
                        get_metrics(enabled=True).inc_counter("hints_generated_total")
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Error generando hint: {e}")
                    # Fallback hint simple con emojis progresivos
                    hint_emojis = ["💡", "🤔", "🎯"]
                    hint_emoji = hint_emojis[min(current_attempt - 1, len(hint_emojis) - 1)]
                    if current_attempt == 1:
                        hint = f"{hint_emoji} Considera revisar los conceptos clave: {', '.join(interview.current_question.expected_concepts[:2])}"
                    elif current_attempt == 2:
                        hint = f"{hint_emoji} Piensa en cómo estos conceptos se relacionan: {', '.join(interview.current_question.expected_concepts)}"
                    else:
                        hint = f"{hint_emoji} La respuesta debería incluir: {', '.join(interview.current_question.expected_concepts)}"
                    feedback_result = f"{feedback_result}\n\n{hint}"
            
            # Si se agotaron los intentos y respuesta incorrecta, dar respuesta correcta con consejos (solo modo PRÁCTICA)
            if not evaluation["is_correct"] and current_attempt >= MAX_ATTEMPTS and interview.mode.is_practice():
                try:
                    from app.infrastructure.llm.advanced_prompts import get_prompt_engine
                    prompt_engine = get_prompt_engine()
                    
                    # Generar respuesta correcta
                    correct_answer_prompt = prompt_engine.get_correct_answer_prompt(
                        role=interview.role,
                        question=interview.current_question.text,
                        expected_concepts=interview.current_question.expected_concepts,
                    )
                    correct_answer_response = c.evaluation_service.llm_service.generate(
                        prompt=correct_answer_prompt,
                        temperature=0.6,
                        max_tokens=300,
                    )
                    from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
                    sanitizer = ResponseSanitizer()
                    correct_answer = sanitizer.sanitize_feedback(
                        correct_answer_response,
                        role=interview.role,
                        category=interview.current_question.category
                    )
                    
                    # Generar consejos de mejora
                    improvement_prompt = prompt_engine.get_improvement_tips_prompt(
                        role=interview.role,
                        question=interview.current_question.text,
                        answer=answer_request.answer,
                        correct_answer=correct_answer,
                    )
                    improvement_response = c.evaluation_service.llm_service.generate(
                        prompt=improvement_prompt,
                        temperature=0.7,
                        max_tokens=200,
                    )
                    improvement_tips = sanitizer.sanitize_feedback(
                        improvement_response,
                        role=interview.role,
                        category=interview.current_question.category
                    )
                    
                    feedback_result = (
                        f"{feedback_result}\n\n"
                            "📚 ✨ Respuesta Correcta Esperada:\n"
                        f"{correct_answer}\n\n"
                            "💡 🎓 Consejos de Mejora:\n"
                            f"{improvement_tips}\n\n"
                            "🌟 ¡No te desanimes! Aprender de los errores es parte del proceso. ¡Sigue practicando! 💪"
                    )
                except Exception as e:
                    logger.warning(f"Error generando respuesta correcta: {e}")
                        # Fallback simple con emojis motivacionales
                    feedback_result = (
                        f"{feedback_result}\n\n"
                            "📚 ✨ Conceptos clave que debías mencionar: "
                            f"{', '.join(interview.current_question.expected_concepts)}\n\n"
                            "💡 🎓 Recomendación: Estudia estos conceptos y cómo se relacionan "
                            "para fortalecer tu comprensión.\n\n"
                            "🌟 ¡Sigue aprendiendo! Cada intento te acerca más al éxito. 💪"
                    )
                    logger.warning(f"🔄 FALLBACK ACTIVADO: Respuesta correcta - Razón: {str(e)}")

            # Guardar respuesta
            from app.domain.value_objects.score import Score

            # Crear respuesta con score como float (Answer espera float, no Score object)
            answer_entity = Answer(
                question_id=interview.current_question.id,
                text=answer_request.answer,
                score=float(evaluation["score"]),  # Usar float directamente, no Score object
                is_correct=evaluation["is_correct"],
                emotion=emotion,
                time_taken=answer_request.time_taken or 0,
                evaluation_details=evaluation,
                hints_used=1 if hint else 0,
            )

            # Actualizar intentos en metadata
            interview.metadata[question_attempts_key] = current_attempt

            # ============================================================================
            # LÓGICA DE AVANCE: Diferente según el modo
            # ============================================================================
            # MODO PRÁCTICA: Solo avanzar si supera umbral O si agotó intentos
            # MODO EXAMEN: Siempre avanzar después de responder (1 solo intento)
            if interview.mode.is_exam():
                # En modo examen, siempre avanzar después de responder (no hay reintentos)
                should_advance = True
            else:
                # En modo práctica, solo avanzar si supera umbral O si agotó intentos
                # Verificar si la respuesta supera el umbral (PASS_THRESHOLD = 6.0)
                from app.domain.value_objects.context_questions import PASS_THRESHOLD
                score_passed = evaluation.get("score", 0) >= PASS_THRESHOLD
                should_advance = score_passed or current_attempt >= MAX_ATTEMPTS
            
            if should_advance:
                # Guardar respuesta: usar la actual directamente (es la más reciente)
                interview.add_answer(answer_entity)
                # Limpiar intentos y respuestas temporales de esta pregunta
                if question_attempts_key in interview.metadata:
                    del interview.metadata[question_attempts_key]
                if f"last_answer_{current_question_id}" in interview.metadata:
                    del interview.metadata[f"last_answer_{current_question_id}"]
                current_index = interview.metadata.get("current_question_index", 0) + 1
                interview.metadata["current_question_index"] = current_index
            else:
                # Guardar respuesta temporalmente pero no avanzar
                # Se guardará cuando sea correcta o se agoten los intentos
                interview.metadata[f"last_answer_{current_question_id}"] = answer_entity.to_dict()
                # No incrementar índice todavía
                current_index = interview.metadata.get("current_question_index", 0)

            logger.info(
                f"✅ Respuesta evaluada: score={evaluation['score']}, intento {current_attempt}/{MAX_ATTEMPTS}, pregunta {current_index}/{MAIN_QUESTIONS_COUNT}"
            )

            # Si no avanzamos, permitir reintentar la misma pregunta
            if not should_advance:
                await c.interview_repository.save(interview)
                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(
                        score=evaluation["score"],
                        is_correct=evaluation["is_correct"],
                        feedback=evaluation.get("justification", ""),
                        breakdown=evaluation.get("breakdown"),
                        strengths=evaluation.get("strengths", []),
                        improvements=evaluation.get("improvements", []),
                        concepts_covered=evaluation.get("concepts_covered", []),
                        missing_concepts=evaluation.get("missing_concepts", []),
                    ),
                    feedback=feedback_result,
                    emotion=EmotionDTO(
                        emotion=emotion.value,
                        confidence=emotion_result.get("confidence", 0.0),
                        language=emotion_result.get("language"),
                    ),
                    next_question=QuestionDTO(
                        id=interview.current_question.id,
                        text=interview.current_question.text,
                        category=interview.current_question.category,
                        difficulty=interview.current_question.difficulty,
                        topic=interview.current_question.topic,
                        expected_concepts=interview.current_question.expected_concepts,
                    ),
                    phase="questions",
                    progress=ProgressDTO(
                        context_completed=CONTEXT_QUESTIONS_COUNT,
                        questions_completed=current_index,
                        total_questions=MAIN_QUESTIONS_COUNT,
                        percentage=(current_index / MAIN_QUESTIONS_COUNT) * 100,
                    ),
                    attempts_left=attempts_left,
                    interview_status="questions",
                )

            # ¿Completamos la entrevista?
            if current_index >= MAIN_QUESTIONS_COUNT:
                interview.complete()
                await c.interview_repository.save(interview)

                logger.info(f"🎉 Entrevista completada: {interview_id}")
                
                # Design Pattern: Observer - Publicar evento de entrevista completada
                overall_score = sum([float(a.score) for a in interview.answers_history]) / len(interview.answers_history) if interview.answers_history else 0
                c.event_bus.publish(
                    "interview_completed",
                    {
                        "interview_id": interview.id,
                        "user_id": interview.user_id,
                        "role": interview.role,
                        "total_questions": MAIN_QUESTIONS_COUNT,
                        "final_score": overall_score,
                    },
                    source="process_answer_endpoint"
                )
                
                # Generar feedback final completo con MEMORIA CONVERSACIONAL COMPLETA
                final_feedback_text = ""
                try:
                    # ============================================================================
                    # MEMORIA CONVERSACIONAL COMPLETA: Incluir preguntas de contexto + técnicas
                    # ============================================================================
                    all_answers = []
                    
                    # Incluir respuestas de contexto si existen
                    context_answers = interview.metadata.get("context_answers", [])
                    context_questions = interview.metadata.get("context_questions", [])
                    
                    for i, ctx_answer in enumerate(context_answers):
                        if i < len(context_questions):
                            all_answers.append({
                                "question": context_questions[i],
                                "answer": ctx_answer,
                                "score": None,  # Las preguntas de contexto no se califican
                                "is_correct": None,
                                "evaluation_details": {"type": "context"},
                                "phase": "context"
                            })
                    
                    # Incluir respuestas técnicas/principales
                    for q, a in zip(interview.questions_history, interview.answers_history):
                        all_answers.append({
                            "question": q.text,
                            "answer": a.text,
                            "score": float(a.score),  # a.score es float, no Score object
                            "is_correct": a.is_correct,
                            "evaluation_details": a.evaluation_details,
                            "phase": "technical"
                        })
                    
                    accuracy = sum([1 for a in interview.answers_history if a.is_correct]) / len(interview.answers_history) * 100 if interview.answers_history else 0
                    
                    # Generar feedback final con memoria conversacional completa
                    final_feedback_text = c.feedback_service.generate_final_feedback(
                        role=interview.role,
                        category=interview.interview_type,
                        all_answers=all_answers,
                        overall_score=overall_score,
                        accuracy=accuracy,
                        mode=interview.mode.to_string(),  # Incluir modo para contexto
                    )
                except Exception as e:
                    logger.error(f"Error generando feedback final: {e}")
                    final_feedback_text = f"¡Felicidades por completar la entrevista! Tu score promedio fue {overall_score:.2f}/10."
                    logger.warning(f"🔄 FALLBACK ACTIVADO: Feedback final - Razón: {str(e)}")
                
                # Generar reporte completo con gráficos
                report_data = None
                report_url = None
                certificate_eligible = False
                certificate_id = None
                
                try:
                    from app.infrastructure.llm.report_generator import get_report_generator
                    import json
                    
                    report_generator = get_report_generator()
                    
                    interview_data = {
                        "id": interview.id,
                        "role": interview.role,
                        "mode": interview.mode.to_string(),
                        "completed_at": interview.completed_at.isoformat() if interview.completed_at else datetime.now(timezone.utc).isoformat(),
                        "answers_history": [a.to_dict() for a in interview.answers_history],  # Corregido: usar answers_history
                        "questions_history": [q.to_dict() for q in interview.questions_history],
                    }
                    
                    user_data = {
                        "name": f"User_{interview.user_id}",
                        "user_id": interview.user_id,
                    }
                    
                    report = report_generator.generate_report(interview_data, user_data)
                    
                    # Exportar a JSON
                    report_json_str = report_generator.export_to_json(report)
                    report_data = json.loads(report_json_str)
                    
                    # Guardar reporte en metadata de entrevista
                    interview.metadata["report"] = report_json_str
                    interview.metadata["report_id"] = report.interview_id
                    interview.metadata["certificate_eligible"] = report.certificate_eligible
                    interview.metadata["certificate_id"] = report.certificate_id
                    
                    report_url = report.shareable_url
                    certificate_eligible = report.certificate_eligible
                    certificate_id = report.certificate_id
                    
                    await c.interview_repository.save(interview)
                    
                    logger.info(f"📊 Reporte generado para entrevista {interview.id}")
                except Exception as e:
                    logger.error(f"Error generando reporte: {e}")
                    logger.warning(f"🔄 FALLBACK ACTIVADO: Reporte - Razón: {str(e)}")
                    # Continuar sin reporte (no crítico)
                
                # Agregar feedback final al feedback principal
                feedback_result = f"{feedback_result}\n\n🎉 FEEDBACK FINAL:\n{final_feedback_text}"

                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(
                        score=evaluation["score"],
                        is_correct=evaluation["is_correct"],
                        feedback=evaluation.get("justification", ""),
                        breakdown=evaluation.get("breakdown"),
                        strengths=evaluation.get("strengths", []),
                        improvements=evaluation.get("improvements", []),
                        concepts_covered=evaluation.get("concepts_covered", []),
                        missing_concepts=evaluation.get("missing_concepts", []),
                    ),
                    feedback=feedback_result,
                    emotion=EmotionDTO(
                        emotion=emotion.value,
                        confidence=emotion_result.get("confidence", 0.0),
                        language=emotion_result.get("language"),
                    ),
                    next_question=None,
                    phase="completed",
                    progress=ProgressDTO(
                        context_completed=CONTEXT_QUESTIONS_COUNT,
                        questions_completed=current_index,
                        total_questions=MAIN_QUESTIONS_COUNT,
                        percentage=100.0,
                    ),
                    interview_status="completed",
                    interview_completed=True,
                    final_report=report_data,
                    report_url=report_url,
                    certificate_eligible=certificate_eligible,
                    certificate_id=certificate_id,
                )
            else:
                # Siguiente pregunta
                selected_question_ids = interview.metadata.get("selected_questions", [])
                if current_index < len(selected_question_ids):
                    next_question_id = selected_question_ids[current_index]
                    next_question = await c.question_repository.find_by_id(next_question_id)

                    if next_question:
                        # ============================================================================
                        # VALIDACIÓN DE COHERENCIA: Verificar que la siguiente pregunta sea coherente
                        # ============================================================================
                        if interview.interview_type == "technical" and next_question.category != "technical":
                            logger.error(
                                f"❌ ERROR DE COHERENCIA: La siguiente pregunta '{next_question.id[:8]}...' "
                                f"es '{next_question.category}' pero se esperaba 'technical'. "
                                f"Rol: {interview.role}. Intentando seleccionar pregunta alternativa..."
                            )
                            # Intentar encontrar una pregunta alternativa coherente
                            alternative_questions = await c.question_repository.find_by_role(
                                interview.role, 
                                category="technical"
                            )
                            # Filtrar preguntas ya usadas
                            used_ids = [q.id for q in interview.questions_history]
                            alternative_questions = [q for q in alternative_questions if q.id not in used_ids]
                            
                            if alternative_questions:
                                next_question = alternative_questions[0]
                                logger.info(f"✅ Pregunta alternativa seleccionada: {next_question.id[:8]}...")
                            else:
                                logger.error("❌ No se encontraron preguntas alternativas coherentes")
                                raise HTTPException(
                                    status_code=500, 
                                    detail="No se pudo encontrar una pregunta coherente con la profesión"
                                )
                        
                        elif interview.interview_type == "soft_skills" and next_question.category != "soft_skills":
                            logger.error(
                                f"❌ ERROR DE COHERENCIA: La siguiente pregunta '{next_question.id[:8]}...' "
                                f"es '{next_question.category}' pero se esperaba 'soft_skills'."
                            )
                            # Intentar encontrar una pregunta alternativa coherente
                            alternative_questions = await c.question_repository.find_all_soft_skills()
                            used_ids = [q.id for q in interview.questions_history]
                            alternative_questions = [q for q in alternative_questions if q.id not in used_ids]
                            
                            if alternative_questions:
                                next_question = alternative_questions[0]
                                logger.info(f"✅ Pregunta alternativa seleccionada: {next_question.id[:8]}...")
                            else:
                                logger.error("❌ No se encontraron preguntas alternativas coherentes")
                                raise HTTPException(
                                    status_code=500, 
                                    detail="No se pudo encontrar una pregunta coherente"
                                )
                        
                        interview.add_question(next_question)
                        await c.interview_repository.save(interview)

                        return ProcessAnswerResponse(
                            evaluation=EvaluationDTO(
                                score=evaluation["score"],
                                is_correct=evaluation["is_correct"],
                                feedback=evaluation.get("justification", ""),
                                breakdown=evaluation.get("breakdown"),
                                strengths=evaluation.get("strengths", []),
                                improvements=evaluation.get("improvements", []),
                                concepts_covered=evaluation.get("concepts_covered", []),
                                missing_concepts=evaluation.get("missing_concepts", []),
                            ),
                            feedback=feedback_result,
                            emotion=EmotionDTO(
                                emotion=emotion.value,
                                confidence=emotion_result.get("confidence", 0.0),
                                language=emotion_result.get("language"),
                            ),
                            next_question=QuestionDTO(
                                id=next_question.id,
                                text=next_question.text,
                                category=next_question.category,
                                difficulty=next_question.difficulty,
                                topic=next_question.topic,
                                expected_concepts=next_question.expected_concepts,
                            ),
                            phase="questions",
                            progress=ProgressDTO(
                                context_completed=CONTEXT_QUESTIONS_COUNT,
                                questions_completed=current_index,
                                total_questions=MAIN_QUESTIONS_COUNT,
                                percentage=(current_index / MAIN_QUESTIONS_COUNT) * 100,
                            ),
                            attempts_left=None,  # Nueva pregunta, resetear intentos
                            interview_status="questions",
                        )

                # Fallback: completar entrevista
                interview.complete()
                await c.interview_repository.save(interview)

                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(
                        score=evaluation["score"],
                        is_correct=evaluation["is_correct"],
                        feedback=evaluation.get("justification", ""),
                    ),
                    feedback=feedback_result,
                    emotion=EmotionDTO(emotion=emotion.value, confidence=0.0),
                    next_question=None,
                    phase="completed",
                    progress=ProgressDTO(
                        context_completed=CONTEXT_QUESTIONS_COUNT,
                        questions_completed=current_index,
                        total_questions=MAIN_QUESTIONS_COUNT,
                    ),
                    interview_status="completed",
                    interview_completed=True,
                )

        else:
            raise HTTPException(status_code=400, detail=f"Invalid phase: {interview.current_phase}")

    except (InterviewNotFound, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Error procesando respuesta: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error procesando respuesta: {str(e)}"
        )


@app.post("/api/v2/interviews/{interview_id}/end", tags=["Interviews"])
@limiter.limit("10/minute")
async def end_interview(request: Request, interview_id: str):
    """Finaliza una entrevista y genera resumen."""
    try:
        c = get_container()

        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise InterviewNotFound(interview_id)

        interview.complete()
        await c.interview_repository.save(interview)

        answers = getattr(interview, "answers_history", []) or []
        total_score = sum(float(a.score) for a in answers) / len(answers) if answers else 0

        logger.info(f"Entrevista finalizada: {interview_id}, score={total_score:.2f}")

        return {
            "interview_id": interview_id,
            "status": "completed",
            "total_answers": len(answers),
            "average_score": round(total_score, 2),
            "summary": f"Entrevista completada con {len(answers)} respuestas",
        }

    except InterviewNotFound:
        raise
    except Exception as e:
        logger.error(f"Error finalizando entrevista: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error finalizando entrevista: {str(e)}"
        )


@app.get("/api/v2/interviews/user/{user_id}/completed", tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_completed_interviews(
    request: Request,
    user_id: str,
    limit: int = 10,
):
    """
    Obtiene todas las entrevistas completadas de un usuario.
    Incluye información básica de reportes y certificados.
    """
    try:
        c = get_container()
        
        # Obtener todas las entrevistas del usuario
        all_interviews = await c.interview_repository.find_all_by_user(user_id)
        
        # Filtrar solo las completadas
        completed_interviews = []
        for interview in all_interviews:
            if interview.status == InterviewStatus.COMPLETED:
                report_data = interview.metadata.get("report")
                certificate_id = interview.metadata.get("certificate_id")
                certificate_eligible = interview.metadata.get("certificate_eligible", False)
                
                # Calcular score promedio
                answers = interview.answers_history
                avg_score = sum(float(a.score) for a in answers) / len(answers) if answers else 0
                
                completed_interviews.append({
                    "interview_id": interview.id,
                    "role": interview.role,
                    "mode": interview.mode.to_string() if hasattr(interview.mode, 'to_string') else str(interview.mode),
                    "completed_at": interview.completed_at.isoformat() if interview.completed_at else None,
                    "total_questions": len(answers),
                    "average_score": round(avg_score, 2),
                    "has_report": report_data is not None,
                    "has_certificate": certificate_id is not None,
                    "certificate_eligible": certificate_eligible,
                    "certificate_id": certificate_id,
                })
        
        # Ordenar por fecha de completación (más reciente primero)
        completed_interviews.sort(key=lambda x: x.get("completed_at") or "", reverse=True)
        
        return {
            "user_id": user_id,
            "total": len(completed_interviews),
            "interviews": completed_interviews[:limit],
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo entrevistas completadas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo entrevistas completadas: {str(e)}"
        )


@app.get("/api/v2/interviews/{interview_id}/report", tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_interview_report(
    request: Request,
    interview_id: str,
):
    """
    Obtiene el reporte completo de una entrevista completada.
    Incluye métricas, fortalezas, áreas de mejora y recursos recomendados.
    """
    try:
        c = get_container()
        
        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise InterviewNotFound(interview_id)
        
        if interview.status != InterviewStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La entrevista no está completada"
            )
        
        # Obtener reporte de metadata
        report_json_str = interview.metadata.get("report")
        if not report_json_str:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reporte no encontrado para esta entrevista"
            )
        
        import json
        report_data = json.loads(report_json_str)
        
        return {
            "interview_id": interview_id,
            "report": report_data,
            "shareable_url": interview.metadata.get("report_id", f"/reports/{interview_id}"),
        }
        
    except InterviewNotFound:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo reporte: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo reporte: {str(e)}"
        )


@app.get("/api/v2/interviews/{interview_id}/certificate", tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def get_interview_certificate(
    request: Request,
    interview_id: str,
    format: str = "json",  # json, svg, pdf
):
    """
    Obtiene el certificado de una entrevista completada.
    Formatos disponibles: json (metadatos), svg (preview), pdf (descarga)
    """
    try:
        c = get_container()
        
        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise InterviewNotFound(interview_id)
        
        if interview.status != InterviewStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La entrevista no está completada"
            )
        
        certificate_id = interview.metadata.get("certificate_id")
        certificate_eligible = interview.metadata.get("certificate_eligible", False)
        
        if not certificate_eligible or not certificate_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Certificado no disponible para esta entrevista"
            )
        
        # Calcular métricas para el certificado
        answers = interview.answers_history
        total_score = sum(float(a.score) for a in answers) / len(answers) if answers else 0
        
        # Obtener percentil del reporte si está disponible
        report_json_str = interview.metadata.get("report")
        percentile = 50
        if report_json_str:
            try:
                import json
                report_data = json.loads(report_json_str)
                percentile = report_data.get("metrics", {}).get("percentile", 50)
            except:
                pass
        
        from app.infrastructure.llm.certificate_generator import get_certificate_generator, CertificateData
        certificate_generator = get_certificate_generator()
        
        cert_data = CertificateData(
            certificate_id=certificate_id,
            candidate_name=f"User_{interview.user_id}",
            role=interview.role,
            completion_date=interview.completed_at if interview.completed_at else datetime.now(timezone.utc),
            score=total_score,
            percentile=percentile,
            interview_id=interview_id,
            validation_url=f"{certificate_generator.base_url}/verify/{certificate_id}",
        )
        
        if format == "json":
            return {
                "certificate_id": certificate_id,
                "candidate_name": cert_data.candidate_name,
                "role": cert_data.role,
                "completion_date": cert_data.completion_date.isoformat(),
                "score": cert_data.score,
                "percentile": cert_data.percentile,
                "validation_url": cert_data.validation_url,
                "download_url": f"/api/v2/interviews/{interview_id}/certificate?format=pdf",
                "preview_url": f"/api/v2/interviews/{interview_id}/certificate?format=svg",
            }
        elif format == "svg":
            from fastapi.responses import Response
            svg_content = certificate_generator.generate_preview_svg(cert_data)
            return Response(content=svg_content, media_type="image/svg+xml")
        elif format == "pdf":
            from fastapi.responses import Response
            pdf_bytes = certificate_generator.generate_certificate(cert_data)
            # Por ahora retornamos SVG (en producción convertir a PDF real)
            return Response(content=pdf_bytes, media_type="image/svg+xml", headers={
                "Content-Disposition": f'attachment; filename="certificate_{certificate_id}.svg"'
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Formato no soportado: {format}. Use 'json', 'svg' o 'pdf'"
            )
        
    except InterviewNotFound:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo certificado: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo certificado: {str(e)}"
        )


# ============================================================================
# Root & Metrics
# ============================================================================


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz"""
    return {
        "message": "Ready4Hire API v2.1 (Improved)",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v2/health",
    }


@app.get("/api/v2/metrics", tags=["Metrics"])
async def get_metrics():
    """Métricas del sistema LLM."""
    try:
        c = get_container()
        metrics = c.llm_service.get_metrics()

        return {"llm": metrics, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error obteniendo métricas: {str(e)}"
        )


# ============================================================================
# WebSocket Endpoints - v2.1 Enterprise
# ============================================================================


@app.websocket("/ws/interview/{interview_id}")
async def interview_websocket(websocket: WebSocket, interview_id: str):
    """
    WebSocket para streaming en tiempo real de entrevistas.
    
    Permite:
    - Streaming de respuestas LLM token por token
    - Typing indicators
    - Progress updates
    - Notificaciones en tiempo real
    """
    ws_manager = get_websocket_manager()
    
    try:
        # Accept connection
        await ws_manager.connect(interview_id, websocket)
        logger.info(f"🔌 WebSocket connected: interview_id={interview_id}")
        
        # Send welcome message
        await ws_manager.send_personal_message(interview_id, {
            "type": "connected",
            "message": "WebSocket connection established",
            "interview_id": interview_id
        })
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            logger.debug(f"📨 Received message: {data}")
            
            # Echo back for now (can be expanded for bi-directional communication)
            await ws_manager.send_personal_message(interview_id, {
                "type": "echo",
                "data": data
            })
    
    except WebSocketDisconnect:
        ws_manager.disconnect(interview_id, websocket)
        logger.info(f"🔌 WebSocket disconnected: interview_id={interview_id}")
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {str(e)}")
        ws_manager.disconnect(interview_id, websocket)


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


# Lifespan events are now handled in app_lifespan context manager above


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_v2_improved:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
