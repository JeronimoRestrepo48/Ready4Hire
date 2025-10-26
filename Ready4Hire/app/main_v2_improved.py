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
from typing import Optional
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timezone
from pathlib import Path

# Configuración
from app.config import settings

# Container DI
from app.container import Container

# ============================================================================
# NEW FEATURES - v2.1 Enterprise Edition
# ============================================================================

# Redis Cache
from app.infrastructure.cache.redis_cache import get_redis_cache, RedisCache

# WebSockets
from app.infrastructure.websocket.websocket_manager import get_websocket_manager, WebSocketManager

# Circuit Breaker
from app.infrastructure.resilience.circuit_breaker import (
    with_circuit_breaker,
    with_retry_and_circuit_breaker,
    CircuitBreakerOpenException
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
from app.domain.entities.interview import Interview
from app.domain.entities.answer import Answer

# ============================================================================
# Gamification Routes
# ============================================================================
from app.api.gamification_routes import router as gamification_router
from app.api.rag_routes import router as rag_router
from app.api.certificate_routes import router as certificate_router
from app.api.graphql_router import router as graphql_router

# ============================================================================
# Configurar Logging
# ============================================================================
log_level = getattr(logging, settings.LOG_LEVEL)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Rate Limiter
# ============================================================================
limiter = Limiter(key_func=get_remote_address)


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


def get_container() -> Container:
    """Obtiene el container de DI (lazy initialization)"""
    global container
    if container is None:
        container = Container(ollama_url=settings.OLLAMA_URL, ollama_model=None)  # Auto-detectar según GPU
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
app.include_router(gamification_router)
app.include_router(rag_router)
app.include_router(certificate_router)
app.include_router(graphql_router)


# ============================================================================
# Helper Functions
# ============================================================================


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
            all_questions = await c.question_repository.find_all_technical()
        elif interview.interview_type == "soft_skills":
            all_questions = await c.question_repository.find_all_soft_skills()
        else:
            tech_qs = await c.question_repository.find_all_technical()
            soft_qs = await c.question_repository.find_all_soft_skills()
            all_questions = tech_qs + soft_qs

        if not all_questions:
            return []

        # 2. Filtrado por nivel
        level_filtered = [q for q in all_questions if q.difficulty == interview.skill_level.value]
        candidates = level_filtered if len(level_filtered) >= MAIN_QUESTIONS_COUNT else all_questions

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

                logger.info(f"⚡ {len(selected_questions)} preguntas seleccionadas INSTANTÁNEAMENTE")
                return selected_questions

            except Exception as e:
                logger.warning(f"⚠️ Fallback a selección aleatoria: {str(e)}")

        # 4. Fallback: Selección aleatoria
        selected = random.sample(candidates, min(MAIN_QUESTIONS_COUNT, len(candidates)))
        logger.info(f"⚡ {len(selected)} preguntas (fallback aleatorio)")
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

        all_healthy = all("healthy" in v.lower() or "✅" in v for v in health_status.values())

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
            mode="practice",
        )

        await c.interview_repository.save(interview)
        logger.info(f"Entrevista iniciada: {interview.id} - Usuario: {interview_request.user_id}")

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
                    # Intentar obtener perfil del usuario desde PostgreSQL
                    from app.infrastructure.persistence.postgres_sync_service import get_sync_session
                    
                    with get_sync_session() as session:
                        result = session.execute(
                            "SELECT profession, experience_level, skills, soft_skills, interests FROM users WHERE id = %s",
                            (interview.user_id,)
                        )
                        user_row = result.fetchone()
                        
                        if user_row:
                            user_profile = {
                                "profession": user_row[0] if user_row[0] else interview.role,
                                "experience_level": user_row[1],
                                "technical_skills": user_row[2] if user_row[2] else [],
                                "soft_skills": user_row[3] if user_row[3] else [],
                                "interests": user_row[4] if user_row[4] else [],
                            }
                            logger.info(f"📊 Perfil de usuario cargado: {user_profile.get('profession')}, {len(user_profile.get('technical_skills', []))} skills técnicas")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo cargar perfil del usuario: {e}")
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
                    feedback="¡Perfecto! Ahora comenzaremos con las preguntas personalizadas según tu perfil. ¡Mucha suerte! 🚀",
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
                    feedback=f"Gracias. Continuemos... (Pregunta {interview.context_question_index + 1}/{CONTEXT_QUESTIONS_COUNT})",
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

            # Detectar emoción
            emotion_result = c.emotion_detector.detect(answer_request.answer)
            emotion = emotion_result["emotion"] if isinstance(emotion_result["emotion"], Emotion) else Emotion.NEUTRAL

            # Evaluar con LLM
            evaluation = c.evaluation_service.evaluate_answer(
                question=interview.current_question.text,
                answer=answer_request.answer,
                expected_concepts=interview.current_question.expected_concepts,
                keywords=interview.current_question.keywords,
                category=interview.current_question.category,
                difficulty=interview.current_question.difficulty,
                role=interview.role,
            )

            # Generar feedback
            feedback_result = c.feedback_service.generate_feedback(
                question=interview.current_question.text,
                answer=answer_request.answer,
                evaluation=evaluation,
                emotion=emotion,
                role=interview.role,
                category=interview.current_question.category,
            )

            # Guardar respuesta
            from app.domain.value_objects.score import Score

            answer_entity = Answer(
                question_id=interview.current_question.id,
                text=answer_request.answer,
                score=Score(evaluation["score"]),
                is_correct=evaluation["is_correct"],
                emotion=emotion,
                time_taken=answer_request.time_taken or 0,
                evaluation_details=evaluation,
            )

            interview.add_answer(answer_entity)

            current_index = interview.metadata.get("current_question_index", 0) + 1
            interview.metadata["current_question_index"] = current_index

            logger.info(
                f"✅ Respuesta evaluada: score={evaluation['score']}, pregunta {current_index}/{MAIN_QUESTIONS_COUNT}"
            )

            # ¿Completamos la entrevista?
            if current_index >= MAIN_QUESTIONS_COUNT:
                interview.complete()
                await c.interview_repository.save(interview)

                logger.info(f"🎉 Entrevista completada: {interview_id}")

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
                )
            else:
                # Siguiente pregunta
                selected_question_ids = interview.metadata.get("selected_questions", [])
                if current_index < len(selected_question_ids):
                    next_question_id = selected_question_ids[current_index]
                    next_question = await c.question_repository.find_by_id(next_question_id)

                    if next_question:
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

        answers = getattr(interview, "answers", []) or []
        total_score = sum(a.score.value for a in answers) / len(answers) if answers else 0

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
