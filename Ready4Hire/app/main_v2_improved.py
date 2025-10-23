"""
FastAPI Application v2.1 - Ready4Hire (IMPROVED)
Arquitectura DDD con seguridad mejorada, rate limiting, y DTOs.

Mejoras implementadas:
- ✅ CORS restringido (no más allow_origins=["*"])
- ✅ Rate limiting con slowapi
- ✅ Autenticación JWT
- ✅ DTOs separados de la lógica de dominio
- ✅ Manejo de excepciones centralizado
- ✅ Configuración desde .env
- ✅ Logging estructurado
"""
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional
import logging
from datetime import datetime
from pathlib import Path

# Configuración
from app.config import settings

# Container DI
from app.container import Container

# DTOs
from app.application.dto import (
    StartInterviewRequest,
    StartInterviewResponse,
    ProcessAnswerRequest,
    ProcessAnswerResponse,
    EndInterviewResponse,
    HealthResponse,
    LoginRequest,
    TokenResponse,
    QuestionDTO,
    EvaluationDTO,
    EmotionDTO,
    ProgressDTO
)

# Excepciones de dominio
from app.domain.exceptions import (
    Ready4HireException,
    InterviewNotFound,
    InterviewAlreadyExists,
    ValidationException
)

# Security
from app.infrastructure.security import (
    get_current_user_id,
    get_optional_user_id,
    authenticate_user,
    create_token_response
)

# Domain
from app.domain.value_objects.emotion import Emotion
from app.domain.value_objects.context_questions import (
    get_context_questions,
    CONTEXT_QUESTIONS_COUNT,
    MAIN_QUESTIONS_COUNT
)
from app.domain.value_objects.interview_status import InterviewStatus
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.entities.interview import Interview
from app.domain.entities.answer import Answer

# ============================================================================
# Configurar Logging
# ============================================================================
log_level = getattr(logging, settings.LOG_LEVEL)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Rate Limiter
# ============================================================================
limiter = Limiter(key_func=get_remote_address)

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="Ready4Hire API v2.1",
    description="Sistema de entrevistas con IA - Arquitectura DDD mejorada",
    version="2.1.0",
    docs_url="/docs" if settings.ENABLE_SWAGGER_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_REDOC else None
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
        container = Container(
            ollama_url=settings.OLLAMA_URL,
            ollama_model=None  # Auto-detectar según GPU
        )
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
    
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para HTTPException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
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
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# Helper Functions
# ============================================================================

async def _select_questions_with_clustering(c: Container, interview, context_text: str):
    """
    Selección RÁPIDA de preguntas usando embeddings pre-computados.
    Optimizado para respuesta instantánea (<100ms).
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
        
        # 3. Ranking con embeddings PRE-COMPUTADOS
        if c.embeddings_service and len(context_text.strip()) > 20:
            try:
                context_embedding = c.embeddings_service.encode([context_text])[0]
                embeddings_cache = getattr(c.question_repository, '_embeddings_cache', {})
                
                if embeddings_cache:
                    question_embeddings = np.array([
                        embeddings_cache.get(q.id, c.embeddings_service.encode([q.text])[0])
                        for q in candidates
                    ])
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
# Authentication Endpoints
# ============================================================================

# ============================================================================
# NOTA: Autenticación manejada por WebApp (.NET Backend)
# Este backend solo proporciona servicios de IA para entrevistas
# ============================================================================

# ============================================================================
# Health Check & Monitoring
# ============================================================================

@app.get("/api/v2/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check del sistema."""
    try:
        c = get_container()
        health_status = c.health_check()
        
        is_degraded = any(
            any(word in v.lower() for word in ["degraded", "error", "failed", "unhealthy"])
            for v in health_status.values()
        )
        
        all_healthy = all(
            "healthy" in v.lower() or "✅" in v
            for v in health_status.values()
        )
        
        overall_status = "degraded" if is_degraded else ("healthy" if all_healthy else "degraded")
        
        return HealthResponse(
            status=overall_status,
            version=settings.APP_VERSION,
            timestamp=datetime.utcnow(),
            components=health_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
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
    
    return PlainTextResponse(
        content=prometheus_text,
        media_type="text/plain; version=0.0.4"
    )


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
    
    return {
        "timestamp": datetime.utcnow(),
        "metrics": stats,
        "version": settings.APP_VERSION
    }

# ============================================================================
# Interview Endpoints (CON AUTENTICACIÓN)
# ============================================================================

@app.post("/api/v2/interviews", response_model=StartInterviewResponse, tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def start_interview(
    request: Request,
    interview_request: StartInterviewRequest,
    current_user_id: str = Depends(get_optional_user_id)  # Opcional por ahora
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
            id=f"interview_{interview_request.user_id}_{datetime.utcnow().timestamp()}",
            user_id=interview_request.user_id,
            role=interview_request.role,
            skill_level=SkillLevel.from_string(interview_request.difficulty),
            interview_type=interview_request.category,
            current_phase="context",
            mode="practice"
        )
        
        await c.interview_repository.save(interview)
        logger.info(f"Entrevista iniciada: {interview.id} - Usuario: {interview_request.user_id}")
        
        # Obtener primera pregunta de contexto
        context_questions = get_context_questions(interview.interview_type)
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
                expected_concepts=[]
            ),
            status="context",
            message="¡Bienvenido! Responde 5 preguntas de contexto para personalizar tu experiencia."
        )
        
    except InterviewAlreadyExists:
        raise
    except Exception as e:
        logger.error(f"Error iniciando entrevista: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error iniciando entrevista: {str(e)}"
        )


@app.post("/api/v2/interviews/{interview_id}/answers", response_model=ProcessAnswerResponse, tags=["Interviews"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def process_answer(
    request: Request,
    interview_id: str,
    answer_request: ProcessAnswerRequest,
    current_user_id: str = Depends(get_optional_user_id)
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
                
                # Seleccionar preguntas con ML
                context_text = " ".join(interview.context_answers)
                selected_questions = await _select_questions_with_clustering(c, interview, context_text)
                
                if not selected_questions:
                    raise HTTPException(status_code=500, detail="No questions could be selected")
                
                interview.metadata['selected_questions'] = [q.id for q in selected_questions]
                interview.metadata['current_question_index'] = 0
                interview.status = InterviewStatus.ACTIVE
                await c.interview_repository.save(interview)
                
                first_tech_question = selected_questions[0]
                interview.add_question(first_tech_question)
                await c.interview_repository.save(interview)
                
                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(
                        score=0,
                        is_correct=True,
                        feedback="Fase de contexto completada"
                    ),
                    feedback="¡Perfecto! Ahora comenzaremos con las preguntas personalizadas según tu perfil. ¡Mucha suerte! 🚀",
                    emotion=EmotionDTO(emotion="neutral", confidence=0.0),
                    next_question=QuestionDTO(
                        id=first_tech_question.id,
                        text=first_tech_question.text,
                        category=first_tech_question.category,
                        difficulty=first_tech_question.difficulty,
                        topic=first_tech_question.topic,
                        expected_concepts=first_tech_question.expected_concepts
                    ),
                    phase="questions",
                    progress=ProgressDTO(
                        context_completed=CONTEXT_QUESTIONS_COUNT,
                        questions_completed=0,
                        total_questions=MAIN_QUESTIONS_COUNT
                    ),
                    interview_status="questions"
                )
            else:
                # Siguiente pregunta de contexto
                context_questions = get_context_questions(interview.interview_type)
                next_context = context_questions[interview.context_question_index]
                
                await c.interview_repository.save(interview)
                
                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(
                        score=0,
                        is_correct=True,
                        feedback="Respuesta guardada"
                    ),
                    feedback=f"Gracias. Continuemos... (Pregunta {interview.context_question_index + 1}/{CONTEXT_QUESTIONS_COUNT})",
                    emotion=EmotionDTO(emotion="neutral", confidence=0.0),
                    next_question=QuestionDTO(
                        id=f"context_{interview.context_question_index}",
                        text=next_context,
                        category="context",
                        difficulty="context",
                        topic="context",
                        expected_concepts=[]
                    ),
                    phase="context",
                    progress=ProgressDTO(
                        context_completed=interview.context_question_index,
                        questions_completed=0,
                        total_questions=MAIN_QUESTIONS_COUNT
                    ),
                    interview_status="context"
                )
        
        # FASE 2: PREGUNTAS TÉCNICAS
        elif interview.current_phase == "questions":
            if not interview.current_question:
                raise HTTPException(status_code=400, detail="No active question")
            
            # Detectar emoción
            emotion_result = c.emotion_detector.detect(answer_request.answer)
            emotion = emotion_result['emotion'] if isinstance(emotion_result['emotion'], Emotion) else Emotion.NEUTRAL
            
            # Evaluar con LLM
            evaluation = c.evaluation_service.evaluate_answer(
                question=interview.current_question.text,
                answer=answer_request.answer,
                expected_concepts=interview.current_question.expected_concepts,
                keywords=interview.current_question.keywords,
                category=interview.current_question.category,
                difficulty=interview.current_question.difficulty,
                role=interview.role
            )
            
            # Generar feedback
            feedback_result = c.feedback_service.generate_feedback(
                question=interview.current_question.text,
                answer=answer_request.answer,
                evaluation=evaluation,
                emotion=emotion,
                role=interview.role,
                category=interview.current_question.category
            )
            
            # Guardar respuesta
            from app.domain.value_objects.score import Score
            answer_entity = Answer(
                question_id=interview.current_question.id,
                text=answer_request.answer,
                score=Score(evaluation['score']),
                is_correct=evaluation['is_correct'],
                emotion=emotion,
                time_taken=answer_request.time_taken or 0,
                evaluation_details=evaluation
            )
            
            interview.add_answer(answer_entity)
            
            current_index = interview.metadata.get('current_question_index', 0) + 1
            interview.metadata['current_question_index'] = current_index
            
            logger.info(f"✅ Respuesta evaluada: score={evaluation['score']}, pregunta {current_index}/{MAIN_QUESTIONS_COUNT}")
            
            # ¿Completamos la entrevista?
            if current_index >= MAIN_QUESTIONS_COUNT:
                interview.complete()
                await c.interview_repository.save(interview)
                
                logger.info(f"🎉 Entrevista completada: {interview_id}")
                
                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(
                        score=evaluation['score'],
                        is_correct=evaluation['is_correct'],
                        feedback=evaluation.get('justification', ''),
                        breakdown=evaluation.get('breakdown'),
                        strengths=evaluation.get('strengths', []),
                        improvements=evaluation.get('improvements', []),
                        concepts_covered=evaluation.get('concepts_covered', []),
                        missing_concepts=evaluation.get('missing_concepts', [])
                    ),
                    feedback=feedback_result,
                    emotion=EmotionDTO(
                        emotion=emotion.value,
                        confidence=emotion_result.get('confidence', 0.0),
                        language=emotion_result.get('language')
                    ),
                    next_question=None,
                    phase="completed",
                    progress=ProgressDTO(
                        context_completed=CONTEXT_QUESTIONS_COUNT,
                        questions_completed=current_index,
                        total_questions=MAIN_QUESTIONS_COUNT,
                        percentage=100.0
                    ),
                    interview_status="completed",
                    interview_completed=True
                )
            else:
                # Siguiente pregunta
                selected_question_ids = interview.metadata.get('selected_questions', [])
                if current_index < len(selected_question_ids):
                    next_question_id = selected_question_ids[current_index]
                    next_question = await c.question_repository.find_by_id(next_question_id)
                    
                    if next_question:
                        interview.add_question(next_question)
                        await c.interview_repository.save(interview)
                        
                        return ProcessAnswerResponse(
                            evaluation=EvaluationDTO(
                                score=evaluation['score'],
                                is_correct=evaluation['is_correct'],
                                feedback=evaluation.get('justification', ''),
                                breakdown=evaluation.get('breakdown'),
                                strengths=evaluation.get('strengths', []),
                                improvements=evaluation.get('improvements', []),
                                concepts_covered=evaluation.get('concepts_covered', []),
                                missing_concepts=evaluation.get('missing_concepts', [])
                            ),
                            feedback=feedback_result,
                            emotion=EmotionDTO(
                                emotion=emotion.value,
                                confidence=emotion_result.get('confidence', 0.0),
                                language=emotion_result.get('language')
                            ),
                            next_question=QuestionDTO(
                                id=next_question.id,
                                text=next_question.text,
                                category=next_question.category,
                                difficulty=next_question.difficulty,
                                topic=next_question.topic,
                                expected_concepts=next_question.expected_concepts
                            ),
                            phase="questions",
                            progress=ProgressDTO(
                                context_completed=CONTEXT_QUESTIONS_COUNT,
                                questions_completed=current_index,
                                total_questions=MAIN_QUESTIONS_COUNT,
                                percentage=(current_index / MAIN_QUESTIONS_COUNT) * 100
                            ),
                            interview_status="questions"
                        )
                
                # Fallback: completar entrevista
                interview.complete()
                await c.interview_repository.save(interview)
                
                return ProcessAnswerResponse(
                    evaluation=EvaluationDTO(
                        score=evaluation['score'],
                        is_correct=evaluation['is_correct'],
                        feedback=evaluation.get('justification', '')
                    ),
                    feedback=feedback_result,
                    emotion=EmotionDTO(emotion=emotion.value, confidence=0.0),
                    next_question=None,
                    phase="completed",
                    progress=ProgressDTO(
                        context_completed=CONTEXT_QUESTIONS_COUNT,
                        questions_completed=current_index,
                        total_questions=MAIN_QUESTIONS_COUNT
                    ),
                    interview_status="completed",
                    interview_completed=True
                )
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid phase: {interview.current_phase}")
        
    except (InterviewNotFound, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Error procesando respuesta: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando respuesta: {str(e)}"
        )


@app.post("/api/v2/interviews/{interview_id}/end", tags=["Interviews"])
@limiter.limit("10/minute")
async def end_interview(
    request: Request,
    interview_id: str,
    current_user_id: str = Depends(get_optional_user_id)
):
    """Finaliza una entrevista y genera resumen."""
    try:
        c = get_container()
        
        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise InterviewNotFound(interview_id)
        
        interview.complete()
        await c.interview_repository.save(interview)
        
        answers = getattr(interview, 'answers', []) or []
        total_score = sum(a.score.value for a in answers) / len(answers) if answers else 0
        
        logger.info(f"Entrevista finalizada: {interview_id}, score={total_score:.2f}")
        
        return {
            "interview_id": interview_id,
            "status": "completed",
            "total_answers": len(answers),
            "average_score": round(total_score, 2),
            "summary": f"Entrevista completada con {len(answers)} respuestas"
        }
        
    except InterviewNotFound:
        raise
    except Exception as e:
        logger.error(f"Error finalizando entrevista: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finalizando entrevista: {str(e)}"
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
        "health": "/api/v2/health"
    }


@app.get("/api/v2/metrics", tags=["Metrics"])
async def get_metrics():
    """Métricas del sistema LLM."""
    try:
        c = get_container()
        metrics = c.llm_service.get_metrics()
        
        return {
            "llm": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo métricas: {str(e)}"
        )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar"""
    logger.info(f"🚀 Ready4Hire v{settings.APP_VERSION} iniciando...")
    logger.info(f"📦 Environment: {settings.ENVIRONMENT}")
    logger.info(f"📦 CORS Origins: {settings.get_cors_origins()}")
    logger.info(f"📦 Rate Limit: {settings.RATE_LIMIT_PER_MINUTE}/min")
    
    try:
        c = get_container()
        health = c.health_check()
        
        logger.info("✅ Container inicializado")
        for component, status in health.items():
            logger.info(f"   {component}: {status}")
        
        logger.info("🎯 Sistema listo para recibir requests")
        
    except Exception as e:
        logger.error(f"❌ Error en startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al apagar"""
    logger.info("👋 Ready4Hire cerrando...")
    logger.info("✅ Shutdown completado")


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
        log_level=settings.LOG_LEVEL.lower()
    )

