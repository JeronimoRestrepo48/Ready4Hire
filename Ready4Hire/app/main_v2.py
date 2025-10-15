"""
FastAPI Application v2 - Ready4Hire
Arquitectura DDD con Ollama local y Dependency Injection
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from app.container import Container
from app.domain.value_objects.emotion import Emotion

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="Ready4Hire API v2",
    description="Sistema de entrevistas con IA - Arquitectura DDD + Ollama",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Container de DI (singleton)
container: Optional[Container] = None


def get_container() -> Container:
    """Obtiene el container de DI (lazy initialization)"""
    global container
    if container is None:
        container = Container(
            ollama_url="http://localhost:11434",
            ollama_model="llama3:latest"
        )
    return container


# ============================================================================
# DTOs (Pydantic Models)
# ============================================================================
class StartInterviewRequest(BaseModel):
    """Request para iniciar entrevista"""
    user_id: str = Field(..., description="ID único del usuario")
    role: str = Field(..., description="Rol/posición (ej: Backend Developer)")
    category: str = Field("technical", description="Categoría: technical o soft_skills")
    difficulty: str = Field("junior", description="Dificultad: junior, mid, senior")


class StartInterviewResponse(BaseModel):
    """Response al iniciar entrevista"""
    interview_id: str
    first_question: Dict[str, Any]
    status: str


class ProcessAnswerRequest(BaseModel):
    """Request para procesar respuesta"""
    answer: str = Field(..., description="Respuesta del candidato")
    time_taken: Optional[int] = Field(None, description="Tiempo en segundos")


class ProcessAnswerResponse(BaseModel):
    """Response al procesar respuesta"""
    evaluation: Dict[str, Any]
    feedback: str
    emotion: Dict[str, Any]
    next_question: Optional[Dict[str, Any]]
    interview_status: str


class HealthResponse(BaseModel):
    """Response del health check"""
    status: str
    components: Dict[str, str]
    version: str
    timestamp: str


# ============================================================================
# Endpoints v2
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz"""
    return {
        "message": "Ready4Hire API v2.0",
        "docs": "/docs",
        "health": "/api/v2/health"
    }


@app.get("/api/v2/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check del sistema.
    Verifica que todos los componentes estén operativos.
    """
    try:
        c = get_container()
        health_status = c.health_check()
        
        return HealthResponse(
            status="healthy" if all("healthy" in v for v in health_status.values()) else "degraded",
            components=health_status,
            version="2.0.0",
            timestamp=datetime.utcnow().isoformat()
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


@app.get("/api/v2/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Métricas del sistema LLM.
    Retorna estadísticas de uso de Ollama.
    """
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


@app.post("/api/v2/interviews", response_model=StartInterviewResponse, tags=["Interviews"])
async def start_interview(request: StartInterviewRequest):
    """
    Inicia una nueva entrevista.
    
    - Crea sesión de entrevista
    - Selecciona primera pregunta según rol y dificultad
    - Retorna pregunta inicial
    """
    try:
        c = get_container()
        
        # Obtener primera pregunta
        import asyncio
        questions = await c.question_repository.find_by_role(
            role=request.role,
            category=request.category
        )
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontraron preguntas para rol '{request.role}'"
            )
        
        # Filtrar por dificultad
        filtered = [q for q in questions if q.difficulty == request.difficulty]
        if not filtered:
            filtered = questions  # Fallback a todas las preguntas
        
        first_question = filtered[0]
        
        # Crear entrevista
        from app.domain.entities.interview import Interview
        from app.domain.value_objects.skill_level import SkillLevel
        
        interview = Interview(
            id=f"interview_{request.user_id}_{datetime.utcnow().timestamp()}",
            user_id=request.user_id,
            role=request.role,
            skill_level=SkillLevel.from_string(request.difficulty),
            interview_type=request.category
        )
        
        # Guardar en repositorio
        await c.interview_repository.save(interview)
        
        logger.info(f"Entrevista iniciada: {interview.id} para {request.user_id}")
        
        return StartInterviewResponse(
            interview_id=interview.id,
            first_question={
                "id": first_question.id,
                "text": first_question.text,
                "category": first_question.category,
                "difficulty": first_question.difficulty,
                "expected_concepts": first_question.expected_concepts,
                "topic": first_question.topic
            },
            status="active"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error iniciando entrevista: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error iniciando entrevista: {str(e)}"
        )


@app.post("/api/v2/interviews/{interview_id}/answers", response_model=ProcessAnswerResponse, tags=["Interviews"])
async def process_answer(interview_id: str, request: ProcessAnswerRequest):
    """
    Procesa la respuesta del candidato.
    
    - Detecta emoción
    - Evalúa respuesta con LLM
    - Genera feedback personalizado
    - Selecciona siguiente pregunta
    """
    try:
        c = get_container()
        
        # Obtener entrevista
        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entrevista {interview_id} no encontrada"
            )
        
        # Detectar emoción
        emotion_result = c.emotion_detector.detect(request.answer)
        emotion = Emotion.from_string(emotion_result['emotion'])
        
        # Evaluar respuesta
        # (Simplificado - en producción usar la pregunta actual)
        evaluation = c.evaluation_service.evaluate_answer(
            question="Pregunta de prueba",
            answer=request.answer,
            expected_concepts=["concepto1", "concepto2"],
            keywords=["keyword1"],
            category=interview.interview_type,
            difficulty=interview.skill_level.value,
            role=interview.role
        )
        
        # Generar feedback
        feedback = c.feedback_service.generate_feedback(
            question="Pregunta de prueba",
            answer=request.answer,
            evaluation=evaluation,
            emotion=emotion,
            role=interview.role,
            category=interview.interview_type
        )
        
        # Seleccionar siguiente pregunta
        next_question = None
        if interview.status == "active":
            questions = await c.question_repository.find_by_difficulty(
                difficulty=interview.skill_level.value,
                category=interview.interview_type
            )
            if questions:
                q = questions[0]
                next_question = {
                    "id": q.id,
                    "text": q.text,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "topic": q.topic
                }
        
        logger.info(f"Respuesta procesada para {interview_id}: score={evaluation['score']}")
        
        return ProcessAnswerResponse(
            evaluation=evaluation,
            feedback=feedback,
            emotion={
                "emotion": emotion.value,
                "confidence": emotion_result.get('confidence', 0.0)
            },
            next_question=next_question,
            interview_status=interview.status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando respuesta: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando respuesta: {str(e)}"
        )


@app.post("/api/v2/interviews/{interview_id}/end", tags=["Interviews"])
async def end_interview(interview_id: str):
    """
    Finaliza una entrevista.
    
    - Genera reporte final
    - Calcula score total
    - Retorna resumen
    """
    try:
        c = get_container()
        
        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entrevista {interview_id} no encontrada"
            )
        
        # Finalizar entrevista
        interview.complete()
        await c.interview_repository.save(interview)
        
        # Calcular estadísticas
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finalizando entrevista: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finalizando entrevista: {str(e)}"
        )


# ============================================================================
# Endpoints de Compatibilidad v1 (Legacy)
# ============================================================================

@app.post("/start_interview", tags=["Legacy v1"])
async def start_interview_v1(request: Dict[str, Any]):
    """
    Endpoint v1 legacy - redirige a v2.
    Mantiene compatibilidad con frontend existente.
    """
    v2_request = StartInterviewRequest(
        user_id=request.get("user_id", "anonymous"),
        role=request.get("role", "Generic"),
        category=request.get("type", "technical"),
        difficulty=request.get("level", "junior")
    )
    
    result = await start_interview(v2_request)
    
    # Adaptar respuesta al formato v1
    return {
        "session_id": result.interview_id,
        "question": result.first_question["text"],
        "question_id": result.first_question["id"]
    }


@app.post("/answer", tags=["Legacy v1"])
async def answer_v1(request: Dict[str, Any]):
    """
    Endpoint v1 legacy - redirige a v2.
    Mantiene compatibilidad con frontend existente.
    """
    session_id = request.get("session_id", "")
    answer = request.get("answer", "")
    
    v2_request = ProcessAnswerRequest(
        answer=answer,
        time_taken=request.get("time_taken")
    )
    result = await process_answer(session_id, v2_request)
    
    # Adaptar respuesta al formato v1
    return {
        "feedback": result.feedback,
        "score": result.evaluation["score"],
        "next_question": result.next_question["text"] if result.next_question else None,
        "emotion": result.emotion["emotion"]
    }


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler para HTTPException"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler para excepciones generales"""
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Error interno del servidor",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar"""
    logger.info("🚀 Ready4Hire v2.0 iniciando...")
    logger.info("📦 Inicializando Container de DI...")
    
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
    logger.info("👋 Ready4Hire v2.0 cerrando...")
    logger.info("✅ Shutdown completado")


# ============================================================================
# Main (para desarrollo)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
