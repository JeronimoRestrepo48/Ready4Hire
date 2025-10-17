"""
FastAPI Application v2 - Ready4Hire
Arquitectura DDD con Ollama local y Dependency Injection
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import inspect
import logging
from datetime import datetime

from app.container import Container
from app.domain.value_objects.emotion import Emotion
from app.domain.value_objects.context_questions import get_context_questions

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
            ollama_model="llama3.2:3b"
        )
    return container


async def _select_questions_with_clustering(c: Container, interview, context_text: str):
    """
    Selección RÁPIDA de preguntas usando embeddings pre-computados.
    Optimizado para respuesta instantánea (<100ms).
    """
    from app.domain.value_objects.context_questions import MAIN_QUESTIONS_COUNT
    import random
    import numpy as np
    
    try:
        # 1. RÁPIDO: Obtener preguntas según tipo (ya cargadas en memoria)
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
        
        # 2. RÁPIDO: Filtrado por nivel (operación en memoria)
        level_filtered = [q for q in all_questions if q.difficulty == interview.skill_level.value]
        candidates = level_filtered if len(level_filtered) >= MAIN_QUESTIONS_COUNT else all_questions
        
        # 3. ⚡⚡ INSTANTÁNEO: Ranking con embeddings PRE-COMPUTADOS
        if c.embeddings_service and len(context_text.strip()) > 20:
            try:
                # Generar embedding del contexto (1 sola llamada)
                context_embedding = c.embeddings_service.encode([context_text])[0]
                
                # ⚡⚡ USAR EMBEDDINGS PRE-COMPUTADOS (sin re-encoding)
                embeddings_cache = getattr(c.question_repository, '_embeddings_cache', {})
                
                if embeddings_cache:
                    # Usar embeddings del cache (instantáneo, sin llamadas a encode)
                    question_embeddings = np.array([
                        embeddings_cache.get(q.id, c.embeddings_service.encode([q.text])[0])
                        for q in candidates
                    ])
                    logger.debug(f"⚡⚡ Usando embeddings pre-computados de {len(candidates)} preguntas")
                else:
                    # Fallback: encodear en batch (más lento pero funciona)
                    question_embeddings = c.embeddings_service.encode([q.text for q in candidates])
                    logger.debug(f"⚠️ Sin cache, encoding on-the-fly de {len(candidates)} preguntas")
                
                # Similitud coseno vectorizada (numpy ultra-rápido)
                similarities = np.dot(question_embeddings, context_embedding) / (
                    np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(context_embedding)
                )
                
                # Top K preguntas (argsort optimizado)
                top_indices = np.argsort(similarities)[::-1][:MAIN_QUESTIONS_COUNT]
                selected_questions = [candidates[i] for i in top_indices]
                
                logger.info(f"⚡ {len(selected_questions)} preguntas seleccionadas INSTANTÁNEAMENTE")
                return selected_questions
                
            except Exception as e:
                logger.warning(f"⚠️ Fallback a selección aleatoria: {str(e)}")
        
        # 4. FALLBACK RÁPIDO: Selección aleatoria
        selected = random.sample(candidates, min(MAIN_QUESTIONS_COUNT, len(candidates)))
        logger.info(f"⚡ {len(selected)} preguntas (fallback aleatorio)")
        return selected
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return []


# ============================================================================
# DTOs (Pydantic Models)
# ============================================================================
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
    next_question: Optional[Dict[str, Any]] = None
    interview_status: str
    interview_completed: bool = False


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
        
        # Determinar estado general
        # Un componente está degradado si contiene "degraded", "error", "failed" o "unhealthy"
        is_degraded = any(
            any(word in v.lower() for word in ["degraded", "error", "failed", "unhealthy"])
            for v in health_status.values()
        )
        
        # Si tiene "healthy" o emoji de check (✅) o ambos, está saludable
        all_healthy = all(
            "healthy" in v.lower() or "✅" in v or "⚠️" not in v
            for v in health_status.values()
        )
        
        overall_status = "degraded" if is_degraded else ("healthy" if all_healthy else "degraded")
        
        return HealthResponse(
            status=overall_status,
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


# ============================================================================
# Endpoints de Entrevistas V2
# ============================================================================

@app.post("/api/v2/interviews", response_model=StartInterviewResponse, tags=["Interviews"])
async def start_interview(request: StartInterviewRequest):
    """
    Inicia una nueva entrevista con preguntas de contexto.
    
    - Crea sesión de entrevista en fase de contexto
    - Retorna primera pregunta de contexto para conocer al candidato
    - Las preguntas técnicas se seleccionarán después del análisis de contexto
    """
    try:
        c = get_container()
        from app.domain.entities.interview import Interview
        from app.domain.value_objects.skill_level import SkillLevel
        
        # Crear nueva entrevista en fase de contexto
        interview = Interview(
            id=f"interview_{request.user_id}_{datetime.utcnow().timestamp()}",
            user_id=request.user_id,
            role=request.role,
            skill_level=SkillLevel.from_string(request.difficulty),
            interview_type=request.category,
            current_phase="context",
            mode="practice"
        )
        
        # Guardar en repositorio
        await c.interview_repository.save(interview)
        logger.info(f"Entrevista iniciada: {interview.id} para {request.user_id} - Fase: contexto")
        
        # Obtener preguntas de contexto
        context_questions = get_context_questions(interview.interview_type)
        if not context_questions or len(context_questions) == 0:
            raise HTTPException(
                status_code=500, 
                detail="No hay preguntas de contexto definidas para este tipo de entrevista."
            )
        
        first_context_question = context_questions[0]
        
        # Estructura de respuesta para frontend conversacional
        return StartInterviewResponse(
            interview_id=interview.id,
            first_question={
                "id": "context_0",
                "text": first_context_question,
                "category": "context",
                "difficulty": "context",
                "expected_concepts": [],
                "topic": "context"
            },
            status="context"  # Indicar que está en fase de contexto
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
    
    Flujo en dos fases:
    1. FASE CONTEXTO (5 preguntas): Solo guarda respuestas, NO evalúa con LLM
    2. FASE PREGUNTAS (10 preguntas): Evalúa, genera feedback y selecciona siguiente
    """
    try:
        c = get_container()
        from app.domain.value_objects.context_questions import get_context_questions, CONTEXT_QUESTIONS_COUNT, MAIN_QUESTIONS_COUNT
        
        # Obtener entrevista
        interview = await c.interview_repository.find_by_id(interview_id)
        if not interview:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entrevista {interview_id} no encontrada"
            )
        
        # ============================================================================
        # FASE 1: PREGUNTAS DE CONTEXTO (NO SE EVALÚAN)
        # ============================================================================
        if interview.current_phase == "context":
            # Guardar respuesta de contexto sin evaluación
            interview.context_answers.append(request.answer)
            interview.context_question_index += 1
            
            logger.info(f"✅ Respuesta de contexto {interview.context_question_index}/{CONTEXT_QUESTIONS_COUNT} guardada (sin evaluación)")
            
            # Verificar si completamos las preguntas de contexto
            if interview.context_question_index >= CONTEXT_QUESTIONS_COUNT:
                # Transición a fase de preguntas
                interview.current_phase = "questions"
                await c.interview_repository.save(interview)
                
                logger.info(f"🎯 Fase de contexto completada. Analizando respuestas para seleccionar preguntas...")
                
                # Generar embeddings de las respuestas de contexto
                context_text = " ".join(interview.context_answers)
                
                # Seleccionar las mejores 10 preguntas usando clustering
                selected_questions = await _select_questions_with_clustering(
                    c=c,
                    interview=interview,
                    context_text=context_text
                )
                
                if not selected_questions:
                    raise HTTPException(
                        status_code=500,
                        detail="No se pudieron seleccionar preguntas adecuadas"
                    )
                
                # Guardar preguntas seleccionadas en metadata
                interview.metadata['selected_questions'] = [q.id for q in selected_questions]
                interview.metadata['current_question_index'] = 0
                
                # IMPORTANTE: Cambiar estado a ACTIVE antes de agregar preguntas
                from app.domain.value_objects.interview_status import InterviewStatus
                interview.status = InterviewStatus.ACTIVE
                await c.interview_repository.save(interview)
                
                # Retornar primera pregunta técnica
                first_tech_question = selected_questions[0]
                interview.add_question(first_tech_question)
                await c.interview_repository.save(interview)
                
                logger.info(f"✅ Primera pregunta técnica seleccionada: {first_tech_question.text[:50]}...")
                
                return ProcessAnswerResponse(
                    evaluation={
                        "score": 0,
                        "is_correct": True,
                        "message": "Fase de contexto completada. Iniciando preguntas técnicas."
                    },
                    feedback="¡Perfecto! He analizado tus respuestas. Ahora comenzaremos con las preguntas técnicas personalizadas según tu perfil. ¡Mucha suerte! 🚀",
                    emotion={"emotion": "neutral", "confidence": 0.0},
                    next_question={
                        "id": first_tech_question.id,
                        "text": first_tech_question.text,
                        "category": first_tech_question.category,
                        "difficulty": first_tech_question.difficulty,
                        "topic": first_tech_question.topic,
                        "expected_concepts": first_tech_question.expected_concepts
                    },
                    interview_status="questions"
                )
            else:
                # Aún hay más preguntas de contexto
                context_questions = get_context_questions(interview.interview_type)
                next_context_question = context_questions[interview.context_question_index]
                
                await c.interview_repository.save(interview)
                
                return ProcessAnswerResponse(
                    evaluation={
                        "score": 0,
                        "is_correct": True,
                        "message": "Respuesta de contexto guardada"
                    },
                    feedback=f"Gracias por tu respuesta. Continuemos conociendo tu perfil... (Pregunta {interview.context_question_index + 1}/{CONTEXT_QUESTIONS_COUNT})",
                    emotion={"emotion": "neutral", "confidence": 0.0},
                    next_question={
                        "id": f"context_{interview.context_question_index}",
                        "text": next_context_question,
                        "category": "context",
                        "difficulty": "context",
                        "topic": "context",
                        "expected_concepts": []
                    },
                    interview_status="context"
                )
        
        # ============================================================================
        # FASE 2: PREGUNTAS TÉCNICAS (SÍ SE EVALÚAN)
        # ============================================================================
        elif interview.current_phase == "questions":
            if not interview.current_question:
                raise HTTPException(
                    status_code=400,
                    detail="No hay pregunta activa para responder"
                )
            
            # Detectar emoción
            emotion_result = c.emotion_detector.detect(request.answer)
            emotion = emotion_result['emotion'] if isinstance(emotion_result['emotion'], Emotion) else Emotion.NEUTRAL
            
            # Evaluar respuesta con el LLM (método síncrono)
            evaluation = c.evaluation_service.evaluate_answer(
                question=interview.current_question.text,
                answer=request.answer,
                expected_concepts=interview.current_question.expected_concepts,
                keywords=interview.current_question.keywords,
                category=interview.current_question.category,
                difficulty=interview.current_question.difficulty,
                role=interview.role
            )
            
            # Generar feedback personalizado (método síncrono)
            from app.domain.entities.answer import Answer
            answer_entity = Answer(
                question_id=interview.current_question.id,
                text=request.answer,
                score=evaluation['score'],
                is_correct=evaluation['is_correct'],
                emotion=emotion,
                time_taken=request.time_taken or 0,
                evaluation_details=evaluation
            )
            
            feedback_result = c.feedback_service.generate_feedback(
                question=interview.current_question.text,
                answer=request.answer,
                evaluation=evaluation,
                emotion=emotion,
                role=interview.role,
                category=interview.current_question.category
            )
            
            # Agregar respuesta a historial
            interview.add_answer(answer_entity)
            
            # Incrementar índice de pregunta
            current_index = interview.metadata.get('current_question_index', 0) + 1
            interview.metadata['current_question_index'] = current_index
            
            logger.info(f"✅ Respuesta evaluada: score={evaluation['score']}, pregunta {current_index}/{MAIN_QUESTIONS_COUNT}")
            
            # Verificar si completamos las 10 preguntas
            if current_index >= MAIN_QUESTIONS_COUNT:
                interview.complete()
                await c.interview_repository.save(interview)
                
                logger.info(f"🎉 Entrevista completada: {interview_id}")
                
                return ProcessAnswerResponse(
                    evaluation=evaluation,
                    feedback=feedback_result,
                    emotion={
                        "emotion": emotion.value,
                        "confidence": emotion_result.get('confidence', 0.0)
                    },
                    next_question=None,
                    interview_status="completed",
                    interview_completed=True
                )
            else:
                # Seleccionar siguiente pregunta de la lista pre-seleccionada
                selected_question_ids = interview.metadata.get('selected_questions', [])
                if current_index < len(selected_question_ids):
                    next_question_id = selected_question_ids[current_index]
                    next_question = await c.question_repository.find_by_id(next_question_id)
                    
                    if next_question:
                        interview.add_question(next_question)
                        await c.interview_repository.save(interview)
                        
                        return ProcessAnswerResponse(
                            evaluation=evaluation,
                            feedback=feedback_result,
                            emotion={
                                "emotion": emotion.value,
                                "confidence": emotion_result.get('confidence', 0.0)
                            },
                            next_question={
                                "id": next_question.id,
                                "text": next_question.text,
                                "category": next_question.category,
                                "difficulty": next_question.difficulty,
                                "topic": next_question.topic,
                                "expected_concepts": next_question.expected_concepts
                            },
                            interview_status="questions"
                        )
                
                # Fallback: completar entrevista si no hay más preguntas
                interview.complete()
                await c.interview_repository.save(interview)
                
                return ProcessAnswerResponse(
                    evaluation=evaluation,
                    feedback=feedback_result,
                    emotion={
                        "emotion": emotion.value,
                        "confidence": emotion_result.get('confidence', 0.0)
                    },
                    next_question=None,
                    interview_status="completed",
                    interview_completed=True
                )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Fase de entrevista inválida: {interview.current_phase}"
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
