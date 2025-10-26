"""
Celery Tasks para procesamiento asíncrono de evaluaciones.
Permite evaluar respuestas en background sin bloquear el API.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from app.infrastructure.tasks.celery_app import celery_app
from app.infrastructure.cache.redis_cache import get_redis_cache
from app.infrastructure.websocket.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="evaluate_answer_async",
    queue="evaluations",
    max_retries=3,
    default_retry_delay=5
)
def evaluate_answer_async(
    self,
    interview_id: str,
    question_id: str,
    answer_text: str,
    user_id: str,
    question_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evalúa una respuesta de forma asíncrona usando Celery.
    
    Args:
        interview_id: ID de la entrevista
        question_id: ID de la pregunta
        answer_text: Respuesta del usuario
        user_id: ID del usuario
        question_data: Datos de la pregunta (text, category, expected_concepts)
        
    Returns:
        Dict con resultado de evaluación
    """
    try:
        logger.info(f"🔄 Evaluating answer async: interview={interview_id}, question={question_id}")
        
        # Importar servicios aquí para evitar circular imports
        from app.container import get_container
        
        container = get_container()
        evaluation_service = container.evaluation_service
        
        # Ejecutar evaluación
        result = evaluation_service.evaluate_answer_sync(
            question_text=question_data["text"],
            answer_text=answer_text,
            expected_concepts=question_data.get("expected_concepts", []),
            category=question_data.get("category", "technical"),
            difficulty=question_data.get("difficulty", "mid")
        )
        
        # Cache del resultado
        cache = get_redis_cache()
        cache_key = f"{interview_id}:{question_id}"
        cache.set("evaluation", cache_key, result, ttl=None)  # Usa TTL default
        
        # Notificar via WebSocket
        ws_manager = get_websocket_manager()
        ws_manager.send_evaluation_result(interview_id, {
            "question_id": question_id,
            "evaluation": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"✅ Evaluation completed: score={result.get('score', 0)}")
        
        return {
            "status": "success",
            "interview_id": interview_id,
            "question_id": question_id,
            "result": result,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"❌ Error in evaluation task: {e}")
        
        # Reintentar si no ha alcanzado max_retries
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=5 * (self.request.retries + 1))
        
        # Si ya agotó reintentos, devolver error
        return {
            "status": "error",
            "interview_id": interview_id,
            "question_id": question_id,
            "error": str(e),
            "task_id": self.request.id
        }


@celery_app.task(
    name="batch_evaluate_answers",
    queue="evaluations",
    time_limit=300
)
def batch_evaluate_answers(
    evaluations: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evalúa múltiples respuestas en batch.
    
    Args:
        evaluations: Lista de dicts con datos de evaluación
        
    Returns:
        Dict con resultados de todas las evaluaciones
    """
    try:
        logger.info(f"🔄 Batch evaluation started: {len(evaluations)} answers")
        
        from app.container import get_container
        container = get_container()
        evaluation_service = container.evaluation_service
        
        results = []
        for eval_data in evaluations:
            try:
                result = evaluation_service.evaluate_answer_sync(
                    question_text=eval_data["question_text"],
                    answer_text=eval_data["answer_text"],
                    expected_concepts=eval_data.get("expected_concepts", []),
                    category=eval_data.get("category", "technical"),
                    difficulty=eval_data.get("difficulty", "mid")
                )
                
                results.append({
                    "interview_id": eval_data["interview_id"],
                    "question_id": eval_data["question_id"],
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"❌ Error in batch item: {e}")
                results.append({
                    "interview_id": eval_data.get("interview_id"),
                    "question_id": eval_data.get("question_id"),
                    "error": str(e),
                    "status": "error"
                })
        
        logger.info(f"✅ Batch evaluation completed: {len(results)} results")
        
        return {
            "status": "success",
            "total": len(evaluations),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Batch evaluation error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "total": len(evaluations)
        }


@celery_app.task(
    name="generate_interview_summary",
    queue="default"
)
def generate_interview_summary(
    interview_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Genera resumen completo de una entrevista.
    
    Args:
        interview_id: ID de la entrevista
        user_id: ID del usuario
        
    Returns:
        Dict con resumen de la entrevista
    """
    try:
        logger.info(f"📊 Generating interview summary: {interview_id}")
        
        from app.container import get_container
        container = get_container()
        interview_service = container.interview_service
        
        # Obtener datos de la entrevista
        interview = interview_service.get_interview(interview_id)
        
        if not interview:
            raise ValueError(f"Interview {interview_id} not found")
        
        # Generar métricas
        total_questions = len(interview.answers)
        correct_answers = sum(1 for a in interview.answers if a.score >= 7.0)
        average_score = sum(a.score for a in interview.answers) / total_questions if total_questions > 0 else 0
        
        # Identificar áreas fuertes y débiles
        strengths = []
        weaknesses = []
        
        for answer in interview.answers:
            if answer.score >= 8.0:
                strengths.append(answer.question.topic)
            elif answer.score < 5.0:
                weaknesses.append(answer.question.topic)
        
        summary = {
            "interview_id": interview_id,
            "user_id": user_id,
            "completed_at": datetime.utcnow().isoformat(),
            "metrics": {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy_percentage": round((correct_answers / total_questions) * 100, 2) if total_questions > 0 else 0,
                "average_score": round(average_score, 2)
            },
            "strengths": list(set(strengths))[:5],
            "weaknesses": list(set(weaknesses))[:5],
            "recommendations": generate_recommendations(weaknesses),
            "status": "completed"
        }
        
        # Cache el resumen
        cache = get_redis_cache()
        cache.set("interview_summary", interview_id, summary, ttl=None)
        
        logger.info(f"✅ Summary generated: avg_score={average_score:.2f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error generating summary: {e}")
        return {
            "status": "error",
            "interview_id": interview_id,
            "error": str(e)
        }


def generate_recommendations(weaknesses: list[str]) -> list[str]:
    """
    Genera recomendaciones basadas en áreas débiles.
    
    Args:
        weaknesses: Lista de topics débiles
        
    Returns:
        Lista de recomendaciones
    """
    recommendations_map = {
        "Python": "Practica ejercicios de Python en LeetCode",
        "JavaScript": "Estudia ES6+ y async/await",
        "SQL": "Revisa queries complejas y optimización",
        "Algorithms": "Refuerza estructuras de datos básicas",
        "System Design": "Lee 'Designing Data-Intensive Applications'",
        "Networking": "Estudia protocolos TCP/IP y HTTP",
        "Security": "Aprende OWASP Top 10",
        "Docker": "Practica creando Dockerfiles y compose",
        "Kubernetes": "Completa tutorial oficial de K8s"
    }
    
    recommendations = []
    for weakness in weaknesses:
        if weakness in recommendations_map:
            recommendations.append(recommendations_map[weakness])
        else:
            recommendations.append(f"Estudia más sobre {weakness}")
    
    return recommendations[:5]


@celery_app.task(
    name="cleanup_old_evaluations",
    queue="low_priority"
)
def cleanup_old_evaluations(days: int = 30) -> Dict[str, Any]:
    """
    Limpia evaluaciones antiguas del cache.
    
    Args:
        days: Días de antigüedad para considerar "viejo"
        
    Returns:
        Dict con estadísticas de limpieza
    """
    try:
        logger.info(f"🗑️ Cleaning evaluations older than {days} days")
        
        cache = get_redis_cache()
        deleted = cache.clear_pattern("evaluation", "*")
        
        logger.info(f"✅ Cleaned {deleted} old evaluations")
        
        return {
            "status": "success",
            "deleted_count": deleted,
            "days": days
        }
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

