"""
Use Case: Procesar Respuesta del Candidato
"""
from dataclasses import dataclass
from typing import List, Dict, Optional

from app.domain.entities.answer import Answer
from app.domain.repositories.interview_repository import InterviewRepository
from app.domain.value_objects.emotion import Emotion


@dataclass
class ProcessAnswerRequest:
    user_id: str
    answer_text: str
    time_taken: int = 0  # segundos


@dataclass
class ProcessAnswerResponse:
    feedback: str
    is_correct: bool
    score: float
    hints: List[str]
    emotion: str
    encouragement: str
    learning_resources: List[Dict]
    next_question: str = ""
    interview_completed: bool = False
    success: bool = True
    error: Optional[str] = None


class ProcessAnswerUseCase:
    """
    Caso de uso: Procesar respuesta del candidato
    
    Flujo:
    1. Recuperar entrevista activa
    2. Evaluar respuesta
    3. Detectar emoción
    4. Generar feedback
    5. Actualizar entrevista
    6. Seleccionar siguiente pregunta
    """
    
    def __init__(
        self,
        interview_repo: InterviewRepository,
        emotion_detector: 'EmotionDetector',
        evaluation_service: 'EvaluationService',
        feedback_service: 'FeedbackService',
        question_selector: 'QuestionSelectorService'
    ):
        """
        Inicializa el caso de uso con todas las dependencias necesarias.
        
        Args:
            interview_repo: Repositorio para gestionar entrevistas
            emotion_detector: Servicio para detectar emociones
            evaluation_service: Servicio para evaluar respuestas
            feedback_service: Servicio para generar feedback
            question_selector: Servicio para seleccionar preguntas
        """
        self.interview_repo = interview_repo
        self.emotion_detector = emotion_detector
        self.evaluation_service = evaluation_service
        self.feedback_service = feedback_service
        self.question_selector = question_selector
    
    async def execute(self, request: ProcessAnswerRequest) -> ProcessAnswerResponse:
        try:
            # 1. Recuperar entrevista activa
            interview = await self.interview_repo.find_active_by_user(request.user_id)
            
            if not interview or not interview.current_question:
                return ProcessAnswerResponse(
                    feedback="",
                    is_correct=False,
                    score=0.0,
                    hints=[],
                    emotion="neutral",
                    encouragement="",
                    learning_resources=[],
                    success=False,
                    error="NO_ACTIVE_INTERVIEW"
                )
            
            # 2. Detectar emoción
            emotion_result = self.emotion_detector.detect(request.answer_text)
            detected_emotion = emotion_result['emotion']
            
            # 3. Evaluar respuesta
            evaluation = await self.evaluation_service.evaluate(
                question=interview.current_question,
                answer_text=request.answer_text,
                interview=interview
            )
            
            # 4. Crear Answer entity
            answer = Answer(
                question_id=interview.current_question.id,
                text=request.answer_text,
                score=evaluation['score'],
                is_correct=evaluation['is_correct'],
                emotion=detected_emotion,
                time_taken=request.time_taken,
                hints_used=0,  # TODO: Implementar sistema de pistas
                evaluation_details=evaluation
            )
            
            # 5. Agregar respuesta a entrevista
            interview.add_answer(answer)
            
            # 6. Generar feedback
            feedback = await self.feedback_service.generate(
                evaluation=evaluation,
                question=interview.current_question,
                answer=answer,
                interview=interview
            )
            
            # 7. Determinar si continuar o terminar
            interview_completed = False
            next_question_text = ""
            
            # Criterio: 10 preguntas por entrevista
            if len(interview.answers_history) >= 10:
                interview.complete()
                interview_completed = True
            else:
                # Seleccionar siguiente pregunta
                next_question = await self.question_selector.select_next(
                    interview=interview,
                    last_evaluation=evaluation
                )
                
                if next_question:
                    interview.add_question(next_question)
                    next_question_text = next_question.text
                else:
                    # No hay más preguntas disponibles
                    interview.complete()
                    interview_completed = True
            
            # 8. Persistir cambios
            await self.interview_repo.save(interview)
            
            # 9. Retornar respuesta
            return ProcessAnswerResponse(
                feedback=feedback.get('message', ''),
                is_correct=evaluation['is_correct'],
                score=evaluation['score'],
                hints=feedback.get('hints', []),
                emotion=detected_emotion.value,
                encouragement=feedback.get('encouragement', ''),
                learning_resources=feedback.get('learning_resources', []),
                next_question=next_question_text,
                interview_completed=interview_completed,
                success=True
            )
        
        except Exception as e:
            print(f"[ERROR] ProcessAnswerUseCase: {e}")
            return ProcessAnswerResponse(
                feedback=f"Error al procesar respuesta: {str(e)}",
                is_correct=False,
                score=0.0,
                hints=[],
                emotion="neutral",
                encouragement="",
                learning_resources=[],
                success=False,
                error="INTERNAL_ERROR"
            )
