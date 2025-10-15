"""
Use Case: Iniciar Entrevista
"""
from dataclasses import dataclass
from typing import Optional

from app.domain.entities.interview import Interview
from app.domain.entities.question import Question
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.repositories.interview_repository import InterviewRepository
from app.domain.repositories.question_repository import QuestionRepository


@dataclass
class StartInterviewRequest:
    user_id: str
    role: str
    interview_type: str = "technical"  # technical | soft_skills | mixed
    skill_level: str = "junior"
    mode: str = "practice"  # practice | exam


@dataclass
class StartInterviewResponse:
    interview_id: str
    first_question: str
    message: str
    success: bool
    error: Optional[str] = None


class StartInterviewUseCase:
    """
    Caso de uso: Iniciar entrevista
    
    Flujo:
    1. Validar que no haya entrevista activa
    2. Crear nueva entrevista
    3. Seleccionar primera pregunta
    4. Persistir entrevista
    5. Retornar respuesta
    """
    
    def __init__(
        self,
        interview_repo: InterviewRepository,
        question_repo: QuestionRepository
    ):
        self.interview_repo = interview_repo
        self.question_repo = question_repo
    
    async def execute(self, request: StartInterviewRequest) -> StartInterviewResponse:
        try:
            # 1. Validar que no exista entrevista activa
            active_interview = await self.interview_repo.find_active_by_user(request.user_id)
            if active_interview:
                return StartInterviewResponse(
                    interview_id="",
                    first_question="",
                    message="Ya tienes una entrevista activa",
                    success=False,
                    error="ACTIVE_INTERVIEW_EXISTS"
                )
            
            # 2. Crear nueva entrevista
            interview = Interview(
                user_id=request.user_id,
                role=request.role,
                interview_type=request.interview_type,
                skill_level=SkillLevel(request.skill_level),
                mode=request.mode
            )
            
            # 3. Iniciar entrevista
            interview.start()
            
            # 4. Seleccionar primera pregunta
            first_question = await self._select_initial_question(
                role=request.role,
                interview_type=request.interview_type,
                level=interview.skill_level
            )
            
            if not first_question:
                return StartInterviewResponse(
                    interview_id="",
                    first_question="",
                    message="No se encontraron preguntas para este rol",
                    success=False,
                    error="NO_QUESTIONS_FOUND"
                )
            
            interview.add_question(first_question)
            
            # 5. Persistir
            await self.interview_repo.save(interview)
            
            # 6. Retornar respuesta
            welcome_message = self._generate_welcome_message(request.role, request.mode)
            
            return StartInterviewResponse(
                interview_id=interview.id,
                first_question=first_question.text,
                message=welcome_message,
                success=True
            )
        
        except Exception as e:
            return StartInterviewResponse(
                interview_id="",
                first_question="",
                message=f"Error al iniciar entrevista: {str(e)}",
                success=False,
                error="INTERNAL_ERROR"
            )
    
    async def _select_initial_question(
        self,
        role: str,
        interview_type: str,
        level: SkillLevel
    ) -> Optional[Question]:
        """Selecciona la pregunta inicial apropiada"""
        # Obtener preguntas del tipo correcto
        if interview_type == "technical":
            questions = await self.question_repo.find_by_role(role, "technical")
        elif interview_type == "soft_skills":
            questions = await self.question_repo.find_all_soft_skills()
        else:  # mixed
            tech_questions = await self.question_repo.find_by_role(role, "technical")
            soft_questions = await self.question_repo.find_all_soft_skills()
            questions = tech_questions + soft_questions
        
        if not questions:
            return None
        
        # Filtrar por dificultad apropiada para el nivel
        difficulty_map = {
            SkillLevel.JUNIOR: "easy",
            SkillLevel.MID: "medium",
            SkillLevel.SENIOR: "hard"
        }
        
        target_difficulty = difficulty_map[level]
        suitable_questions = [q for q in questions if q.difficulty == target_difficulty]
        
        # Si no hay preguntas del nivel exacto, usar todas
        if not suitable_questions:
            suitable_questions = questions
        
        # Retornar la primera
        import random
        return random.choice(suitable_questions)
    
    def _generate_welcome_message(self, role: str, mode: str) -> str:
        """Genera mensaje de bienvenida personalizado"""
        if mode == "exam":
            return f"🎓 ¡Bienvenido/a al examen de {role}! Responde 10 preguntas. ¡Éxito!"
        else:
            return f"👋 ¡Hola! Iniciemos tu entrevista de práctica para {role}. Tómate tu tiempo y aprende. 🚀"
