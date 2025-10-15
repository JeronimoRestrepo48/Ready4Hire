"""
Entity: Entrevista (Aggregate Root)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from app.domain.value_objects.interview_status import InterviewStatus
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.entities.question import Question
from app.domain.entities.answer import Answer


@dataclass
class Interview:
    """
    Aggregate Root: Entrevista
    
    Invariantes:
    - Una entrevista siempre tiene un user_id
    - Solo puede haber una pregunta activa a la vez
    - No se pueden agregar respuestas si status != ACTIVE
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    role: str = ""
    interview_type: str = "technical"  # technical | soft_skills | mixed
    status: InterviewStatus = InterviewStatus.CREATED
    skill_level: SkillLevel = SkillLevel.JUNIOR
    mode: str = "practice"  # practice | exam
    
    current_question: Optional[Question] = None
    questions_history: List[Question] = field(default_factory=list)
    answers_history: List[Answer] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    # Métricas
    current_streak: int = 0
    max_streak: int = 0
    total_hints_used: int = 0
    
    def start(self) -> None:
        """Inicia la entrevista"""
        if self.status != InterviewStatus.CREATED:
            raise ValueError(f"Cannot start interview in {self.status} status")
        
        self.status = InterviewStatus.ACTIVE
        self.started_at = datetime.now()
    
    def pause(self) -> None:
        """Pausa la entrevista"""
        if self.status != InterviewStatus.ACTIVE:
            raise ValueError("Only active interviews can be paused")
        
        self.status = InterviewStatus.PAUSED
    
    def resume(self) -> None:
        """Reanuda la entrevista"""
        if self.status != InterviewStatus.PAUSED:
            raise ValueError("Only paused interviews can be resumed")
        
        self.status = InterviewStatus.ACTIVE
    
    def add_question(self, question: Question) -> None:
        """Agrega una nueva pregunta a la entrevista"""
        if not self.status.can_add_questions():
            raise ValueError(f"Cannot add questions in {self.status} status")
        
        self.current_question = question
        self.questions_history.append(question)
    
    def add_answer(self, answer: Answer) -> None:
        """
        Registra una respuesta del usuario.
        Actualiza automáticamente las métricas (streak, hints, etc.)
        """
        if self.status != InterviewStatus.ACTIVE:
            raise ValueError(f"Cannot add answers in {self.status} status")
        
        if not self.current_question:
            raise ValueError("No active question to answer")
        
        self.answers_history.append(answer)
        self.current_question = None  # Limpia pregunta actual
        
        # Actualizar métricas
        if answer.is_correct:
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
        else:
            self.current_streak = 0
        
        self.total_hints_used += answer.hints_used
    
    def complete(self) -> None:
        """Finaliza la entrevista"""
        if self.status not in (InterviewStatus.ACTIVE, InterviewStatus.PAUSED):
            raise ValueError(f"Cannot complete interview in {self.status} status")
        
        self.status = InterviewStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def cancel(self) -> None:
        """Cancela la entrevista"""
        if self.status == InterviewStatus.COMPLETED:
            raise ValueError("Cannot cancel a completed interview")
        
        self.status = InterviewStatus.CANCELLED
        self.completed_at = datetime.now()
    
    # Métodos de consulta
    
    def get_score_average(self) -> float:
        """Calcula el score promedio de la entrevista"""
        if not self.answers_history:
            return 0.0
        
        return sum(a.score for a in self.answers_history) / len(self.answers_history)
    
    def get_correct_count(self) -> int:
        """Cuenta respuestas correctas"""
        return sum(1 for a in self.answers_history if a.is_correct)
    
    def get_incorrect_count(self) -> int:
        """Cuenta respuestas incorrectas"""
        return len(self.answers_history) - self.get_correct_count()
    
    def get_accuracy(self) -> float:
        """Calcula el porcentaje de aciertos"""
        if not self.answers_history:
            return 0.0
        
        return (self.get_correct_count() / len(self.answers_history)) * 100
    
    def get_total_time(self) -> int:
        """Calcula el tiempo total en segundos"""
        return sum(a.time_taken for a in self.answers_history)
    
    def get_average_time_per_question(self) -> float:
        """Calcula el tiempo promedio por pregunta"""
        if not self.answers_history:
            return 0.0
        
        return self.get_total_time() / len(self.answers_history)
    
    def needs_difficulty_adjustment(self) -> bool:
        """
        Verifica si se debe ajustar la dificultad.
        Criterio: Después de 3 respuestas consecutivas correctas o incorrectas.
        """
        if len(self.answers_history) < 3:
            return False
        
        recent_3 = self.answers_history[-3:]
        all_correct = all(a.is_correct for a in recent_3)
        all_incorrect = all(not a.is_correct for a in recent_3)
        
        return all_correct or all_incorrect
    
    def should_increase_difficulty(self) -> bool:
        """Verifica si se debe aumentar la dificultad"""
        if len(self.answers_history) < 3:
            return False
        
        recent_3 = self.answers_history[-3:]
        return all(a.is_correct for a in recent_3)
    
    def should_decrease_difficulty(self) -> bool:
        """Verifica si se debe disminuir la dificultad"""
        if len(self.answers_history) < 3:
            return False
        
        recent_3 = self.answers_history[-3:]
        return all(not a.is_correct for a in recent_3)
    
    def is_complete(self) -> bool:
        """Verifica si la entrevista está completa"""
        return self.status == InterviewStatus.COMPLETED
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para serialización"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'role': self.role,
            'interview_type': self.interview_type,
            'status': self.status.value,
            'skill_level': self.skill_level.value,
            'mode': self.mode,
            'current_question': self.current_question.to_dict() if self.current_question else None,
            'questions_history': [q.to_dict() for q in self.questions_history],
            'answers_history': [a.to_dict() for a in self.answers_history],
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'context': self.context,
            'metadata': self.metadata,
            'current_streak': self.current_streak,
            'max_streak': self.max_streak,
            'total_hints_used': self.total_hints_used,
            'metrics': {
                'score_average': self.get_score_average(),
                'correct_count': self.get_correct_count(),
                'incorrect_count': self.get_incorrect_count(),
                'accuracy': self.get_accuracy(),
                'total_time': self.get_total_time(),
                'avg_time_per_question': self.get_average_time_per_question()
            }
        }
