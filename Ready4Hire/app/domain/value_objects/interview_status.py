"""
Value Object: Estado de la entrevista
"""
from enum import Enum


class InterviewStatus(str, Enum):
    """Estados posibles de una entrevista"""
    
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    
    def is_active(self) -> bool:
        """Verifica si la entrevista está activa"""
        return self == InterviewStatus.ACTIVE
    
    def can_add_questions(self) -> bool:
        """Verifica si se pueden agregar preguntas"""
        return self in (InterviewStatus.ACTIVE, InterviewStatus.PAUSED)
