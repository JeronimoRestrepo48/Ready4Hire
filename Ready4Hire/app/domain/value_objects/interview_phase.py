"""
Value Object: Interview Phase
Define las fases de una entrevista
"""
from enum import Enum


class InterviewPhase(Enum):
    """
    Fases de la entrevista.
    
    CONFIGURATION: Configurando parámetros de la entrevista
    CONTEXT: Recopilando información contextual del candidato
    QUESTIONS: Fase principal de preguntas
    COMPLETED: Entrevista finalizada
    """
    CONFIGURATION = "configuration"
    CONTEXT = "context"
    QUESTIONS = "questions"
    COMPLETED = "completed"
    
    @classmethod
    def from_string(cls, value: str) -> 'InterviewPhase':
        """Convierte string a InterviewPhase"""
        mapping = {
            "configuration": cls.CONFIGURATION,
            "context": cls.CONTEXT,
            "questions": cls.QUESTIONS,
            "completed": cls.COMPLETED
        }
        return mapping.get(value.lower(), cls.CONFIGURATION)
    
    def __str__(self) -> str:
        return self.value
