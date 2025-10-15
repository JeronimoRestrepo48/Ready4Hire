"""
Repository Interface: Question Repository
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.entities.question import Question


class QuestionRepository(ABC):
    """
    Repositorio de preguntas.
    """
    
    @abstractmethod
    async def find_by_id(self, question_id: str) -> Optional[Question]:
        """Busca pregunta por ID"""
        pass
    
    @abstractmethod
    async def find_by_role(self, role: str, category: str = "technical") -> List[Question]:
        """Busca preguntas por rol y categoría"""
        pass
    
    @abstractmethod
    async def find_by_difficulty(self, difficulty: str, category: str = "technical") -> List[Question]:
        """Busca preguntas por dificultad"""
        pass
    
    @abstractmethod
    async def find_all_technical(self) -> List[Question]:
        """Obtiene todas las preguntas técnicas"""
        pass
    
    @abstractmethod
    async def find_all_soft_skills(self) -> List[Question]:
        """Obtiene todas las preguntas de soft skills"""
        pass
    
    @abstractmethod
    async def search(self, query: str, category: Optional[str] = None) -> List[Question]:
        """Busca preguntas por texto"""
        pass
