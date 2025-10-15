"""
Repository Interface: Interview Repository
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from app.domain.entities.interview import Interview
from app.domain.value_objects.interview_status import InterviewStatus


class InterviewRepository(ABC):
    """
    Repositorio de entrevistas.
    Define el contrato para persistencia sin implementación concreta.
    """
    
    @abstractmethod
    async def save(self, interview: Interview) -> None:
        """Persiste o actualiza una entrevista"""
        pass
    
    @abstractmethod
    async def find_by_id(self, interview_id: str) -> Optional[Interview]:
        """Busca entrevista por ID"""
        pass
    
    @abstractmethod
    async def find_active_by_user(self, user_id: str) -> Optional[Interview]:
        """Busca entrevista activa del usuario"""
        pass
    
    @abstractmethod
    async def find_all_by_user(self, user_id: str) -> List[Interview]:
        """Obtiene todas las entrevistas de un usuario"""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: InterviewStatus) -> List[Interview]:
        """Obtiene todas las entrevistas con un estado específico"""
        pass
    
    @abstractmethod
    async def delete(self, interview_id: str) -> None:
        """Elimina una entrevista"""
        pass
    
    @abstractmethod
    async def exists(self, interview_id: str) -> bool:
        """Verifica si existe una entrevista"""
        pass
