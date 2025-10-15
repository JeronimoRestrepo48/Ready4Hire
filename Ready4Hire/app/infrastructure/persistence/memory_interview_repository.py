"""
Memory Implementation: Interview Repository
Almacenamiento en memoria con sesiones (compatible con código actual)
"""
from typing import Optional, List, Dict

from app.domain.entities.interview import Interview
from app.domain.repositories.interview_repository import InterviewRepository
from app.domain.value_objects.interview_status import InterviewStatus


class MemoryInterviewRepository(InterviewRepository):
    """
    Implementación en memoria del repositorio de entrevistas.
    Compatible con el sistema de sesiones actual.
    """
    
    def __init__(self):
        """
        Inicializa el repositorio en memoria.
        
        Crea estructuras de almacenamiento para entrevistas e índices por usuario.
        """
        self._storage: Dict[str, Interview] = {}
        self._user_index: Dict[str, List[str]] = {}  # user_id -> [interview_ids]
    
    async def save(self, interview: Interview) -> None:
        """Persiste o actualiza una entrevista en memoria"""
        self._storage[interview.id] = interview
        
        # Actualizar índice por usuario
        if interview.user_id not in self._user_index:
            self._user_index[interview.user_id] = []
        
        if interview.id not in self._user_index[interview.user_id]:
            self._user_index[interview.user_id].append(interview.id)
    
    async def find_by_id(self, interview_id: str) -> Optional[Interview]:
        """Busca entrevista por ID"""
        return self._storage.get(interview_id)
    
    async def find_active_by_user(self, user_id: str) -> Optional[Interview]:
        """Busca entrevista activa del usuario"""
        interview_ids = self._user_index.get(user_id, [])
        
        for interview_id in reversed(interview_ids):  # Más reciente primero
            interview = self._storage.get(interview_id)
            if interview and interview.status == InterviewStatus.ACTIVE:
                return interview
        
        return None
    
    async def find_all_by_user(self, user_id: str) -> List[Interview]:
        """Obtiene todas las entrevistas de un usuario"""
        interview_ids = self._user_index.get(user_id, [])
        interviews = []
        
        for interview_id in interview_ids:
            interview = self._storage.get(interview_id)
            if interview:
                interviews.append(interview)
        
        return sorted(interviews, key=lambda i: i.created_at, reverse=True)
    
    async def find_by_status(self, status: InterviewStatus) -> List[Interview]:
        """Obtiene todas las entrevistas con un estado específico"""
        return [
            interview 
            for interview in self._storage.values() 
            if interview.status == status
        ]
    
    async def delete(self, interview_id: str) -> None:
        """Elimina una entrevista"""
        interview = self._storage.get(interview_id)
        
        if interview:
            # Eliminar del storage
            del self._storage[interview_id]
            
            # Eliminar del índice de usuario
            if interview.user_id in self._user_index:
                if interview_id in self._user_index[interview.user_id]:
                    self._user_index[interview.user_id].remove(interview_id)
    
    async def exists(self, interview_id: str) -> bool:
        """Verifica si existe una entrevista"""
        return interview_id in self._storage
    
    def clear(self) -> None:
        """Limpia todo el storage (útil para tests)"""
        self._storage.clear()
        self._user_index.clear()
