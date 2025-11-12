"""
Repository Pattern Implementation.

Proporciona una abstracción mejorada para repositorios,
con soporte para múltiples implementaciones y registro dinámico.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type, Generic, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Repositorio base abstracto.
    
    Define la interfaz común para todos los repositorios.
    """
    
    @abstractmethod
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Busca una entidad por ID."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[T]:
        """Retorna todas las entidades."""
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Guarda una entidad."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Elimina una entidad por ID."""
        pass
    
    def get_repository_type(self) -> str:
        """Retorna el tipo de repositorio."""
        return self.__class__.__name__


class RepositoryRegistry:
    """
    Registro centralizado de repositorios.
    
    Permite registrar y obtener repositorios de forma dinámica.
    """
    
    def __init__(self):
        self._repositories: Dict[str, Type[BaseRepository]] = {}
        self._instances: Dict[str, BaseRepository] = {}
    
    def register(
        self,
        name: str,
        repository_class: Type[BaseRepository],
        singleton: bool = True
    ):
        """
        Registra un tipo de repositorio.
        
        Args:
            name: Nombre del repositorio
            repository_class: Clase del repositorio
            singleton: Si True, crea una única instancia
        """
        self._repositories[name] = repository_class
        logger.info(f"✅ Repository '{name}' registrado")
        
        if singleton:
            # Crear instancia singleton
            try:
                self._instances[name] = repository_class()
            except Exception as e:
                logger.warning(f"⚠️ No se pudo crear instancia singleton de '{name}': {e}")
    
    def get(
        self,
        name: str,
        **kwargs
    ) -> BaseRepository:
        """
        Obtiene un repositorio.
        
        Args:
            name: Nombre del repositorio
            **kwargs: Parámetros para crear instancia si no existe
            
        Returns:
            Instancia del repositorio
        """
        # Intentar obtener instancia existente
        if name in self._instances:
            return self._instances[name]
        
        # Crear nueva instancia
        if name not in self._repositories:
            raise ValueError(
                f"Repository '{name}' no registrado. "
                f"Disponibles: {list(self._repositories.keys())}"
            )
        
        repository_class = self._repositories[name]
        instance = repository_class(**kwargs)
        
        # Guardar como singleton si se creó sin parámetros
        if not kwargs:
            self._instances[name] = instance
        
        return instance
    
    def list_repositories(self) -> List[str]:
        """Lista repositorios disponibles."""
        return list(self._repositories.keys())


# Singleton instance
_repository_registry: Optional[RepositoryRegistry] = None


def get_repository_registry() -> RepositoryRegistry:
    """Retorna instancia singleton del registro de repositorios."""
    global _repository_registry
    if _repository_registry is None:
        _repository_registry = RepositoryRegistry()
        # Registrar repositorios por defecto
        from app.infrastructure.persistence.memory_interview_repository import MemoryInterviewRepository
        from app.infrastructure.persistence.json_question_repository import JsonQuestionRepository
        
        _repository_registry.register("memory_interview", MemoryInterviewRepository)
        _repository_registry.register("json_question", JsonQuestionRepository)
    return _repository_registry

