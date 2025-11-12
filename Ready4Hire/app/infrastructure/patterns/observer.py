"""
Observer Pattern Implementation.

Proporciona un sistema de eventos y notificaciones
para comunicación desacoplada entre componentes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Representa un evento en el sistema."""
    name: str
    data: Dict[str, Any]
    timestamp: datetime = None
    source: str = "system"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventObserver(ABC):
    """
    Observador abstracto de eventos.
    
    Los observadores se suscriben a eventos y reciben notificaciones.
    """
    
    @abstractmethod
    def on_event(self, event: Event) -> None:
        """
        Maneja un evento.
        
        Args:
            event: Evento recibido
        """
        pass
    
    def get_observed_events(self) -> List[str]:
        """
        Retorna lista de eventos que este observador quiere recibir.
        Si retorna [], observa todos los eventos.
        """
        return []


class EventPublisher:
    """
    Publicador de eventos.
    
    Permite registrar observadores y publicar eventos.
    """
    
    def __init__(self):
        self._observers: List[EventObserver] = []
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, observer: EventObserver):
        """Suscribe un observador."""
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(f"✅ Observer '{observer.__class__.__name__}' suscrito")
    
    def unsubscribe(self, observer: EventObserver):
        """Desuscribe un observador."""
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(f"❌ Observer '{observer.__class__.__name__}' desuscrito")
    
    def subscribe_handler(self, event_name: str, handler: Callable[[Event], None]):
        """Suscribe un handler específico para un evento."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
        logger.debug(f"✅ Handler registrado para evento '{event_name}'")
    
    def publish(self, event: Event):
        """
        Publica un evento.
        
        Args:
            event: Evento a publicar
        """
        # Notificar observadores
        observed_events = {}
        for observer in self._observers:
            events = observer.get_observed_events()
            if not events or event.name in events:
                try:
                    observer.on_event(event)
                    observed_events[observer.__class__.__name__] = True
                except Exception as e:
                    logger.error(f"Error en observer '{observer.__class__.__name__}': {e}")
                    observed_events[observer.__class__.__name__] = False
        
        # Ejecutar handlers específicos
        if event.name in self._event_handlers:
            for handler in self._event_handlers[event.name]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error en handler para '{event.name}': {e}")
        
        logger.debug(f"📢 Evento '{event.name}' publicado a {len(observed_events)} observadores")


class EventBus:
    """
    Bus de eventos global.
    
    Singleton que proporciona comunicación centralizada entre componentes.
    """
    
    _instance: Optional['EventBus'] = None
    _publisher: Optional[EventPublisher] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._publisher = EventPublisher()
        return cls._instance
    
    def publish(self, event_name: str, data: Dict[str, Any], source: str = "system"):
        """Publica un evento."""
        event = Event(name=event_name, data=data, source=source)
        self._publisher.publish(event)
    
    def subscribe(self, observer: EventObserver):
        """Suscribe un observador."""
        self._publisher.subscribe(observer)
    
    def unsubscribe(self, observer: EventObserver):
        """Desuscribe un observador."""
        self._publisher.unsubscribe(observer)
    
    def subscribe_handler(self, event_name: str, handler: Callable[[Event], None]):
        """Suscribe un handler específico."""
        self._publisher.subscribe_handler(event_name, handler)
    
    def get_publisher(self) -> EventPublisher:
        """Retorna el publicador interno."""
        return self._publisher


def get_event_bus() -> EventBus:
    """Retorna instancia singleton del bus de eventos."""
    return EventBus()

