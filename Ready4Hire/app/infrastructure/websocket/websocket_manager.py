"""
WebSocket Manager para streaming de respuestas LLM en tiempo real.
Permite enviar tokens del LLM mientras se generan para mejor UX.
"""

import logging
import json
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Gestiona conexiones WebSocket activas.
    Permite broadcasting y comunicación 1-1.
    """
    
    def __init__(self):
        # {interview_id: Set[WebSocket]}
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # {websocket_id: interview_id} para lookup reverso
        self._ws_to_interview: Dict[int, str] = {}
    
    async def connect(self, websocket: WebSocket, interview_id: str) -> None:
        """
        Acepta una nueva conexión WebSocket.
        
        Args:
            websocket: Conexión WebSocket
            interview_id: ID de la entrevista
        """
        await websocket.accept()
        
        if interview_id not in self.active_connections:
            self.active_connections[interview_id] = set()
        
        self.active_connections[interview_id].add(websocket)
        self._ws_to_interview[id(websocket)] = interview_id
        
        logger.info(f"✅ WebSocket connected: interview={interview_id}, total={len(self.active_connections[interview_id])}")
        
        # Enviar mensaje de bienvenida
        await self.send_personal_message(
            {
                "type": "connected",
                "interview_id": interview_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Conexión establecida. Listo para recibir respuestas en tiempo real."
            },
            websocket
        )
    
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Desconecta un WebSocket.
        
        Args:
            websocket: Conexión a desconectar
        """
        ws_id = id(websocket)
        interview_id = self._ws_to_interview.get(ws_id)
        
        if interview_id and interview_id in self.active_connections:
            self.active_connections[interview_id].discard(websocket)
            
            # Si no quedan conexiones, eliminar la entrada
            if not self.active_connections[interview_id]:
                del self.active_connections[interview_id]
        
        if ws_id in self._ws_to_interview:
            del self._ws_to_interview[ws_id]
        
        logger.info(f"👋 WebSocket disconnected: interview={interview_id}")
    
    async def send_personal_message(
        self,
        message: Dict[str, Any],
        websocket: WebSocket
    ) -> None:
        """
        Envía un mensaje a un WebSocket específico.
        
        Args:
            message: Mensaje a enviar (será serializado a JSON)
            websocket: Conexión destino
        """
        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            logger.warning("WebSocket already disconnected")
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"❌ Error sending personal message: {e}")
    
    async def broadcast(self, interview_id: str, message: Dict[str, Any]) -> None:
        """
        Envía un mensaje a todos los WebSockets de una entrevista.
        
        Args:
            interview_id: ID de la entrevista
            message: Mensaje a broadcast
        """
        if interview_id not in self.active_connections:
            logger.warning(f"No active connections for interview {interview_id}")
            return
        
        # Copiar set para evitar modificación durante iteración
        connections = self.active_connections[interview_id].copy()
        
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                logger.warning(f"WebSocket disconnected during broadcast")
                self.disconnect(websocket)
            except Exception as e:
                logger.error(f"❌ Error in broadcast: {e}")
    
    async def stream_llm_response(
        self,
        interview_id: str,
        response_generator,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Streams respuesta del LLM token por token a los WebSockets.
        
        Args:
            interview_id: ID de la entrevista
            response_generator: Async generator que yielda tokens
            metadata: Metadata adicional a incluir
            
        Returns:
            Respuesta completa acumulada
        """
        full_response = ""
        metadata = metadata or {}
        
        try:
            # Enviar señal de inicio
            await self.broadcast(interview_id, {
                "type": "stream_start",
                "timestamp": datetime.utcnow().isoformat(),
                **metadata
            })
            
            # Stream de tokens
            async for token in response_generator:
                full_response += token
                
                await self.broadcast(interview_id, {
                    "type": "stream_token",
                    "token": token,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Small delay para evitar overflow de mensajes
                await asyncio.sleep(0.01)
            
            # Enviar señal de fin
            await self.broadcast(interview_id, {
                "type": "stream_end",
                "full_response": full_response,
                "timestamp": datetime.utcnow().isoformat(),
                **metadata
            })
            
            logger.info(f"✅ Streamed {len(full_response)} chars to interview {interview_id}")
            return full_response
            
        except Exception as e:
            logger.error(f"❌ Error streaming LLM response: {e}")
            
            # Enviar error a los clientes
            await self.broadcast(interview_id, {
                "type": "stream_error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return full_response
    
    async def send_typing_indicator(
        self,
        interview_id: str,
        is_typing: bool
    ) -> None:
        """
        Envía indicador de "AI está escribiendo...".
        
        Args:
            interview_id: ID de la entrevista
            is_typing: True si está escribiendo, False si terminó
        """
        await self.broadcast(interview_id, {
            "type": "typing",
            "is_typing": is_typing,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_progress(
        self,
        interview_id: str,
        current: int,
        total: int,
        message: Optional[str] = None
    ) -> None:
        """
        Envía indicador de progreso.
        
        Args:
            interview_id: ID de la entrevista
            current: Progreso actual
            total: Total
            message: Mensaje opcional
        """
        percentage = int((current / total) * 100) if total > 0 else 0
        
        await self.broadcast(interview_id, {
            "type": "progress",
            "current": current,
            "total": total,
            "percentage": percentage,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_notification(
        self,
        interview_id: str,
        notification_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Envía notificación (badge desbloqueado, logro, etc).
        
        Args:
            interview_id: ID de la entrevista
            notification_type: Tipo de notificación (badge, achievement, etc)
            message: Mensaje
            data: Datos adicionales
        """
        await self.broadcast(interview_id, {
            "type": "notification",
            "notification_type": notification_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_evaluation_result(
        self,
        interview_id: str,
        evaluation: Dict[str, Any]
    ) -> None:
        """
        Envía resultado de evaluación completa.
        
        Args:
            interview_id: ID de la entrevista
            evaluation: Resultado de evaluación
        """
        await self.broadcast(interview_id, {
            "type": "evaluation_complete",
            "evaluation": evaluation,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_active_connections_count(self, interview_id: Optional[str] = None) -> int:
        """
        Obtiene el número de conexiones activas.
        
        Args:
            interview_id: Si se provee, cuenta solo para esa entrevista
            
        Returns:
            Número de conexiones activas
        """
        if interview_id:
            return len(self.active_connections.get(interview_id, set()))
        
        return sum(len(conns) for conns in self.active_connections.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del WebSocket manager.
        
        Returns:
            Dict con estadísticas
        """
        return {
            "total_interviews": len(self.active_connections),
            "total_connections": self.get_active_connections_count(),
            "connections_by_interview": {
                iid: len(conns)
                for iid, conns in self.active_connections.items()
            }
        }


# Instancia global del connection manager
ws_manager = ConnectionManager()


def get_websocket_manager() -> ConnectionManager:
    """Factory para obtener el manager global de WebSockets"""
    return ws_manager

