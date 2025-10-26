"""
Streaming Ollama Client.
Cliente con soporte para streaming de respuestas para mejor UX.
"""

import httpx
import json
import logging
from typing import AsyncIterator, Optional, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)


class StreamingOllamaClient:
    """
    Cliente Ollama con soporte para streaming.

    Features:
    - Streaming de respuestas en tiempo real
    - Mejor percepción de velocidad para el usuario
    - Compatible con Server-Sent Events (SSE)
    - Async generator para integración con FastAPI
    """

    def __init__(self, base_url: str = None, default_model: str = None, timeout: int = 60):
        """
        Inicializa el cliente streaming.

        Args:
            base_url: URL de Ollama
            default_model: Modelo por defecto
            timeout: Timeout en segundos
        """
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.default_model = default_model or settings.OLLAMA_MODEL
        self.timeout = timeout

        # Cliente HTTP con soporte para streaming
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

        logger.info(f"StreamingOllamaClient initialized: {self.base_url}")

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """
        Genera respuesta en streaming.

        Args:
            prompt: Prompt de entrada
            model: Modelo a usar (None = default)
            system: System prompt opcional
            temperature: Temperatura (0.0-1.0)
            max_tokens: Máximo de tokens

        Yields:
            Chunks de texto generados

        Example:
            async for chunk in client.generate_stream("¿Qué es Python?"):
                print(chunk, end="", flush=True)
        """
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": True,  # ⚡ Habilitar streaming
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        if system:
            payload["system"] = system

        try:
            # Realizar request con streaming
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"Ollama error {response.status_code}: {error_text}")

                # Leer chunks en streaming
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        # Parsear JSON de cada línea
                        chunk_data = json.loads(line)

                        # Extraer texto del chunk
                        if "response" in chunk_data:
                            text = chunk_data["response"]
                            if text:
                                yield text

                        # Verificar si es el último chunk
                        if chunk_data.get("done", False):
                            logger.debug("Streaming completed")
                            break

                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing streaming chunk: {e}")
                        continue

        except httpx.TimeoutException:
            logger.error("Streaming timeout")
            raise Exception("Timeout during streaming")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    async def generate_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Genera respuesta completa (recolecta todos los chunks).

        Args:
            prompt: Prompt de entrada
            model: Modelo a usar
            system: System prompt opcional
            temperature: Temperatura
            max_tokens: Máximo de tokens

        Returns:
            Texto generado completo
        """
        chunks = []

        async for chunk in self.generate_stream(
            prompt=prompt, model=model, system=system, temperature=temperature, max_tokens=max_tokens
        ):
            chunks.append(chunk)

        return "".join(chunks)

    async def close(self):
        """Cierra el cliente HTTP."""
        await self.client.aclose()
        logger.info("StreamingOllamaClient closed")


async def stream_to_sse(generator: AsyncIterator[str], event_type: str = "message") -> AsyncIterator[str]:
    """
    Convierte un async generator a Server-Sent Events (SSE) format.

    Args:
        generator: Generador async de chunks
        event_type: Tipo de evento SSE

    Yields:
        Eventos SSE formateados

    Example:
        async def sse_endpoint():
            chunks = client.generate_stream("prompt")
            return StreamingResponse(
                stream_to_sse(chunks),
                media_type="text/event-stream"
            )
    """
    try:
        async for chunk in generator:
            # Formato SSE: data: {chunk}\n\n
            yield f"event: {event_type}\n"
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        # Evento final
        yield f"event: done\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        # Evento de error
        yield f"event: error\n"
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# Global instance
_streaming_client: Optional[StreamingOllamaClient] = None


async def get_streaming_client() -> StreamingOllamaClient:
    """Obtiene la instancia global del cliente streaming."""
    global _streaming_client

    if _streaming_client is None:
        _streaming_client = StreamingOllamaClient()

    return _streaming_client
