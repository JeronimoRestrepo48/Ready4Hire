"""
Cliente Ollama Asíncrono usando httpx.
Mucho más eficiente bajo carga que requests síncrono.
"""
import httpx
import json
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from .circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class AsyncOllamaClient:
    """
    Cliente asíncrono para Ollama con soporte de:
    - httpx async (no bloqueante)
    - Connection pooling
    - Circuit breaker
    - Retry con backoff
    - Métricas de latencia
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:3b",
        timeout: int = 30,
        max_retries: int = 2,
        circuit_breaker_enabled: bool = True,
        pool_limits: Optional[httpx.Limits] = None
    ):
        """
        Inicializa cliente Ollama asíncrono.
        
        Args:
            base_url: URL base de Ollama
            default_model: Modelo por defecto
            timeout: Timeout en segundos
            max_retries: Reintentos máximos
            circuit_breaker_enabled: Habilitar circuit breaker
            pool_limits: Límites de connection pool
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Connection pool limits (async)
        if pool_limits is None:
            pool_limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            )
        
        # Cliente HTTP asíncrono con connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=pool_limits
            # http2=True  # Deshabilitado temporalmente (requiere pip install httpx[http2])
        )
        
        # Circuit Breaker
        self.circuit_breaker_enabled = circuit_breaker_enabled
        if circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception,
                name="async_ollama_client"
            )
        else:
            self.circuit_breaker = None
        
        # Métricas
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'circuit_open_rejections': 0,
            'total_latency': 0.0,
            'avg_latency': 0.0
        }
        
        logger.info(f"✅ AsyncOllamaClient initialized (pool_size=50)")
    
    async def _check_health(self) -> bool:
        """Verifica que Ollama esté disponible (async)."""
        try:
            response = await self.client.get("/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"✅ Ollama conectado (async). Modelos: {[m['name'] for m in models]}")
                return True
            return False
        except Exception as e:
            logger.warning(f"⚠️ Ollama health check failed: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera texto usando Ollama (async).
        
        Args:
            prompt: Prompt de entrada
            model: Modelo a usar
            system: System prompt
            temperature: Temperatura
            max_tokens: Tokens máximos
        
        Returns:
            Dict con respuesta y métricas
        """
        # Circuit breaker check
        if self.circuit_breaker_enabled and self.circuit_breaker:
            try:
                return await asyncio.to_thread(
                    self.circuit_breaker.call,
                    self._generate_internal_sync,
                    prompt, model, system, temperature, max_tokens, **kwargs
                )
            except CircuitBreakerError as e:
                self.metrics['circuit_open_rejections'] += 1
                logger.error(f"🔴 Circuit OPEN: {e}")
                raise Exception(str(e)) from e
        else:
            return await self._generate_internal(
                prompt, model, system, temperature, max_tokens, **kwargs
            )
    
    def _generate_internal_sync(
        self,
        prompt: str,
        model: Optional[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ):
        """Wrapper síncrono para circuit breaker."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._generate_internal(prompt, model, system, temperature, max_tokens, **kwargs)
            )
        finally:
            loop.close()
    
    async def _generate_internal(
        self,
        prompt: str,
        model: Optional[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generación interna (async)."""
        model = model or self.default_model
        start_time = time.time()
        
        self.metrics['total_requests'] += 1
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        if system:
            payload["system"] = system
        
        # Retry con backoff exponencial
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post("/api/generate", json=payload)
                
                if response.status_code != 200:
                    self.metrics['failed_requests'] += 1
                    raise Exception(f"Ollama error {response.status_code}: {response.text}")
                
                result = response.json()
                latency = (time.time() - start_time) * 1000
                
                self.metrics['successful_requests'] += 1
                self.metrics['total_latency'] += latency
                self.metrics['avg_latency'] = (
                    self.metrics['total_latency'] / self.metrics['successful_requests']
                )
                
                return {
                    'response': result.get('response', ''),
                    'model': result.get('model', model),
                    'latency_ms': latency,
                    'total_duration': result.get('total_duration', 0),
                    'eval_count': result.get('eval_count', 0)
                }
            
            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = 0.5 * (2 ** attempt)
                    logger.warning(f"Timeout en intento {attempt + 1}/{self.max_retries}. "
                                 f"Reintentando en {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    self.metrics['failed_requests'] += 1
                    raise Exception(f"Timeout después de {self.max_retries} intentos") from e
            
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = 0.5 * (2 ** attempt)
                    logger.warning(f"Error en intento {attempt + 1}/{self.max_retries}: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.metrics['failed_requests'] += 1
                    raise
        
        if last_exception:
            raise last_exception
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Modo chat (async).
        
        Args:
            messages: Lista de mensajes
            model: Modelo a usar
            temperature: Temperatura
            max_tokens: Tokens máximos
        
        Returns:
            Dict con respuesta
        """
        model = model or self.default_model
        start_time = time.time()
        
        self.metrics['total_requests'] += 1
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        try:
            response = await self.client.post("/api/chat", json=payload)
            
            if response.status_code != 200:
                self.metrics['failed_requests'] += 1
                raise Exception(f"Ollama chat error {response.status_code}: {response.text}")
            
            result = response.json()
            latency = (time.time() - start_time) * 1000
            
            self.metrics['successful_requests'] += 1
            self.metrics['total_latency'] += latency
            self.metrics['avg_latency'] = (
                self.metrics['total_latency'] / self.metrics['successful_requests']
            )
            
            return {
                'response': result.get('message', {}).get('content', ''),
                'model': result.get('model', model),
                'latency_ms': latency,
                'role': result.get('message', {}).get('role', 'assistant')
            }
        
        except Exception as e:
            self.metrics['failed_requests'] += 1
            raise Exception(f"Error inesperado: {str(e)}") from e
    
    async def list_models(self) -> List[str]:
        """Lista modelos disponibles (async)."""
        try:
            response = await self.client.get("/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del cliente."""
        return self.metrics.copy()
    
    async def close(self):
        """Cierra el cliente HTTP."""
        await self.client.aclose()
        logger.info("AsyncOllamaClient closed")
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

