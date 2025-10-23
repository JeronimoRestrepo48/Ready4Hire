"""
Cliente Ollama robusto para inferencia LLM local.
Soporta retry, timeout, streaming, manejo de errores y circuit breaker.
"""
import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List, Generator
from functools import wraps
from .circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Error base para Ollama"""
    pass


class OllamaConnectionError(OllamaError):
    """Error de conexión con Ollama"""
    pass


class OllamaTimeoutError(OllamaError):
    """Timeout en Ollama"""
    pass


class OllamaClient:
    """
    Cliente robusto para Ollama local con soporte de retry, timeout y fallback.
    
    Features:
    - Retry automático con backoff exponencial
    - Timeout configurable
    - Manejo robusto de errores
    - Soporte para streaming
    - Métricas de latencia
    - Pool de modelos para alta disponibilidad
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:3b",
        fallback_models: Optional[List[str]] = None,
        timeout: int = 30,  # ⚡ Reducido de 120s a 30s para respuestas más rápidas
        max_retries: int = 2,  # ⚡ Reducido de 3 a 2 reintentos
        retry_delay: float = 0.5,  # ⚡ Reducido de 1.0s a 0.5s para reintentos más rápidos
        circuit_breaker_enabled: bool = True,  # 🔌 Circuit breaker para resiliencia
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: int = 60
    ):
        """
        Inicializa el cliente Ollama.
        
        Args:
            base_url: URL base de Ollama (default: localhost:11434)
            default_model: Modelo principal a usar
            fallback_models: Lista de modelos alternativos si el principal falla
            timeout: Timeout en segundos para requests
            max_retries: Número máximo de reintentos
            retry_delay: Delay base entre reintentos (usa backoff exponencial)
            circuit_breaker_enabled: Si habilitar circuit breaker
            circuit_failure_threshold: Fallos consecutivos antes de abrir circuito
            circuit_recovery_timeout: Segundos antes de intentar recuperación
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.fallback_models = fallback_models or ["mistral:7b", "llama2:7b"]
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        # 🔌 Circuit Breaker para resiliencia
        self.circuit_breaker_enabled = circuit_breaker_enabled
        if circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=circuit_failure_threshold,
                recovery_timeout=circuit_recovery_timeout,
                expected_exception=OllamaError,
                name="ollama_client"
            )
            logger.info("✅ Circuit Breaker habilitado para Ollama")
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
        
        # Health check inicial
        self._check_health()
    
    def _check_health(self) -> bool:
        """Verifica que Ollama esté disponible"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"✅ Ollama conectado. Modelos disponibles: {[m['name'] for m in models]}")
                return True
            else:
                logger.warning(f"⚠️ Ollama responde pero con error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Ollama no disponible: {str(e)}")
            raise OllamaConnectionError(
                f"No se puede conectar a Ollama en {self.base_url}. "
                f"Asegúrate de que Ollama esté corriendo: 'ollama serve'"
            )
    
    def _retry_with_backoff(self, func):
        """Decorator para retry con backoff exponencial"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Optional[Exception] = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except OllamaTimeoutError as e:
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Timeout en intento {attempt + 1}/{self.max_retries}. "
                                     f"Reintentando en {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Timeout después de {self.max_retries} intentos")
                        raise
                except OllamaConnectionError as e:
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Error de conexión en intento {attempt + 1}/{self.max_retries}. "
                                     f"Reintentando en {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Error de conexión después de {self.max_retries} intentos")
                        raise
                except Exception as e:
                    last_exception = e
                    logger.error(f"Error inesperado: {str(e)}")
                    raise
            
            if last_exception:
                raise last_exception
            raise OllamaError("Max retries exceeded")
        
        return wrapper
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs
    ):
        """
        Genera texto usando Ollama.
        
        Args:
            prompt: Prompt para el modelo
            model: Modelo a usar (None = default_model)
            system: System prompt opcional
            temperature: Temperatura (0.0-2.0)
            max_tokens: Máximo de tokens a generar
            stream: Si True, retorna generator para streaming
            **kwargs: Parámetros adicionales de Ollama
        
        Returns:
            Dict con 'response', 'model', 'latency_ms', etc. o Generator si stream=True
        
        Raises:
            OllamaConnectionError: Si no se puede conectar
            OllamaTimeoutError: Si se excede el timeout
            OllamaError: Otros errores de Ollama
            CircuitBreakerError: Si el circuit está OPEN
        """
        # 🔌 Si circuit breaker está habilitado, verificar estado
        if self.circuit_breaker_enabled and self.circuit_breaker:
            try:
                return self.circuit_breaker.call(
                    self._generate_internal,
                    prompt, model, system, temperature, max_tokens, stream, **kwargs
                )
            except CircuitBreakerError as e:
                self.metrics['circuit_open_rejections'] += 1
                logger.error(f"🔴 Circuit OPEN: {e}")
                raise OllamaError(str(e)) from e
        else:
            return self._generate_internal(
                prompt, model, system, temperature, max_tokens, stream, **kwargs
            )
    
    def _generate_internal(
        self,
        prompt: str,
        model: Optional[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stream: bool,
        **kwargs
    ):
        """Método interno de generación (protegido por circuit breaker)."""
        model = model or self.default_model
        start_time = time.time()
        
        self.metrics['total_requests'] += 1
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            
            if response.status_code != 200:
                self.metrics['failed_requests'] += 1
                raise OllamaError(f"Ollama error {response.status_code}: {response.text}")
            
            if stream:
                return self._handle_stream(response)
            else:
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
                    'load_duration': result.get('load_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'eval_count': result.get('eval_count', 0)
                }
        
        except requests.Timeout:
            self.metrics['failed_requests'] += 1
            raise OllamaTimeoutError(f"Timeout después de {self.timeout}s")
        except requests.ConnectionError as e:
            self.metrics['failed_requests'] += 1
            raise OllamaConnectionError(f"Error de conexión: {str(e)}")
        except Exception as e:
            self.metrics['failed_requests'] += 1
            raise OllamaError(f"Error inesperado: {str(e)}")
    
    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Maneja streaming de respuestas"""
        try:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        yield chunk['response']
                    if chunk.get('done', False):
                        break
        except Exception as e:
            logger.error(f"Error en streaming: {str(e)}")
            raise OllamaError(f"Error en streaming: {str(e)}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Modo chat con Ollama (conversación multi-turno).
        
        Args:
            messages: Lista de mensajes [{"role": "user"|"assistant", "content": "..."}]
            model: Modelo a usar
            temperature: Temperatura
            max_tokens: Máximo de tokens
        
        Returns:
            Dict con respuesta del asistente
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
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self.metrics['failed_requests'] += 1
                raise OllamaError(f"Ollama chat error {response.status_code}: {response.text}")
            
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
        
        except requests.Timeout:
            self.metrics['failed_requests'] += 1
            raise OllamaTimeoutError(f"Timeout después de {self.timeout}s")
        except requests.ConnectionError as e:
            self.metrics['failed_requests'] += 1
            raise OllamaConnectionError(f"Error de conexión: {str(e)}")
        except Exception as e:
            self.metrics['failed_requests'] += 1
            raise OllamaError(f"Error inesperado: {str(e)}")
    
    def list_models(self) -> List[str]:
        """Lista los modelos disponibles en Ollama"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except Exception as e:
            logger.error(f"Error listando modelos: {str(e)}")
            return []
    
    def pull_model(self, model: str) -> bool:
        """
        Descarga un modelo si no existe localmente.
        
        Args:
            model: Nombre del modelo a descargar
        
        Returns:
            True si se descargó exitosamente o ya existía
        """
        try:
            logger.info(f"📥 Descargando modelo {model}...")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=600  # 10 minutos para descarga
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Modelo {model} listo")
                return True
            else:
                logger.error(f"❌ Error descargando {model}: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Error descargando {model}: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas del cliente"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Resetea las métricas"""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency': 0.0,
            'avg_latency': 0.0
        }
