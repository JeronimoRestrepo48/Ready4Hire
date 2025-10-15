"""
Servicio LLM genérico - Abstracción para usar diferentes proveedores.
Por ahora solo Ollama, pero puede extenderse fácilmente.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .ollama_client import OllamaClient


class LLMService(ABC):
    """Interfaz abstracta para servicios LLM"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto dado un prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Genera respuesta en modo chat"""
        pass


class OllamaLLMService(LLMService):
    """
    Implementación de LLMService usando Ollama local.
    Esta es la implementación principal para Ready4Hire.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Inicializa el servicio Ollama.
        
        Args:
            base_url: URL de Ollama
            model: Modelo por defecto
            temperature: Temperatura por defecto
            max_tokens: Máximo de tokens por defecto
        """
        self.client = OllamaClient(
            base_url=base_url,
            default_model=model
        )
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Genera texto usando Ollama.
        
        Args:
            prompt: Prompt de entrada
            system: System prompt opcional
            temperature: Temperatura (None = usar default)
            max_tokens: Max tokens (None = usar default)
            **kwargs: Otros parámetros
        
        Returns:
            Texto generado
        """
        result = self.client.generate(
            prompt=prompt,
            system=system,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            stream=False,  # Siempre no-streaming para simplificar
            **kwargs
        )
        # result es Dict cuando stream=False
        if isinstance(result, dict):
            return result['response']
        # Fallback por si acaso
        return str(result)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Chat multi-turno usando Ollama.
        
        Args:
            messages: Lista de mensajes [{"role": "user/assistant", "content": "..."}]
            temperature: Temperatura (None = usar default)
            max_tokens: Max tokens (None = usar default)
            **kwargs: Otros parámetros
        
        Returns:
            Respuesta del asistente
        """
        result = self.client.chat(
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            **kwargs
        )
        # Consume the generator to get the full response
        return ''.join(result) if hasattr(result, '__iter__') and not isinstance(result, (str, dict)) else result.get('response', str(result))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas del cliente"""
        return self.client.get_metrics()
