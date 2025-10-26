"""
Optimizador de rendimiento para Ollama LLM.
Implementa técnicas para mejorar velocidad de respuesta.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache
import hashlib
import json

logger = logging.getLogger(__name__)


class OllamaPerformanceOptimizer:
    """Optimiza el rendimiento de llamadas a Ollama"""
    
    def __init__(self, cache_size: int = 128):
        """
        Inicializa el optimizador.
        
        Args:
            cache_size: Tamaño del cache LRU
        """
        self.cache_size = cache_size
        self._response_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def get_prompt_hash(self, prompt: str, model: str) -> str:
        """Genera hash único para un prompt y modelo"""
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_response(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene respuesta del cache si existe.
        
        Args:
            prompt: Prompt enviado al LLM
            model: Modelo utilizado
            
        Returns:
            Respuesta cacheada o None
        """
        cache_key = self.get_prompt_hash(prompt, model)
        
        if cache_key in self._response_cache:
            self._cache_hits += 1
            logger.info(f"⚡ Cache HIT para prompt (hits: {self._cache_hits})")
            return self._response_cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def cache_response(self, prompt: str, model: str, response: Dict[str, Any]) -> None:
        """
        Guarda respuesta en cache.
        
        Args:
            prompt: Prompt enviado
            model: Modelo utilizado
            response: Respuesta del LLM
        """
        cache_key = self.get_prompt_hash(prompt, model)
        
        # Limitar tamaño del cache
        if len(self._response_cache) >= self.cache_size:
            # Eliminar entrada más antigua (FIFO simple)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = response
        logger.debug(f"💾 Respuesta cacheada (total: {len(self._response_cache)})")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Dict con hits, misses y total
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._response_cache),
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    def clear_cache(self) -> None:
        """Limpia el cache de respuestas"""
        self._response_cache.clear()
        logger.info("🗑️ Cache limpiado")
    
    def optimize_prompt(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Optimiza el prompt para reducir tiempo de procesamiento.
        
        Args:
            prompt: Prompt original
            max_tokens: Límite de tokens para la respuesta
            
        Returns:
            Prompt optimizado
        """
        # Agregar instrucciones para respuestas concisas
        optimized = prompt
        
        if "evalúa" in prompt.lower() or "evaluate" in prompt.lower():
            optimized += "\n\nResponde de forma CONCISA y DIRECTA."
        
        return optimized
    
    async def batch_generate(
        self,
        prompts: List[str],
        model: str,
        llm_client: Any,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Genera respuestas para múltiples prompts en batch con concurrencia limitada.
        
        Args:
            prompts: Lista de prompts
            model: Modelo a usar
            llm_client: Cliente LLM
            max_concurrent: Máximo de requests concurrentes
            
        Returns:
            Lista de respuestas
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                # Verificar cache primero
                cached = self.get_cached_response(prompt, model)
                if cached:
                    return cached
                
                # Generar si no está en cache
                response = await llm_client.generate(prompt, model=model)
                self.cache_response(prompt, model, response)
                return response
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        logger.info(f"⚡ Batch generation completado: {len(prompts)} prompts")
        return results
    
    def get_recommended_params(self, hardware: str = "cpu") -> Dict[str, Any]:
        """
        Obtiene parámetros recomendados según el hardware.
        
        Args:
            hardware: Tipo de hardware ('cpu', 'gpu', 'gpu_high')
            
        Returns:
            Dict con parámetros optimizados
        """
        params = {
            "cpu": {
                "num_ctx": 2048,  # Contexto reducido para CPU
                "num_predict": 256,  # Tokens predichos reducidos
                "num_thread": 8,  # Threads para CPU
                "num_gpu": 0,
                "temperature": 0.3,  # Menos creativo = más rápido
                "top_p": 0.9,
                "top_k": 40,
            },
            "gpu": {
                "num_ctx": 4096,
                "num_predict": 512,
                "num_thread": 4,
                "num_gpu": 1,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
            },
            "gpu_high": {
                "num_ctx": 8192,
                "num_predict": 1024,
                "num_thread": 2,
                "num_gpu": 1,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
            }
        }
        
        return params.get(hardware, params["cpu"])


# Instancia global
_optimizer_instance: Optional[OllamaPerformanceOptimizer] = None


def get_performance_optimizer(cache_size: int = 128) -> OllamaPerformanceOptimizer:
    """Factory para obtener optimizador de rendimiento"""
    global _optimizer_instance
    
    if _optimizer_instance is None:
        _optimizer_instance = OllamaPerformanceOptimizer(cache_size=cache_size)
        logger.info("⚡ Performance optimizer initialized")
    
    return _optimizer_instance

