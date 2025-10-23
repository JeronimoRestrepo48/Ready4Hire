"""
Redis Cache Implementation.
Cache persistente para evaluaciones y embeddings.
"""
import redis.asyncio as redis
import json
import hashlib
import logging
from typing import Optional, Any, Dict
from datetime import timedelta

from app.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Implementación de caché usando Redis.
    
    Features:
    - Async Redis client
    - TTL configurab por tipo de dato
    - Serialización JSON automática
    - Connection pooling
    - Manejo robusto de errores
    """
    
    def __init__(
        self,
        redis_url: str = None,
        default_ttl_seconds: int = 3600,
        max_connections: int = 10
    ):
        """
        Inicializa el cliente Redis.
        
        Args:
            redis_url: URL de Redis (ej: redis://localhost:6379/0)
            default_ttl_seconds: TTL por defecto en segundos
            max_connections: Máximo de conexiones en el pool
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = default_ttl_seconds
        self.max_connections = max_connections
        self.client: Optional[redis.Redis] = None
        logger.info(f"RedisCache initialized with URL: {self.redis_url}")
    
    async def connect(self):
        """Establece conexión con Redis."""
        if self.client is None:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.max_connections
            )
            logger.info("✅ Connected to Redis")
    
    async def disconnect(self):
        """Cierra la conexión con Redis."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Redis connection closed")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del caché.
        
        Args:
            key: Clave del caché
            
        Returns:
            Valor deserializado o None si no existe
        """
        if not self.client:
            await self.connect()
        
        try:
            value = await self.client.get(key)
            if value is None:
                return None
            
            # Deserializar JSON
            return json.loads(value)
        
        except Exception as e:
            logger.error(f"Error getting key '{key}' from Redis: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Guarda un valor en el caché.
        
        Args:
            key: Clave del caché
            value: Valor a guardar (será serializado a JSON)
            ttl_seconds: TTL en segundos (None = default)
            
        Returns:
            True si se guardó correctamente
        """
        if not self.client:
            await self.connect()
        
        try:
            # Serializar a JSON
            serialized = json.dumps(value)
            
            # Guardar con TTL
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            await self.client.setex(key, ttl, serialized)
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting key '{key}' in Redis: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Elimina una clave del caché.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si se eliminó
        """
        if not self.client:
            await self.connect()
        
        try:
            await self.client.delete(key)
            return True
        
        except Exception as e:
            logger.error(f"Error deleting key '{key}' from Redis: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Elimina todas las claves que coinciden con un patrón.
        
        Args:
            pattern: Patrón (ej: "evaluation:*")
            
        Returns:
            Número de claves eliminadas
        """
        if not self.client:
            await self.connect()
        
        try:
            keys = await self.client.keys(pattern)
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching '{pattern}'")
                return deleted
            return 0
        
        except Exception as e:
            logger.error(f"Error clearing pattern '{pattern}' from Redis: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Verifica si una clave existe.
        
        Args:
            key: Clave a verificar
            
        Returns:
            True si existe
        """
        if not self.client:
            await self.connect()
        
        try:
            return await self.client.exists(key) > 0
        
        except Exception as e:
            logger.error(f"Error checking key '{key}' in Redis: {e}")
            return False
    
    def make_key(self, *args, **kwargs) -> str:
        """
        Genera una clave de caché a partir de argumentos.
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Clave generada (hash MD5)
        """
        # Concatenar todos los argumentos
        parts = [str(arg) for arg in args]
        parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        
        # Generar hash MD5
        combined = "|".join(parts)
        hash_key = hashlib.md5(combined.encode()).hexdigest()
        
        return hash_key


class EvaluationCache:
    """
    Caché específico para evaluaciones de respuestas.
    Usa Redis como backend.
    """
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
        self.prefix = "evaluation:"
        self.ttl = 7 * 24 * 3600  # 7 días
    
    async def get(
        self,
        question: str,
        answer: str,
        model: str,
        temperature: float,
        **kwargs
    ) -> Optional[Dict]:
        """
        Obtiene una evaluación del caché.
        
        Args:
            question: Texto de la pregunta
            answer: Respuesta del candidato
            model: Modelo LLM usado
            temperature: Temperatura usada
            **kwargs: Otros parámetros
            
        Returns:
            Evaluación si existe en caché
        """
        key = self._make_key(question, answer, model, temperature, **kwargs)
        result = await self.cache.get(key)
        
        if result:
            logger.debug(f"✅ Evaluation cache HIT for key: {key[:16]}...")
        
        return result
    
    async def set(
        self,
        question: str,
        answer: str,
        model: str,
        temperature: float,
        evaluation: Dict,
        **kwargs
    ) -> bool:
        """
        Guarda una evaluación en el caché.
        
        Args:
            question: Texto de la pregunta
            answer: Respuesta del candidato
            model: Modelo LLM usado
            temperature: Temperatura usada
            evaluation: Resultado de la evaluación
            **kwargs: Otros parámetros
            
        Returns:
            True si se guardó
        """
        key = self._make_key(question, answer, model, temperature, **kwargs)
        success = await self.cache.set(key, evaluation, ttl_seconds=self.ttl)
        
        if success:
            logger.debug(f"✅ Evaluation cached with key: {key[:16]}...")
        
        return success
    
    def _make_key(
        self,
        question: str,
        answer: str,
        model: str,
        temperature: float,
        **kwargs
    ) -> str:
        """Genera clave de caché para una evaluación."""
        return self.prefix + self.cache.make_key(
            question,
            answer,
            model,
            temperature,
            **kwargs
        )


# Global instance
_redis_cache: Optional[RedisCache] = None
_evaluation_cache: Optional[EvaluationCache] = None


async def get_redis_cache() -> RedisCache:
    """Obtiene la instancia global de RedisCache."""
    global _redis_cache
    
    if _redis_cache is None:
        _redis_cache = RedisCache(
            redis_url=settings.REDIS_URL,
            default_ttl_seconds=settings.REDIS_CACHE_TTL_SECONDS,
            max_connections=settings.REDIS_MAX_CONNECTIONS
        )
        await _redis_cache.connect()
    
    return _redis_cache


async def get_evaluation_cache() -> EvaluationCache:
    """Obtiene la instancia global de EvaluationCache."""
    global _evaluation_cache
    
    if _evaluation_cache is None:
        redis_cache = await get_redis_cache()
        _evaluation_cache = EvaluationCache(redis_cache)
    
    return _evaluation_cache

