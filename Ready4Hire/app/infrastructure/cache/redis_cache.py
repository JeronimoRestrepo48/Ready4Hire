"""
Redis Cache Service para cache distribuido y persistente.
Reemplaza el cache en memoria local con Redis para escalabilidad.
"""

import json
import hashlib
import logging
from typing import Any, Optional, Dict, List
from datetime import timedelta
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

logger = logging.getLogger(__name__)


class RedisCacheService:
    """
    Servicio de cache distribuido usando Redis.
    
    Características:
    - Cache persistente (sobrevive reinicios)
    - Distribuido (múltiples instancias backend)
    - TTL configurable por tipo de dato
    - Serialización automática JSON
    - Manejo de errores robusto
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True,
        max_connections: int = 50
    ):
        """
        Inicializa el servicio de cache Redis.
        
        Args:
            host: Host de Redis
            port: Puerto de Redis
            db: Database number (0-15)
            password: Password de Redis (si aplica)
            decode_responses: Auto-decode bytes a strings
            max_connections: Máximo de conexiones en el pool
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._client: Optional[Redis] = None
        self._pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            max_connections=max_connections
        )
        
        # TTLs por tipo de cache
        self.ttls = {
            "evaluation": timedelta(days=7),
            "embedding": timedelta(days=30),
            "session": timedelta(hours=24),
            "question": timedelta(days=90),
            "rate_limit": timedelta(minutes=1),
            "user_stats": timedelta(hours=1),
        }
        
        # Prefixes para namespacing
        self.prefixes = {
            "evaluation": "eval:",
            "embedding": "embed:",
            "session": "session:",
            "question": "question:",
            "rate_limit": "rl:",
            "user_stats": "stats:",
        }
    
    async def connect(self) -> None:
        """Establece conexión con Redis"""
        try:
            self._client = Redis(connection_pool=self._pool)
            await self._client.ping()
            logger.info(f"✅ Redis connected: {self.host}:{self.port} (db={self.db})")
        except RedisConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Cierra conexión con Redis"""
        if self._client:
            await self._client.aclose()
            logger.info("👋 Redis disconnected")
    
    def _get_key(self, cache_type: str, key: str) -> str:
        """Genera key con prefix y namespace"""
        prefix = self.prefixes.get(cache_type, "cache:")
        return f"{prefix}{key}"
    
    def _hash_key(self, data: Any) -> str:
        """Genera hash MD5 de datos para usar como key"""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        elif not isinstance(data, str):
            data = str(data)
        return hashlib.md5(data.encode()).hexdigest()
    
    async def set(
        self,
        cache_type: str,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Guarda un valor en cache.
        
        Args:
            cache_type: Tipo de cache (evaluation, embedding, etc)
            key: Key única
            value: Valor a cachear (será serializado a JSON)
            ttl: Time to live (usa default si no se provee)
            
        Returns:
            True si se guardó exitosamente
        """
        if not self._client:
            logger.warning("Redis not connected, skipping cache set")
            return False
        
        try:
            cache_key = self._get_key(cache_type, key)
            ttl = ttl or self.ttls.get(cache_type, timedelta(hours=1))
            
            # Serializar a JSON
            serialized = json.dumps(value, default=str)
            
            # Guardar con TTL
            await self._client.setex(
                cache_key,
                int(ttl.total_seconds()),
                serialized
            )
            
            logger.debug(f"💾 Cached [{cache_type}] {key} (TTL: {ttl})")
            return True
            
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"❌ Cache set error: {e}")
            return False
    
    async def get(self, cache_type: str, key: str) -> Optional[Any]:
        """
        Recupera un valor del cache.
        
        Args:
            cache_type: Tipo de cache
            key: Key única
            
        Returns:
            Valor deserializado o None si no existe
        """
        if not self._client:
            logger.warning("Redis not connected, skipping cache get")
            return None
        
        try:
            cache_key = self._get_key(cache_type, key)
            value = await self._client.get(cache_key)
            
            if value is None:
                logger.debug(f"❌ Cache MISS [{cache_type}] {key}")
                return None
            
            # Deserializar JSON
            deserialized = json.loads(value)
            logger.debug(f"✅ Cache HIT [{cache_type}] {key}")
            return deserialized
            
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"❌ Cache get error: {e}")
            return None
    
    async def delete(self, cache_type: str, key: str) -> bool:
        """Elimina un valor del cache"""
        if not self._client:
            return False
        
        try:
            cache_key = self._get_key(cache_type, key)
            deleted = await self._client.delete(cache_key)
            logger.debug(f"🗑️ Deleted [{cache_type}] {key}")
            return deleted > 0
        except RedisError as e:
            logger.error(f"❌ Cache delete error: {e}")
            return False
    
    async def exists(self, cache_type: str, key: str) -> bool:
        """Verifica si existe una key en cache"""
        if not self._client:
            return False
        
        try:
            cache_key = self._get_key(cache_type, key)
            return await self._client.exists(cache_key) > 0
        except RedisError as e:
            logger.error(f"❌ Cache exists error: {e}")
            return False
    
    async def increment(
        self,
        cache_type: str,
        key: str,
        amount: int = 1,
        ttl: Optional[timedelta] = None
    ) -> int:
        """
        Incrementa un contador (útil para rate limiting).
        
        Args:
            cache_type: Tipo de cache
            key: Key única
            amount: Cantidad a incrementar
            ttl: TTL para el contador
            
        Returns:
            Valor después del incremento
        """
        if not self._client:
            return 0
        
        try:
            cache_key = self._get_key(cache_type, key)
            new_value = await self._client.incrby(cache_key, amount)
            
            # Si es primera vez, establecer TTL
            if new_value == amount and ttl:
                await self._client.expire(cache_key, int(ttl.total_seconds()))
            
            return new_value
        except RedisError as e:
            logger.error(f"❌ Cache increment error: {e}")
            return 0
    
    async def get_many(self, cache_type: str, keys: List[str]) -> Dict[str, Any]:
        """
        Recupera múltiples valores del cache de forma eficiente.
        
        Args:
            cache_type: Tipo de cache
            keys: Lista de keys
            
        Returns:
            Dict con {key: value} para keys encontradas
        """
        if not self._client or not keys:
            return {}
        
        try:
            cache_keys = [self._get_key(cache_type, k) for k in keys]
            values = await self._client.mget(cache_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to deserialize {key}")
            
            logger.debug(f"✅ Batch get: {len(result)}/{len(keys)} hits")
            return result
            
        except RedisError as e:
            logger.error(f"❌ Cache get_many error: {e}")
            return {}
    
    async def set_many(
        self,
        cache_type: str,
        items: Dict[str, Any],
        ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Guarda múltiples valores de forma eficiente.
        
        Args:
            cache_type: Tipo de cache
            items: Dict con {key: value}
            ttl: TTL para todos los items
            
        Returns:
            True si se guardaron exitosamente
        """
        if not self._client or not items:
            return False
        
        try:
            ttl = ttl or self.ttls.get(cache_type, timedelta(hours=1))
            
            # Serializar todos los valores
            serialized_items = {}
            for key, value in items.items():
                cache_key = self._get_key(cache_type, key)
                serialized_items[cache_key] = json.dumps(value, default=str)
            
            # Guardar en batch
            await self._client.mset(serialized_items)
            
            # Establecer TTL para cada key (en batch)
            if ttl:
                pipeline = self._client.pipeline()
                ttl_seconds = int(ttl.total_seconds())
                for cache_key in serialized_items.keys():
                    pipeline.expire(cache_key, ttl_seconds)
                await pipeline.execute()
            
            logger.debug(f"💾 Batch set: {len(items)} items")
            return True
            
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"❌ Cache set_many error: {e}")
            return False
    
    async def clear_pattern(self, cache_type: str, pattern: str = "*") -> int:
        """
        Limpia todas las keys que coincidan con un patrón.
        
        Args:
            cache_type: Tipo de cache
            pattern: Patrón de búsqueda (wildcards permitidos)
            
        Returns:
            Número de keys eliminadas
        """
        if not self._client:
            return 0
        
        try:
            full_pattern = self._get_key(cache_type, pattern)
            keys = []
            
            # Buscar keys que coincidan
            async for key in self._client.scan_iter(match=full_pattern):
                keys.append(key)
            
            if keys:
                deleted = await self._client.delete(*keys)
                logger.info(f"🗑️ Cleared {deleted} keys matching {full_pattern}")
                return deleted
            
            return 0
            
        except RedisError as e:
            logger.error(f"❌ Cache clear_pattern error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache Redis.
        
        Returns:
            Dict con estadísticas de uso
        """
        if not self._client:
            return {"connected": False}
        
        try:
            info = await self._client.info("stats")
            memory_info = await self._client.info("memory")
            
            # Contar keys por tipo
            key_counts = {}
            for cache_type in self.prefixes.keys():
                pattern = self._get_key(cache_type, "*")
                count = 0
                async for _ in self._client.scan_iter(match=pattern):
                    count += 1
                key_counts[cache_type] = count
            
            return {
                "connected": True,
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate_percent": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "memory_used_mb": round(memory_info.get("used_memory", 0) / 1024 / 1024, 2),
                "keys_by_type": key_counts,
                "total_keys": sum(key_counts.values())
            }
            
        except RedisError as e:
            logger.error(f"❌ Failed to get cache stats: {e}")
            return {"connected": False, "error": str(e)}
    
    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        """Calcula tasa de hits del cache"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)


# Factory para obtener instancia global
_redis_cache_instance: Optional[RedisCacheService] = None


async def get_redis_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0
) -> RedisCacheService:
    """
    Factory para obtener instancia del cache Redis.
    Usa singleton pattern para reutilizar conexión.
    """
    global _redis_cache_instance
    
    if _redis_cache_instance is None:
        _redis_cache_instance = RedisCacheService(host=host, port=port, db=db)
        await _redis_cache_instance.connect()
        logger.info("⚡ Redis cache service initialized")
    
    return _redis_cache_instance


async def close_redis_cache() -> None:
    """Cierra la conexión global de Redis"""
    global _redis_cache_instance
    
    if _redis_cache_instance:
        await _redis_cache_instance.disconnect()
        _redis_cache_instance = None
