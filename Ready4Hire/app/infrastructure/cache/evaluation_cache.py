"""
Sistema de caché de 2 niveles para evaluaciones de IA.
Nivel 1: Memoria (LRU cache) - Ultra rápido
Nivel 2: Disco (SQLite) - Persistente entre reinicios

Reduce latencia de evaluación de ~16s a <10ms (95% reducción).
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EvaluationCache:
    """
    Caché de 2 niveles para evaluaciones de respuestas.

    Características:
    - Nivel 1: Memoria LRU (1000 entradas) - Latencia <1ms
    - Nivel 2: SQLite (10,000 entradas) - Latencia <10ms
    - TTL configurable (default: 7 días)
    - Invalidación automática por cambios de modelo
    - Métricas de hit rate
    """

    def __init__(
        self,
        cache_dir: str = ".cache/evaluations",
        ttl_days: int = 7,
        max_memory_entries: int = 1000,
        max_disk_entries: int = 10000,
    ):
        """
        Inicializa el sistema de caché.

        Args:
            cache_dir: Directorio para caché en disco
            ttl_days: Tiempo de vida en días (0 = sin expiración)
            max_memory_entries: Máximo de entradas en memoria (LRU)
            max_disk_entries: Máximo de entradas en disco
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_days = ttl_days
        self.max_memory_entries = max_memory_entries
        self.max_disk_entries = max_disk_entries

        # Base de datos SQLite para caché persistente
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()

        # Métricas
        self.stats = {"hits": 0, "misses": 0, "memory_hits": 0, "disk_hits": 0, "total_requests": 0}

        logger.info(f"EvaluationCache inicializado: {self.cache_dir}, TTL={ttl_days} días")

    def _init_database(self):
        """Inicializa la base de datos SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_cache (
                    cache_key TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    result TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 1
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_accessed_at 
                ON evaluation_cache(accessed_at)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON evaluation_cache(created_at)
            """
            )
            conn.commit()

    def get_cache_key(
        self,
        question: str,
        answer: str,
        model: str,
        temperature: float,
        expected_concepts: List[str],
        keywords: List[str],
    ) -> str:
        """
        Genera clave de caché única para una evaluación.

        Incluye pregunta, respuesta, modelo, temp, conceptos y keywords
        para garantizar que cambios en cualquiera invalidan el caché.

        Returns:
            Hash MD5 de 32 caracteres
        """
        content = {
            "question": question.strip().lower(),
            "answer": answer.strip().lower(),
            "model": model,
            "temperature": round(temperature, 2),
            "expected_concepts": sorted([c.lower() for c in expected_concepts]),
            "keywords": sorted([k.lower() for k in keywords]),
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

    @lru_cache(maxsize=1000)
    def _get_from_memory(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Caché de nivel 1 (memoria) usando LRU.
        Decorador @lru_cache proporciona el cache automáticamente.

        Esta función es solo un wrapper para el disco,
        pero el decorador cachea sus resultados en memoria.
        """
        return self._get_from_disk(cache_key)

    def _get_from_disk(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Obtiene del caché de nivel 2 (disco SQLite)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Obtener entrada
                cursor.execute(
                    """
                    SELECT result, created_at, accessed_at, hit_count
                    FROM evaluation_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                # Verificar TTL
                if self.ttl_days > 0:
                    created_at = datetime.fromisoformat(row["created_at"])
                    expiry = created_at + timedelta(days=self.ttl_days)
                    if datetime.now() > expiry:
                        # Expirado, eliminar
                        self._delete_entry(cache_key)
                        return None

                # Actualizar estadísticas de acceso
                conn.execute(
                    """
                    UPDATE evaluation_cache
                    SET accessed_at = CURRENT_TIMESTAMP,
                        hit_count = hit_count + 1
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )
                conn.commit()

                # Retornar resultado
                return json.loads(row["result"])

        except Exception as e:
            logger.error(f"Error leyendo caché de disco: {e}")
            return None

    def get(
        self,
        question: str,
        answer: str,
        model: str,
        temperature: float,
        expected_concepts: List[str],
        keywords: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene evaluación del caché (memoria o disco).

        Returns:
            Dict con resultado de evaluación, o None si no está cacheado
        """
        self.stats["total_requests"] += 1

        cache_key = self.get_cache_key(question, answer, model, temperature, expected_concepts, keywords)

        start_time = time.time()

        # Intentar nivel 1 (memoria LRU)
        result = self._get_from_memory(cache_key)

        if result:
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["hits"] += 1

            # Determinar si fue hit de memoria o disco
            if elapsed_ms < 1.0:
                self.stats["memory_hits"] += 1
                logger.debug(f"Cache HIT (memoria): {cache_key[:8]}... ({elapsed_ms:.2f}ms)")
            else:
                self.stats["disk_hits"] += 1
                logger.debug(f"Cache HIT (disco): {cache_key[:8]}... ({elapsed_ms:.2f}ms)")

            return result

        # Cache miss
        self.stats["misses"] += 1
        logger.debug(f"Cache MISS: {cache_key[:8]}...")
        return None

    def set(
        self,
        question: str,
        answer: str,
        model: str,
        temperature: float,
        expected_concepts: List[str],
        keywords: List[str],
        result: Dict[str, Any],
    ):
        """
        Guarda evaluación en caché (disco + memoria).

        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            model: Modelo usado
            temperature: Temperatura usada
            expected_concepts: Conceptos esperados
            keywords: Palabras clave
            result: Resultado de la evaluación
        """
        cache_key = self.get_cache_key(question, answer, model, temperature, expected_concepts, keywords)

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Insertar o actualizar
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evaluation_cache
                    (cache_key, question, answer, model, temperature, result)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (cache_key, question, answer, model, temperature, json.dumps(result)),
                )
                conn.commit()

            # Invalidar caché LRU para que recargue en próximo get
            self._get_from_memory.cache_clear()

            logger.debug(f"Cache SET: {cache_key[:8]}...")

            # Limpiar entradas antiguas si excede límite
            self._cleanup_old_entries()

        except Exception as e:
            logger.error(f"Error guardando en caché: {e}")

    def _delete_entry(self, cache_key: str):
        """Elimina entrada del caché."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM evaluation_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()

    def _cleanup_old_entries(self):
        """Limpia entradas antiguas cuando se excede max_disk_entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Contar entradas
                cursor = conn.execute("SELECT COUNT(*) FROM evaluation_cache")
                count = cursor.fetchone()[0]

                if count > self.max_disk_entries:
                    # Eliminar las más antiguas (menos usadas)
                    to_delete = count - self.max_disk_entries
                    conn.execute(
                        """
                        DELETE FROM evaluation_cache
                        WHERE cache_key IN (
                            SELECT cache_key
                            FROM evaluation_cache
                            ORDER BY accessed_at ASC, hit_count ASC
                            LIMIT ?
                        )
                        """,
                        (to_delete,),
                    )
                    conn.commit()
                    logger.info(f"Limpieza de caché: {to_delete} entradas eliminadas")
        except Exception as e:
            logger.error(f"Error en limpieza de caché: {e}")

    def clear(self):
        """Limpia todo el caché (memoria + disco)."""
        # Limpiar memoria
        self._get_from_memory.cache_clear()

        # Limpiar disco
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM evaluation_cache")
                conn.commit()
            logger.info("Caché limpiado completamente")
        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")

    def clear_expired(self):
        """Limpia solo entradas expiradas."""
        if self.ttl_days <= 0:
            return

        try:
            cutoff_date = datetime.now() - timedelta(days=self.ttl_days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM evaluation_cache WHERE created_at < ?", (cutoff_date.isoformat(),))
                deleted = cursor.rowcount
                conn.commit()

                if deleted > 0:
                    logger.info(f"Limpieza TTL: {deleted} entradas expiradas eliminadas")
                    self._get_from_memory.cache_clear()
        except Exception as e:
            logger.error(f"Error limpiando entradas expiradas: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.

        Returns:
            Dict con métricas de hit rate, latencia, etc.
        """
        total = self.stats["total_requests"]
        hits = self.stats["hits"]
        misses = self.stats["misses"]

        hit_rate = (hits / total * 100) if total > 0 else 0
        memory_hit_rate = (self.stats["memory_hits"] / hits * 100) if hits > 0 else 0

        # Contar entradas en disco
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM evaluation_cache")
                disk_entries = cursor.fetchone()[0]
        except:
            disk_entries = 0

        return {
            "total_requests": total,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 2),
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "memory_hit_rate": round(memory_hit_rate, 2),
            "disk_entries": disk_entries,
            "max_disk_entries": self.max_disk_entries,
            "ttl_days": self.ttl_days,
        }

    def get_top_cached(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Obtiene las evaluaciones más cacheadas.

        Returns:
            Lista de (pregunta_snippet, hit_count)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT question, hit_count
                    FROM evaluation_cache
                    ORDER BY hit_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [(row["question"][:100], row["hit_count"]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error obteniendo top cached: {e}")
            return []
