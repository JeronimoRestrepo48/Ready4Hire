"""
Módulo de caché para optimizar rendimiento de IA.
Implementa caché de 2 niveles (memoria + disco).
"""

from .evaluation_cache import EvaluationCache

__all__ = ["EvaluationCache"]
