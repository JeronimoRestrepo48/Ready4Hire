"""
A/B Testing Framework para experimentación con variantes.
Permite testear diferentes features, prompts, y UX de forma científica.
"""

import logging
import hashlib
import random
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class VariantType(Enum):
    """Tipos de variantes"""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_C = "variant_c"


@dataclass
class Experiment:
    """
    Definición de un experimento A/B.
    """
    name: str
    description: str
    variants: Dict[str, float]  # {"control": 0.5, "variant_a": 0.5}
    active: bool = True
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    target_metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validar que las probabilidades sumen 1.0
        total = sum(self.variants.values())
        if not (0.99 <= total <= 1.01):  # Tolerancia para float precision
            raise ValueError(f"Variant probabilities must sum to 1.0, got {total}")


@dataclass
class ExperimentResult:
    """Resultado de una métrica en un experimento"""
    experiment_name: str
    variant: str
    metric_name: str
    metric_value: float
    user_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ABTestingFramework:
    """
    Framework completo de A/B Testing.
    
    Features:
    - Assignment consistente de usuarios a variantes
    - Tracking de métricas por variante
    - Análisis estadístico básico
    - Persistencia en DB
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # {user_id: {exp_name: variant}}
        self.results: List[ExperimentResult] = []
        
        logger.info("✅ A/B Testing Framework initialized")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: Dict[str, float],
        target_metrics: Optional[List[str]] = None,
        duration_days: Optional[int] = None
    ) -> Experiment:
        """
        Crea un nuevo experimento.
        
        Args:
            name: Nombre único del experimento
            description: Descripción del experimento
            variants: Dict con variantes y sus probabilidades
            target_metrics: Métricas a trackear
            duration_days: Duración del experimento en días
            
        Returns:
            Experiment creado
            
        Example:
            >>> ab.create_experiment(
            ...     name="evaluation_prompt_v2",
            ...     description="Test new evaluation prompt",
            ...     variants={"control": 0.5, "variant_a": 0.5},
            ...     target_metrics=["evaluation_score", "evaluation_time"],
            ...     duration_days=14
            ... )
        """
        end_date = None
        if duration_days:
            end_date = datetime.utcnow() + timedelta(days=duration_days)
        
        experiment = Experiment(
            name=name,
            description=description,
            variants=variants,
            target_metrics=target_metrics or [],
            end_date=end_date
        )
        
        self.experiments[name] = experiment
        logger.info(f"✅ Experiment created: {name} with variants {list(variants.keys())}")
        
        return experiment
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """
        Obtiene la variante asignada a un usuario para un experimento.
        El assignment es consistente (mismo user_id siempre obtiene misma variante).
        
        Args:
            experiment_name: Nombre del experimento
            user_id: ID del usuario
            
        Returns:
            Nombre de la variante asignada
            
        Raises:
            ValueError: Si el experimento no existe
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        experiment = self.experiments[experiment_name]
        
        # Verificar si ya tiene assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        
        if experiment_name not in self.user_assignments[user_id]:
            # Asignar variante de forma consistente basada en hash
            variant = self._assign_variant(experiment_name, user_id, experiment.variants)
            self.user_assignments[user_id][experiment_name] = variant
            logger.debug(f"User {user_id} assigned to variant '{variant}' for experiment '{experiment_name}'")
        
        return self.user_assignments[user_id][experiment_name]
    
    def _assign_variant(
        self,
        experiment_name: str,
        user_id: str,
        variants: Dict[str, float]
    ) -> str:
        """
        Asigna una variante de forma determinística basada en hash.
        
        Args:
            experiment_name: Nombre del experimento
            user_id: ID del usuario
            variants: Dict de variantes con probabilidades
            
        Returns:
            Variante asignada
        """
        # Generar hash consistente
        hash_input = f"{experiment_name}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Normalizar a rango [0, 1)
        normalized = (hash_value % 10000) / 10000.0
        
        # Asignar basado en probabilidades acumuladas
        cumulative = 0.0
        for variant, probability in variants.items():
            cumulative += probability
            if normalized < cumulative:
                return variant
        
        # Fallback (no debería llegar aquí si probabilidades suman 1.0)
        return list(variants.keys())[0]
    
    def track_metric(
        self,
        experiment_name: str,
        user_id: str,
        metric_name: str,
        metric_value: float
    ) -> None:
        """
        Trackea una métrica para un usuario en un experimento.
        
        Args:
            experiment_name: Nombre del experimento
            user_id: ID del usuario
            metric_name: Nombre de la métrica
            metric_value: Valor de la métrica
            
        Example:
            >>> ab.track_metric(
            ...     "evaluation_prompt_v2",
            ...     "user_123",
            ...     "evaluation_score",
            ...     8.5
            ... )
        """
        if experiment_name not in self.experiments:
            logger.warning(f"Experiment '{experiment_name}' not found, skipping metric")
            return
        
        variant = self.get_variant(experiment_name, user_id)
        
        result = ExperimentResult(
            experiment_name=experiment_name,
            variant=variant,
            metric_name=metric_name,
            metric_value=metric_value,
            user_id=user_id
        )
        
        self.results.append(result)
        logger.debug(f"Tracked metric: {metric_name}={metric_value} for {variant}")
    
    def get_experiment_results(
        self,
        experiment_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene resultados agregados de un experimento.
        
        Args:
            experiment_name: Nombre del experimento
            
        Returns:
            Dict con estadísticas por variante:
            {
                "control": {
                    "users_count": 100,
                    "metrics": {
                        "evaluation_score": {"mean": 7.5, "std": 1.2, "count": 100},
                        "evaluation_time": {"mean": 25.3, "std": 5.1, "count": 100}
                    }
                },
                "variant_a": {...}
            }
        """
        if experiment_name not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_name]
        
        # Filtrar resultados del experimento
        exp_results = [r for r in self.results if r.experiment_name == experiment_name]
        
        # Agrupar por variante
        results_by_variant: Dict[str, Dict[str, Any]] = {}
        
        for variant in experiment.variants.keys():
            variant_results = [r for r in exp_results if r.variant == variant]
            
            # Contar usuarios únicos
            unique_users = set(r.user_id for r in variant_results)
            
            # Calcular estadísticas por métrica
            metrics_stats = {}
            for metric_name in experiment.target_metrics:
                metric_values = [
                    r.metric_value
                    for r in variant_results
                    if r.metric_name == metric_name
                ]
                
                if metric_values:
                    metrics_stats[metric_name] = {
                        "mean": sum(metric_values) / len(metric_values),
                        "std": self._calculate_std(metric_values),
                        "count": len(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values)
                    }
            
            results_by_variant[variant] = {
                "users_count": len(unique_users),
                "metrics": metrics_stats
            }
        
        return results_by_variant
    
    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calcula desviación estándar"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def analyze_experiment(
        self,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Analiza un experimento y determina si hay ganador.
        
        Args:
            experiment_name: Nombre del experimento
            
        Returns:
            Dict con análisis:
            {
                "has_winner": bool,
                "winner": str,
                "confidence": float,
                "results_by_variant": {...}
            }
        """
        results = self.get_experiment_results(experiment_name)
        
        if not results:
            return {
                "has_winner": False,
                "message": "No data available"
            }
        
        # Encontrar variante con mejor métrica principal
        experiment = self.experiments[experiment_name]
        if not experiment.target_metrics:
            return {
                "has_winner": False,
                "message": "No target metrics defined"
            }
        
        primary_metric = experiment.target_metrics[0]
        
        best_variant = None
        best_mean = float('-inf')
        
        for variant, data in results.items():
            if primary_metric in data["metrics"]:
                mean = data["metrics"][primary_metric]["mean"]
                if mean > best_mean:
                    best_mean = mean
                    best_variant = variant
        
        # Calcular "confidence" simple (diferencia relativa con control)
        confidence = 0.0
        if best_variant and "control" in results:
            control_mean = results["control"]["metrics"].get(primary_metric, {}).get("mean", 0)
            if control_mean > 0:
                improvement = ((best_mean - control_mean) / control_mean) * 100
                confidence = min(abs(improvement) / 10.0, 1.0)  # Simple heuristic
        
        return {
            "has_winner": best_variant != "control" and confidence > 0.3,
            "winner": best_variant,
            "confidence": confidence,
            "improvement_percent": ((best_mean - results.get("control", {}).get("metrics", {}).get(primary_metric, {}).get("mean", 0)) / results.get("control", {}).get("metrics", {}).get(primary_metric, {}).get("mean", 1)) * 100 if "control" in results else 0,
            "primary_metric": primary_metric,
            "best_mean": best_mean,
            "results_by_variant": results
        }
    
    def stop_experiment(self, experiment_name: str) -> None:
        """Detiene un experimento"""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].active = False
            logger.info(f"⏸️ Experiment stopped: {experiment_name}")


# Decorator para A/B testing
def ab_test(experiment_name: str, metric_name: Optional[str] = None):
    """
    Decorator para aplicar A/B testing a una función.
    
    Args:
        experiment_name: Nombre del experimento
        metric_name: Nombre de métrica a trackear automáticamente
        
    Example:
        >>> @ab_test("evaluation_prompt_v2", metric_name="evaluation_score")
        ... async def evaluate_answer(answer: str, user_id: str, ab_variant: str = "control"):
        ...     if ab_variant == "variant_a":
        ...         prompt = NEW_PROMPT
        ...     else:
        ...         prompt = CURRENT_PROMPT
        ...     return await llm.evaluate(prompt, answer)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, user_id: str, **kwargs):
            ab_framework = get_ab_framework()
            variant = ab_framework.get_variant(experiment_name, user_id)
            
            # Agregar variant a kwargs
            kwargs["ab_variant"] = variant
            
            # Ejecutar función
            result = await func(*args, user_id=user_id, **kwargs)
            
            # Track métrica si está especificada
            if metric_name and isinstance(result, (int, float)):
                ab_framework.track_metric(experiment_name, user_id, metric_name, result)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, user_id: str, **kwargs):
            ab_framework = get_ab_framework()
            variant = ab_framework.get_variant(experiment_name, user_id)
            
            kwargs["ab_variant"] = variant
            result = func(*args, user_id=user_id, **kwargs)
            
            if metric_name and isinstance(result, (int, float)):
                ab_framework.track_metric(experiment_name, user_id, metric_name, result)
            
            return result
        
        # Detectar si es async o sync
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global instance
_ab_framework_instance: Optional[ABTestingFramework] = None


def get_ab_framework() -> ABTestingFramework:
    """Factory para obtener instancia global del framework"""
    global _ab_framework_instance
    
    if _ab_framework_instance is None:
        _ab_framework_instance = ABTestingFramework()
    
    return _ab_framework_instance

