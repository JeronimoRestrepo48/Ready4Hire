"""
Value Object: Puntuación
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    """
    Puntuación validada entre 0 y 10.
    Inmutable y con validación automática.
    """
    
    value: float
    
    def __post_init__(self):
        """
        Valida que el valor del score esté entre 0 y 10.
        
        Raises:
            ValueError: Si el valor no está en el rango válido.
        """
        if not 0 <= self.value <= 10:
            raise ValueError(f"Score must be between 0 and 10, got {self.value}")
    
    def is_passing(self, threshold: float = 6.0) -> bool:
        """Verifica si el score es aprobatorio"""
        return self.value >= threshold
    
    def is_excellent(self) -> bool:
        """Verifica si es excelente (>= 9)"""
        return self.value >= 9.0
    
    def is_good(self) -> bool:
        """Verifica si es bueno (>= 7)"""
        return self.value >= 7.0
    
    def is_poor(self) -> bool:
        """Verifica si es pobre (< 5)"""
        return self.value < 5.0
    
    def to_percentage(self) -> float:
        """Convierte a porcentaje"""
        return self.value * 10.0
    
    def __str__(self) -> str:
        """
        Representación en string del score.
        
        Returns:
            str: Score formateado como "X.X/10"
        """
        return f"{self.value:.1f}/10"
