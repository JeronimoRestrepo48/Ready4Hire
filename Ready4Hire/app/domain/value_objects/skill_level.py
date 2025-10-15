"""
Value Object: Nivel de habilidad
"""
from enum import Enum


class SkillLevel(str, Enum):
    """Niveles de habilidad del candidato"""
    
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    
    def next_level(self) -> 'SkillLevel':
        """Retorna el siguiente nivel"""
        if self == SkillLevel.JUNIOR:
            return SkillLevel.MID
        elif self == SkillLevel.MID:
            return SkillLevel.SENIOR
        return self  # Ya es senior
    
    def previous_level(self) -> 'SkillLevel':
        """Retorna el nivel anterior"""
        if self == SkillLevel.SENIOR:
            return SkillLevel.MID
        elif self == SkillLevel.MID:
            return SkillLevel.JUNIOR
        return self  # Ya es junior
    
    def to_numeric(self) -> int:
        """Convierte a valor numérico para comparaciones"""
        return {
            SkillLevel.JUNIOR: 1,
            SkillLevel.MID: 2,
            SkillLevel.SENIOR: 3
        }[self]
    
    @classmethod
    def from_numeric(cls, value: int) -> 'SkillLevel':
        """Crea desde valor numérico"""
        mapping = {1: cls.JUNIOR, 2: cls.MID, 3: cls.SENIOR}
        return mapping.get(value, cls.JUNIOR)
    
    @classmethod
    def from_string(cls, value: str) -> 'SkillLevel':
        """Crea desde string (junior, mid, senior)"""
        value_lower = value.lower().strip()
        mapping = {
            "junior": cls.JUNIOR,
            "mid": cls.MID,
            "senior": cls.SENIOR
        }
        return mapping.get(value_lower, cls.JUNIOR)
