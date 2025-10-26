"""
Value Object: Nivel de habilidad
"""

from enum import Enum


class SkillLevel(str, Enum):
    """Niveles de habilidad del candidato"""

    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"

    def next_level(self) -> "SkillLevel":
        """Retorna el siguiente nivel"""
        if self == SkillLevel.JUNIOR:
            return SkillLevel.MID
        elif self == SkillLevel.MID:
            return SkillLevel.SENIOR
        return self  # Ya es senior

    def previous_level(self) -> "SkillLevel":
        """Retorna el nivel anterior"""
        if self == SkillLevel.SENIOR:
            return SkillLevel.MID
        elif self == SkillLevel.MID:
            return SkillLevel.JUNIOR
        return self  # Ya es junior

    def to_numeric(self) -> int:
        """Convierte a valor numérico para comparaciones (0=junior, 1=mid, 2=senior)"""
        return {SkillLevel.JUNIOR: 0, SkillLevel.MID: 1, SkillLevel.SENIOR: 2}[self]

    @classmethod
    def from_numeric(cls, value: int) -> "SkillLevel":
        """Crea desde valor numérico"""
        mapping = {0: cls.JUNIOR, 1: cls.MID, 2: cls.SENIOR}
        return mapping.get(value, cls.JUNIOR)

    def can_increase_to(self, target: "SkillLevel") -> bool:
        """Verifica si se puede aumentar al nivel objetivo (solo incrementos de 1 nivel)"""
        return target.to_numeric() == self.to_numeric() + 1

    @classmethod
    def from_string(cls, value: str) -> "SkillLevel":
        """Crea desde string (junior, mid, senior)"""
        value_lower = value.lower().strip()
        mapping = {"junior": cls.JUNIOR, "mid": cls.MID, "senior": cls.SENIOR}
        if value_lower not in mapping:
            raise ValueError(f"Invalid skill level: {value}")
        return mapping[value_lower]
