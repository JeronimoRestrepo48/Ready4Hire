"""
Value Object: Interview Mode
Tipos de entrevista con lógica específica
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InterviewModeType(Enum):
    """Tipos de modo de entrevista"""

    PRACTICE = "practice"  # 🎓 Modo aprendizaje
    EXAM = "exam"  # 📝 Modo evaluación


@dataclass(frozen=True)
class InterviewMode:
    """
    Value Object que encapsula la lógica de cada modo de entrevista.

    MODO PRÁCTICA 🎓:
    - Objetivo: Aprendizaje y mejora continua
    - Permite múltiples intentos (hasta 3)
    - Ofrece pistas progresivas
    - Feedback extendido y constructivo
    - Sugiere recursos de aprendizaje
    - Sin límite de tiempo estricto
    - No afecta ranking global

    MODO EXAMEN 📝:
    - Objetivo: Evaluación objetiva y certificación
    - Un solo intento por pregunta
    - Sin pistas ni hints
    - Feedback conciso y profesional
    - Límite de tiempo por pregunta (5 min)
    - Score definitivo e inmutable
    - Afecta ranking y habilita certificación
    """

    mode_type: InterviewModeType

    def is_practice(self) -> bool:
        """¿Es modo práctica?"""
        return self.mode_type == InterviewModeType.PRACTICE

    def is_exam(self) -> bool:
        """¿Es modo examen?"""
        return self.mode_type == InterviewModeType.EXAM

    def max_attempts_per_question(self) -> int:
        """Máximo de intentos permitidos por pregunta"""
        if self.is_practice():
            return 3  # 3 intentos en práctica
        return 1  # 1 solo intento en examen

    def hints_enabled(self) -> bool:
        """¿Se permiten pistas?"""
        return self.is_practice()

    def time_limit_seconds(self) -> Optional[int]:
        """Límite de tiempo por pregunta en segundos"""
        if self.is_exam():
            return 300  # 5 minutos en modo examen
        return None  # Sin límite en práctica

    def feedback_style(self) -> str:
        """Estilo de feedback a generar"""
        if self.is_practice():
            return "extended_constructive"  # Feedback detallado y motivacional
        return "concise_professional"  # Feedback breve y objetivo

    def affects_global_ranking(self) -> bool:
        """¿Afecta el ranking global?"""
        return self.is_exam()

    def allows_retake(self) -> bool:
        """¿Permite repetir la entrevista?"""
        return self.is_practice()

    def enables_certification(self) -> bool:
        """¿Habilita certificación al completar?"""
        return self.is_exam()

    def score_is_mutable(self) -> bool:
        """¿El score puede cambiar después?"""
        return self.is_practice()  # En práctica sí, en examen no

    def requires_minimum_score_for_completion(self) -> bool:
        """¿Requiere score mínimo para completar?"""
        return self.is_exam()  # Examen requiere mínimo 6.0, práctica no

    def minimum_score_required(self) -> float:
        """Score mínimo requerido para aprobar"""
        if self.is_exam():
            return 6.0  # 60% para aprobar examen
        return 0.0  # No hay mínimo en práctica

    def show_correct_answers_immediately(self) -> bool:
        """¿Mostrar respuestas correctas inmediatamente?"""
        return False  # En ambos modos se espera al final

    def show_progress_bar(self) -> bool:
        """¿Mostrar barra de progreso?"""
        return True  # En ambos modos

    def allow_skip_questions(self) -> bool:
        """¿Permite saltar preguntas?"""
        return self.is_practice()  # Solo en práctica

    def description(self) -> str:
        """Descripción del modo"""
        if self.is_practice():
            return "🎓 Modo Práctica - Aprende sin presión, con pistas y múltiples intentos"
        return "📝 Modo Examen - Evaluación objetiva para certificación"

    def emoji(self) -> str:
        """Emoji representativo"""
        return "🎓" if self.is_practice() else "📝"

    def color_theme(self) -> str:
        """Tema de color para UI"""
        if self.is_practice():
            return "#10b981"  # Verde - aprendizaje
        return "#6366f1"  # Azul/Morado - evaluación

    @classmethod
    def from_string(cls, mode_str: str) -> "InterviewMode":
        """Crea InterviewMode desde string"""
        if mode_str.lower() == "practice":
            return cls(InterviewModeType.PRACTICE)
        elif mode_str.lower() == "exam":
            return cls(InterviewModeType.EXAM)
        else:
            raise ValueError(f"Invalid interview mode: {mode_str}")

    @classmethod
    def practice(cls) -> "InterviewMode":
        """Factory method para modo práctica"""
        return cls(InterviewModeType.PRACTICE)

    @classmethod
    def exam(cls) -> "InterviewMode":
        """Factory method para modo examen"""
        return cls(InterviewModeType.EXAM)

    def to_string(self) -> str:
        """Convierte a string"""
        return self.mode_type.value

    def __str__(self) -> str:
        return f"InterviewMode({self.mode_type.value})"

    def __repr__(self) -> str:
        return self.__str__()


# Constantes para fácil acceso
PRACTICE_MODE = InterviewMode.practice()
EXAM_MODE = InterviewMode.exam()
