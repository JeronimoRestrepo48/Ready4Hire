"""
Value Object: Emoción detectada
"""
from enum import Enum


class Emotion(str, Enum):
    """Emociones detectables en respuestas"""
    
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    
    @classmethod
    def from_string(cls, emotion_str: str) -> 'Emotion':
        """Crea una instancia de Emotion desde un string"""
        emotion_upper = emotion_str.upper()
        if hasattr(cls, emotion_upper):
            return getattr(cls, emotion_upper)
        # Fallback a un valor por defecto si no existe
        return cls.NEUTRAL if hasattr(cls, 'NEUTRAL') else list(cls)[0]
    
    def is_positive(self) -> bool:
        """Verifica si es una emoción positiva"""
        return self in (Emotion.JOY, Emotion.SURPRISE, Emotion.NEUTRAL)
    
    def is_negative(self) -> bool:
        """Verifica si es una emoción negativa"""
        return self in (Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR)
    
    def to_spanish(self) -> str:
        """Traduce la emoción al español"""
        translations = {
            Emotion.JOY: "Alegría/Confianza",
            Emotion.SADNESS: "Tristeza/Desánimo",
            Emotion.ANGER: "Frustración",
            Emotion.FEAR: "Inseguridad/Miedo",
            Emotion.SURPRISE: "Sorpresa",
            Emotion.NEUTRAL: "Neutral"
        }
        return translations[self]
