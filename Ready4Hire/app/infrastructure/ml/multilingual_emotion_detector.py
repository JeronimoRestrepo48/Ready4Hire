"""
Detector de emociones multilenguaje con modelos específicos por idioma.
Reemplaza al emotion_analyzer.py actual que solo soporta inglés.
"""

from transformers import pipeline
import torch
from functools import lru_cache
from typing import Dict, List, Optional
import langid

from app.domain.value_objects.emotion import Emotion


class MultilingualEmotionDetector:
    """
    Detector de emociones con soporte para español e inglés.
    Usa modelos específicos por idioma para máxima precisión.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._lazy_load = True  # Carga lazy para mejor performance inicial

    def _get_model(self, lang: str):
        """Carga lazy del modelo por idioma"""
        if lang not in self.models:
            if lang == "es":
                # Modelo optimizado para español
                self.models["es"] = pipeline(
                    "text-classification",
                    model="finiteautomata/bertweet-base-emotion-analysis",
                    device=0 if self.device == "cuda" else -1,
                    top_k=None,
                )
            else:
                # Modelo para inglés (fallback)
                self.models["en"] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if self.device == "cuda" else -1,
                    top_k=None,
                )

        return self.models[lang]

    def detect(self, text: str) -> Dict:
        """
        Detecta emoción en texto con detección automática de idioma.

        Args:
            text: Texto a analizar

        Returns:
            {
                'emotion': Emotion enum,
                'confidence': 0.0-1.0,
                'all_emotions': [{'emotion': Emotion, 'score': float}, ...],
                'language': 'es' | 'en'
            }
        """
        if not text or len(text.strip()) < 3:
            return {"emotion": Emotion.NEUTRAL, "confidence": 1.0, "all_emotions": [], "language": "unknown"}

        try:
            # Detectar idioma
            detected_lang, _ = langid.classify(text)
            lang = "es" if detected_lang == "es" else "en"

            # Seleccionar y cargar modelo apropiado
            model = self._get_model(lang)

            # Predecir
            results = model(text)

            # El pipeline con top_k=None retorna una lista de listas [[{...}, {...}]]
            # Si results es una lista de listas, tomar el primer elemento
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]

            # Normalizar etiquetas
            normalized = self._normalize_emotions(results, lang)

            return {
                "emotion": normalized[0]["emotion"],
                "confidence": normalized[0]["score"],
                "all_emotions": normalized,
                "language": lang,
            }

        except Exception as e:
            print(f"[ERROR] Emotion detection failed: {e}")
            import traceback

            traceback.print_exc()
            return {"emotion": Emotion.NEUTRAL, "confidence": 0.5, "all_emotions": [], "language": "unknown"}

    def _normalize_emotions(self, results: List[Dict], lang: str) -> List[Dict]:
        """Normaliza nombres de emociones a Emotion enum"""
        emotion_map = {
            # Español
            "alegría": Emotion.JOY,
            "tristeza": Emotion.SADNESS,
            "enojo": Emotion.ANGER,
            "ira": Emotion.ANGER,
            "miedo": Emotion.FEAR,
            "sorpresa": Emotion.SURPRISE,
            "neutro": Emotion.NEUTRAL,
            # Inglés
            "joy": Emotion.JOY,
            "sadness": Emotion.SADNESS,
            "anger": Emotion.ANGER,
            "fear": Emotion.FEAR,
            "surprise": Emotion.SURPRISE,
            "neutral": Emotion.NEUTRAL,
            "disgust": Emotion.ANGER,  # Map disgust -> anger
        }

        normalized = []
        for r in results:
            label = r["label"].lower()
            emotion = emotion_map.get(label, Emotion.NEUTRAL)
            normalized.append({"emotion": emotion, "score": r["score"]})

        # Ordenar por score
        normalized.sort(key=lambda x: x["score"], reverse=True)

        return normalized

    def analyze_emotion_trend(self, recent_texts: List[str]) -> Dict:
        """
        Analiza tendencia emocional en múltiples textos.
        Útil para detectar frustración o confianza creciente.
        """
        if not recent_texts:
            return {"trend": "neutral", "predominant_emotion": Emotion.NEUTRAL}

        emotions = []
        for text in recent_texts:
            result = self.detect(text)
            emotions.append(result["emotion"])

        # Contar emociones
        positive = sum(1 for e in emotions if Emotion(e).is_positive())
        negative = sum(1 for e in emotions if Emotion(e).is_negative())

        # Determinar tendencia
        if positive > negative * 1.5:
            trend = "improving"
        elif negative > positive * 1.5:
            trend = "declining"
        else:
            trend = "stable"

        # Emoción predominante
        from collections import Counter

        emotion_counts = Counter(emotions)
        predominant = emotion_counts.most_common(1)[0][0]

        return {
            "trend": trend,
            "predominant_emotion": Emotion(predominant),
            "positive_ratio": positive / len(emotions),
            "negative_ratio": negative / len(emotions),
        }


# Instancia global singleton (lazy loading)
_detector_instance = None


def get_emotion_detector() -> MultilingualEmotionDetector:
    """Obtiene instancia singleton del detector"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MultilingualEmotionDetector()
    return _detector_instance


def analyze_emotion(text: str) -> Dict:
    """
    Función de conveniencia compatible con la API anterior.
    Mantiene compatibilidad con código existente.
    """
    detector = get_emotion_detector()
    result = detector.detect(text)

    # Convertir al formato antiguo para compatibilidad
    return {
        "emotion": result["emotion"].value,
        "score": result["confidence"],
        "all_results": [{"label": e["emotion"].value, "score": e["score"]} for e in result["all_emotions"]],
    }
