# emotion_analyzer.py
"""
Módulo para análisis de emociones en texto usando modelos de transformers (HuggingFace).
"""
from transformers.pipelines import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def get_emotion_pipeline():
    # Modelo multilingüe para emociones (puedes cambiar por otro si lo deseas)
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

def analyze_emotion(text: str):
    try:
        pipe = get_emotion_pipeline()
        # El modelo es inglés, pero funciona razonablemente bien en español para emociones básicas
        results = pipe(text)
        # Tomar la emoción dominante
        if results and isinstance(results, list):
            # results[0] puede ser una lista de dicts o un dict
            if isinstance(results[0], list):
                dicts = [x for x in results[0] if isinstance(x, dict)]
                if dicts:
                    best = max(dicts, key=lambda x: x.get('score', 0))
                else:
                    best = {'label': 'neutral', 'score': 0.0}
            elif isinstance(results[0], dict):
                best = results[0] if isinstance(results[0], dict) else {'label': 'neutral', 'score': 0.0}
            else:
                best = {'label': 'neutral', 'score': 0.0}
            return {'emotion': best.get('label', 'neutral'), 'score': best.get('score', 0.0), 'all': results}
        return {'emotion': 'neutral', 'score': 0.0, 'all': results}
    except Exception as e:
        print(f"[ERROR] EmotionAnalyzer: {e}")
        return {'emotion': 'error', 'score': 0.0, 'all': []}
