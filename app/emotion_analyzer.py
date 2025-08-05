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
    pipe = get_emotion_pipeline()
    # El modelo es inglés, pero funciona razonablemente bien en español para emociones básicas
    results = pipe(text)
    # results: list of dicts with 'label' and 'score'
    return results
