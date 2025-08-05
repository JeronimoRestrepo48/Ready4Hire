# Stub alternativo para recomendaciones personalizadas si no hay datos de clustering
def suggest_learning_path(events):
    # Analiza eventos y sugiere rutas de aprendizaje
    if not events:
        return ["Practica más para obtener recomendaciones personalizadas."]
    correct = sum(1 for e in events if e.get('is_correct'))
    total = len(events)
    if total and correct/total < 0.5:
        return ["Revisa fundamentos básicos antes de avanzar.", "Enfócate en tus áreas débiles detectadas."]
    if total and correct/total > 0.8:
        return ["¡Excelente desempeño! Considera practicar preguntas avanzadas."]
    return ["Buen trabajo, sigue practicando para mejorar tu precisión."]
# Algoritmos de clustering y recomendaciones personalizadas para Ready4Hire
from sklearn.cluster import KMeans
import numpy as np

# Simulación: embeddings de usuarios y entrevistas
# En producción, obtén los embeddings reales de respuestas y entrevistas

def cluster_users(user_embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(user_embeddings)
    return labels, kmeans.cluster_centers_

