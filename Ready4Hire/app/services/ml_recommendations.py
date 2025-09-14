
# Algoritmos de clustering y recomendaciones personalizadas para Ready4Hire
from sklearn.cluster import KMeans
import numpy as np

# Simulación: embeddings de usuarios y entrevistas
# En producción, obtén los embeddings reales de respuestas y entrevistas

def cluster_responses(response_embeddings, n_clusters=5):
    """
    Agrupa respuestas de candidatos usando embeddings para identificar perfiles y necesidades comunes.
    Devuelve etiquetas de cluster y centros.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(response_embeddings)
    return labels, kmeans.cluster_centers_

def personalized_recommendations(user_embedding, cluster_centers, resources_by_cluster):
    """
    Recomienda recursos personalizados según el cluster más cercano al embedding de la respuesta del usuario.
    """
    dists = np.linalg.norm(cluster_centers - user_embedding, axis=1)
    user_cluster = int(np.argmin(dists))
    return resources_by_cluster.get(user_cluster, [])

def chain_of_thought_explanation(concept, user_answer, expected_answer, context=None):
    """
    Genera una explicación profunda y paso a paso (chain-of-thought) para ayudar al usuario a entender un concepto.
    Integra contexto, errores comunes y analogías si es posible.
    """
    explanation = f"Vamos a desglosar el concepto de '{concept}':\n"
    if context:
        explanation += f"Contexto relevante: {context}\n"
    explanation += f"1. Analicemos tu respuesta: '{user_answer}'.\n"
    explanation += f"2. Lo esperado era: '{expected_answer}'.\n"
    # Paso a paso: identificar brechas
    if user_answer and expected_answer and user_answer.lower() not in expected_answer.lower():
        explanation += "3. Observa que tu respuesta no cubre algunos puntos clave.\n"
        explanation += "4. Piensa en ejemplos prácticos o analogías que te ayuden a conectar el concepto con tu experiencia.\n"
    else:
        explanation += "3. ¡Muy bien! Tu respuesta está alineada con lo esperado.\n"
    explanation += "4. Recuerda: comprender el 'por qué' detrás de cada concepto te ayudará a aplicarlo en situaciones reales.\n"
    return explanation
