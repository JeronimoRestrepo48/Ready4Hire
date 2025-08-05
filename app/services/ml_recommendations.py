# Algoritmos de clustering y recomendaciones personalizadas para Ready4Hire
from sklearn.cluster import KMeans
import numpy as np

# Simulación: embeddings de usuarios y entrevistas
# En producción, obtén los embeddings reales de respuestas y entrevistas

def cluster_users(user_embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(user_embeddings)
    return labels, kmeans.cluster_centers_

# Sugerir rutas de aprendizaje personalizadas según cluster
# (En producción, usaría historial real y recursos)
def suggest_learning_path(user_cluster, cluster_centers, resources_by_cluster):
    return resources_by_cluster.get(user_cluster, [])
