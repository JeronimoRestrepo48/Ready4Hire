
# Documentación Técnica y Práctica: NLP, Embeddings, Clustering, RankNet y Emociones en Ready4Hire

Este documento explica en detalle, con ejemplos y diagramas, los componentes de IA, NLP y aprendizaje profundo usados en Ready4Hire. Está pensado para técnicos y no técnicos.

---


## ¿Qué es un Embedding?

Un **embedding** es una representación matemática (vector de números) de un texto, que captura su significado y contexto. Permite comparar textos de manera eficiente y encontrar similitudes semánticas.

**Ejemplo visual:**

| Texto                        | Embedding (vector simplificado)         |
|------------------------------|-----------------------------------------|
| "¿Qué es Python?"            | [0.12, -0.33, 0.88, ...]                |
| "Explica el lenguaje Python" | [0.13, -0.31, 0.85, ...]                |
| "¿Qué es una manzana?"       | [-0.77, 0.22, 0.01, ...]                |

Los dos primeros textos tienen embeddings similares, el tercero es diferente.

**¿Cómo se calculan?**
- Se usa un modelo preentrenado (ej: SentenceTransformers MiniLM-L6-v2, 384 dimensiones).
- El texto se tokeniza, se pasa por la red neuronal y se obtiene el vector.

**¿Para qué sirve?**
- Buscar preguntas similares a la del usuario.
- Medir la relevancia entre la respuesta del usuario y la esperada.

**Ejemplo de código:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(["¿Qué es Python?"])
```

## Búsqueda Semántica y Similitud

Se calcula la **similitud coseno** entre embeddings para encontrar los textos más parecidos.

**Ejemplo:**
```python
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embedding1], [embedding2])
```


## Clustering Temático (UMAP + HDBSCAN)

**¿Por qué agrupar preguntas?**
Para asegurar variedad temática y evitar repeticiones, agrupamos preguntas similares en "clusters".

- **UMAP:** Reduce la dimensionalidad de los embeddings (de 384 a 2-10 dimensiones) para visualizarlos y agruparlos mejor.
- **HDBSCAN:** Detecta grupos (clusters) de preguntas similares y también identifica preguntas "raras" (outliers).

**Ejemplo visual:**

![Diagrama UMAP+HDBSCAN](https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/master/images/hdbscan_clusters.png)

**¿Cómo se usa?**
1. Se calculan los embeddings de todas las preguntas.
2. UMAP reduce la dimensionalidad.
3. HDBSCAN agrupa las preguntas.
4. El sistema selecciona preguntas de clusters poco cubiertos para diversificar.

**Ejemplo de código:**
```python
import umap, hdbscan
umap_embeddings = umap.UMAP(n_components=10).fit_transform(all_embeddings)
clusters = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(umap_embeddings)
```


## Penalización y Diversificación

- **Penalización por repetición:** Si una pregunta ya fue vista, su score baja para evitar repeticiones.
- **Bonus por cluster poco cubierto:** Se incentiva cubrir todos los temas.
- **Softmax:** Transforma los scores en probabilidades, permitiendo selección variada y no determinista.

**Ejemplo de softmax:**
```python
import numpy as np
scores = np.array([2.0, 1.0, 0.1])
probs = np.exp(scores) / np.sum(np.exp(scores))
```


## RankNet (Deep Learning para Ranking)

**¿Qué es RankNet?**
Es una red neuronal profunda que aprende a ordenar preguntas según su relevancia para el usuario y el contexto.

**¿Cómo funciona?**
1. Recibe pares de preguntas (A, B) y aprende cuál es más relevante.
2. Usa los embeddings y el historial del usuario como entrada.
3. Optimiza una función de pérdida basada en el orden correcto.

**Entrenamiento:**
- Se generan pares de ejemplos (pregunta relevante vs. no relevante) usando el dataset histórico.
- RankNet aprende a asignar mayor score a la pregunta relevante.

**Ejemplo de arquitectura:**
```python
import torch.nn as nn
class RankNet(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.fc = nn.Sequential(
      nn.Linear(input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
    )
  def forward(self, x):
    return self.fc(x)
```

**Integración en Ready4Hire:**
- Si RankNet está entrenado, reordena las preguntas candidatas para máxima personalización.
- Si no, se usa el ranking tradicional por similitud.

**Métricas de evaluación:**
- NDCG, MAP, Precision@k para medir la calidad del ranking.


## Análisis Emocional

**¿Por qué analizar emociones?**
Para adaptar el feedback y la dificultad según el estado emocional del usuario (ej: motivar si hay frustración).

**¿Cómo funciona?**
- Se usa un modelo transformer (ej: distilbert-base-uncased-finetuned-sst-2-english) para clasificar emociones en la respuesta del usuario.
- El sistema adapta el feedback y las pistas según la emoción detectada (alegría, frustración, duda, etc.).

**Ejemplo de código:**
```python
from transformers import pipeline
emotion_classifier = pipeline('sentiment-analysis')
result = emotion_classifier("No entiendo la pregunta y me siento frustrado")
```


## Pistas Conceptuales y Prompts Adaptativos

**¿Cómo se generan las pistas?**
- Si el usuario falla, el agente genera:
  - Explicación conceptual
  - Analogía
  - Ejemplo concreto
  - Caso de uso real
- El prompt enviado al LLM incluye contexto, historial, respuesta esperada y tono motivador.
- Si el LLM está limitado, se genera una pista sintética basada en el dataset y reglas heurísticas.


## Aprendizaje Continuo y Fine-tuning

- Cada interacción relevante se almacena para mejorar el modelo y el dataset.
- El sistema puede ser re-entrenado periódicamente con nuevas buenas respuestas y errores frecuentes.
- Se pueden ajustar los pesos de penalización y bonus según el feedback de usuarios reales.


## Optimización y Recursos

- Límite de hilos y uso de CPU para evitar sobrecarga.
- Modelos ligeros y eficientes para uso en servidores modestos.
- Uso de batch processing para acelerar el cálculo de embeddings.

---

## Glosario de Términos Clave

- **Embedding:** Vector numérico que representa un texto.
- **Cluster:** Grupo de preguntas similares.
- **UMAP:** Algoritmo de reducción de dimensionalidad.
- **HDBSCAN:** Algoritmo de clustering robusto.
- **RankNet:** Red neuronal para ranking de relevancia.
- **Similitud coseno:** Medida de parecido entre dos vectores.
- **Softmax:** Función que convierte scores en probabilidades.
- **Fine-tuning:** Reentrenamiento de un modelo con nuevos datos.

## Preguntas Frecuentes Técnicas

**¿Puedo cambiar el modelo de embeddings?**
Sí, pero debe ser compatible con SentenceTransformers y tener buena cobertura semántica.

**¿Cómo entreno RankNet con mis propios datos?**
Usa el script `app/train_ranknet.py` y proporciona pares de preguntas relevantes/no relevantes.

**¿Qué pasa si el modelo emocional falla?**
El sistema usa un fallback neutral y sigue generando pistas.

## Flujo de Datos (Diagrama)

```mermaid
flowchart TD
  A[Pregunta del usuario] --> B[Embedding]
  B --> C[Clustering UMAP+HDBSCAN]
  C --> D[Selección avanzada (RankNet)]
  D --> E[Análisis emocional]
  E --> F[Generación de pista y feedback]
```

---

¿Dudas técnicas? Contacta a JeronimoRestrepo48.

---

¿Dudas técnicas? Contacta a JeronimoRestrepo48.
