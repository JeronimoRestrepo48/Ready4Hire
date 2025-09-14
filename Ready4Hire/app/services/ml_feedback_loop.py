# Analítica de desempeño y feedback loop para mejora continua
import numpy as np

def compute_performance_metrics(interviews):
    # interviews: lista de objetos con score, satisfaction, etc.
    scores = np.array([i.score for i in interviews if i.score is not None])
    satisf = np.array([i.satisfaction for i in interviews if i.satisfaction is not None])
    return {
        'score_mean': float(np.mean(scores)) if len(scores) else 0,
        'score_std': float(np.std(scores)) if len(scores) else 0,
        'satisfaction_mean': float(np.mean(satisf)) if len(satisf) else 0,
        'satisfaction_std': float(np.std(satisf)) if len(satisf) else 0,
    }

# Esta función puede llamarse periódicamente para ajustar el modelo, dificultad, recursos, etc.
# Stub para feedback ML adaptativo. Reemplazar con lógica real de ML/DL.
def generate_feedback_ml(answer, expected=None):
    # Aquí se puede usar un modelo ML real para analizar la respuesta
    if not answer:
        return "Respuesta vacía."
    if expected and expected.lower() in answer.lower():
        return "¡Respuesta muy alineada con lo esperado!"
    if len(answer) > 30:
        return "Respuesta extensa, bien desarrollada."
    return "Respuesta recibida. Sigue practicando para mejorar."
