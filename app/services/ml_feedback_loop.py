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
    # Feedback ML/NLP avanzado: similitud semántica, cobertura de conceptos, feedback personalizado
    if not answer:
        return "Respuesta vacía."
    import re
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sim_score = 0.0
    if expected:
        user_emb = model.encode([answer], convert_to_tensor=True)
        expected_emb = model.encode([expected], convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(user_emb, expected_emb)[0][0])
    expected_keywords = set(re.findall(r"\w+", (expected or '').lower()))
    answer_keywords = set(re.findall(r"\w+", answer.lower()))
    common_keywords = expected_keywords.intersection(answer_keywords)
    coverage = len(common_keywords) / len(expected_keywords) if expected_keywords else 0
    if sim_score > 0.75 or coverage > 0.6:
        return "¡Respuesta muy alineada con lo esperado! Excelente comprensión."
    if len(answer) > 30:
        return "Respuesta extensa, bien desarrollada. Intenta incluir conceptos clave para mejorar aún más."
    if expected_keywords:
        missing = expected_keywords - answer_keywords
        if missing:
            return f"Te faltó mencionar: {', '.join(list(missing)[:3])}. Repasa esos conceptos."
    return "Respuesta recibida. Sigue practicando para mejorar."
