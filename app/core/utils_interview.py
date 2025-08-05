"""
Utilidades reutilizables para lógica de entrevistas, feedback, logging y filtrado de preguntas.
"""
import time
from typing import List, Dict, Any

def log_llm_error(llm_errors: list, error, answer, expected):
    """Agrega un error de LLM/embeddings a la lista de errores."""
    llm_errors.append({
        'error': str(error),
        'answer': answer,
        'expected': expected,
        'timestamp': time.time(),
        'error_type': type(error).__name__,
        'length_answer': len(answer) if answer else 0
    })

def filter_questions(questions: List[dict], filters: dict) -> List[dict]:
    """Filtra preguntas usando todos los criterios disponibles en filters."""
    def match(q):
        for key, val in filters.items():
            if val is None or val == []:
                continue
            if key in ["knowledge", "tools"]:
                if not set(val).intersection(set(q.get(key, []))):
                    return False
            elif key == "years":
                if q.get("years") and (int(q["years"]) > int(val)+2 or int(q["years"]) < int(val)-2):
                    return False
            elif key == "difficulty":
                if q.get("difficulty", 1) != int(val):
                    return False
            elif key == "tags":
                if not set(val).intersection(set(q.get("tags", []))):
                    return False
            else:
                if str(q.get(key, "")).lower() != str(val).lower():
                    return False
        return True
    filtered = [q for q in questions if match(q)]
    # Priorizar preguntas con mayor dificultad si hay muchas
    if filtered and len(filtered) > 5:
        filtered = sorted(filtered, key=lambda x: x.get('difficulty', 1), reverse=True)
    return filtered if filtered else questions.copy()

def process_survey_metrics(metrics: dict, rating: int, comments: str):
    """Almacena el feedback de encuesta para aprendizaje continuo."""
    metrics['satisfaction_scores'].append(rating)
    if 'survey_comments' not in metrics:
        metrics['survey_comments'] = []
    if comments:
        metrics['survey_comments'].append(comments)
    # Análisis IA de comentarios (stub)
    if comments and len(comments) > 10:
        metrics['last_comment_sentiment'] = 'positivo' if 'bien' in comments.lower() or 'excelente' in comments.lower() else 'neutral'
    return True

def generate_final_feedback(session: dict) -> dict:
    """Genera feedback global de la entrevista usando ML/DL (stub inicial)."""
    # Feedback IA más robusto: análisis de desempeño, errores, tiempos, badges, etc.
    num_preguntas = len([h for h in session["history"] if "agent" in h])
    num_respuestas = len([h for h in session["history"] if "user" in h])
    puntos = session.get("points", num_respuestas * 10)
    nivel = session.get("level", session.get("level_num", "N/A"))
    errores = [h for h in session["history"] if "feedback" in h and "No es la respuesta esperada" in h["feedback"]]
    tiempo_total = session.get("last_activity", 0) - session.get("start_time", 0) if session.get("start_time") else 0
    badges = list(session.get("badges", []))
    resumen = f"Respondiste {num_respuestas} de {num_preguntas} preguntas. Puntaje: {puntos}. Nivel: {nivel}. Errores: {len(errores)}. Tiempo total: {int(tiempo_total)}s. Badges: {', '.join(badges) if badges else 'Ninguno'}."
    return {
        "summary": resumen,
        "score": puntos,
        "level": nivel,
        "points": puntos,
        "badges": badges,
        "errors": len(errores),
        "time": int(tiempo_total)
    }
