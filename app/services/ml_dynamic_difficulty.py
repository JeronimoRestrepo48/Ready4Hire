# Stub alternativo para ajuste dinámico de dificultad
def adjust_difficulty(session, performance):
    # Ajusta la dificultad de la entrevista según el desempeño
    if performance < 0.5:
        session['difficulty'] = 'baja'
    elif performance > 0.8:
        session['difficulty'] = 'alta'
    else:
        session['difficulty'] = 'media'
    return session['difficulty']
# Ajuste dinámico de dificultad según nivel y desempeño

def adjust_question_difficulty(user_level, last_score):
    # Convertir user_level a int de forma segura
    try:
        user_level = int(user_level)
    except (ValueError, TypeError):
        user_level = 0
    # Ejemplo simple: si el usuario tiene nivel alto y buen score, sube dificultad
    if user_level >= 3 and last_score >= 8:
        return 'hard'
    elif user_level >= 2 and last_score >= 6:
        return 'medium'
    else:
        return 'easy'
