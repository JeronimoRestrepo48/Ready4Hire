"""
Context Questions Configuration
Preguntas iniciales para conocer el contexto del candidato antes de la entrevista formal.
"""
from typing import Dict, List


# Preguntas de contexto por tipo de entrevista
CONTEXT_QUESTIONS: Dict[str, List[str]] = {
    "technical": [
        "¿Cuál es tu experiencia previa en desarrollo de software? Por favor, describe los proyectos más relevantes en los que has trabajado.",
        "¿Con qué tecnologías y lenguajes de programación te sientes más cómodo trabajando?",
        "¿Cuál ha sido el desafío técnico más complejo que has enfrentado y cómo lo resolviste?",
        "¿Qué áreas técnicas te gustaría fortalecer o en qué te gustaría especializarte?",
        "¿Cómo te mantienes actualizado con las nuevas tecnologías y mejores prácticas de desarrollo?"
    ],
    "soft_skills": [
        "Cuéntame sobre una situación en la que tuviste que trabajar en equipo bajo presión. ¿Cómo manejaste la situación?",
        "¿Cómo describirías tu estilo de comunicación cuando trabajas con otros miembros del equipo?",
        "¿Qué te motiva profesionalmente y qué tipo de ambiente de trabajo buscas?",
        "Describe una ocasión en la que tuviste que resolver un conflicto con un compañero de trabajo.",
        "¿Cómo priorizas tus tareas cuando tienes múltiples proyectos con deadlines ajustados?"
    ],
    "mixed": [
        "¿Cuál es tu experiencia previa profesional? Cuéntame sobre tu trayectoria.",
        "¿Qué tipo de rol estás buscando y qué te gustaría aportar al equipo?",
        "Describe un proyecto del que te sientas orgulloso y tu rol en él.",
        "¿Cómo balanceas las habilidades técnicas con las habilidades interpersonales en tu trabajo diario?",
        "¿Qué aspectos tanto técnicos como personales te gustaría desarrollar en tu próximo rol?"
    ]
}


def get_context_questions(interview_type: str) -> List[str]:
    """
    Obtiene las preguntas de contexto según el tipo de entrevista.
    
    Args:
        interview_type: Tipo de entrevista ("technical", "soft_skills", "mixed")
        
    Returns:
        Lista de preguntas de contexto
    """
    return CONTEXT_QUESTIONS.get(interview_type, CONTEXT_QUESTIONS["mixed"])


# Número de preguntas de contexto antes de comenzar la entrevista formal
CONTEXT_QUESTIONS_COUNT = 5

# Número de preguntas principales de la entrevista
MAIN_QUESTIONS_COUNT = 10

# Número máximo de intentos por pregunta
MAX_ATTEMPTS_PER_QUESTION = 3

# Umbral de score para considerar una respuesta correcta
PASS_THRESHOLD = 6.0

# Umbral de score para considerar una respuesta excelente
EXCELLENT_THRESHOLD = 9.0
