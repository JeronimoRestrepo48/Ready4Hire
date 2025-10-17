"""
Context Questions Configuration
Preguntas iniciales para conocer el contexto del candidato antes de la entrevista formal.

IMPORTANTE: Estas preguntas están diseñadas para generar embeddings que alimentan
el sistema de clustering y selección inteligente de preguntas.

Los factores clave que capturan son:
1. Experiencia y nivel técnico
2. Tecnologías y herramientas conocidas
3. Áreas de especialización y expertise
4. Estilo de trabajo y preferencias
5. Objetivos de aprendizaje y desarrollo
"""
from typing import Dict, List


# Preguntas de contexto por tipo de entrevista
# Diseñadas para capturar factores relevantes para embeddings y clustering
CONTEXT_QUESTIONS: Dict[str, List[str]] = {
    "technical": [
        # Factor 1: Experiencia y nivel técnico
        "¿Cuántos años de experiencia tienes en desarrollo de software y cuál ha sido tu progresión profesional? Menciona roles específicos y responsabilidades.",
        
        # Factor 2: Stack tecnológico y herramientas
        "¿Con qué lenguajes de programación, frameworks y herramientas trabajas regularmente? Indica tu nivel de dominio en cada uno (básico, intermedio, avanzado).",
        
        # Factor 3: Áreas de especialización técnica
        "¿En qué áreas técnicas te especializas? (ej: arquitectura de software, bases de datos, cloud computing, machine learning, seguridad, DevOps, frontend, backend, etc.)",
        
        # Factor 4: Complejidad y desafíos resueltos
        "Describe el proyecto técnico más complejo en el que has trabajado: ¿Qué desafíos enfrentaste, qué tecnologías utilizaste y cuál fue tu rol específico?",
        
        # Factor 5: Objetivos de aprendizaje y crecimiento
        "¿Qué tecnologías o áreas técnicas te gustaría aprender o profundizar en tu próximo rol? ¿Por qué te interesan?"
    ],
    "soft_skills": [
        # Factor 1: Estilo de liderazgo y trabajo en equipo
        "¿Cómo describirías tu estilo de trabajo en equipo? ¿Prefieres liderar, colaborar o trabajar de forma independiente? Da ejemplos concretos.",
        
        # Factor 2: Comunicación y resolución de conflictos
        "Describe una situación donde tuviste que comunicar ideas complejas o resolver un conflicto en el equipo. ¿Qué estrategias utilizaste?",
        
        # Factor 3: Adaptabilidad y manejo del cambio
        "Cuéntame sobre una ocasión donde tuviste que adaptarte rápidamente a un cambio significativo (proyecto, tecnología, metodología). ¿Cómo lo manejaste?",
        
        # Factor 4: Toma de decisiones y gestión de presión
        "¿Cómo tomas decisiones importantes cuando trabajas bajo presión y con plazos ajustados? Proporciona un ejemplo específico.",
        
        # Factor 5: Motivación y valores profesionales
        "¿Qué te motiva profesionalmente y qué valores son importantes para ti en un ambiente de trabajo? ¿Cómo impactan en tu desempeño diario?"
    ],
    "mixed": [
        # Factor 1: Perfil profesional integral
        "Describe tu trayectoria profesional: roles, tecnologías utilizadas, logros principales y cómo has evolucionado.",
        
        # Factor 2: Balance técnico-interpersonal
        "¿Cómo combinas tus habilidades técnicas con tus habilidades interpersonales en proyectos? Da ejemplos de situaciones donde ambas fueron críticas.",
        
        # Factor 3: Fortalezas distintivas
        "¿Cuáles son tus principales fortalezas tanto técnicas como personales? ¿Cómo te diferencian de otros profesionales?",
        
        # Factor 4: Experiencia en proyectos complejos
        "Describe un proyecto donde tuviste que aplicar tanto habilidades técnicas avanzadas como soft skills (comunicación, liderazgo, resolución de conflictos).",
        
        # Factor 5: Visión de carrera y desarrollo
        "¿Qué tipo de rol buscas y qué aspectos (técnicos, interpersonales, liderazgo) te gustaría desarrollar en tu próxima posición?"
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
