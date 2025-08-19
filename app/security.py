"""
security.py - Protección contra jailbreaking y prompt injection para Ready4Hire

Incluye:
- Sanitización de entradas
- Detección de patrones maliciosos
- Validación de salidas
- Logging de intentos sospechosos
"""
import re
import logging
from typing import Any

# Configuración de logging
logger = logging.getLogger("llm_security")
logger.setLevel(logging.INFO)

# Patrones comunes de prompt injection y jailbreaking
PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore (all|previous|above) instructions",
    r"(?i)disregard (all|previous|above) instructions",
    r"(?i)act as (an? )?(admin|system|developer|jailbreak)",
    r"(?i)you are now (an? )?(admin|system|developer|jailbreak)",
    r"(?i)repeat after me",
    r"(?i)simulate (an? )?(error|crash|hack)",
    r"(?i)output .* as code",
    r"(?i)write (malicious|harmful|offensive) code",
    r"(?i)forget you are an ai",
    r"(?i)prompt injection",
    r"(?i)jailbreak",
    r"(?i)system prompt",
    r"(?i)as a language model",
    r"(?i)\{\{.*\}\}",  # Plantillas sospechosas
]


def sanitize_input(user_input: str) -> str:
    """
    Elimina o reemplaza patrones peligrosos en la entrada del usuario.
    """
    sanitized = user_input
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized)
    return sanitized


def detect_prompt_injection(user_input: str) -> bool:
    """
    Retorna True si detecta patrones de prompt injection/jailbreak.
    """
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, user_input):
            logger.warning(f"Intento de prompt injection detectado: {user_input}")
            return True
    return False


def validate_llm_output(output: str) -> str:
    """
    Post-procesa la salida del LLM para filtrar instrucciones peligrosas o fugas de sistema.
    """
    # Ejemplo: bloquear respuestas que revelen instrucciones internas
    if re.search(r"(?i)as a language model|system prompt|ignore instructions", output):
        logger.warning(f"Salida potencialmente peligrosa: {output}")
        return "[Respuesta bloqueada por seguridad]"
    return output


def log_security_event(event: str, data: Any = None):
    logger.info(f"[SECURITY] {event} | Data: {data}")

# Ejemplo de integración:
# sanitized = sanitize_input(user_input)
# if detect_prompt_injection(sanitized):
#     log_security_event("Prompt injection detectada", sanitized)
#     ...
# output = validate_llm_output(llm_response)
