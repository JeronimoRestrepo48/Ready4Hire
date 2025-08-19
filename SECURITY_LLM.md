# Seguridad LLM: Protección contra Jailbreaking y Prompt Injection

Este documento describe el plan, la implementación y las mejores prácticas para proteger Ready4Hire contra ataques de jailbreaking y prompt injection en modelos de lenguaje.

## 1. ¿Qué es Prompt Injection y Jailbreaking?
- **Prompt Injection:** Manipulación de la entrada para que el modelo ignore instrucciones, filtre información o ejecute acciones no deseadas.
- **Jailbreaking:** Técnicas para saltarse restricciones del modelo y obtener respuestas prohibidas o peligrosas.

## 2. Estrategias de Protección

### a) Sanitización y Filtrado de Entradas
- Se eliminan o reemplazan patrones peligrosos (regex, listas negras).
- Ejemplo: "ignore previous instructions" → "[REDACTED]"

### b) Detección de Patrones Maliciosos
- Se detectan frases y estructuras típicas de ataques.
- Si se detecta, se loguea el intento y se puede bloquear la interacción.

### c) Instrucciones de Sistema Robustas
- Separar claramente el prompt de sistema y el del usuario.
- No interpolar entradas del usuario directamente en instrucciones críticas.

### d) Validación de Salidas
- Se filtran respuestas que revelen instrucciones internas o acepten comportamientos peligrosos.
- Ejemplo: Si la salida contiene "as a language model", se bloquea.

### e) Logging y Monitoreo
- Todos los intentos sospechosos quedan registrados para análisis y mejora continua.

### f) Pruebas de Red Team
- Se realizan pruebas automáticas y manuales con prompts maliciosos para validar la robustez.

## 3. Implementación Técnica
- El módulo `app/security.py` contiene funciones para sanitizar, detectar y validar entradas/salidas.
- Se integra en el pipeline de entrada y salida del agente.

## 4. Ejemplo de Uso
```python
from app import security

entrada = input("Usuario: ")
entrada_segura = security.sanitize_input(entrada)
if security.detect_prompt_injection(entrada_segura):
    security.log_security_event("Intento de prompt injection", entrada_segura)
    print("Entrada bloqueada por seguridad.")
else:
    respuesta = modelo_llm(entrada_segura)
    respuesta_segura = security.validate_llm_output(respuesta)
    print(respuesta_segura)
```

## 5. Recomendaciones y Advertencias
- Nunca interpolar directamente la entrada del usuario en prompts de sistema.
- Actualizar y ampliar los patrones de detección regularmente.
- Revisar logs y ajustar reglas según nuevos vectores de ataque.
- Educar a los usuarios y desarrolladores sobre riesgos y buenas prácticas.

## 6. Referencias
- [OpenAI: Prompt Injection](https://platform.openai.com/docs/guides/prompt-injection)
- [OWASP: LLM Security](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---
¿Dudas o sugerencias? Contacta al equipo de seguridad de Ready4Hire.
