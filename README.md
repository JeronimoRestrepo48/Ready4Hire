# Ready4Hire - Simulador de Entrevistas IA

## Descripción General
Ready4Hire es un sistema avanzado de simulación de entrevistas técnicas y de habilidades blandas, impulsado por IA generativa (LLMs locales vía Ollama), embeddings semánticos, análisis emocional y gamificación. Permite practicar y evaluar competencias en roles de ingeniería, con feedback adaptativo, aprendizaje continuo y experiencia inmersiva por chat web (voz y texto).

## Arquitectura y Componentes

- **Backend FastAPI**: expone endpoints REST para entrevista, audio, historial, recursos y control de sesión.
- **Agente de Entrevista** (`app/interview_agent.py`): lógica central, selección de preguntas, feedback, motivación, pistas, gamificación, análisis emocional, aprendizaje y fine-tuning.
- **Gestor de Embeddings** (`app/embeddings/embeddings_manager.py`): generación y búsqueda semántica de embeddings para preguntas, respuestas y recursos.
- **Frontend Webchat** (`app/static/`): interfaz moderna, accesible, con integración de audio, temporizador, gamificación y feedback visual.
- **Datasets** (`app/datasets/`): preguntas técnicas, soft skills, recursos, interacciones para fine-tuning (JSONL).
- **Audio** (`app/audio_utils.py`): STT (Whisper), TTS (pyttsx3), manejo robusto de archivos y voces.

## Flujo de Entrevista y Aprendizaje

1. **Inicio y Contexto**: el usuario responde preguntas de contexto (rol, nivel, años, herramientas, expectativas). El agente adapta la entrevista según el tipo (técnica/blanda).
2. **Selección Inteligente de Preguntas**: se filtran y rankean las preguntas del dataset usando embeddings y criterios de contexto (rol, nivel, conocimientos, etc.).
3. **Ciclo de Pregunta-Respuesta**:
   - En **modo práctica**: feedback inmediato, emocional y motivador, con pistas generadas por LLM si la respuesta es incorrecta. El usuario debe responder correctamente o agotar pistas antes de avanzar.
   - En **modo examen**: sin feedback inmediato, solo confirmación y temporizador activo. Gamificación avanzada (puntos, niveles, logros).
4. **Análisis Emocional**: el feedback se adapta según la emoción detectada en la respuesta (alegría, tristeza, frustración, etc.).
5. **Aprendizaje y Fine-tuning**: cada interacción relevante (pregunta, respuesta, feedback, pista, acierto/error) se guarda para mejorar el modelo y el dataset en el futuro.
6. **Resumen Final**: estadísticas, fortalezas, puntos de mejora, habilidades destacadas, gamificación y encuesta de satisfacción.

## Endpoints Principales (API REST)

- `/start_interview` (POST): Inicia la entrevista. Requiere `user_id`, `role`, `type`, `mode`.
- `/next_question` (POST): Siguiente pregunta personalizada.
- `/answer` (POST): Envía respuesta del usuario. Devuelve feedback, pistas y control de avance.
- `/end_interview` (POST): Finaliza y entrega resumen detallado.
- `/stt` (POST): Audio a texto (STT, Whisper).
- `/tts` (POST): Texto a audio (TTS, pyttsx3).
- `/get_roles`, `/get_levels`, `/get_question_bank`: Utilidades para frontend.
- `/interview_history`, `/reset_interview`: Historial y reinicio de sesión.

## Modos de Entrevista

- **Práctica**: feedback inmediato, pistas, recursos, aprendizaje adaptativo. No se avanza hasta responder bien o agotar pistas.
- **Examen**: sin feedback inmediato, temporizador, gamificación avanzada (puntos, niveles, logros), resumen final.

## Selección y Validación de Preguntas

- Filtrado por contexto: rol, nivel, años, herramientas, conocimientos.
- Búsqueda semántica: embeddings de contexto vs. embeddings de preguntas.
- Validación de respuestas: comparación semántica (embeddings) con campos `answer`/`expected` del dataset.
- Pistas: generadas por LLM, adaptadas al error y contexto.

## Feedback, Motivación y Emociones

- Feedback adaptativo: frases motivacionales, emojis, recursos de aprendizaje.
- Análisis emocional: el feedback se ajusta según la emoción detectada en la respuesta.
- Aprendizaje automático: el sistema aprende nuevas frases y emojis motivacionales.

## Gamificación

- Puntos por respuesta correcta (modo examen).
- Niveles y nombres personalizados.
- Logros temáticos y mensajes de avance.
- Visualización destacada en el chat.

## Aprendizaje Continuo y Fine-tuning

- Cada interacción relevante se guarda en `datasets/finetune_interactions.jsonl`.
- Permite mejorar el modelo LLM y el dataset con ejemplos reales.
- Soporta etiquetado y revisión para ciclos de fine-tuning.

## Seguridad y Robustez

- Sanitización de entradas y protección anti-inyección de prompts.
- Límite de hilos y recursos para evitar sobrecarga.
- Manejo robusto de errores y excepciones.

## Integración Frontend (Webchat)

- Interfaz moderna, accesible, con soporte de voz y texto.
- Temporizador, gamificación visual, feedback emocional.
- Flujo robusto: en práctica, no avanza hasta respuesta correcta o agotar pistas.
- Accesibilidad: controles de audio, teclado y mensajes claros.

## Audio (STT/TTS)

- Reconocimiento de voz (Whisper) y síntesis de voz (pyttsx3).
- Integración directa en el chat web.


## Arquitectura Detallada del Agente Ready4Hire

El agente Ready4Hire está diseñado para simular entrevistas técnicas y de soft skills de forma adaptativa, eficiente y personalizada, integrando IA generativa, NLP, clustering, análisis emocional y gamificación. Su arquitectura modular garantiza relevancia, robustez y aprendizaje continuo.

### 1. Flujo General

1. **Inicio de sesión/contexto**: El usuario responde preguntas de contexto (rol, nivel, años, conocimientos, herramientas, expectativas). El agente construye un embedding de perfil y ajusta el flujo según el tipo de entrevista.
2. **Selección semántica estricta**: Solo se seleccionan preguntas relevantes al rol/contexto usando embeddings y clustering, evitando preguntas fuera de dominio (ej: no se pregunta de redes a IA).
3. **Ciclo de entrevista**:
    - **Modo práctica**: Feedback inmediato, emocional y motivador, con pistas generadas por LLM. El usuario debe responder correctamente o agotar pistas antes de avanzar.
    - **Modo examen**: Sin feedback inmediato, solo confirmación y temporizador activo. Gamificación avanzada (puntos, niveles, logros).
4. **Análisis y feedback**:
    - **Scoring robusto**: Se pondera similitud semántica, completitud, precisión y soft skills clave.
    - **Clustering de desempeño**: Se agrupan respuestas por embeddings para personalizar consejos y recursos.
    - **Prompts adaptativos**: El LLM recibe contexto, historial y desempeño para generar feedback y pistas personalizadas.
    - **Análisis emocional**: El feedback se adapta según la emoción detectada (alegría, frustración, etc.).
5. **Memoria conversacional**: Todo el historial de la sesión se almacena y utiliza para adaptar el flujo y el feedback.
6. **Aprendizaje activo**: Cada interacción relevante se guarda para mejorar el modelo, el dataset y los embeddings en ciclos de fine-tuning.
7. **Gamificación y resumen**: Se calculan puntos, niveles, logros y se genera un resumen final con fortalezas, áreas de mejora y recomendaciones personalizadas.

### 2. Componentes Principales

- **Backend FastAPI**: Orquesta el flujo, expone endpoints REST y gestiona sesiones.
- **Agente de Entrevista** (`app/interview_agent.py`):
   - Selección inteligente de preguntas (embeddings, clustering, contexto).
   - Validación y scoring de respuestas (similitud, completitud, soft skills).
   - Feedback motivador, emocional y adaptativo (prompts adaptativos, análisis emocional, pistas LLM).
   - Gamificación (puntos, niveles, logros, consejos personalizados).
   - Memoria conversacional y temporizador de respuesta.
   - Aprendizaje activo y registro para fine-tuning.
- **Gestor de Embeddings** (`app/embeddings/embeddings_manager.py`):
   - Generación y búsqueda semántica eficiente (SentenceTransformers optimizado para CPU).
   - Filtrado estricto por rol/contexto.
   - Actualización dinámica de embeddings con nuevas buenas respuestas.
- **Clustering y recomendaciones** (`app/services/ml_recommendations.py`):
   - Agrupa respuestas por similitud para personalizar consejos y recursos.
   - Explicaciones chain-of-thought y feedback profundo.
- **Frontend Webchat** (`app/static/`):
   - Interfaz moderna, accesible, con integración de audio, temporizador, gamificación y feedback visual.
   - Flujo robusto: en práctica, no avanza hasta respuesta correcta o agotar pistas.
- **Datasets** (`app/datasets/`):
   - Preguntas técnicas y blandas alineadas a contexto, respuestas esperadas, recursos y ejemplos.
   - Interacciones para fine-tuning y aprendizaje activo.
- **Audio**: STT (Whisper), TTS (pyttsx3), integración directa en el chat web.

### 3. Eficiencia y Robustez

- **Optimización para CPU**: Modelos ligeros, batch processing, límite de hilos y recursos para evitar sobrecarga.
- **Seguridad**: Sanitización de entradas, protección anti-inyección, manejo robusto de errores.
- **Extensibilidad**: Modularidad para agregar nuevos modelos, algoritmos ML, recursos y visualizaciones.

### 4. Aprendizaje y Adaptación

- **Aprendizaje activo**: El sistema aprende de cada sesión, ajusta el banco de preguntas, el feedback y los recursos recomendados.
- **Fine-tuning**: Las buenas respuestas y errores frecuentes se usan para mejorar el modelo y el dataset.
- **Personalización profunda**: El feedback, las pistas y los consejos se adaptan dinámicamente al perfil, desempeño y emociones del usuario.

---

## Preguntas Frecuentes (FAQ)

**¿Cómo se seleccionan las preguntas?**
Por contexto (rol, nivel, conocimientos) y similitud semántica usando embeddings.

**¿Cómo se valida una respuesta?**
Comparando embeddings de la respuesta del usuario con los campos `answer`/`expected` del dataset.

**¿Qué pasa si respondo mal?**
Recibes feedback motivador y una pista generada por IA. No avanzas hasta responder bien o agotar pistas.

**¿Cómo aprende el sistema?**
Guarda cada interacción relevante para mejorar el modelo y el dataset en ciclos de fine-tuning.

**¿Qué tecnologías usa?**
FastAPI, Ollama (LLM local), SentenceTransformers, HuggingFace, pyttsx3, Whisper, HTML/CSS/JS.

**¿Es seguro?**
Sí, sanitiza entradas y protege contra inyección de prompts.

## Troubleshooting

- Si el sistema no avanza tras una pista, revisa que el frontend respete el campo `retry`.
- Si hay errores de recursos, ajusta los límites de hilos en `embeddings_manager.py`.
- Para problemas de audio, verifica permisos de micrófono y compatibilidad del navegador.

## Créditos y Licencia

Desarrollado por JeronimoRestrepo48 y colaboradores. Licencia MIT.
- Endpoints robustos y validados.

  
## Aprendizaje Continuo

- Las buenas respuestas se almacenan y refrescan los embeddings automáticamente para mejorar la selección y feedback.

---

  
## Audio y Webchat

El frontend webchat (`/`) permite:

- Responder por texto o voz (STT).
- Escuchar las respuestas del agente (TTS).
- Visualizar el temporizador en modo examen.
- Cambiar entre modo práctica y examen.
- Interfaz moderna, responsiva y accesible.

  
## Aprendizaje Automático y Deep Learning

- `app/emotion_analyzer.py`: Análisis de emociones en texto usando modelos transformers de HuggingFace.
- Feedback adaptativo: el agente ajusta su respuesta y motivación según la emoción detectada en el usuario.
- Aprendizaje automático de frases motivacionales y emojis: el sistema expande dinámicamente su repertorio usando LLMs.
- Generación automática de nuevas preguntas técnicas: el agente crea preguntas inéditas y las integra al dataset y embeddings.

  
## Documentación Técnica: NLP, Embeddings y Emociones

Ver `NLP_EMBEDDINGS_TECH.md` para detalles matemáticos, estadísticos, de arquitectura y de integración de análisis emocional y aprendizaje automático.

## Seguridad LLM: Protección contra Prompt Injection y Jailbreaking

Ready4Hire implementa un plan robusto para proteger el sistema y los usuarios frente a ataques de prompt injection y jailbreaking en modelos de lenguaje:

- **Sanitización y filtrado de entradas**: Todas las respuestas del usuario se limpian y filtran antes de ser procesadas. Se eliminan o reemplazan patrones peligrosos (regex, listas negras).
- **Detección automática de patrones maliciosos**: Se detectan frases y estructuras típicas de ataques. Si se detecta, se bloquea la interacción y se registra el intento.
- **Validación de salidas del modelo**: Toda respuesta generada por el LLM se valida antes de ser mostrada. Si contiene instrucciones internas o frases peligrosas, se bloquea y se registra.
- **Logging y monitoreo**: Todos los intentos sospechosos quedan registrados para análisis y mejora continua.
- **Pruebas automáticas de robustez**: El archivo `app/test_security_llm.py` contiene tests unitarios que simulan intentos de prompt injection y validan la protección.

### Ejemplo de integración

```python
from app import security

# En el pipeline del agente:
try:
    entrada_segura = security.sanitize_input(entrada_usuario)
    if security.detect_prompt_injection(entrada_segura):
        security.log_security_event("Prompt injection detectada", entrada_segura)
        raise ValueError("Entrada bloqueada por seguridad.")
except ValueError as e:
    print(str(e))

# Validación de salida del LLM
respuesta = modelo_llm(entrada_segura)
respuesta_segura = security.validate_llm_output(respuesta)
```

### Pruebas automáticas

Ejecuta:
```bash
python3 -m unittest app/test_security_llm.py
```

Para más detalles, consulta `SECURITY_LLM.md`.

## Explicabilidad (XAI) en la Selección de Preguntas y Feedback

## Diversidad y Cobertura Temática en la Selección de Preguntas

## Personalización Dinámica y Trayectoria del Usuario

## Feedback Enriquecido: Ejemplos y Recursos Personalizados

## Detección y Mitigación de Sesgos

## Robustez y Resiliencia ante Errores

## Trazabilidad y Auditoría (Logging Estructurado)

Cada decisión relevante del agente (selección de pregunta, feedback, detección de sesgo, adaptaciones, etc.) se registra en un log estructurado (`audit_log.jsonl`). Esto permite auditar, analizar y mejorar el comportamiento del sistema, garantizando transparencia y facilitando la mejora continua basada en datos reales de uso.

El sistema maneja de forma robusta los errores de dependencias externas (por ejemplo, caídas del LLM, fallos de red o de recursos). Si ocurre un error, el agente utiliza respuestas automáticas de respaldo, registra el incidente y permite continuar la entrevista sin interrumpir la experiencia del usuario. Esto garantiza una experiencia fluida y resiliente incluso ante fallos inesperados.

El sistema detecta patrones de sesgo durante la entrevista, como secuencias de emociones negativas, repetición de temas o feedback poco diverso. Cuando se detecta un posible sesgo, el agente alterna el tipo de pregunta (técnica/blanda), sugiere preguntas objetivas y ajusta el feedback para equilibrar la experiencia, promoviendo una evaluación más justa, variada y libre de sesgos.

Cuando el usuario comete un error o muestra un área de mejora, el agente ahora sugiere ejemplos concretos y recursos de aprendizaje personalizados (artículos, videos, documentación, cursos, etc.) relevantes al tema y al error detectado. Esto permite que cada feedback sea más útil, práctico y orientado a la mejora continua, facilitando el aprendizaje autónomo y la profundización en los conceptos clave.

El sistema ahora adapta la selección de preguntas no solo al contexto inicial (rol, nivel, intereses), sino también al desempeño histórico y la trayectoria del usuario durante la entrevista. Se penalizan temas ya cubiertos y se ajusta la dificultad en tiempo real según aciertos, errores y patrones detectados, personalizando la experiencia de aprendizaje y evaluación.

Esto permite que cada entrevista evolucione de forma única, maximizando el valor formativo y la relevancia para cada usuario.

Para maximizar la diversidad y cobertura temática, el sistema prioriza la variedad de temas (clusters) en cada ronda de preguntas. El selector avanzado (`advanced_question_selector`) utiliza los resultados de clustering (UMAP + HDBSCAN) para asegurar que las preguntas seleccionadas provengan de la mayor cantidad posible de clusters distintos, evitando repeticiones de tema y maximizando la cobertura del conocimiento evaluado. Si no hay suficientes clusters distintos, se completan los cupos con las preguntas más relevantes restantes.

Esto garantiza entrevistas más completas, variadas y representativas del dominio, mejorando la experiencia y la robustez del proceso de evaluación.

Ready4Hire ahora explica por qué selecciona cada pregunta relevante para el usuario:

- Para cada pregunta sugerida, se muestra:
  - Similitud semántica con el contexto del usuario.
  - Penalización por repetición (si ya fue vista).
  - Bonus por pertenecer a un cluster temático poco cubierto.
  - Cluster temático asignado.
  - Score final y probabilidad de selección.

Esto permite a usuarios técnicos auditar y entender el proceso de selección, y a usuarios no técnicos confiar en la transparencia del sistema.

**Ejemplo de uso:**
```python
explanations = emb_mgr.explain_selection('contexto del usuario', history=[], top_k=3, technical=True)
for exp in explanations:
    print(exp['question'], exp['explanation'])
```

Esta función facilita la depuración, la mejora continua y la confianza en el sistema de IA.

## Ajuste Dinámico de Dificultad (Adaptive Testing)

El agente Ready4Hire ahora adapta automáticamente la dificultad de las preguntas (junior, mid, senior) según el desempeño reciente del usuario:

- Si el usuario responde correctamente 2 o más veces seguidas, sube el nivel de dificultad.
- Si falla 2 o más veces seguidas, baja el nivel de dificultad.
- Si no hay historial suficiente, mantiene el nivel actual o el inicial.

Esto permite una experiencia personalizada y retadora, acelerando el aprendizaje y evitando frustración o aburrimiento.

**Ejemplo:**
- Usuario acierta 2 preguntas junior → recibe una pregunta mid.
- Usuario falla 2 preguntas senior → recibe una pregunta mid o junior.

La lógica es transparente y auditable en el código (`_get_adaptive_level`).

## Aprendizaje Activo y Auto-Fine-Tuning

El sistema Ready4Hire ahora identifica y registra automáticamente:
- **Respuestas ambiguas**: aquellas cuya similitud semántica está cerca del umbral de corrección, para revisión o feedback adicional.
- **Errores frecuentes**: preguntas que el usuario falla varias veces, marcadas para priorizar en ciclos de mejora.

Estas interacciones se almacenan en el dataset de fine-tuning (`finetune_interactions.jsonl`) con flags `ambiguous` y `frequent_error`.

Esto permite:
- Mejorar el modelo y el dataset con ejemplos reales y casos límite.
- Solicitar feedback adicional al usuario o a expertos para refinar la evaluación.
- Automatizar el ciclo de aprendizaje y adaptación del agente.

## Mejoras Recientes (resumen y documentación)

- `advanced_question_selector` ahora acepta un parámetro opcional `custom_pool` para restringir la selección a un subconjunto de preguntas provisto por el llamador. Esto facilita integraciones donde el frontend o la lógica de sesión ya han pre-filtrado preguntas.
- `process_answer` fue corregido y robustecido: sanitización, logging, detección de emociones, cálculo de similitud y manejo de pistas quedan dentro del alcance correcto y protegidos contra fallos de logging/IO.
- Aprendizaje activo mejorado: se registran interacciones ambiguas y errores frecuentes tanto para respuestas incorrectas como para aciertos que resultaron ambiguos, y se exponen en `session['active_learning']` para revisión humana o procesos automáticos de etiquetado.
- Se añadieron pruebas unitarias básicas para los helpers `filter_by_level` y `penalize_covered_topics` en `tests/test_helpers.py`.

Estas mejoras están orientadas a confiabilidad en producción y a facilitar ciclos de mejora supervisada para los datasets y modelos.
