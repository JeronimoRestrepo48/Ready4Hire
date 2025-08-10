## API REST Detallada

Ready4Hire expone una API REST robusta y modular para la integración de frontend, automatización y pruebas. Todos los endpoints devuelven respuestas JSON estructuradas y gestionan el estado de la sesión, el feedback, el scoring y la gamificación.

### Endpoints principales

#### `POST /start_interview`
Inicia una nueva entrevista personalizada.
- **Parámetros:**
   - `user_id` (str, requerido): identificador único de usuario/sesión.
   - `role` (str, requerido): rol objetivo (ej: "IA", "devops").
   - `type` (str, requerido): "technical" o "soft".
   - `mode` (str, opcional): "practice" o "exam" (por defecto "practice").
- **Respuesta:**
   - `question`: primera pregunta de contexto.
   - `session_state`: estado inicial de la sesión.

#### `POST /next_question`
Devuelve la siguiente pregunta relevante según el contexto, desempeño y flujo adaptativo.
- **Parámetros:**
   - `user_id` (str, requerido)
- **Respuesta:**
   - `question`: texto de la pregunta (o encuesta de satisfacción si finalizó el bloque principal).
   - `type`: "technical" o "soft".
   - `session_state`: estado actualizado.

#### `POST /answer`
Envía la respuesta del usuario a la pregunta actual.
- **Parámetros:**
   - `user_id` (str, requerido)
   - `answer` (str, requerido)
- **Respuesta:**
   - `feedback`: mensaje motivador/adaptativo, con consejos personalizados y pistas si aplica.
   - `score`: puntaje obtenido (si modo examen).
   - `hint`: pista generada por LLM (si aplica).
   - `session_state`: estado actualizado, historial, nivel, logros, etc.

#### `POST /end_interview`
Finaliza la entrevista y entrega un resumen detallado.
- **Parámetros:**
   - `user_id` (str, requerido)
- **Respuesta:**
   - `summary`: resumen de desempeño, fortalezas, áreas de mejora, logros, recomendaciones y estadísticas.

#### `POST /stt`
Convierte audio a texto usando Whisper.
- **Parámetros:**
   - `audio` (archivo, requerido)
   - `lang` (str, opcional, por defecto "es")
- **Respuesta:**
   - `text`: transcripción del audio.

#### `POST /tts`
Convierte texto a audio usando pyttsx3.
- **Parámetros:**
   - `text` (str, requerido)
   - `lang` (str, opcional, por defecto "es")
- **Respuesta:**
   - Archivo de audio generado (stream).

#### Otros endpoints útiles
- `/get_roles`, `/get_levels`, `/get_question_bank`: Listados para autocompletado y selección en frontend.
- `/interview_history`: Devuelve el historial completo de la sesión.
- `/reset_interview`: Reinicia la sesión del usuario.

### Ejemplo de flujo API

```bash
# Iniciar entrevista
curl -X POST http://localhost:8000/start_interview -H 'Content-Type: application/json' -d '{"user_id": "test1", "role": "ia", "type": "technical", "mode": "practice"}'
# Siguiente pregunta
curl -X POST http://localhost:8000/next_question -H 'Content-Type: application/json' -d '{"user_id": "test1"}'
# Enviar respuesta
curl -X POST http://localhost:8000/answer -H 'Content-Type: application/json' -d '{"user_id": "test1", "answer": "mi respuesta"}'
# Finalizar entrevista
curl -X POST http://localhost:8000/end_interview -H 'Content-Type: application/json' -d '{"user_id": "test1"}'
# Audio a texto (STT)
curl -X POST http://localhost:8000/stt -F 'audio=@audio.wav' -F 'lang=es'
# Texto a audio (TTS)
curl -X POST http://localhost:8000/tts -F 'text=Hola, ¿cómo estás?' -F 'lang=es' --output tts.wav
```


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
