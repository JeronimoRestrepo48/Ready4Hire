

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

## Ejemplo de Consumo API

```bash
# Iniciar entrevista
curl -X POST http://localhost:8000/start_interview -H 'Content-Type: application/json' -d '{"user_id": "test1", "role": "devops", "type": "technical", "mode": "practice"}'
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
