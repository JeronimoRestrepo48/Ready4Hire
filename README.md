
# Ready4Hire - Simulador de Entrevistas IA

## Descripción General
Ready4Hire es un agente de IA que simula entrevistas técnicas y de soft skills para roles de ingeniería de sistemas, usando LLMs locales (Ollama), embeddings robustos y un frontend web tipo chat con audio (voz y texto), modo examen/práctica, feedback emocional y recursos de aprendizaje.

## Características
- Selección dinámica de preguntas técnicas y blandas usando embeddings y NLP.
- Feedback motivador, humano, emocional y personalizado (con emojis y frases motivacionales).
- Aprendizaje continuo: el sistema aprende de buenas respuestas y refresca embeddings.
- Resumen final con estadísticas, fortalezas y puntos de mejora.
- Seguridad: validación de entradas y protección contra inyección de prompts.
- Audio integrado: reconocimiento de voz (STT con Whisper) y síntesis de voz (TTS con pyttsx3).
- Modo examen (sin feedback inmediato, con temporizador) y modo práctica (feedback inmediato).
- Recursos de aprendizaje sugeridos tras cada respuesta incorrecta.
- Frontend webchat moderno, interactivo y accesible.
- **Análisis de emociones en las respuestas del usuario (transformers):** el feedback se adapta empáticamente según la emoción detectada (alegría, tristeza, frustración, etc.).
- **Aprendizaje automático de frases y emojis:** el sistema genera y aprende nuevas frases motivacionales y de refuerzo, y nuevos emojis, usando LLMs y deep learning.
- **Generación automática de nuevas preguntas técnicas:** tras cada buena respuesta, el sistema crea preguntas inéditas y las integra al dataset.

  
## Estructura del Proyecto

- `app/main.py`: Backend FastAPI, endpoints REST (entrevista, audio STT/TTS, seguridad, historial, bancos de preguntas) y webchat.
- `app/interview_agent.py`: Lógica del agente, selección de preguntas, feedback emocional/motivador, aprendizaje, modos examen/práctica, resumen final.
- `app/embeddings/embeddings_manager.py`: Gestión de embeddings, búsqueda semántica, refresco dinámico.
- `app/audio_utils.py`: Utilidades de audio (STT Whisper, TTS pyttsx3, manejo robusto de voces y archivos temporales).
- `app/datasets/`: Datasets de preguntas técnicas, soft skills y buenas respuestas (JSONL, con campo opcional 'resources').
- `app/static/`: Frontend webchat (HTML, CSS, JS) con controles de audio, temporizador y experiencia moderna.

  
## Consumo desde Backend Web

1. Iniciar el servidor FastAPI (`uvicorn app.main:app --reload`).

2. Consumir los endpoints principales vía POST con JSON:
   - `/start_interview`: Inicia la entrevista (requiere `user_id`, `role`, `type`, `mode`).
   - `/next_question`: Siguiente pregunta.
   - `/answer`: Envía respuesta del usuario.
   - `/end_interview`: Finaliza y entrega resumen.
   - `/stt`: Audio a texto (STT, POST con archivo de audio).
   - `/tts`: Texto a audio (TTS, POST con texto).
   - `/get_roles`, `/get_levels`, `/get_question_bank`: Utilidades para frontend.
   - `/interview_history`, `/reset_interview`: Historial y reinicio de sesión.

3. El frontend webchat ya está integrado en `/` y soporta audio, temporizador y modos.

  
## Ejemplo de Consumo API

```bash
# Iniciar entrevista
curl -X POST http://localhost:8000/start_interview -H 'Content-Type: application/json' -d '{"user_id": "test1", "role": "devops", "type": "technical", "mode": "practice"}'
# Siguiente pregunta
curl -X POST http://localhost:8000/next_question -H 'Content-Type: application/json' -d '{"user_id": "test1"}'
# Enviar respuesta
curl -X POST http://localhost:8000/answer -H 'Content-Type: application/json' -d '{"user_id": "test1", "answer": "mi respuesta"}'
# Finalizar entrevista
curl -X POST http://localhost:8000/end_interview?user_id=test1
# Audio a texto (STT)
curl -X POST http://localhost:8000/stt -F 'audio=@audio.wav' -F 'lang=es'
# Texto a audio (TTS)
curl -X POST http://localhost:8000/tts -F 'text=Hola, ¿cómo estás?' -F 'lang=es' --output tts.wav
```

  
## Flujo de Entrevista

1. El usuario responde preguntas de contexto (rol, nivel, años, conocimientos, herramientas, expectativas).
2. El agente selecciona las 10 preguntas técnicas y blandas más relevantes usando embeddings y contexto.
3. El usuario responde cada pregunta:
   - En modo práctica: recibe feedback inmediato, emocional y motivador (con emojis y frases), y recursos de aprendizaje si falla.
   - En modo examen: no recibe feedback inmediato, solo confirmación y temporizador activo.
4. Al finalizar, se entrega un resumen con estadísticas, fortalezas, puntos de mejora y tiempo total (modo examen).

  
## Seguridad

- Sanitización de entradas.
- Política anti-inyección de prompts.
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
