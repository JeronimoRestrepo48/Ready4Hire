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
