# Ready4Hire - Documentación de API y Seguridad (2025)

## Endpoints RESTful Principales

### 1. `/start_interview` (POST)
Inicia una nueva entrevista personalizada para un usuario y rol.
- **Body:**
  - `user_id` (str, requerido)
  - `role` (str, opcional)
  - `level` (str, opcional)
  - `mode` (str, opcional: 'practice'|'exam')
- **Respuesta:**
  - `{ "question": "...", "context_questions": { ... } }`

### 2. `/next_question` (POST)
Obtiene la siguiente pregunta personalizada (técnica o soft skill).
- **Body:**
  - `user_id` (str, requerido)
- **Respuesta:**
  - `{ "question": "..." }`

### 3. `/answer` (POST)
Procesa la respuesta del usuario, evalúa con IA y devuelve feedback adaptativo.
- **Body:**
  - `user_id` (str, requerido)
  - `answer` (str, requerido)
- **Respuesta:**
  - `{ "feedback": "...", "retry": bool, "points": int, "level": int, "badges": [str], "next": "...", "end": bool }`
  - Si `retry` es true, el usuario debe mejorar la respuesta antes de avanzar.
  - Si `end` es true, la entrevista ha finalizado.

### 4. `/end_interview` (POST)
Finaliza la entrevista y entrega feedback final, clustering de temas, recursos y analítica avanzada.
- **Body:**
  - `user_id` (str, requerido)
  - `satisfaction` (float, opcional)
- **Respuesta:**
  - `{ "summary": "...", "temas_debiles": [str], "recursos": [str], "reconocimiento": "...", "score": int, "level": int, "points": int, "badges": [str], "errors": int, "time": int }`

### 5. `/stt` (POST)
Convierte audio a texto (Speech-to-Text).
- **Form-data:**
  - `audio` (file, requerido)
  - `lang` (str, opcional)
- **Respuesta:**
  - `{ "text": "..." }`

### 6. `/tts` (POST)
Convierte texto a audio (Text-to-Speech).
- **Form-data:**
  - `text` (str, requerido)
  - `lang` (str, opcional)
- **Respuesta:**
  - Archivo de audio (WAV/MP3)

### 7. `/get_roles` (GET)
Lista todos los roles/cargos disponibles en el dataset técnico.
- **Respuesta:**
  - `{ "roles": ["IA", "DevOps", ...] }`

### 8. `/get_levels` (GET)
Lista los niveles de experiencia soportados.
- **Respuesta:**
  - `{ "levels": ["junior", "senior", ...] }`

### 9. `/get_question_bank` (GET)
Devuelve todas las preguntas filtradas por rol, nivel y dificultad.
- **Query params:**
  - `role` (str, opcional)
  - `level` (str, opcional)
  - `difficulty` (int, opcional)
- **Respuesta:**
  - `{ "questions": [ ... ] }`

### 10. `/interview_history` (GET)
Consulta el historial de la entrevista en curso de un usuario.
- **Query params:**
  - `user_id` (str, requerido)
- **Respuesta:**
  - `{ "history": [ ... ] }`

### 11. `/reset_interview` (POST)
Reinicia la entrevista para un usuario.
- **Body:**
  - `user_id` (str, requerido)
- **Respuesta:**
  - `{ "status": "reset" }`

### 12. `/survey` (POST)
Registra la encuesta de satisfacción y comentarios del usuario.
- **Body:**
  - `user_id` (str, requerido)
  - `rating` (int, requerido)
  - `comments` (str, opcional)
- **Respuesta:**
  - `{ "status": "ok" }`

---

## Seguridad y Buenas Prácticas
- Todos los endpoints requieren autenticación JWT (excepto `/stt` y `/tts`).
- Validación robusta de datos, protección contra inyección y XSS.
- Rate limiting, CORS estricto, HTTPS/HSTS, headers y cookies seguras.
- Logging y analítica de eventos para monitoreo y mejora continua.

---

## Respuestas y Campos Avanzados
- El feedback puede incluir campos adicionales: `temas_debiles`, `recursos`, `reconocimiento`, `badges`, `errors`, `time`, `learning_path`.
- El feedback y los recursos se adaptan dinámicamente según el desempeño, emociones y clustering de temas del usuario.

---
Última actualización: agosto 2025
