# API Documentation - Ready4Hire

## 🌐 Base URL
- **Development**: `http://localhost:8001`
- **Production (SSL)**: `https://localhost` or `https://ready4hire.local`

## 🔑 Authentication
The API supports JWT-based authentication for protected endpoints. Public endpoints (health check, metrics) don't require authentication.

### Security Features
- 🔐 **Input Validation**: Protección contra prompt injection, XSS y SQL injection
- 🛡️ **Output Sanitization**: Limpieza automática de respuestas del LLM
- 🚦 **Rate Limiting**: Límite de requests por IP
- 🔌 **Circuit Breaker**: Resiliencia ante fallos de servicios externos
- 📊 **Monitoring**: Métricas de Prometheus en `/metrics`

## 📡 Endpoints

### Interview Management

#### 1. Start Interview
**Endpoint**: `POST /start_interview`

**Description**: Initializes a new interview session for a user.

**Request Body**:
```json
{
  "user_id": "string (required)",
  "role": "string (required) - e.g., 'DevOps Engineer', 'Backend Developer'",
  "type": "string (required) - 'technical' or 'soft_skills'",
  "mode": "string (optional) - 'practice' or 'real', default: 'practice'"
}
```

**Response** (200 OK):
```json
{
  "message": "Entrevista iniciada exitosamente",
  "session": {
    "user_id": "string",
    "role": "string",
    "type": "string",
    "mode": "string",
    "questions_asked": 0,
    "current_score": 0,
    "level": 1,
    "points": 0
  },
  "first_question": "string"
}
```

**Error Responses**:
- `400 Bad Request`: Missing or invalid parameters
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST "https://localhost/start_interview" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "role": "DevOps Engineer",
    "type": "technical",
    "mode": "practice"
  }'
```

---

#### 2. Next Question
**Endpoint**: `POST /next_question`

**Description**: Retrieves the next question for the user's active interview session.

**Request Body**:
```json
{
  "user_id": "string (required)"
}
```

**Response** (200 OK):
```json
{
  "question": "string",
  "question_type": "technical|soft",
  "difficulty": "junior|mid|senior",
  "category": "string",
  "hints_available": "boolean"
}
```

**Error Responses**:
- `404 Not Found`: No active interview session
- `400 Bad Request`: user_id is required

**Example**:
```bash
curl -X POST "https://localhost/next_question" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

---

#### 3. Submit Answer
**Endpoint**: `POST /answer`

**Description**: Processes the user's answer and returns feedback with gamification elements.

**Request Body**:
```json
{
  "user_id": "string (required)",
  "answer": "string (required)"
}
```

**Response** (200 OK):
```json
{
  "feedback": "string - Detailed feedback about the answer",
  "score": "number - Score for this answer (0-100)",
  "is_correct": "boolean",
  "emotion_detected": "string - joy|sadness|fear|anger|neutral|surprise",
  "hints": ["array of strings - Conceptual hints if answer was incorrect"],
  "motivational_message": "string - Encouraging message",
  "next_question": "string",
  "gamification": {
    "points_earned": "number",
    "total_points": "number",
    "level": "number",
    "level_up": "boolean",
    "achievements_unlocked": ["array of achievement objects"],
    "badges": ["array of badge names"]
  },
  "progress": {
    "questions_answered": "number",
    "correct_answers": "number",
    "accuracy": "number - percentage"
  }
}
```

**Security Features**:
- Input sanitization applied automatically
- Prompt injection detection
- Forbidden patterns blocked

**Error Responses**:
- `400 Bad Request`: Input blocked by security policy or missing parameters
- `404 Not Found`: No active interview session
- `500 Internal Server Error`: Processing error

**Example**:
```bash
curl -X POST "https://localhost/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "answer": "Docker is a containerization platform that..."
  }'
```

---

#### 4. End Interview
**Endpoint**: `POST /end_interview`

**Description**: Finalizes the interview session and returns comprehensive results.

**Query Parameters**:
- `user_id` (string, required): User identifier

**Response** (200 OK):
```json
{
  "message": "Entrevista finalizada",
  "summary": {
    "user_id": "string",
    "role": "string",
    "total_questions": "number",
    "correct_answers": "number",
    "accuracy": "number",
    "total_score": "number",
    "final_level": "number",
    "total_points": "number",
    "achievements": ["array"],
    "badges": ["array"],
    "duration_minutes": "number"
  },
  "recommendations": [
    "string - Areas to improve"
  ],
  "strengths": [
    "string - Strong areas identified"
  ]
}
```

**Example**:
```bash
curl -X POST "https://localhost/end_interview?user_id=user123"
```

---

### Audio Processing

#### 5. Speech-to-Text (STT)
**Endpoint**: `POST /stt`

**Description**: Transcribes audio to text using OpenAI Whisper.

**Request** (multipart/form-data):
- `audio` (file, required): Audio file (WAV, MP3, M4A, etc.)
- `lang` (string, optional): Language code ('es', 'en', 'pt', etc.), default: 'es'

**Response** (200 OK):
```json
{
  "text": "string - Transcribed text"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid audio file
- `500 Internal Server Error`: Transcription error

**Example**:
```bash
curl -X POST "https://localhost/stt" \
  -F "audio=@recording.wav" \
  -F "lang=es"
```

---

#### 6. Text-to-Speech (TTS)
**Endpoint**: `POST /tts`

**Description**: Converts text to audio using pyttsx3.

**Request** (form-data):
- `text` (string, required): Text to synthesize
- `lang` (string, optional): Language code, default: 'es'

**Response** (200 OK):
- Content-Type: `audio/wav`
- Binary audio file

**Error Responses**:
- `400 Bad Request`: Missing text parameter
- `500 Internal Server Error`: Synthesis error

**Example**:
```bash
curl -X POST "https://localhost/tts" \
  -F "text=Bienvenido a la entrevista" \
  -F "lang=es" \
  -o output.wav
```

---

### Utility Endpoints

#### 7. Get Roles
**Endpoint**: `GET /get_roles`

**Description**: Returns all available job roles from the technical question bank.

**Response** (200 OK):
```json
{
  "roles": [
    "Backend Developer",
    "DevOps Engineer",
    "Frontend Developer",
    "Data Engineer",
    "Cloud Architect",
    "Site Reliability Engineer"
  ]
}
```

**Example**:
```bash
curl "https://localhost/get_roles"
```

---

#### 8. Get Levels
**Endpoint**: `GET /get_levels`

**Description**: Returns all experience levels supported.

**Response** (200 OK):
```json
{
  "levels": [
    "junior",
    "mid",
    "senior"
  ]
}
```

---

#### 9. Get Question Bank
**Endpoint**: `GET /get_question_bank`

**Description**: Returns filtered questions from the database.

**Query Parameters**:
- `role` (string, optional): Filter by job role
- `level` (string, optional): Filter by experience level

**Response** (200 OK):
```json
{
  "questions": [
    {
      "question": "string",
      "answer": "string",
      "role": "string",
      "level": "string",
      "category": "string",
      "tags": ["array"]
    }
  ]
}
```

**Example**:
```bash
curl "https://localhost/get_question_bank?role=DevOps%20Engineer&level=mid"
```

---

#### 10. Interview History
**Endpoint**: `GET /interview_history`

**Description**: Retrieves the complete history of an active interview session.

**Query Parameters**:
- `user_id` (string, required): User identifier

**Response** (200 OK):
```json
{
  "history": [
    {
      "question": "string",
      "answer": "string",
      "feedback": "string",
      "score": "number",
      "timestamp": "ISO 8601 datetime",
      "emotion": "string"
    }
  ]
}
```

**Error Responses**:
- `404 Not Found`: No active interview for this user

**Example**:
```bash
curl "https://localhost/interview_history?user_id=user123"
```

---

#### 11. Reset Interview
**Endpoint**: `POST /reset_interview`

**Description**: Resets/clears the interview session for a user.

**Query Parameters**:
- `user_id` (string, required): User identifier

**Response** (200 OK):
```json
{
  "message": "Entrevista reiniciada."
}
```

**Example**:
```bash
curl -X POST "https://localhost/reset_interview?user_id=user123"
```

---

## 🔒 Security Policies

### Input Sanitization
All user inputs are automatically:
1. Unicode normalized
2. Stripped of dangerous characters: `<>"'\``
3. Checked against forbidden patterns
4. Cleaned of excessive whitespace

### Blocked Patterns
The following patterns are automatically blocked:
- Prompt injection attempts
- System commands
- Code injection patterns
- Jailbreak attempts

### Rate Limiting (Recommended for Production)
- **Default**: 100 requests/minute per IP
- **Authenticated**: 1000 requests/minute per user

### CORS Policy
```python
# Default CORS (modify in production)
allow_origins = ["*"]
allow_methods = ["GET", "POST"]
allow_headers = ["*"]
```

---

## 📊 Response Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters or blocked by security |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

---

## 🧪 Testing Endpoints

### Using cURL
```bash
# Start interview
curl -X POST "https://localhost/start_interview" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test1","role":"DevOps Engineer","type":"technical"}'

# Get next question
curl -X POST "https://localhost/next_question" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test1"}'

# Submit answer
curl -X POST "https://localhost/answer" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test1","answer":"Docker permite..."}'
```

### Using Python requests
```python
import requests

BASE_URL = "https://localhost"

# Start interview
response = requests.post(
    f"{BASE_URL}/start_interview",
    json={
        "user_id": "test1",
        "role": "DevOps Engineer",
        "type": "technical"
    },
    verify=False  # Only for self-signed certs in dev
)
print(response.json())
```

### Using JavaScript fetch
```javascript
const BASE_URL = 'https://localhost';

// Start interview
fetch(`${BASE_URL}/start_interview`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'test1',
    role: 'DevOps Engineer',
    type: 'technical'
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 🔄 Workflow Example

1. **Initialize Interview**
   ```
   POST /start_interview → Get first question
   ```

2. **Interview Loop**
   ```
   POST /answer → Get feedback + next question
   (Repeat until interview complete)
   ```

3. **Finalize**
   ```
   POST /end_interview → Get summary and recommendations
   ```

4. **Optional: Query History**
   ```
   GET /interview_history → Review all Q&A
   ```

---

## 📈 Monitoring & Logs

### Audit Logs Location
- Path: `logs/audit_log.jsonl`
- Format: JSON Lines (one JSON object per line)

### Log Entry Example
```json
{
  "event": "answer_processed",
  "user_id": "user123",
  "timestamp": "2025-10-14T10:30:45.123Z",
  "answer_length": 150,
  "score": 85,
  "emotion": "joy",
  "sanitized": true,
  "ip_address": "192.168.1.100"
}
```

---

## 🚀 Performance Tips

1. **Use HTTP/2**: Enabled by default with Nginx
2. **Compress Responses**: Gzip enabled for JSON responses
3. **Cache Static Assets**: Frontend files cached by Nginx
4. **Connection Pooling**: Reuse connections to backend
5. **Async Processing**: All endpoints use async/await

---

## 📞 Support

For API issues or questions:
- GitHub Issues: [Report Bug](https://github.com/JeronimoRestrepo48/Ready4Hire/issues)
- Email: api-support@ready4hire.com
- Swagger UI: `https://localhost/docs`
- ReDoc: `https://localhost/redoc`

---

## 🔍 Monitoring & Observability

### Health Check
**Endpoint**: `GET /api/v2/health`

**Description**: Verifica el estado de todos los componentes del sistema.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-10-21T10:30:00Z",
  "components": {
    "llm_service": "healthy",
    "repositories": "healthy",
    "audio": "STT: ✅ TTS: ✅",
    "security": "healthy",
    "ml": "Embeddings: ✅"
  }
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "error": "Circuit breaker OPEN: Ollama unavailable",
  "timestamp": "2025-10-21T10:30:00Z"
}
```

---

### Metrics (Prometheus)
**Endpoint**: `GET /metrics`

**Description**: Exporta métricas en formato Prometheus.

**Métricas disponibles**:

- **Contadores**:
  - `ready4hire_http_requests_total`: Total de requests HTTP
  - `ready4hire_http_errors_total`: Total de errores HTTP
  - `ready4hire_llm_requests_total`: Total de requests al LLM
  - `ready4hire_llm_errors_total`: Total de errores del LLM
  - `ready4hire_evaluations_total`: Total de evaluaciones
  - `ready4hire_cache_hits_total`: Total de cache hits
  - `ready4hire_cache_misses_total`: Total de cache misses
  - `ready4hire_circuit_breaker_opens_total`: Total de circuit breaker opens
  - `ready4hire_prompt_injections_blocked_total`: Total de prompt injections bloqueados

- **Gauges**:
  - `ready4hire_active_interviews`: Entrevistas activas
  - `ready4hire_circuit_breaker_state`: Estado del circuit breaker (0=closed, 2=open)
  - `ready4hire_cache_hit_rate`: Porcentaje de cache hit rate
  - `ready4hire_llm_avg_latency_ms`: Latencia promedio del LLM

- **Histogramas (p50, p95, p99)**:
  - `ready4hire_http_request_duration_ms_*`: Latencia de requests HTTP
  - `ready4hire_llm_request_duration_ms`: Latencia de requests al LLM
  - `ready4hire_evaluation_duration_ms`: Latencia de evaluaciones

**Response** (200 OK):
```text
# HELP ready4hire_metrics Ready4Hire Application Metrics
# TYPE ready4hire_metrics untyped

# HELP ready4hire_counters Counter metrics
# TYPE ready4hire_counters counter
ready4hire_http_requests_total 1234
ready4hire_llm_requests_total 456
...
```

---

## 🔐 Security Best Practices

### Input Validation
Todos los inputs del usuario son validados automáticamente:

- **Longitud máxima**: 5000 caracteres
- **Caracteres permitidos**: UTF-8 sin caracteres de control
- **Patrones bloqueados**:
  - Prompt injection: `ignore previous instructions`, `system: you are`, etc.
  - XSS: `<script>`, `javascript:`, `onerror=`, etc.
  - SQL injection: `DROP`, `DELETE`, etc.
  - Repetición excesiva (spam)

**Ejemplo de error**:
```json
{
  "detail": "Contenido no permitido detectado: patrón sospechoso"
}
```

### Output Sanitization
Todas las respuestas del LLM son sanitizadas antes de enviarse al cliente:

- Remoción de scripts HTML
- Filtrado de información sensible (API keys, passwords)
- Truncado de outputs muy largos
- Normalización de espacios y newlines

### Rate Limiting
- **Por defecto**: 100 requests/minuto por IP
- **Login**: 5 intentos/minuto
- **Health check**: 30 requests/minuto

**Respuesta al exceder límite** (429 Too Many Requests):
```json
{
  "error": "Rate limit exceeded"
}
```

