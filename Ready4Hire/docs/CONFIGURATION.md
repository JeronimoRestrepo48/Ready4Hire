# 🛠️ Ready4Hire - Guía de Configuración

## Tabla de Contenidos

1. [Variables de Entorno](#variables-de-entorno)
2. [Configuración de Ollama](#configuración-de-ollama)
3. [Configuración de API](#configuración-de-api)
4. [Configuración de Seguridad](#configuración-de-seguridad)
5. [Configuración de ML](#configuración-de-ml)
6. [Configuración de Audio](#configuración-de-audio)
7. [Personalización de Preguntas](#personalización-de-preguntas)
8. [Configuración de Logging](#configuración-de-logging)

---

## Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=512

# =============================================================================
# API CONFIGURATION
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=true
API_TITLE="Ready4Hire API v2"
API_VERSION="2.0.0"

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================
MAX_INPUT_LENGTH=2000
ENABLE_PROMPT_INJECTION_DETECTION=true
PROMPT_INJECTION_THRESHOLD=0.5
ENABLE_INPUT_SANITIZATION=true

# =============================================================================
# ML MODELS CONFIGURATION
# =============================================================================
EMOTION_MODEL_ES=finiteautomata/bertweet-base-emotion-analysis
EMOTION_MODEL_EN=j-hartmann/emotion-english-distilroberta-base
EMBEDDING_MODEL=all-MiniLM-L6-v2
RANKNET_MODEL_PATH=app/datasets/ranknet_model.pt

# =============================================================================
# AUDIO SERVICES (OPCIONAL)
# =============================================================================
ENABLE_STT=false
ENABLE_TTS=false
WHISPER_MODEL=base
TTS_RATE=150
TTS_VOLUME=1.0

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL=INFO
LOG_FILE=logs/ready4hire.log
AUDIT_LOG_FILE=logs/audit_log.jsonl

# =============================================================================
# DATABASE (FUTURO)
# =============================================================================
# DATABASE_URL=postgresql://user:password@localhost:5432/ready4hire
# DATABASE_POOL_SIZE=10
# DATABASE_POOL_TIMEOUT=30

# =============================================================================
# REDIS CACHE (FUTURO)
# =============================================================================
# REDIS_URL=redis://localhost:6379/0
# REDIS_TTL=3600
```

---

## Configuración de Ollama

### 1. Instalación

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows (requiere WSL2)
wsl curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Modelos Recomendados

| Modelo | Tamaño | RAM Mínima | Velocidad | Calidad |
|--------|--------|------------|-----------|---------|
| `llama3.2:1b` | 1.3GB | 4GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| `llama3.2:3b` | 2GB | 8GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| `llama3:latest` (8B) | 4.7GB | 16GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| `llama3.1:8b` | 4.7GB | 16GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |

### 3. Descargar Modelos

```bash
# Recomendado para desarrollo
ollama pull llama3.2:3b

# Alternativas
ollama pull llama3:latest
ollama pull llama3.1:8b
```

### 4. Configuración del Servidor

**Archivo**: `/etc/systemd/system/ollama.service` (Linux)

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"

[Install]
WantedBy=default.target
```

**Comandos**:

```bash
# Iniciar servicio
sudo systemctl start ollama

# Habilitar en inicio
sudo systemctl enable ollama

# Ver estado
sudo systemctl status ollama

# Ver logs
sudo journalctl -u ollama -f
```

### 5. Variables de Entorno de Ollama

```bash
# Host y puerto
export OLLAMA_HOST=0.0.0.0:11434

# Directorio de modelos
export OLLAMA_MODELS=/path/to/models

# Límite de memoria (GB)
export OLLAMA_MAX_LOADED_MODELS=1

# Timeout de carga
export OLLAMA_LOAD_TIMEOUT=5m

# Keep alive (tiempo de inactividad antes de descargar)
export OLLAMA_KEEP_ALIVE=5m
```

---

## Configuración de API

### 1. FastAPI Settings

**Archivo**: `app/config.py` (crear si no existe)

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = True
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_MAX_TOKENS: int = 512
    
    # Security
    MAX_INPUT_LENGTH: int = 2000
    ENABLE_PROMPT_INJECTION_DETECTION: bool = True
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 2. Configuración de CORS

En `app/main_v2.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### 3. Rate Limiting (Opcional)

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v2/interviews")
@limiter.limit("10/minute")
async def start_interview(request: Request, ...):
    ...
```

---

## Configuración de Seguridad

### 1. Input Sanitization

En `app/infrastructure/security/input_sanitizer.py`:

```python
class InputSanitizer:
    def __init__(
        self,
        max_length: int = 2000,
        remove_html: bool = True,
        remove_sql: bool = True
    ):
        self.max_length = max_length
        self.remove_html = remove_html
        self.remove_sql = remove_sql
```

**Uso**:

```python
from app.infrastructure.security import get_sanitizer

sanitizer = get_sanitizer()
clean_text = sanitizer.sanitize(user_input)
```

### 2. Prompt Injection Detection

En `app/infrastructure/security/prompt_injection_guard.py`:

```python
guard = PromptInjectionGuard(threshold=0.5)
result = guard.detect(user_input)

if result['is_injection']:
    raise HTTPException(
        status_code=400,
        detail=f"Prompt injection detected: {result['patterns']}"
    )
```

### 3. Patrones Detectados

- `ignore previous instructions`
- `system: you are now`
- `</s>`, `<|endoftext|>`
- `{{`, `}}`, `[[`, `]]`
- SQL injection patterns
- XSS patterns

---

## Configuración de ML

### 1. Emotion Detection

**Modelos por idioma**:

```python
EMOTION_MODELS = {
    'es': 'finiteautomata/bertweet-base-emotion-analysis',
    'en': 'j-hartmann/emotion-english-distilroberta-base',
    'fr': 'j-hartmann/emotion-english-distilroberta-base'  # Fallback
}
```

**Configuración**:

```python
detector = MultilingualEmotionDetector(
    device='cpu',  # o 'cuda' si tienes GPU
    models=EMOTION_MODELS
)
```

### 2. Difficulty Adjustment (RankNet)

**Entrenamiento del modelo**:

```bash
cd app
python train_ranknet.py \
    --data datasets/finetune_interactions.jsonl \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

**Uso**:

```python
adjuster = NeuralDifficultyAdjuster(
    model_path='app/datasets/ranknet_model.pt'
)

prediction = adjuster.predict(
    question_text="¿Qué es Docker?",
    current_difficulty=1,  # junior=1, mid=2, senior=3
    streak=3,
    time_taken=45
)
```

### 3. Question Embeddings

**Modelo**:

```python
embeddings_service = get_embeddings_service(
    model_name='all-MiniLM-L6-v2',
    ranknet_model_path='app/datasets/ranknet_model.pt'
)
```

**Generar embeddings**:

```bash
cd app
python -c "
from infrastructure.ml.question_embeddings import get_embeddings_service
service = get_embeddings_service()
service.generate_embeddings_for_questions('datasets/tech_questions.jsonl')
"
```

---

## Configuración de Audio

### 1. Speech-to-Text (Whisper)

**Instalación**:

```bash
pip install openai-whisper
```

**Configuración**:

```python
from app.infrastructure.audio import get_stt_service

stt = get_stt_service(
    model_name='base',  # tiny, base, small, medium, large
    device='cpu'
)

text = stt.transcribe('audio.wav')
```

**Modelos Disponibles**:

| Modelo | Tamaño | RAM | Velocidad | Calidad |
|--------|--------|-----|-----------|---------|
| `tiny` | 39M | 1GB | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| `base` | 74M | 1GB | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| `small` | 244M | 2GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `medium` | 769M | 5GB | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| `large` | 1550M | 10GB | ⚡ | ⭐⭐⭐⭐⭐ |

### 2. Text-to-Speech (pyttsx3)

**Instalación**:

```bash
pip install pyttsx3
```

**Configuración**:

```python
from app.infrastructure.audio import get_tts_service

tts = get_tts_service(
    rate=150,  # Palabras por minuto
    volume=1.0  # 0.0 a 1.0
)

tts.speak("Hola, bienvenido a Ready4Hire")
```

---

## Personalización de Preguntas

### 1. Formato JSONL

**Archivo**: `app/datasets/tech_questions.jsonl`

```json
{
  "text": "¿Qué es Docker y para qué se utiliza?",
  "category": "technical",
  "difficulty": "junior",
  "topic": "DevOps",
  "keywords": ["docker", "contenedores", "virtualización"],
  "expected_concepts": ["contenedores", "aislamiento", "portabilidad"],
  "hints": [
    "Piensa en cómo empaquetar aplicaciones",
    "Es una tecnología de virtualización ligera"
  ],
  "sample_answer": "Docker es una plataforma que permite crear, implementar y ejecutar aplicaciones en contenedores..."
}
```

### 2. Estructura de Campos

| Campo | Tipo | Obligatorio | Descripción |
|-------|------|-------------|-------------|
| `text` | string | ✅ | Texto de la pregunta |
| `category` | string | ✅ | `technical` o `soft_skills` |
| `difficulty` | string | ✅ | `junior`, `mid`, `senior` |
| `topic` | string | ❌ | Tema (DevOps, Databases, etc.) |
| `keywords` | array | ❌ | Palabras clave esperadas |
| `expected_concepts` | array | ❌ | Conceptos a detectar en respuesta |
| `hints` | array | ❌ | Pistas progresivas |
| `sample_answer` | string | ❌ | Respuesta de ejemplo |

### 3. Agregar Nuevas Preguntas

```bash
# Editar archivo
vim app/datasets/tech_questions.jsonl

# Agregar línea (formato JSONL - una línea por pregunta)
{"text": "Nueva pregunta?", "category": "technical", "difficulty": "mid"}

# Reiniciar API para cargar cambios
pkill -f uvicorn
./scripts/start_api.sh
```

### 4. Validar Preguntas

```python
import json
import jsonschema

schema = {
    "type": "object",
    "required": ["text", "category", "difficulty"],
    "properties": {
        "text": {"type": "string", "minLength": 10},
        "category": {"enum": ["technical", "soft_skills"]},
        "difficulty": {"enum": ["junior", "mid", "senior"]}
    }
}

with open('app/datasets/tech_questions.jsonl', 'r') as f:
    for line in f:
        question = json.loads(line)
        jsonschema.validate(question, schema)
        print(f"✅ {question['text'][:50]}")
```

---

## Configuración de Logging

### 1. Configuración Básica

**Archivo**: `app/logging_config.py`

```python
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO"):
    """Configura logging para toda la aplicación"""
    
    # Crear directorio de logs
    Path("logs").mkdir(exist_ok=True)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para archivo
    file_handler = logging.FileHandler('logs/ready4hire.log')
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Silenciar loggers ruidosos
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

### 2. Uso en la Aplicación

```python
import logging

logger = logging.getLogger(__name__)

# Niveles
logger.debug("Información detallada")
logger.info("Información general")
logger.warning("Advertencia")
logger.error("Error")
logger.critical("Error crítico")
```

### 3. Audit Log

**Archivo**: `logs/audit_log.jsonl`

```python
import json
from datetime import datetime

def log_audit_event(event_type: str, data: dict):
    """Registra evento de auditoría"""
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        **data
    }
    
    with open('logs/audit_log.jsonl', 'a') as f:
        f.write(json.dumps(event) + '\n')

# Ejemplo
log_audit_event("prompt_injection_detected", {
    "user_id": "user_123",
    "input": "ignore previous instructions",
    "patterns": ["ignore previous instructions"]
})
```

---

## Configuración Avanzada

### 1. Docker

**Archivo**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434

  ready4hire:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OLLAMA_MODEL=llama3.2:3b
    depends_on:
      - ollama
    volumes:
      - ./logs:/app/logs

volumes:
  ollama_data:
```

### 2. Nginx Reverse Proxy

**Archivo**: `nginx/ready4hire.conf`

```nginx
server {
    listen 80;
    server_name ready4hire.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. Systemd Service

**Archivo**: `/etc/systemd/system/ready4hire.service`

```ini
[Unit]
Description=Ready4Hire API Service
After=network.target ollama.service

[Service]
Type=simple
User=ready4hire
WorkingDirectory=/opt/ready4hire
ExecStart=/opt/ready4hire/venv/bin/uvicorn app.main_v2:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

---

## Referencias

- [FastAPI Settings](https://fastapi.tiangolo.com/advanced/settings/)
- [Ollama Documentation](https://ollama.com/docs)
- [Pydantic Settings](https://docs.pydantic.dev/latest/usage/settings/)
- [Python Logging](https://docs.python.org/3/library/logging.html)

---

**Configuración completada ✅**
