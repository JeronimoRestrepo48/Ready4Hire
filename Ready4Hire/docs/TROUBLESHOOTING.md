# 🔧 Ready4Hire - Guía de Resolución de Problemas

## Tabla de Contenidos

1. [Problemas de Instalación](#problemas-de-instalación)
2. [Problemas con Ollama](#problemas-con-ollama)
3. [Problemas con la API](#problemas-con-la-api)
4. [Problemas con Tests](#problemas-con-tests)
5. [Problemas de Rendimiento](#problemas-de-rendimiento)
6. [Errores Comunes](#errores-comunes)
7. [Debugging Avanzado](#debugging-avanzado)

---

## Problemas de Instalación

### ❌ Error: `ModuleNotFoundError: No module named 'app'`

**Síntoma**:
```bash
$ uvicorn app.main_v2:app
ModuleNotFoundError: No module named 'app'
```

**Causas**:
- Ejecutando desde directorio incorrecto
- PYTHONPATH no configurado
- Estructura de paquetes incorrecta

**Solución 1**: Ejecutar desde raíz del proyecto

```bash
cd /path/to/Ready4Hire
python -m uvicorn app.main_v2:app --host 0.0.0.0 --port 8000
```

**Solución 2**: Configurar PYTHONPATH

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Ready4Hire"
uvicorn app.main_v2:app
```

**Solución 3**: Usar script de inicio

```bash
./scripts/start_api.sh
```

---

### ❌ Error: `pip install` falla con dependencias

**Síntoma**:
```bash
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**Solución 1**: Actualizar pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

**Solución 2**: Instalar dependencias por grupos

```bash
# Core dependencies
pip install fastapi uvicorn pydantic httpx

# ML dependencies
pip install torch transformers sentence-transformers

# Audio dependencies (opcional)
pip install openai-whisper pyttsx3
```

**Solución 3**: Usar requirements específicos

```bash
# Instalación mínima (sin audio)
pip install -r requirements.txt --no-deps
pip install -r requirements.txt

# Con cache de pip
pip install -r requirements.txt --cache-dir ~/.pip/cache
```

---

### ❌ Error: Permisos insuficientes

**Síntoma**:
```bash
PermissionError: [Errno 13] Permission denied: '/usr/local/bin/ollama'
```

**Solución**:

```bash
# Instalar con permisos de usuario
pip install --user -r requirements.txt

# O usar virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Problemas con Ollama

### ❌ Error: `Connection refused` al conectar con Ollama

**Síntoma**:
```python
httpx.ConnectError: [Errno 111] Connection refused
```

**Diagnóstico**:

```bash
# Verificar si Ollama está corriendo
ps aux | grep ollama

# Verificar puerto
lsof -i :11434
```

**Solución 1**: Iniciar Ollama

```bash
# Inicio manual
ollama serve > /tmp/ollama.log 2>&1 &

# Como servicio (Linux)
sudo systemctl start ollama
sudo systemctl enable ollama

# Verificar estado
curl http://localhost:11434/api/tags
```

**Solución 2**: Verificar configuración

```bash
# Ver variables de entorno
env | grep OLLAMA

# Configurar correctamente
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS="*"
```

---

### ❌ Error: Modelo no encontrado

**Síntoma**:
```json
{"error": "model 'llama3.2:3b' not found"}
```

**Diagnóstico**:

```bash
# Listar modelos instalados
ollama list

# Verificar espacio en disco
df -h ~/.ollama
```

**Solución**:

```bash
# Descargar modelo
ollama pull llama3.2:3b

# Alternativa con modelo más pequeño
ollama pull llama3.2:1b

# Verificar instalación
ollama run llama3.2:3b "Hello"
```

---

### ❌ Error: Ollama muy lento o se queda colgado

**Síntoma**:
- Respuestas tardan >60 segundos
- CPU al 100% constante
- OOM (Out of Memory)

**Diagnóstico**:

```bash
# Monitorear recursos
top -p $(pgrep ollama)

# Ver memoria disponible
free -h

# Ver uso de GPU (si aplica)
nvidia-smi
```

**Solución 1**: Usar modelo más pequeño

```bash
# Modelo pequeño (1-2GB RAM)
ollama pull llama3.2:1b
export OLLAMA_MODEL=llama3.2:1b

# Modelo mediano (4-8GB RAM)
ollama pull llama3.2:3b
export OLLAMA_MODEL=llama3.2:3b
```

**Solución 2**: Limitar modelos cargados

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
ollama serve
```

**Solución 3**: Ajustar parámetros

En `.env`:
```bash
OLLAMA_MAX_TOKENS=256  # Reducir de 512
OLLAMA_TEMPERATURE=0.5  # Reducir de 0.7
```

---

## Problemas con la API

### ❌ Error: API no responde en el puerto esperado

**Síntoma**:
```bash
$ curl http://localhost:8000/api/v2/health
curl: (7) Failed to connect to localhost port 8000
```

**Diagnóstico**:

```bash
# Ver procesos en puerto 8000
lsof -i :8000

# Ver logs de API
tail -f logs/ready4hire.log

# Verificar uvicorn
ps aux | grep uvicorn
```

**Solución**:

```bash
# Matar proceso anterior
pkill -f uvicorn

# Iniciar API correctamente
cd /path/to/Ready4Hire
python -m uvicorn app.main_v2:app --host 0.0.0.0 --port 8000

# Verificar health endpoint
curl http://localhost:8000/api/v2/health
```

---

### ❌ Error 500: Internal Server Error

**Síntoma**:
```json
{"detail": "Internal Server Error"}
```

**Diagnóstico**:

```bash
# Ver logs detallados
tail -100 logs/ready4hire.log

# Modo debug
LOG_LEVEL=DEBUG python -m uvicorn app.main_v2:app --reload
```

**Solución 1**: Verificar dependencias

```python
# test_dependencies.py
from app.container import Container

try:
    container = Container()
    result = container.health_check()
    print(f"✅ Container OK: {result}")
except Exception as e:
    print(f"❌ Error: {e}")
```

**Solución 2**: Revisar configuración

```bash
# Verificar .env existe
ls -la .env

# Verificar variables críticas
grep -E "OLLAMA|API" .env
```

---

### ❌ Error: CORS bloqueando requests

**Síntoma**:
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solución**:

En `app/main_v2.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Problemas con Tests

### ❌ Tests fallan: `fixture 'container' not found`

**Síntoma**:
```bash
E   fixture 'container' not found
```

**Solución**:

Verificar `conftest.py` existe:

```python
# tests/conftest.py
import pytest
from app.container import Container

@pytest.fixture
def container():
    """Fixture que proporciona Container"""
    return Container()
```

---

### ❌ Tests fallan: Timeout al conectar con Ollama

**Síntoma**:
```
httpx.ReadTimeout: Read operation timed out
```

**Solución 1**: Iniciar Ollama antes de tests

```bash
# Iniciar Ollama
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

# Ejecutar tests
pytest tests/ -v
```

**Solución 2**: Aumentar timeout en tests

```python
# tests/test_integration.py
import httpx

client = httpx.Client(
    base_url="http://localhost:11434",
    timeout=60.0  # Aumentar de 30 a 60 segundos
)
```

**Solución 3**: Usar mocks

```python
# tests/mocks.py
class MockOllamaClient:
    def generate(self, prompt: str) -> str:
        return "Respuesta mock: Docker es una plataforma..."

@pytest.fixture
def mock_ollama(monkeypatch):
    monkeypatch.setattr("app.infrastructure.llm.ollama_client.OllamaClient", MockOllamaClient)
```

---

### ❌ Tests pasan individualmente pero fallan en suite

**Síntoma**:
```bash
$ pytest tests/test_interview.py  # ✅ PASS
$ pytest tests/  # ❌ FAIL en test_interview.py
```

**Causa**: Estado compartido entre tests

**Solución**:

```python
import pytest

@pytest.fixture(autouse=True)
def reset_state():
    """Reset estado antes de cada test"""
    # Limpiar cache
    from app.infrastructure.cache import cache
    cache.clear()
    
    # Reset singletons
    from app.container import Container
    Container._instance = None
    
    yield
    
    # Cleanup después del test
    cache.clear()
```

---

## Problemas de Rendimiento

### 🐌 API responde muy lento (>5 segundos)

**Diagnóstico**:

```python
import time

def benchmark_endpoint(url: str, payload: dict):
    start = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start
    print(f"⏱️ {url} tomó {elapsed:.2f}s")
    return response

benchmark_endpoint(
    "http://localhost:8000/api/v2/interviews/123/answer",
    {"answer": "Docker es una plataforma..."}
)
```

**Solución 1**: Cachear embeddings

```python
# app/infrastructure/ml/embeddings_service.py
from functools import lru_cache

class EmbeddingsService:
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str):
        return self.model.encode(text)
```

**Solución 2**: Usar modelo más rápido

```bash
# Cambiar de llama3:8b a llama3.2:3b
export OLLAMA_MODEL=llama3.2:3b

# O incluso llama3.2:1b para máxima velocidad
export OLLAMA_MODEL=llama3.2:1b
```

**Solución 3**: Reducir max_tokens

```python
# .env
OLLAMA_MAX_TOKENS=256  # En lugar de 512
```

---

### 🐌 Primer request es muy lento (~30 segundos)

**Causa**: Cold start - modelo se está cargando en memoria

**Solución**: Warm-up al inicio

```python
# app/main_v2.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: cargar modelo
    container = Container()
    llm_service = container.llm_service()
    await llm_service.generate("Hello")
    
    yield
    
    # Cleanup
    pass

app = FastAPI(lifespan=lifespan)
```

---

### 💾 Uso excesivo de memoria (>8GB)

**Diagnóstico**:

```bash
# Ver uso de memoria
ps aux --sort=-%mem | head -10

# Memory profiling
pip install memory_profiler
python -m memory_profiler app/main_v2.py
```

**Solución 1**: Limitar modelos ML cargados

```python
# app/container.py
class Container:
    def __init__(self):
        # Solo cargar si es necesario
        self._emotion_detector = None
    
    def emotion_detector(self):
        if self._emotion_detector is None:
            self._emotion_detector = MultilingualEmotionDetector(device='cpu')
        return self._emotion_detector
```

**Solución 2**: Desactivar servicios opcionales

```bash
# .env
ENABLE_STT=false
ENABLE_TTS=false
ENABLE_EMOTION_DETECTION=false
```

---

## Errores Comunes

### ❌ `RuntimeError: Expected all tensors to be on the same device`

**Síntoma**:
```python
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0
```

**Solución**:

```python
# Forzar CPU en todos los modelos
import torch

device = 'cpu'  # No usar 'cuda'

model = AutoModel.from_pretrained('model-name').to(device)
```

---

### ❌ `JSONDecodeError: Expecting value: line 1 column 1`

**Síntoma**:
```python
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Causa**: Respuesta vacía o no-JSON de Ollama

**Solución**:

```python
import httpx
import json

def safe_ollama_request(prompt: str):
    try:
        response = client.post("/api/generate", json={
            "model": "llama3.2:3b",
            "prompt": prompt
        })
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")
        print(f"Raw response: {response.text}")
        return None
```

---

### ❌ `PromptInjectionDetected: Detected prompt injection patterns`

**Síntoma**:
```json
{"detail": "Prompt injection detected: ['ignore previous instructions']"}
```

**Causa**: Input del usuario contiene patrones sospechosos

**Solución 1**: Ajustar threshold

```python
# app/infrastructure/security/prompt_injection_guard.py
guard = PromptInjectionGuard(threshold=0.7)  # Más permisivo (era 0.5)
```

**Solución 2**: Desactivar temporalmente

```bash
# .env
ENABLE_PROMPT_INJECTION_DETECTION=false
```

**Solución 3**: Whitelist de patrones legítimos

```python
ALLOWED_PATTERNS = [
    "ignore case",  # SQL
    "system design",  # Pregunta legítima
]

if any(pattern in user_input.lower() for pattern in ALLOWED_PATTERNS):
    # Permitir
    pass
```

---

## Debugging Avanzado

### 🔍 Activar logs detallados

```bash
# .env
LOG_LEVEL=DEBUG

# O en runtime
LOG_LEVEL=DEBUG uvicorn app.main_v2:app --reload
```

**Ver logs específicos**:

```python
import logging

# Activar logger de Ollama
logging.getLogger("app.infrastructure.llm").setLevel(logging.DEBUG)

# Activar logger de evaluación
logging.getLogger("app.application.services.evaluation").setLevel(logging.DEBUG)
```

---

### 🔍 Profiling de rendimiento

```bash
# Instalar
pip install py-spy

# Profiling en tiempo real
py-spy top -- python -m uvicorn app.main_v2:app

# Generar flamegraph
py-spy record -o profile.svg -- python -m uvicorn app.main_v2:app
```

---

### 🔍 Debugging de container DI

```python
# debug_container.py
from app.container import Container

container = Container()

# Ver todas las dependencias
print("🔍 Verificando container...")
print(f"✅ LLM Service: {container.llm_service()}")
print(f"✅ Evaluation Service: {container.evaluation_service()}")
print(f"✅ Question Service: {container.question_service()}")
print(f"✅ Interview Service: {container.interview_service()}")

# Health check
result = container.health_check()
print(f"\n📊 Health Check: {result}")
```

---

### 🔍 Debugging de Ollama requests

```python
# debug_ollama.py
import httpx
import json

client = httpx.Client(base_url="http://localhost:11434")

# Test 1: List models
response = client.get("/api/tags")
print(f"📦 Models: {response.json()}")

# Test 2: Generate
response = client.post("/api/generate", json={
    "model": "llama3.2:3b",
    "prompt": "Say hello",
    "stream": False
})
print(f"💬 Response: {response.json()}")

# Test 3: Embeddings
response = client.post("/api/embeddings", json={
    "model": "llama3.2:3b",
    "prompt": "Hello world"
})
print(f"🔢 Embeddings shape: {len(response.json()['embedding'])}")
```

---

## Obtener Ayuda

### 📝 Reporte de Bug

Incluir:

1. **Versión**: `cat README.md | grep "Version"`
2. **Sistema**: `uname -a`
3. **Python**: `python --version`
4. **Logs**: `cat logs/ready4hire.log | tail -100`
5. **Configuración**: `.env` (sin secretos)
6. **Pasos para reproducir**

### 📧 Contacto

- **GitHub Issues**: [Ready4Hire/issues](https://github.com/yourusername/Ready4Hire/issues)
- **Email**: support@ready4hire.example.com
- **Discord**: [Ready4Hire Community](https://discord.gg/ready4hire)

---

## Checklist de Diagnóstico

Antes de reportar un bug, verificar:

- [ ] Ollama está corriendo: `curl http://localhost:11434/api/tags`
- [ ] Modelo descargado: `ollama list | grep llama3.2`
- [ ] Dependencias instaladas: `pip check`
- [ ] Tests pasan: `pytest tests/ -v`
- [ ] API responde: `curl http://localhost:8000/api/v2/health`
- [ ] Logs sin errores: `tail logs/ready4hire.log`
- [ ] Puerto libre: `lsof -i :8000`
- [ ] Variables de entorno: `cat .env`

---

## Problemas Resueltos

✅ = Problema común ya solucionado en versión actual

- ✅ ModuleNotFoundError al iniciar API
- ✅ Tests fallan por falta de Ollama
- ✅ CORS bloqueando frontend
- ✅ Primer request muy lento (cold start)
- ✅ Prompt injection demasiado estricto

---

## Conclusión

Si después de revisar esta guía el problema persiste:

1. **Activar DEBUG**: `LOG_LEVEL=DEBUG`
2. **Revisar logs**: `tail -f logs/ready4hire.log`
3. **Ejecutar tests**: `pytest tests/ -v`
4. **Health check**: `curl http://localhost:8000/api/v2/health`
5. **Reportar issue** con toda la información

¡Buena suerte! 🚀
