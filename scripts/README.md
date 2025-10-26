# 🧪 Scripts de Testing Local

Scripts para ejecutar tests localmente sin necesidad de desplegar toda la infraestructura.

---

## 📋 Scripts Disponibles

### 1. `test-local.sh` (Linux/macOS)
Script bash para sistemas Unix que ejecuta tests locales.

### 2. `test-local.ps1` (Windows)
Script PowerShell para Windows que ejecuta tests locales.

---

## 🚀 Uso Rápido

### Linux/macOS

```bash
# Desde la raíz del proyecto
./scripts/test-local.sh
```

### Windows (PowerShell)

```powershell
# Desde la raíz del proyecto
.\scripts\test-local.ps1
```

---

## 🔧 ¿Qué Hacen los Scripts?

Los scripts ejecutan automáticamente:

1. ✅ **Verificación de Prerrequisitos**
   - Python 3.11+
   - .NET 9.0 (opcional, para tests de frontend)
   - Node.js (opcional, para tests E2E)

2. ✅ **Setup del Entorno**
   - Crea entorno virtual Python si no existe
   - Instala dependencias automáticamente

3. ✅ **Tests Unitarios Backend**
   - Ejecuta todos los tests unitarios
   - Genera reporte de coverage
   - **No requiere servicios externos** (Redis, PostgreSQL, etc)

4. ⚠️ **Tests de Integración (Opcional)**
   - Requiere servicios corriendo (Ollama, PostgreSQL, Redis)
   - Se pregunta al usuario si desea ejecutarlos

5. 🎨 **Linting & Code Quality (Opcional)**
   - Black (formateo de código)
   - Flake8 (guía de estilo)

6. 🖥️ **Tests Frontend (Opcional)**
   - Compilación del proyecto .NET
   - Se pregunta al usuario si desea ejecutarlos

---

## 📦 Prerrequisitos

### Mínimos (Solo Tests Unitarios)

- **Python 3.11+**
  ```bash
  python3 --version  # Linux/macOS
  python --version   # Windows
  ```

### Opcionales (Tests Completos)

- **.NET 9.0 SDK** (para tests de frontend)
  ```bash
  dotnet --version
  ```

- **Node.js 18+** (para tests E2E)
  ```bash
  node --version
  ```

---

## 🎯 Ejemplos de Uso

### Ejecutar Solo Tests Unitarios

```bash
# Linux/macOS
./scripts/test-local.sh

# Windows
.\scripts\test-local.ps1
```

El script preguntará si deseas ejecutar tests opcionales (integración, linting, frontend).
Responde `N` para saltarlos y ejecutar solo los tests unitarios.

### Ejecutar Tests Completos

Responde `s` o `S` cuando el script pregunte sobre tests opcionales.

### Ejecutar Tests de Forma Manual

Si prefieres ejecutar los tests manualmente:

```bash
# Backend - Tests Unitarios
cd Ready4Hire
source venv/bin/activate  # Linux/macOS
# venv\Scripts\Activate.ps1  # Windows

export PYTHONPATH=$PWD:$PYTHONPATH  # Linux/macOS
# $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"  # Windows

pytest tests/unit/ -v

# Backend - Tests con Coverage
pytest tests/unit/ --cov=app --cov-report=html

# Linting
black --check app/ tests/
flake8 app/ --max-line-length=120

# Frontend
cd ../WebApp
dotnet build
dotnet test
```

---

## 📊 Salida Esperada

El script mostrará:

```
╔══════════════════════════════════════════════════════════════════════════╗
║              🧪 Ready4Hire v2.1 - Testing Local                          ║
╚══════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Verificando Prerrequisitos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Python: 3.11.9
✅ .NET: 9.0.0
✅ Node.js: v18.17.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Configurando Backend Python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ️  Entorno virtual ya existe
✅ Backend configurado

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ejecutando Tests Backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

═══════════════════════════════════════════════════════════════════
  TESTS UNITARIOS
═══════════════════════════════════════════════════════════════════

tests/unit/test_question_repository.py::TestJsonQuestionRepository::test_repository_initialization PASSED
tests/unit/test_question_repository.py::TestJsonQuestionRepository::test_technical_questions_count PASSED
...
tests/unit/test_question_repository.py::TestQuestionQuality::test_difficulty_distribution PASSED

========================= 16 passed in 3.20s =========================

✅ Tests unitarios pasaron

═══════════════════════════════════════════════════════════════════
  COVERAGE REPORT
═══════════════════════════════════════════════════════════════════

Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
app/infrastructure/persistence/json_question_repository.py    71     10    86%
---------------------------------------------------------------------------
TOTAL                                          7474   7196     4%

╔══════════════════════════════════════════════════════════════════════════╗
║                          📊 RESUMEN FINAL                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

✅ Tests locales completados
ℹ️  Para deployment completo, usa: docker-compose --profile dev up -d
ℹ️  Para más tests, ejecuta: pytest tests/ -v --cov=app
```

---

## 🐛 Troubleshooting

### Error: "Python no encontrado"

**Linux/macOS:**
```bash
# Instalar Python 3.11+
sudo apt install python3 python3-pip python3-venv  # Ubuntu/Debian
brew install python@3.11                           # macOS
```

**Windows:**
```powershell
# Descargar e instalar desde:
# https://www.python.org/downloads/
```

### Error: "ModuleNotFoundError"

Asegúrate de que el entorno virtual esté activado:

```bash
# Linux/macOS
source Ready4Hire/venv/bin/activate

# Windows
Ready4Hire\venv\Scripts\Activate.ps1
```

### Error: "Tests de integración fallan"

Esto es **esperado** si no tienes los servicios corriendo. Los tests de integración requieren:
- Ollama (LLM)
- PostgreSQL
- Redis
- Qdrant

Para ejecutar estos servicios:
```bash
docker-compose --profile dev up -d
```

---

## 🔗 Más Información

- **Tests Completos con Docker**: Ver [docker-compose.yml](../docker-compose.yml)
- **Documentación Backend**: Ver [Ready4Hire/docs/](../Ready4Hire/docs/)
- **README Principal**: Ver [README.md](../README.md)

---

## 📝 Notas

- Los scripts **no modifican** tu sistema, solo crean un entorno virtual local
- Puedes ejecutar los scripts **múltiples veces** sin problemas
- Los scripts son **idempotentes**: si ya está configurado, lo detectan
- Para **limpiar** el entorno: `rm -rf Ready4Hire/venv`

---

**Made with ❤️ by the Ready4Hire Team**

