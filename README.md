# 🚀 Ready4Hire

**Sistema Inteligente de Entrevistas Técnicas con IA**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![.NET](https://img.shields.io/badge/.NET-9.0-purple.svg)](https://dotnet.microsoft.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Blazor](https://img.shields.io/badge/Blazor-Server-blueviolet.svg)](https://dotnet.microsoft.com/apps/aspnet/web-apps/blazor)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Plataforma de entrevistas técnicas impulsada por IA que simula entrevistas reales, evalúa respuestas en tiempo real y proporciona feedback inteligente para preparación profesional.

---

## 📋 Tabla de Contenidos

- [Características](#-características-principales)
- [Arquitectura](#️-arquitectura)
- [Requisitos](#-requisitos-previos)
- [Instalación](#-instalación-rápida)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [API](#-api)
- [Seguridad](#-seguridad)
- [Contribuir](#-contribuir)
- [Troubleshooting](#-troubleshooting)
- [Licencia](#-licencia)

---

## ✨ Características Principales

### 🎯 Sistema de Entrevistas Inteligente

- **Entrevistas Contextuales**: Adaptación dinámica según respuestas del candidato
- **Dos Modos de Operación**:
  - 🎓 **Práctica**: Feedback inmediato y evaluación continua
  - 📝 **Examen**: Evaluación final con timer y puntuación
- **Categorías Múltiples**: Backend, Frontend, DevOps, Data Science, Mobile, etc.
- **Niveles de Dificultad**: Easy, Medium, Hard

### 🤖 IA Avanzada

- **LLM Local con Ollama**: Modelo llama3.2:3b optimizado
- **Embeddings Semánticos**: Búsqueda inteligente de preguntas relevantes
- **Evaluación Contextual**: Análisis semántico de respuestas
- **Follow-up Inteligente**: Preguntas de seguimiento automáticas

### 💻 Frontend Moderno

- **Diseño Inspirado en ChatGPT/Perplexity**: UI profesional y elegante
- **Avatares con Iniciales**: Experiencia personalizada
- **Sidebar con Historial**: Gestión de conversaciones
- **Welcome Screen**: Pantalla de bienvenida personalizada
- **Responsive Design**: Compatible con todos los dispositivos

### 🔒 Seguridad Robusta

- **Autenticación Obligatoria**: Sistema de login con sesión protegida
- **BCrypt Hashing**: Contraseñas seguras con algoritmo industry-standard
- **Anti-XSS/CSRF**: Protección contra ataques comunes
- **Headers de Seguridad**: X-Frame-Options, CSP, etc.
- **Input Sanitization**: Validación y limpieza de todos los inputs

### 📊 Persistencia de Datos

- **PostgreSQL**: Base de datos relacional robusta
- **Entity Framework Core**: ORM moderno para .NET
- **Historial de Conversaciones**: Guardado automático
- **Perfil de Usuario**: Skills, intereses, experiencia

---

## 🏗️ Arquitectura

### Stack Tecnológico

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (Blazor Server)               │
│  - .NET 9.0                                                 │
│  - Blazor Server Components                                 │
│  - Entity Framework Core                                    │
│  - Modern CSS (ChatGPT-inspired)                            │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                      │
│  - Python 3.13                                              │
│  - FastAPI (async)                                          │
│  - Clean Architecture (DDD)                                 │
│  - Dependency Injection                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Ollama     │  │  PostgreSQL  │  │  Embeddings  │
│  (LLM API)   │  │   Database   │  │   (Numpy)    │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Capas de la Aplicación (Backend)

```
app/
├── domain/           # Lógica de negocio pura
│   ├── entities/     # Entidades de dominio
│   ├── value_objects/  # Objetos de valor
│   ├── repositories/   # Interfaces de repos
│   └── services/     # Servicios de dominio
├── application/      # Casos de uso
│   ├── use_cases/    # Lógica de aplicación
│   ├── services/     # Servicios de aplicación
│   └── dto/          # Data Transfer Objects
└── infrastructure/   # Implementaciones concretas
    ├── llm/          # Integración Ollama
    ├── ml/           # Embeddings, evaluación
    ├── persistence/  # Repositorios in-memory
    ├── audio/        # STT/TTS
    └── security/     # Auth, JWT
```

---

## 📦 Requisitos Previos

### Software Necesario

- **Python**: 3.13+ ([Descargar](https://www.python.org/downloads/))
- **.NET SDK**: 9.0+ ([Descargar](https://dotnet.microsoft.com/download))
- **PostgreSQL**: 14+ ([Descargar](https://www.postgresql.org/download/))
- **Ollama**: Latest ([Descargar](https://ollama.ai/download))

### Hardware Recomendado

- **RAM**: 8 GB mínimo, 16 GB recomendado
- **Disco**: 10 GB libres
- **GPU**: Opcional (NVIDIA para aceleración CUDA)

---

## ⚡ Instalación Rápida

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/Ready4Hire.git
cd Ready4Hire
```

### 2. Instalar Modelo de Ollama

```bash
ollama pull llama3.2:3b
```

### 3. Configurar PostgreSQL

```bash
# Crear base de datos
sudo -u postgres psql
CREATE DATABASE ready4hire;
CREATE USER ready4hire_user WITH PASSWORD 'tu_contraseña_segura';
GRANT ALL PRIVILEGES ON DATABASE ready4hire TO ready4hire_user;
\q
```

### 4. Configurar Variables de Entorno

**Backend** (`Ready4Hire/.env`):
```env
# API
API_HOST=0.0.0.0
API_PORT=8001

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Security
SECRET_KEY=tu_secret_key_muy_seguro_cambiar_en_produccion
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

**Frontend** (`WebApp/.env`):
```env
POSTGRES_CONNECTION=Host=localhost;Port=5432;Database=ready4hire;Username=ready4hire_user;Password=tu_contraseña_segura
```

### 5. Instalar Dependencias

```bash
# Python (Backend)
cd Ready4Hire
pip install -r ../requirements.txt

# .NET (Frontend)
cd ../WebApp
dotnet restore
```

### 6. Aplicar Migraciones de Base de Datos

```bash
cd WebApp
dotnet ef database update
```

### 7. Iniciar el Sistema

**Opción A: Script Automatizado (Recomendado)**
```bash
./ready4hire.sh start
```

**Opción B: Usando Make**
```bash
make start
```

**Opción C: Manual**
```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: Backend
cd Ready4Hire
python3 -m uvicorn app.main_v2_improved:app --host 0.0.0.0 --port 8001

# Terminal 3: Frontend
cd WebApp
dotnet run
```

### 8. Acceder a la Aplicación

- **Frontend**: http://localhost:5214
- **API Backend**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

---

## 🎮 Uso

### Comandos Rápidos

```bash
# Iniciar todos los servicios
make start

# Reiniciar todos los servicios
make restart

# Ver estado del sistema
make status

# Ver logs en tiempo real
make logs

# Detener todos los servicios
make stop

# Modo interactivo (menú)
./ready4hire.sh
```

### Flujo de Usuario

1. **Registro/Login**: 
   - Crear cuenta con email y contraseña segura
   - Completar perfil (skills, intereses, experiencia)

2. **Configurar Entrevista**:
   - Elegir modo (Práctica/Examen)
   - Seleccionar categoría (Backend, Frontend, etc.)
   - Elegir dificultad (Easy, Medium, Hard)

3. **Realizar Entrevista**:
   - Responder preguntas del asistente IA
   - Recibir feedback en tiempo real (modo Práctica)
   - Ver evaluación final (modo Examen)

4. **Revisar Historial**:
   - Sidebar con conversaciones anteriores
   - Métricas y puntuaciones
   - Progreso a lo largo del tiempo

---

## 📁 Estructura del Proyecto

```
Ready4Hire/
├── Ready4Hire/              # Backend (Python/FastAPI)
│   ├── app/
│   │   ├── domain/          # Lógica de negocio
│   │   ├── application/     # Casos de uso
│   │   ├── infrastructure/  # Implementaciones
│   │   ├── container.py     # Dependency Injection
│   │   └── main_v2_improved.py  # Entry point
│   ├── docs/                # Documentación técnica
│   ├── scripts/             # Scripts de entrenamiento
│   └── tests/               # Tests unitarios
├── WebApp/                  # Frontend (C#/Blazor)
│   ├── Components/          # Componentes Blazor
│   ├── MVVM/
│   │   ├── Models/          # Modelos de datos
│   │   ├── ViewModels/      # Lógica de vista
│   │   └── Views/           # Vistas Blazor
│   ├── Services/            # AuthService, SecurityService
│   ├── Data/                # DbContext, Migrations
│   └── wwwroot/             # Assets estáticos
├── logs/                    # Logs del sistema
├── ready4hire.sh            # Script de control principal
├── Makefile                 # Comandos rápidos
└── requirements.txt         # Dependencias Python
```

---

## 🔌 API

### Endpoints Principales

#### V2 - Flujo Conversacional

```http
POST /api/v2/interview/start
Body: {
  "user_id": "string",
  "role": "string",
  "category": "string",
  "difficulty": "easy|medium|hard"
}
Response: {
  "interview_id": "uuid",
  "first_question": "string",
  "phase": "context_gathering"
}
```

```http
POST /api/v2/interview/{interview_id}/answer
Body: {
  "answer": "string",
  "time_taken": 120
}
Response: {
  "next_question": "string",
  "evaluation": {...},
  "phase": "technical_questions",
  "progress": {...}
}
```

```http
GET /api/v2/interview/{interview_id}/end
Response: {
  "final_evaluation": {...},
  "score": 85,
  "recommendations": [...]
}
```

```http
GET /api/v2/health
Response: {
  "status": "healthy",
  "version": "2.0.0",
  "components": {...}
}
```

### Documentación Completa

Accede a la documentación interactiva en:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## 🔒 Seguridad

### Implementaciones de Seguridad

✅ **Autenticación**
- Session-based auth con `ProtectedSessionStorage`
- Tokens únicos por sesión
- Validación en todas las rutas protegidas

✅ **Protección de Contraseñas**
- BCrypt hashing (industry standard)
- Salt automático
- Requisitos de complejidad (8+ chars, mayúsculas, minúsculas, números)

✅ **Prevención de Ataques**
- **XSS**: HTML encoding + sanitización de inputs
- **CSRF**: Anti-forgery tokens automáticos
- **SQL Injection**: Entity Framework (queries parametrizadas)
- **Clickjacking**: `X-Frame-Options: DENY`

✅ **Headers HTTP de Seguridad**
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Content-Security-Policy: default-src 'self'; ...
```

✅ **Validación de Inputs**
- Regex estricto para emails
- Límites de longitud
- Sanitización automática
- Prevención de inyección

### Servicios de Seguridad

- **`AuthService`**: Gestión de sesiones y autenticación
- **`SecurityService`**: Validación, sanitización, prevención de ataques

---

## 🤝 Contribuir

### Guía de Contribución

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

### Estándares de Código

- **Python**: PEP 8, type hints
- **C#**: Microsoft conventions, async/await
- **Commits**: Conventional Commits
- **Tests**: Coverage mínimo 70%

### Ejecutar Tests

```bash
# Backend
cd Ready4Hire
pytest tests/ -v --cov=app

# Frontend
cd WebApp
dotnet test
```

---

## 🐛 Troubleshooting

### Backend no inicia

```bash
# Ver logs
tail -f logs/ready4hire_api.log

# Verificar puerto
lsof -i :8001

# Verificar Ollama
curl http://localhost:11434/api/tags
```

### Frontend no inicia

```bash
# Ver logs
tail -f logs/webapp.log

# Compilar manualmente
cd WebApp
dotnet clean
dotnet build
dotnet run
```

### PostgreSQL no conecta

```bash
# Verificar servicio
sudo systemctl status postgresql

# Iniciar servicio
sudo systemctl start postgresql

# Verificar conexión
psql -h localhost -U ready4hire_user -d ready4hire
```

### Ollama no responde

```bash
# Reiniciar Ollama
pkill ollama
ollama serve

# Verificar modelo
ollama list

# Descargar modelo si falta
ollama pull llama3.2:3b
```

### Ver todos los logs

```bash
make logs
```

### Reiniciar todo

```bash
make restart
```

---

## 📚 Documentación Adicional

- **[Arquitectura](Ready4Hire/docs/ARCHITECTURE.md)**: Detalles técnicos de la arquitectura
- **[API Documentation](Ready4Hire/docs/API_DOCUMENTATION.md)**: Guía completa de la API
- **[Configuration](Ready4Hire/docs/CONFIGURATION.md)**: Opciones de configuración
- **[Deployment](Ready4Hire/docs/DEPLOYMENT.md)**: Guía de despliegue en producción
- **[Performance](Ready4Hire/docs/PERFORMANCE_OPTIMIZATIONS.md)**: Optimizaciones de rendimiento

---

## 🗺️ Roadmap

### Q1 2025
- [ ] Integración con LinkedIn para importar perfil
- [ ] Modo multi-entrevistador (panel de entrevistas)
- [ ] Análisis de sentimientos en respuestas
- [ ] Recomendaciones personalizadas de estudio

### Q2 2025
- [ ] Soporte multiidioma (ES, EN, PT)
- [ ] Video entrevistas con análisis de expresiones
- [ ] Gamificación y sistema de logros
- [ ] Marketplace de preguntas de entrevista

### Q3 2025
- [ ] Mobile app (React Native)
- [ ] Integración con sistemas ATS
- [ ] API pública para terceros
- [ ] Analytics avanzados y reportes

---

## 📊 Estado del Proyecto

- ✅ Backend API completamente funcional
- ✅ Frontend moderno con Blazor
- ✅ Sistema de autenticación robusto
- ✅ Integración con Ollama (LLM)
- ✅ Evaluación semántica de respuestas
- ✅ Persistencia con PostgreSQL
- ✅ Scripts de automatización
- ✅ Documentación completa

---

## 👥 Equipo

Desarrollado con ❤️ por el equipo de Ready4Hire.

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- [Ollama](https://ollama.ai/) por el runtime de LLM local
- [FastAPI](https://fastapi.tiangolo.com/) por el excelente framework
- [Blazor](https://dotnet.microsoft.com/apps/aspnet/web-apps/blazor) por el framework frontend
- Comunidad open-source por las librerías utilizadas

---

## 📞 Contacto

- **Email**: contact@ready4hire.com
- **GitHub**: [github.com/ready4hire](https://github.com/ready4hire)
- **Twitter**: [@ready4hire](https://twitter.com/ready4hire)

---

<div align="center">

**⭐ Si este proyecto te ayuda, dale una estrella en GitHub ⭐**

[Reportar Bug](https://github.com/ready4hire/issues) · [Solicitar Feature](https://github.com/ready4hire/issues) · [Documentación](Ready4Hire/docs/)

</div>
