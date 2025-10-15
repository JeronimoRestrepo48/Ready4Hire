# 🚀 Ready4Hire - Sistema de Entrevistas con IA

## 🎯 Estado Actual

### ✅ SISTEMA COMPLETAMENTE FUNCIONAL Y PROBADO

Sistema avanzado de entrevistas técnicas y soft skills con evaluación automática mediante IA, que incluye:

- ✅ **Fase de Contexto**: 5 preguntas iniciales para conocer al candidato
- ✅ **Selección Inteligente de Preguntas**: Basada en análisis de contexto con ML
- ✅ **Evaluación en Tiempo Real**: Feedback personalizado con detección de emociones
- ✅ **Sistema de Reintentos**: Hasta 3 intentos por pregunta con hints progresivos
- ✅ **Gamificación**: Motivación adaptativa según desempeño
- ✅ **Arquitectura DDD**: Domain-Driven Design con Dependency Injection

---

## ⚡ Inicio Rápido (1 Comando)

```bash
cd /home/jeronimorestrepoangel/Documentos/Integracion
./scripts/run.sh
```

O directamente:

```bash
./scripts/run.sh
```

Esto iniciará automáticamente:

- ✅ **Ollama Server** (LLM en puerto 11434)
- ✅ **Backend FastAPI** (puerto 8001) - Arquitectura DDD
- ✅ **Frontend Blazor** (puerto 5214, si tienes .NET 9.0)

---

## 🌐 Acceder a la Aplicación

Una vez iniciado:

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **WebApp (Interfaz)** | <http://localhost:5214> | Frontend Blazor con chat interactivo |
| **API Backend** | <http://localhost:8001> | Backend FastAPI con arquitectura DDD |
| **API Docs (Swagger)** | <http://localhost:8001/docs> | Documentación interactiva de la API |
| **Health Check** | <http://localhost:8001/api/v2/health> | Estado del sistema (LLM, STT, ML) |
| **Ollama Server** | <http://localhost:11434> | Servidor LLM local |

---

## 🔄 Flujo Conversacional Completo

### Arquitectura del Sistema

```text
┌─────────────────────────────────────────────────────────────┐
│  Ready4Hire - Full Stack Integration v2.0                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐                                          │
│  │  Ollama LLM   │  ← Modelo ready4hire:latest             │
│  │  Port: 11434  │                                          │
│  └───────┬───────┘                                          │
│          │                                                  │
│          ↓                                                  │
│  ┌───────────────────────────────────────┐                 │
│  │  FastAPI Backend (DDD Architecture)   │                 │
│  │  Port: 8001                           │                 │
│  │                                       │                 │
│  │  Domain Layer:                        │                 │
│  │  ├─ Interview Entity      ✅          │                 │
│  │  ├─ Question Entity       ✅          │                 │
│  │  ├─ Context Questions     ✅          │                 │
│  │  └─ Interview Phases      ✅          │                 │
│  │                                       │                 │
│  │  Application Layer:                   │                 │
│  │  ├─ Evaluation Service    ✅          │                 │
│  │  ├─ Feedback Service      ✅          │                 │
│  │  ├─ Question Selector     ✅          │                 │
│  │  └─ ML Integration        ✅          │                 │
│  │                                       │                 │
│  │  Infrastructure Layer:                │                 │
│  │  ├─ LLM Service (Ollama)  ✅          │                 │
│  │  ├─ Audio (Whisper STT)   ✅          │                 │
│  │  ├─ ML Embeddings         ✅          │                 │
│  │  ├─ RankNet Model         ✅          │                 │
│  │  └─ Security Layer        ✅          │                 │
│  └───────────────┬───────────────────────┘                 │
│                  │                                          │
│                  ↓                                          │
│  ┌───────────────────────────────────────┐                 │
│  │  Blazor WebApp (.NET 9.0)             │                 │
│  │  Port: 5214                           │                 │
│  │                                       │                 │
│  │  MVVM Architecture:                   │                 │
│  │  ├─ Chat Page (Conversational UI) ✅  │                 │
│  │  ├─ Interview API Service V2     ✅   │                 │
│  │  ├─ Login Page                    ✅  │                 │
│  │  └─ Bootstrap UI                  ✅  │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de la Entrevista Conversacional

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. CONFIGURACIÓN INICIAL                                    │
│    Usuario selecciona:                                      │
│    • Rol (Backend Developer, Frontend, etc.)                │
│    • Tipo (Technical / Soft Skills)                         │
│    • Dificultad (Junior / Mid / Senior)                     │
│    • Modo (Practice / Exam)                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. INICIO DE ENTREVISTA                                     │
│    POST /api/v2/interviews                                  │
│    • Crea Interview Entity (phase="context")                │
│    • Retorna primera pregunta de contexto                   │
│    • Frontend muestra mensaje de bienvenida                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FASE DE CONTEXTO (5 preguntas)                          │
│    POST /api/v2/interviews/{id}/answers                     │
│    • Usuario responde cada pregunta                         │
│    • Se guardan en interview.context_answers                │
│    • Cuando se completan las 5:                             │
│       ✓ Analiza respuestas con LLM                         │
│       ✓ Extrae: nivel, fortalezas, áreas de mejora        │
│       ✓ Transiciona a phase="questions"                    │
│       ✓ Mensaje: "¡Análisis completado! Iniciando..."     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. SELECCIÓN INTELIGENTE DE PREGUNTAS                      │
│    • Usa ML Selector (clustering + MAB + embeddings)        │
│    • Selecciona las 10 mejores preguntas basadas en:        │
│       ✓ Análisis de contexto del candidato                 │
│       ✓ Nivel de experiencia detectado                     │
│       ✓ Fortalezas identificadas                           │
│       ✓ Áreas a evaluar prioritarias                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. FASE DE PREGUNTAS (10 preguntas seleccionadas)          │
│    POST /api/v2/interviews/{id}/answers                     │
│    • Para cada pregunta:                                    │
│       ✓ Usuario responde                                   │
│       ✓ Detecta emoción (Whisper + NLP)                   │
│       ✓ Evalúa con LLM (score 0-10)                       │
│       ✓ Si score >= 6.0: ✅ Siguiente pregunta            │
│       ✓ Si score < 6.0:                                    │
│         - Intento 1: Feedback + hint sutil                 │
│         - Intento 2: Feedback + hint más directo           │
│         - Intento 3: Feedback + hint explícito             │
│         - Después de 3: ⚠️ Siguiente pregunta             │
│       ✓ Genera feedback personalizado (fine-tuned LLM)     │
│       ✓ Genera motivación adaptativa según desempeño       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. COMPLETAR ENTREVISTA                                     │
│    POST /api/v2/interviews/{id}/end                         │
│    • Después de 10 preguntas correctas                      │
│    • phase="completed"                                      │
│    • Genera resumen final con:                              │
│       ✓ Score total                                        │
│       ✓ Fortalezas demostradas                             │
│       ✓ Áreas de mejora                                    │
│       ✓ Recomendaciones personalizadas                     │
│       ✓ Siguientes pasos sugeridos                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Endpoints de la API v2

### 1. Iniciar Entrevista (Con Contexto)

**POST** `/api/v2/interviews`

**Request Body:**

```json
{
  "user_id": "user-12345",
  "role": "Backend Developer",
  "category": "technical",
  "difficulty": "mid"
}
```

**Response:**

```json
{
  "interview_id": "interview_user-12345_1729012345.67",
  "first_question": {
    "id": "context_0",
    "text": "¿Cuántos años de experiencia tienes en desarrollo de software?",
    "category": "context",
    "difficulty": "context",
    "expected_concepts": [],
    "topic": "context"
  },
  "status": "context"
}
```

### 2. Procesar Respuesta (Contexto o Pregunta)

**POST** `/api/v2/interviews/{interview_id}/answers`

**Request Body:**

```json
{
  "answer": "Tengo 3 años de experiencia en desarrollo backend con Python y Java.",
  "time_taken": 45
}
```

**Response (Durante Contexto):**

```json
{
  "evaluation": {
    "score": 0,
    "is_correct": false,
    "feedback": ""
  },
  "feedback": "",
  "emotion": {
    "emotion": "neutral",
    "confidence": 0.85
  },
  "next_question": {
    "id": "context_1",
    "text": "¿Qué tecnologías y frameworks dominas?",
    "category": "context",
    "difficulty": "context"
  },
  "phase": "context",
  "progress": {
    "context_completed": 1,
    "questions_completed": 0
  },
  "interview_status": "active"
}
```

**Response (Durante Preguntas Técnicas):**

```json
{
  "evaluation": {
    "score": 7.5,
    "is_correct": true,
    "feedback": "Excelente respuesta. Demostraste comprensión sólida de los conceptos."
  },
  "feedback": "Tu explicación sobre la arquitectura REST fue clara y precisa. Mencionaste los aspectos clave como verbos HTTP, recursos y estado. Para mejorar, podrías profundizar en HATEOAS.",
  "emotion": {
    "emotion": "confident",
    "confidence": 0.92
  },
  "next_question": {
    "id": "tech_42",
    "text": "¿Cómo implementarías un sistema de caché distribuido?",
    "category": "technical",
    "difficulty": "mid",
    "topic": "arquitectura"
  },
  "motivation": "¡Vas muy bien! Tu comprensión de REST es sólida. Sigamos con el siguiente desafío.",
  "phase": "questions",
  "progress": {
    "context_completed": 5,
    "questions_completed": 3
  },
  "attempts_left": 3,
  "interview_status": "active"
}
```

### 3. Finalizar Entrevista

**POST** `/api/v2/interviews/{interview_id}/end`

**Response:**

```json
{
  "interview_id": "interview_user-12345_1729012345.67",
  "summary": {
    "total_score": 8.2,
    "questions_answered": 10,
    "correct_answers": 8,
    "strengths": [
      "Arquitectura de software",
      "Diseño de APIs REST",
      "Patrones de diseño"
    ],
    "areas_to_improve": [
      "Optimización de queries SQL",
      "Seguridad en autenticación"
    ],
    "recommendations": [
      "Estudiar índices de base de datos",
      "Profundizar en OAuth 2.0 y JWT"
    ]
  },
  "status": "completed"
}
```

### 4. Health Check

**GET** `/api/v2/health`

**Response:**

```json
{
  "status": "healthy",
  "components": {
    "llm_service": "healthy",
    "audio_stt": "healthy",
    "ml_embeddings": "healthy",
    "question_repository": "healthy"
  },
  "version": "2.0.0",
  "timestamp": "2025-10-15T10:30:00Z"
}
```

---

## 🎨 Frontend - Interfaz Conversacional

### Componentes Principales

#### 1. **ChatPage.razor / ChatPage.razor.cs**

- ✅ Interfaz de chat interactiva
- ✅ Mensajes del usuario (derecha, azul)
- ✅ Mensajes del agente (izquierda, gris con icono de robot)
- ✅ Scroll automático al final
- ✅ Modal de configuración con:
  - Selección de Rol
  - Tipo de Entrevista
  - Nivel de Dificultad
  - Modo (Práctica/Examen)

#### 2. **InterviewApiService.cs**

Servicio que consume todos los endpoints de la API v2:

```csharp
public class InterviewApiService
{
    // API V2 - Flujo Conversacional
    Task<JsonElement> StartInterviewV2Async(string userId, string role, string category, string difficulty);
    Task<JsonElement> ProcessAnswerV2Async(string interviewId, string answer, int? timeTaken);
    Task<JsonElement> EndInterviewV2Async(string interviewId);
    Task<JsonElement> HealthCheckV2Async();
    
    // API V1 - Legacy (mantener compatibilidad)
    Task<JsonElement> StartInterviewAsync(string userId, string role, string type, string mode);
    Task<JsonElement> AnswerAsync(string userId, string answer);
    // ... otros métodos legacy
}
```

### Flujo de Interacción en el Frontend

1. **Usuario abre la página de chat**
   - Ve botón "Configurar"

2. **Usuario hace clic en "Configurar"**
   - Se abre modal con opciones:
     - Rol: Backend Developer, Frontend Developer, etc.
     - Tipo: Technical / Soft Skills
     - Dificultad: Junior / Mid / Senior
     - Modo: Practice / Exam

3. **Usuario guarda configuración**
   - Modal se cierra
   - Botón "Comenzar Entrevista" se habilita

4. **Usuario hace clic en "Comenzar Entrevista"**
   - Frontend llama a `StartInterviewV2Async()`
   - Backend retorna:
     - `interview_id` (se guarda para siguientes llamadas)
     - Primera pregunta de contexto
   - Frontend muestra:
     - Mensaje de bienvenida del agente
     - Primera pregunta de contexto

5. **Usuario responde cada pregunta**
   - Usuario escribe respuesta y hace clic en "Enviar"
   - Frontend llama a `ProcessAnswerV2Async(interviewId, answer)`
   - Backend retorna:
     - Feedback (si aplica)
     - Motivación (si aplica)
     - Siguiente pregunta
     - Progreso (contexto/preguntas completadas)
   - Frontend muestra secuencialmente:
     - Mensaje del usuario (derecha)
     - Feedback del agente (si existe)
     - Motivación del agente (si existe)
     - Siguiente pregunta del agente

6. **Transición de Contexto a Preguntas Técnicas**
   - Al completar 5 preguntas de contexto
   - Backend analiza respuestas con LLM
   - Selecciona 10 preguntas personalizadas
   - Frontend muestra mensaje de transición:
     - "✅ ¡Análisis de contexto completado! Ahora comenzaremos con las preguntas técnicas personalizadas según tu perfil."

7. **Evaluación con Reintentos**
   - Si `score < 6.0`:
     - Backend retorna `retry: true` y `attempts_left: 2/1`
     - Frontend muestra feedback + hint
     - Usuario puede reintentar (hasta 3 veces)
   - Si `score >= 6.0`:
     - Backend retorna siguiente pregunta
     - Contador de preguntas completadas aumenta

8. **Finalización**
   - Después de 10 preguntas correctas
   - Backend retorna `phase: "completed"`
   - Frontend muestra mensaje de felicitación
   - (Opcional) Llama a `EndInterviewV2Async()` para resumen final

---

## 📋 Comandos Disponibles

```bash
# Iniciar servicios (modo normal)
./scripts/run.sh

# Iniciar en modo desarrollo (auto-reload)
./scripts/run.sh --dev

# Ver estado de servicios
./scripts/run.sh --status

# Detener todos los servicios
./scripts/run.sh --stop

# Ayuda
./scripts/run.sh --help
```

---

## 🔧 Solución Rápida de Problemas

### Puerto ocupado

```bash
# Detener todo y reiniciar
./scripts/run.sh --stop
./scripts/run.sh
```

### Ver logs

```bash
# Logs del backend
tail -f Ready4Hire/logs/ready4hire_api.log

# Logs de Ollama
tail -f Ready4Hire/logs/ollama.log

# Evaluaciones
tail -f Ready4Hire/logs/audit_log.jsonl
```

### Verificar servicios manualmente

```bash
# Ollama
curl http://localhost:11434/api/tags

# Backend (Health Check)
curl http://localhost:8001/api/v2/health

# WebApp
curl http://localhost:5214/

# Modelo
ollama list | grep ready4hire
```

---

## 🧪 Ejecutar Pruebas de Integración

Para validar que todo el sistema está funcionando correctamente:

```bash
./scripts/test_integration.sh
```

Esto ejecutará 16 pruebas automatizadas que verifican:

- ✅ Ollama Server y modelo ready4hire:latest
- ✅ API Python (todos los componentes: LLM, STT, ML)
- ✅ WebApp Blazor (login, bootstrap, etc.)
- ✅ Integración entre servicios

Ver más detalles en [TESTING.md](TESTING.md)

---

## 📚 Estructura del Proyecto

```text
Integracion/
├── start.sh                 # ⚡ Inicio rápido
├── scripts/
│   ├── run.sh              # 🎯 Script maestro completo
│   └── README.md           # 📖 Documentación de scripts
├── QUICKSTART.md           # 🚀 Guía de inicio completa
├── Ready4Hire/             # 🐍 Backend Python (FastAPI)
│   ├── app/               # Código de aplicación
│   ├── scripts/           # Scripts de ML/Data
│   │   ├── 1_data/       # Generación de datos
│   │   ├── 2_training/   # Fine-tuning
│   │   ├── 3_deployment/ # Deployment
│   │   └── 4_testing/    # Testing
│   ├── logs/             # Logs del sistema
│   └── .env              # Configuración
└── WebApp/                # 🎨 Frontend Blazor (.NET)
    ├── Ready4Hire.csproj
    ├── Program.cs
    ├── appsettings.json       # Config (puerto 8001)
    └── MVVM/
        ├── Models/
        │   └── InterviewApiService.cs  # Cliente API (CORREGIDO)
        └── Views/
            ├── LoginView.razor
            └── ChatPage.razor
```

---

## 🎓 Documentación Completa

- **Inicio Rápido**: `QUICKSTART.md`
- **Scripts**: `scripts/README.md`
- **Pipeline ML**: `Ready4Hire/scripts/README.md`
- **Fase 1 - Datos**: `Ready4Hire/scripts/1_data/README.md`
- **Fase 2 - Training**: `Ready4Hire/scripts/2_training/README.md`
- **Fase 3 - Deployment**: `Ready4Hire/scripts/3_deployment/README.md`
- **Fase 4 - Testing**: `Ready4Hire/scripts/4_testing/README.md`

---

## 🤖 Pipeline de ML

Si quieres mejorar el modelo:

### 1. Generar más datos

```bash
cd Ready4Hire
python3 scripts/1_data/step1_generate_demo_data.py --num-samples 1000
python3 scripts/1_data/step2_convert_to_training.py
python3 scripts/1_data/step3_create_dataset.py
```

### 2. Fine-tuning en Google Colab

```bash
# Abre el notebook en Colab
Ready4Hire/scripts/2_training/COLAB_FINETUNE.ipynb
```

- Activa GPU T4 (gratis)
- Sube datasets
- Ejecuta todas las celdas
- Descarga modelo .gguf
- Importa a Ollama

### 3. Testear modelo

```bash
cd Ready4Hire
python3 scripts/4_testing/step1_test_model.py --model ready4hire:latest
```

---

## 📊 Estado Actual del Sistema

### Datos

- ✅ 500 evaluaciones generadas
- ✅ 214 ejemplos de entrenamiento
- ✅ 54 ejemplos de validación
- ✅ Dataset listo para fine-tuning

### Modelos

- ✅ `ready4hire:latest` - Modelo personalizado (llama3.2:3b + system prompt)
- ✅ `llama3.2:3b` - Modelo base
- ✅ `llama3:latest` - Modelo alternativo

### Servicios

- ✅ Ollama Server configurado
- ✅ Backend FastAPI funcionando
- ✅ Frontend Blazor (opcional)

---

## 💡 Próximos Pasos

1. **Usar la aplicación**: <http://localhost:8001>
2. **Explorar API**: <http://localhost:8001/docs>
3. **Generar más datos**: Para mejor fine-tuning
4. **Fine-tune en Colab**: Mejorar accuracy a >80%
5. **Conectar frontend**: Si tienes WebApp Blazor

---

## 🆘 Soporte

- **Documentación completa**: `QUICKSTART.md`
- **Logs**: `Ready4Hire/logs/`
- **Script ayuda**: `./scripts/run.sh --help`
- **Estado**: `./scripts/run.sh --status`

---

## 🎉 ¡Todo Listo!

El sistema está completamente operativo. Solo ejecuta:

```bash
./start.sh
```

Y abre <http://localhost:8001> en tu navegador.

### ¡A entrevistar con IA! 🚀

---

**Version**: 2.0.0 (DDD Architecture)  
**Stack**: Python + FastAPI + Ollama + Blazor  
**ML**: LLM Fine-tuning con Unsloth
