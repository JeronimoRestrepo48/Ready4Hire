# 🔍 ANÁLISIS EXHAUSTIVO - IA DE READY4HIRE

**Fecha**: 14 de octubre de 2025  
**Autor**: GitHub Copilot - Análisis Profundo del Sistema de IA  
**Versión**: 1.0

---

## 📊 RESUMEN EJECUTIVO

### Estado Actual del Sistema

**✅ Fortalezas Identificadas**:

- ✓ Sistema funcional con entrevistas interactivas completas
- ✓ SSL/TLS implementado con reverse proxy Nginx
- ✓ Embeddings avanzados con UMAP + HDBSCAN clustering
- ✓ Detección de sesgos y análisis de emociones básico
- ✓ Documentación completa y organizada
- ✓ Docker containerizado con multi-stage builds
- ✓ Sanitización de inputs y protección contra inyecciones

**🔴 Debilidades Críticas Detectadas**:

- ❌ **Arquitectura monolítica**: `interview_agent.py` con 1422 líneas
- ❌ **Funciones ML stub**: No son productivas, solo simulaciones
- ❌ **Cobertura de tests baja**: ~40% del código
- ❌ **Sin CI/CD**: Despliegues manuales propensos a errores
- ❌ **Modelo de emociones**: Solo inglés (sistema en español)
- ❌ **Sin monitoreo**: No hay métricas de performance en producción
- ❌ **Sin base de datos**: Todo en memoria o JSON files
- ❌ **Sin caché**: Llamadas LLM repetitivas no cachadas

### Métricas de Calidad

| Aspecto | Estado Actual | Prioridad | Esfuerzo |
|---------|---------------|-----------|----------|
| **Arquitectura** | 🔴 Monolítica | 🔴 CRÍTICA | 2-3 semanas |
| **Tests** | 🟡 40% cobertura | 🔴 ALTA | 1-2 semanas |
| **CI/CD** | 🔴 Inexistente | 🔴 ALTA | 1 semana |
| **ML Real** | 🔴 Stubs | 🟡 MEDIA | 3-4 semanas |
| **Monitoreo** | 🟡 Logs básicos | 🟡 MEDIA | 1 semana |
| **Performance** | 🟡 No optimizado | 🟡 MEDIA | 1-2 semanas |
| **Seguridad** | 🟢 SSL + Sanitización | ✓ COMPLETADO | - |
| **Documentación** | 🟢 Completa | ✓ COMPLETADO | - |

---

## 🎯 PROBLEMAS CRÍTICOS IDENTIFICADOS

### 1. 🔴 ARQUITECTURA MONOLÍTICA

**Severidad**: CRÍTICA  
**Impacto**: Mantenibilidad, Escalabilidad, Testing  
**Archivo**: `app/interview_agent.py` (1422 líneas)

#### Problema Actual

```python
# interview_agent.py - TODO MEZCLADO EN UNA CLASE GIGANTE
class InterviewAgent:
    def __init__(self): 
        # 20+ atributos mezclados sin separación de responsabilidades
        self.llm = ChatOllama(...)  # Cliente LLM
        self.emb_mgr = EmbeddingsManager()  # Embeddings
        self.sessions = {}  # Gestión de sesiones
        self.tech_data = []  # Datos técnicos
        self.soft_data = []  # Datos soft skills
        self.emotion_analyzer = EmotionAnalyzer()
        # ... más atributos
    
    # 50+ MÉTODOS MEZCLADOS:
    
    # Gestión de sesiones
    def start_interview(...)  
    def end_interview(...)
    def _get_session(...)
    
    # Selección de preguntas
    def next_question(...)
    def _filter_questions(...)
    def _select_next_question(...)
    
    # Evaluación
    def process_answer(...)
    def _evaluate_answer(...)
    def _calculate_score(...)
    
    # Feedback
    def _generate_feedback(...)
    def _get_learning_resources(...)
    def _get_example(...)
    
    # Análisis
    def _detect_bias(...)
    def _analyze_emotion(...)
    
    # Gamificación
    def _get_adaptive_level(...)
    def _check_achievements(...)
    def _update_streaks(...)
    
    # Encuestas
    def process_satisfaction_answer(...)
    
    # ... 30+ métodos más
```

#### Impacto en el Sistema

- ❌ **Violación masiva de SOLID**:
  - **SRP (Single Responsibility)**: Una clase con 15+ responsabilidades
  - **OCP (Open/Closed)**: Imposible extender sin modificar todo
  - **ISP (Interface Segregation)**: Clientes forzados a depender de todo
  - **DIP (Dependency Inversion)**: Alto acoplamiento a implementaciones concretas

- ❌ **Testing imposible de mantener**:
  - Tests requieren mock de 20+ dependencias
  - Cambios pequeños rompen tests no relacionados
  - Difícil probar lógica aislada

- ❌ **Imposible escalar horizontalmente**:
  - Sesiones en memoria (no distribuible)
  - No se pueden separar servicios
  - Cuellos de botella en un solo proceso

#### Solución: Domain-Driven Design (DDD)

##### Nueva Estructura Propuesta

```
app/
├── domain/                    # ⭐ LÓGICA DE NEGOCIO PURA
│   ├── entities/              # Entidades del dominio
│   │   ├── interview.py       # Interview aggregate root
│   │   ├── question.py        # Question entity
│   │   ├── answer.py          # Answer value object
│   │   └── user_profile.py    # User profile entity
│   │
│   ├── value_objects/         # Objetos de valor inmutables
│   │   ├── skill_level.py     # Enum: junior, mid, senior
│   │   ├── emotion.py         # Enum: joy, sadness, anger...
│   │   ├── interview_status.py # Enum: active, completed...
│   │   └── score.py           # Score validation (0-10)
│   │
│   ├── repositories/          # Interfaces de persistencia
│   │   ├── interview_repository.py
│   │   ├── question_repository.py
│   │   └── user_profile_repository.py
│   │
│   └── services/              # Servicios de dominio
│       ├── difficulty_calculator.py
│       └── achievement_tracker.py
│
├── application/               # ⭐ CASOS DE USO (ORQUESTACIÓN)
│   ├── use_cases/
│   │   ├── start_interview_use_case.py
│   │   ├── process_answer_use_case.py
│   │   ├── evaluate_response_use_case.py
│   │   ├── generate_feedback_use_case.py
│   │   └── end_interview_use_case.py
│   │
│   └── dto/                   # Data Transfer Objects
│       ├── interview_dto.py
│       ├── answer_dto.py
│       └── feedback_dto.py
│
├── infrastructure/            # ⭐ IMPLEMENTACIONES TÉCNICAS
│   ├── llm/                   # Clients LLM
│   │   ├── base_llm_client.py
│   │   ├── ollama_client.py
│   │   ├── openai_client.py
│   │   └── fallback_llm.py
│   │
│   ├── embeddings/            # Servicios de embeddings
│   │   ├── sentence_transformer_service.py
│   │   └── openai_embeddings_service.py
│   │
│   ├── ml/                    # Modelos ML
│   │   ├── emotion_detector.py
│   │   ├── bias_detector.py
│   │   ├── difficulty_adjuster.py
│   │   └── ranknet_ranker.py
│   │
│   ├── persistence/           # Implementaciones de repos
│   │   ├── memory_interview_repository.py
│   │   ├── postgres_interview_repository.py
│   │   ├── redis_cache_repository.py
│   │   └── json_question_repository.py
│   │
│   └── audio/                 # Servicios de audio
│       ├── whisper_stt.py
│       └── pyttsx3_tts.py
│
├── api/                       # ⭐ CAPA DE API
│   ├── v1/
│   │   ├── interview_routes.py
│   │   ├── audio_routes.py
│   │   ├── admin_routes.py
│   │   └── health_routes.py
│   │
│   ├── middleware/
│   │   ├── security_middleware.py
│   │   ├── rate_limit_middleware.py
│   │   ├── logging_middleware.py
│   │   └── error_handler_middleware.py
│   │
│   └── dependencies/          # FastAPI dependencies
│       └── dependency_injection.py
│
└── services/                  # ⭐ SERVICIOS DE APLICACIÓN
    ├── interview_orchestrator.py
    ├── evaluation_service.py
    ├── feedback_service.py
    └── analytics_service.py
```

##### Implementación Paso a Paso

**PASO 1: Definir Entidades del Dominio**

```python
# domain/entities/interview.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4
from domain.value_objects.interview_status import InterviewStatus
from domain.value_objects.skill_level import SkillLevel
from domain.entities.question import Question
from domain.entities.answer import Answer

@dataclass
class Interview:
    """
    Aggregate Root: Entrevista
    
    Invariantes:
    - Una entrevista siempre tiene un user_id
    - Solo puede haber una pregunta activa a la vez
    - No se pueden agregar respuestas si status != ACTIVE
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    role: str = ""
    interview_type: str = "technical"  # technical | soft_skills | mixed
    status: InterviewStatus = InterviewStatus.CREATED
    skill_level: SkillLevel = SkillLevel.JUNIOR
    
    current_question: Optional[Question] = None
    questions_history: List[Question] = field(default_factory=list)
    answers_history: List[Answer] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    def start(self) -> None:
        """Inicia la entrevista"""
        if self.status != InterviewStatus.CREATED:
            raise ValueError("Interview already started")
        
        self.status = InterviewStatus.ACTIVE
        self.started_at = datetime.now()
    
    def add_question(self, question: Question) -> None:
        """Agrega una nueva pregunta a la entrevista"""
        if self.status != InterviewStatus.ACTIVE:
            raise ValueError("Cannot add questions to inactive interview")
        
        self.current_question = question
        self.questions_history.append(question)
    
    def add_answer(self, answer: Answer) -> None:
        """Registra una respuesta del usuario"""
        if self.status != InterviewStatus.ACTIVE:
            raise ValueError("Cannot add answers to inactive interview")
        
        if not self.current_question:
            raise ValueError("No active question to answer")
        
        self.answers_history.append(answer)
        self.current_question = None  # Limpia pregunta actual
    
    def complete(self) -> None:
        """Finaliza la entrevista"""
        if self.status != InterviewStatus.ACTIVE:
            raise ValueError("Interview not active")
        
        self.status = InterviewStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def get_score_average(self) -> float:
        """Calcula el score promedio de la entrevista"""
        if not self.answers_history:
            return 0.0
        
        return sum(a.score for a in self.answers_history) / len(self.answers_history)
    
    def get_correct_count(self) -> int:
        """Cuenta respuestas correctas"""
        return sum(1 for a in self.answers_history if a.is_correct)
    
    def get_current_streak(self) -> int:
        """Calcula racha actual de aciertos"""
        streak = 0
        for answer in reversed(self.answers_history):
            if answer.is_correct:
                streak += 1
            else:
                break
        return streak
```

```python
# domain/value_objects/skill_level.py
from enum import Enum

class SkillLevel(str, Enum):
    """Nivel de habilidad del candidato"""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    
    def next_level(self) -> 'SkillLevel':
        """Retorna el siguiente nivel"""
        if self == SkillLevel.JUNIOR:
            return SkillLevel.MID
        elif self == SkillLevel.MID:
            return SkillLevel.SENIOR
        return self  # Ya es senior
    
    def previous_level(self) -> 'SkillLevel':
        """Retorna el nivel anterior"""
        if self == SkillLevel.SENIOR:
            return SkillLevel.MID
        elif self == SkillLevel.MID:
            return SkillLevel.JUNIOR
        return self  # Ya es junior
```

**PASO 2: Definir Repositorios (Interfaces)**

```python
# domain/repositories/interview_repository.py
from abc import ABC, abstractmethod
from typing import Optional, List
from domain.entities.interview import Interview

class InterviewRepository(ABC):
    """
    Repositorio de entrevistas.
    Define el contrato para persistencia, sin implementación concreta.
    """
    
    @abstractmethod
    async def save(self, interview: Interview) -> None:
        """Persiste o actualiza una entrevista"""
        pass
    
    @abstractmethod
    async def find_by_id(self, interview_id: str) -> Optional[Interview]:
        """Busca entrevista por ID"""
        pass
    
    @abstractmethod
    async def find_active_by_user(self, user_id: str) -> Optional[Interview]:
        """Busca entrevista activa del usuario"""
        pass
    
    @abstractmethod
    async def find_all_by_user(self, user_id: str) -> List[Interview]:
        """Obtiene todas las entrevistas de un usuario"""
        pass
    
    @abstractmethod
    async def delete(self, interview_id: str) -> None:
        """Elimina una entrevista"""
        pass
```

**PASO 3: Implementar Use Cases**

```python
# application/use_cases/start_interview_use_case.py
from dataclasses import dataclass
from domain.entities.interview import Interview
from domain.value_objects.skill_level import SkillLevel
from domain.repositories.interview_repository import InterviewRepository

@dataclass
class StartInterviewRequest:
    user_id: str
    role: str
    interview_type: str = "technical"
    skill_level: str = "junior"

@dataclass
class StartInterviewResponse:
    interview_id: str
    first_question: str
    message: str

class StartInterviewUseCase:
    """
    Caso de uso: Iniciar entrevista
    
    Responsabilidades:
    1. Validar que no haya entrevista activa
    2. Crear nueva entrevista
    3. Persistir entrevista
    4. Retornar primera pregunta
    """
    
    def __init__(
        self,
        interview_repo: InterviewRepository,
        question_selector: 'QuestionSelectorService'
    ):
        self.interview_repo = interview_repo
        self.question_selector = question_selector
    
    async def execute(self, request: StartInterviewRequest) -> StartInterviewResponse:
        # 1. Validar que no exista entrevista activa
        active_interview = await self.interview_repo.find_active_by_user(request.user_id)
        if active_interview:
            raise ValueError(f"User {request.user_id} already has an active interview")
        
        # 2. Crear nueva entrevista
        interview = Interview(
            user_id=request.user_id,
            role=request.role,
            interview_type=request.interview_type,
            skill_level=SkillLevel(request.skill_level)
        )
        
        # 3. Iniciar entrevista
        interview.start()
        
        # 4. Seleccionar primera pregunta
        first_question = await self.question_selector.select_initial_question(
            role=request.role,
            level=interview.skill_level,
            interview_type=request.interview_type
        )
        
        interview.add_question(first_question)
        
        # 5. Persistir
        await self.interview_repo.save(interview)
        
        # 6. Retornar respuesta
        return StartInterviewResponse(
            interview_id=interview.id,
            first_question=first_question.text,
            message=f"¡Hola! Iniciemos tu entrevista para {request.role}. 🚀"
        )
```

```python
# application/use_cases/process_answer_use_case.py
from dataclasses import dataclass
from typing import List, Dict
from domain.repositories.interview_repository import InterviewRepository
from services.evaluation_service import EvaluationService
from services.feedback_service import FeedbackService

@dataclass
class ProcessAnswerRequest:
    user_id: str
    answer_text: str

@dataclass
class ProcessAnswerResponse:
    feedback: str
    is_correct: bool
    score: float
    hints: List[str]
    emotion: str
    encouragement: str
    learning_resources: List[Dict]
    next_question: str = ""

class ProcessAnswerUseCase:
    """
    Caso de uso: Procesar respuesta del candidato
    
    Flujo:
    1. Recuperar entrevista activa
    2. Evaluar respuesta con ML/LLM
    3. Generar feedback personalizado
    4. Actualizar estado de entrevista
    5. Seleccionar siguiente pregunta (si aplica)
    """
    
    def __init__(
        self,
        interview_repo: InterviewRepository,
        evaluation_service: EvaluationService,
        feedback_service: FeedbackService,
        question_selector: 'QuestionSelectorService'
    ):
        self.interview_repo = interview_repo
        self.evaluation_service = evaluation_service
        self.feedback_service = feedback_service
        self.question_selector = question_selector
    
    async def execute(self, request: ProcessAnswerRequest) -> ProcessAnswerResponse:
        # 1. Recuperar entrevista activa
        interview = await self.interview_repo.find_active_by_user(request.user_id)
        if not interview or not interview.current_question:
            raise ValueError("No active interview found")
        
        # 2. Evaluar respuesta
        evaluation = await self.evaluation_service.evaluate(
            question=interview.current_question,
            answer_text=request.answer_text,
            interview_context=interview
        )
        
        # 3. Crear Answer entity
        from domain.entities.answer import Answer
        answer = Answer(
            question_id=interview.current_question.id,
            text=request.answer_text,
            score=evaluation.score,
            is_correct=evaluation.is_correct,
            emotion=evaluation.emotion,
            evaluation_details=evaluation.to_dict()
        )
        
        # 4. Agregar respuesta a entrevista
        interview.add_answer(answer)
        
        # 5. Generar feedback personalizado
        feedback = await self.feedback_service.generate(
            evaluation=evaluation,
            question=interview.current_question,
            answer=answer,
            interview=interview
        )
        
        # 6. Seleccionar siguiente pregunta (si no terminó)
        next_question_text = ""
        if len(interview.answers_history) < 10:  # 10 preguntas por entrevista
            next_question = await self.question_selector.select_next_question(
                interview=interview,
                last_evaluation=evaluation
            )
            interview.add_question(next_question)
            next_question_text = next_question.text
        else:
            interview.complete()
        
        # 7. Persistir cambios
        await self.interview_repo.save(interview)
        
        # 8. Retornar respuesta
        return ProcessAnswerResponse(
            feedback=feedback.message,
            is_correct=evaluation.is_correct,
            score=evaluation.score,
            hints=feedback.hints,
            emotion=evaluation.emotion.value,
            encouragement=feedback.encouragement,
            learning_resources=feedback.learning_resources,
            next_question=next_question_text
        )
```

**PASO 4: Implementar Servicios de Infraestructura**

```python
# infrastructure/persistence/postgres_interview_repository.py
import asyncpg
import json
from typing import Optional, List
from domain.entities.interview import Interview
from domain.repositories.interview_repository import InterviewRepository
from domain.value_objects.interview_status import InterviewStatus
from domain.value_objects.skill_level import SkillLevel

class PostgresInterviewRepository(InterviewRepository):
    """
    Implementación de repositorio con PostgreSQL.
    Usa asyncpg para operaciones asíncronas.
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def save(self, interview: Interview) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interviews (
                    id, user_id, role, interview_type, status, skill_level,
                    current_question, questions_history, answers_history,
                    created_at, started_at, completed_at, context, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    current_question = EXCLUDED.current_question,
                    questions_history = EXCLUDED.questions_history,
                    answers_history = EXCLUDED.answers_history,
                    completed_at = EXCLUDED.completed_at,
                    context = EXCLUDED.context,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                interview.id,
                interview.user_id,
                interview.role,
                interview.interview_type,
                interview.status.value,
                interview.skill_level.value,
                json.dumps(interview.current_question.to_dict()) if interview.current_question else None,
                json.dumps([q.to_dict() for q in interview.questions_history]),
                json.dumps([a.to_dict() for a in interview.answers_history]),
                interview.created_at,
                interview.started_at,
                interview.completed_at,
                json.dumps(interview.context),
                json.dumps(interview.metadata)
            )
    
    async def find_by_id(self, interview_id: str) -> Optional[Interview]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interviews WHERE id = $1",
                interview_id
            )
            return self._row_to_interview(row) if row else None
    
    async def find_active_by_user(self, user_id: str) -> Optional[Interview]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM interviews 
                WHERE user_id = $1 AND status = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id,
                InterviewStatus.ACTIVE.value
            )
            return self._row_to_interview(row) if row else None
    
    async def find_all_by_user(self, user_id: str) -> List[Interview]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM interviews 
                WHERE user_id = $1
                ORDER BY created_at DESC
                """,
                user_id
            )
            return [self._row_to_interview(row) for row in rows]
    
    async def delete(self, interview_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM interviews WHERE id = $1",
                interview_id
            )
    
    def _row_to_interview(self, row: asyncpg.Record) -> Interview:
        """Convierte una fila de DB a entidad Interview"""
        from domain.entities.question import Question
        from domain.entities.answer import Answer
        
        current_q = None
        if row['current_question']:
            current_q = Question.from_dict(json.loads(row['current_question']))
        
        questions_hist = [
            Question.from_dict(q) 
            for q in json.loads(row['questions_history'])
        ]
        
        answers_hist = [
            Answer.from_dict(a) 
            for a in json.loads(row['answers_history'])
        ]
        
        return Interview(
            id=row['id'],
            user_id=row['user_id'],
            role=row['role'],
            interview_type=row['interview_type'],
            status=InterviewStatus(row['status']),
            skill_level=SkillLevel(row['skill_level']),
            current_question=current_q,
            questions_history=questions_hist,
            answers_history=answers_hist,
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            context=json.loads(row['context']),
            metadata=json.loads(row['metadata'])
        )
```

---

### 2. 🔴 FUNCIONES ML STUB (NO PRODUCTIVAS)

**Severidad**: CRÍTICA para calidad del producto  
**Archivos afectados**:
- `app/services/ml_feedback_loop.py`
- `app/services/ml_dynamic_difficulty.py`
- `app/services/ml_recommendations.py`

#### Código Actual (NO Productivo)

```python
# ml_feedback_loop.py - STUB
def generate_feedback_ml(answer, expected=None):
    """
    Genera feedback adaptativo usando ML.
    """
    # TODO: Stub para feedback ML adaptativo  <-- ❌ NO ES ML REAL
    if expected and answer and expected.lower() in answer.lower():
        return "¡Muy bien! Tu respuesta está alineada con lo esperado."
    return "Intenta profundizar más en tu respuesta."

# ml_dynamic_difficulty.py - RULE-BASED
def adjust_question_difficulty(user_level, last_score):
    """
    Ajusta dificultad de preguntas usando ML.
    """
    # NO ES ML - Solo if/else hardcodeados
    if user_level >= 3 and last_score >= 8:
        return "hard"
    elif user_level <= 1 or last_score <= 5:
        return "easy"
    return "medium"
```

#### Solución: ML/AI Real

##### 2.1. Implementar Detector de Emociones Multilenguaje

```python
# infrastructure/ml/multilingual_emotion_detector.py
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from functools import lru_cache
from typing import Dict, List
import langid

class MultilingualEmotionDetector:
    """
    Detector de emociones con soporte para español e inglés.
    Modelos específicos por idioma para máxima precisión.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = self._load_models()
        
    @lru_cache(maxsize=2)
    def _load_models(self) -> Dict:
        """Carga modelos optimizados por idioma"""
        return {
            'es': pipeline(
                "text-classification",
                model="finiteautomata/bertweet-base-emotion-analysis",  # Español
                device=0 if self.device == 'cuda' else -1,
                top_k=None
            ),
            'en': pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",  # Inglés
                device=0 if self.device == 'cuda' else -1,
                top_k=None
            )
        }
    
    def detect(self, text: str) -> Dict:
        """
        Detecta emoción en texto con detección automática de idioma.
        
        Returns:
            {
                'emotion': 'joy' | 'sadness' | 'anger' | 'fear' | 'surprise' | 'neutral',
                'confidence': 0.0-1.0,
                'all_emotions': [{' emotion': str, 'score': float}, ...],
                'language': 'es' | 'en'
            }
        """
        if not text or len(text.strip()) < 3:
            return {
                'emotion': 'neutral',
                'confidence': 1.0,
                'all_emotions': [],
                'language': 'unknown'
            }
        
        # Detectar idioma
        detected_lang, confidence = langid.classify(text)
        lang = 'es' if detected_lang == 'es' else 'en'
        
        # Seleccionar modelo apropiado
        model = self.models[lang]
        
        # Predecir
        results = model(text)
        
        # Normalizar etiquetas
        normalized = self._normalize_emotions(results, lang)
        
        return {
            'emotion': normalized[0]['emotion'],
            'confidence': normalized[0]['score'],
            'all_emotions': normalized,
            'language': lang
        }
    
    def _normalize_emotions(self, results: List[Dict], lang: str) -> List[Dict]:
        """Normaliza nombres de emociones a estándar inglés"""
        emotion_map = {
            # Español
            'alegría': 'joy',
            'tristeza': 'sadness',
            'enojo': 'anger',
            'ira': 'anger',
            'miedo': 'fear',
            'sorpresa': 'surprise',
            'neutro': 'neutral',
            # Inglés (ya normalized)
            'joy': 'joy',
            'sadness': 'sadness',
            'anger': 'anger',
            'fear': 'fear',
            'surprise': 'surprise',
            'neutral': 'neutral',
            'disgust': 'anger'  # Map disgust -> anger
        }
        
        normalized = []
        for r in results:
            emotion = emotion_map.get(r['label'].lower(), 'neutral')
            normalized.append({
                'emotion': emotion,
                'score': r['score']
            })
        
        # Ordenar por score
        normalized.sort(key=lambda x: x['score'], reverse=True)
        
        return normalized
```

##### 2.2. Ajuste Dinámico de Dificultad con Red Neuronal

```python
# infrastructure/ml/difficulty_adjuster_nn.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

class DifficultyNet(nn.Module):
    """
    Red neuronal feedforward para predecir dificultad óptima.
    
    Input: 12 features del historial del usuario
    Output: Probabilidades para [easy, medium, hard]
    """
    
    def __init__(self, input_size=12, hidden_sizes=[64, 32], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, 3))  # 3 clases
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=1)

class NeuralDifficultyAdjuster:
    """
    Ajustador de dificultad con ML real.
    Aprende patrones de desempeño para optimizar el desafío.
    """
    
    DIFFICULTY_MAP = {0: 'easy', 1: 'medium', 2: 'hard'}
    DIFFICULTY_TO_IDX = {'easy': 0, 'medium': 1, 'hard': 2}
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DifficultyNet().to(self.device)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        
        self.model.eval()
    
    def extract_features(self, history: List[Dict]) -> np.ndarray:
        """
        Extrae 12 features del historial:
        
        1. Accuracy últimas 3 respuestas
        2. Accuracy últimas 5 respuestas
        3. Accuracy últimas 10 respuestas
        4. Accuracy total
        5. Racha actual (normalizada)
        6. Tiempo promedio de respuesta (normalizado)
        7. Uso de pistas (ratio)
        8. Nivel emocional (0=negativo, 1=positivo)
        9. Variabilidad de scores (std)
        10. Tendencia (mejorando/empeorando)
        11. Dificultad promedio enfrentada
        12. Ratio de respuestas rápidas (<30s)
        """
        if not history:
            # Usuario nuevo - features por defecto (nivel medio)
            return np.array([0.5] * 12, dtype=np.float32)
        
        # Segmentos de historial
        recent_3 = history[-3:]
        recent_5 = history[-5:]
        recent_10 = history[-10:]
        
        # Feature 1-4: Accuracy en diferentes ventanas
        acc_3 = self._calculate_accuracy(recent_3)
        acc_5 = self._calculate_accuracy(recent_5)
        acc_10 = self._calculate_accuracy(recent_10)
        acc_total = self._calculate_accuracy(history)
        
        # Feature 5: Racha actual
        streak = self._calculate_streak(history)
        streak_norm = min(streak / 10.0, 1.0)  # Normalizar a [0, 1]
        
        # Feature 6: Tiempo promedio
        avg_time = np.mean([h.get('time_taken', 30) for h in recent_10])
        time_norm = min(avg_time / 180.0, 1.0)  # Normalizar (máx 3 min)
        
        # Feature 7: Uso de pistas
        hints_ratio = sum(1 for h in recent_10 if h.get('hint_used', False)) / max(len(recent_10), 1)
        
        # Feature 8: Nivel emocional
        emotion_level = self._calculate_emotion_level(recent_5)
        
        # Feature 9: Variabilidad de scores
        scores = [h.get('score', 5.0) for h in recent_10]
        score_std = np.std(scores) / 10.0 if len(scores) > 1 else 0.2
        
        # Feature 10: Tendencia
        trend = self._calculate_trend(history)
        
        # Feature 11: Dificultad promedio
        avg_difficulty = self._calculate_avg_difficulty(recent_10)
        
        # Feature 12: Ratio respuestas rápidas
        quick_ratio = sum(1 for h in recent_10 if h.get('time_taken', 30) < 30) / max(len(recent_10), 1)
        
        features = np.array([
            acc_3, acc_5, acc_10, acc_total,
            streak_norm, time_norm, hints_ratio,
            emotion_level, score_std, trend,
            avg_difficulty, quick_ratio
        ], dtype=np.float32)
        
        return features
    
    def predict(self, history: List[Dict]) -> Tuple[str, float]:
        """
        Predice la dificultad óptima.
        
        Returns:
            (difficulty: str, confidence: float)
        """
        features = self.extract_features(history)
        
        with torch.no_grad():
            X = torch.from_numpy(features).unsqueeze(0).to(self.device)
            probs = self.model(X)
            predicted_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_idx].item()
        
        return self.DIFFICULTY_MAP[predicted_idx], confidence
    
    def train(
        self,
        training_data: List[Dict],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Entrena el modelo con datos históricos.
        
        training_data format:
        [
            {
                'history': [...],  # Historial hasta ese momento
                'optimal_difficulty': 'medium',  # Dificultad óptima observada
                'outcome': 'success' | 'struggle' | 'bored'  # Resultado
            },
            ...
        ]
        """
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Preparar datos
        X_train = []
        y_train = []
        
        for sample in training_data:
            features = self.extract_features(sample['history'])
            X_train.append(features)
            y_train.append(self.DIFFICULTY_TO_IDX[sample['optimal_difficulty']])
        
        X_train = torch.from_numpy(np.array(X_train)).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f} - Acc: {accuracy:.2f}%")
        
        self.model.eval()
    
    def save(self, path: str):
        """Guarda el modelo entrenado"""
        torch.save(self.model.state_dict(), path)
    
    # Métodos auxiliares privados
    def _calculate_accuracy(self, history_segment: List[Dict]) -> float:
        if not history_segment:
            return 0.5
        correct = sum(1 for h in history_segment if h.get('is_correct', False))
        return correct / len(history_segment)
    
    def _calculate_streak(self, history: List[Dict]) -> int:
        streak = 0
        for h in reversed(history):
            if h.get('is_correct', False):
                streak += 1
            else:
                break
        return streak
    
    def _calculate_emotion_level(self, recent_history: List[Dict]) -> float:
        """0 = negativo, 1 = positivo"""
        if not recent_history:
            return 0.5
        
        positive_emotions = {'joy', 'neutral', 'surprise'}
        emotions = [h.get('emotion', 'neutral') for h in recent_history]
        positive_count = sum(1 for e in emotions if e in positive_emotions)
        
        return positive_count / len(emotions)
    
    def _calculate_trend(self, history: List[Dict]) -> float:
        """Retorna -1 (empeorando) a 1 (mejorando)"""
        if len(history) < 10:
            return 0.0
        
        mid = len(history) // 2
        first_half_acc = self._calculate_accuracy(history[:mid])
        second_half_acc = self._calculate_accuracy(history[mid:])
        
        return second_half_acc - first_half_acc
    
    def _calculate_avg_difficulty(self, recent_history: List[Dict]) -> float:
        """Dificultad promedio: 0.33=easy, 0.66=medium, 1.0=hard"""
        if not recent_history:
            return 0.66
        
        diff_map = {'easy': 0.33, 'medium': 0.66, 'hard': 1.0}
        difficulties = [diff_map.get(h.get('difficulty', 'medium'), 0.66) for h in recent_history]
        
        return np.mean(difficulties)
```

##### 2.3. Sistema de Feedback Adaptativo con LLM

```python
# services/adaptive_feedback_service.py
from langchain import ChatOllama
from langchain.prompts import ChatPromptTemplate
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Evaluation:
    is_correct: bool
    score: float
    emotion: str
    key_concepts_covered: List[str]
    key_concepts_missing: List[str]
    response_quality: str  # 'excellent' | 'good' | 'fair' | 'poor'

@dataclass
class Feedback:
    main_message: str
    specific_points: List[str]
    hints: List[str]
    learning_resources: List[Dict]
    encouragement: str
    next_steps: List[str]

class AdaptiveFeedbackService:
    """
    Genera feedback personalizado y contextual usando LLM.
    Adapta tono, profundidad y contenido según el usuario.
    """
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt_template())
        ])
    
    def _get_system_prompt(self) -> str:
        return """Eres un mentor experto en entrevistas técnicas con estas características:

🎯 TU ROL:
- Mentor empático pero profesional
- Proporcionas feedback específico y accionable
- Te adaptas al nivel emocional del candidato
- Celebras logros y apoyas en dificultades

📋 REGLAS ESTRICTAS:
1. Feedback en 3-5 líneas máximo (conciso)
2. Inicia con aspecto positivo (si hay alguno)
3. Menciona 1-2 puntos específicos a mejorar
4. Proporciona ejemplo concreto cuando sea útil
5. Termina con ánimo o siguiente paso
6. Usa emojis con moderación (1-2 máx)
7. Lenguaje profesional pero cercano

🎭 ADAPTACIÓN EMOCIONAL:
- Si detectas FRUSTRACIÓN → Aumenta empatía, simplifica mensaje
- Si detectas TRISTEZA → Más ánimo, enfatiza progreso
- Si detectas ALEGRÍA/CONFIANZA → Celebra, pero mantén desafío
- Si detectas MIEDO → Tranquiliza, refuerza capacidades
- Si detectas NEUTRAL → Tono balanceado estándar

📊 NIVELES DE DETALLE:
- Junior: Más explicativo, ejemplos concretos
- Mid: Balance teoría-práctica
- Senior: Directo, enfoque en arquitectura/decisiones

SIEMPRE en español."""
    
    def _get_human_prompt_template(self) -> str:
        return """📝 CONTEXTO DE LA RESPUESTA:

Pregunta:
{question}

Respuesta del candidato:
{answer}

📊 EVALUACIÓN:
- ✅ Correcta: {is_correct}
- 📈 Puntuación: {score}/10
- 😊 Emoción: {emotion}
- ✓ Conceptos cubiertos: {concepts_covered}
- ✗ Conceptos faltantes: {concepts_missing}
- 📝 Calidad: {quality}

👤 PERFIL DEL USUARIO:
- Nivel: {level}
- Rol objetivo: {role}
- Historial reciente: {history_summary}
- Racha actual: {streak}

🎯 GENERA FEEDBACK PERSONALIZADO siguiendo las reglas del system prompt."""
    
    async def generate(
        self,
        evaluation: Evaluation,
        question: str,
        answer: str,
        user_profile: Dict,
        history: List[Dict]
    ) -> Feedback:
        """
        Genera feedback completo y personalizado.
        """
        
        # 1. Preparar contexto
        history_summary = self._summarize_history(history[-5:])
        
        # 2. Generar mensaje principal con LLM
        prompt_values = {
            'question': question,
            'answer': answer,
            'is_correct': 'Sí ✓' if evaluation.is_correct else 'No ✗',
            'score': evaluation.score,
            'emotion': self._translate_emotion(evaluation.emotion),
            'concepts_covered': ', '.join(evaluation.key_concepts_covered) or 'Ninguno identificado',
            'concepts_missing': ', '.join(evaluation.key_concepts_missing) or 'Ninguno',
            'quality': self._translate_quality(evaluation.response_quality),
            'level': user_profile.get('level', 'junior'),
            'role': user_profile.get('role', 'Desarrollador'),
            'history_summary': history_summary,
            'streak': user_profile.get('current_streak', 0)
        }
        
        messages = self.feedback_prompt.format_messages(**prompt_values)
        response = await self.llm.ainvoke(messages)
        main_message = response.content
        
        # 3. Generar puntos específicos
        specific_points = self._generate_specific_points(evaluation, answer)
        
        # 4. Generar pistas progresivas
        hints = self._generate_progressive_hints(
            evaluation,
            question,
            user_profile.get('level', 'junior')
        )
        
        # 5. Recursos de aprendizaje
        resources = self._get_learning_resources(
            evaluation.key_concepts_missing,
            user_profile.get('role', '')
        )
        
        # 6. Mensaje de ánimo adaptado
        encouragement = self._generate_encouragement(
            evaluation.emotion,
            evaluation.is_correct,
            user_profile.get('current_streak', 0)
        )
        
        # 7. Próximos pasos
        next_steps = self._generate_next_steps(
            evaluation,
            user_profile
        )
        
        return Feedback(
            main_message=main_message,
            specific_points=specific_points,
            hints=hints[:3],  # Máx 3 pistas
            learning_resources=resources[:2],  # Top 2 recursos
            encouragement=encouragement,
            next_steps=next_steps
        )
    
    def _summarize_history(self, recent: List[Dict]) -> str:
        """Resume las últimas respuestas"""
        if not recent:
            return "Sin historial previo"
        
        correct = sum(1 for h in recent if h.get('is_correct', False))
        total = len(recent)
        
        summary = f"{correct}/{total} correctas en últimas preguntas. "
        
        if correct == total:
            summary += "Desempeño excelente 🌟"
        elif correct == 0:
            summary += "Necesita apoyo adicional"
        elif correct / total >= 0.7:
            summary += "Buen progreso"
        else:
            summary += "Progreso irregular"
        
        return summary
    
    def _translate_emotion(self, emotion: str) -> str:
        """Traduce emoción a español con descripción"""
        emotions = {
            'joy': 'Alegría/Confianza 😊',
            'sadness': 'Tristeza/Desánimo 😔',
            'anger': 'Frustración 😤',
            'fear': 'Inseguridad/Miedo 😰',
            'surprise': 'Sorpresa 😮',
            'neutral': 'Neutral 😐'
        }
        return emotions.get(emotion, 'Neutral')
    
    def _translate_quality(self, quality: str) -> str:
        """Traduce calidad de respuesta"""
        qualities = {
            'excellent': 'Excelente (completa, clara, precisa)',
            'good': 'Buena (cubre puntos clave)',
            'fair': 'Aceptable (falta profundidad)',
            'poor': 'Débil (conceptos incorrectos)'
        }
        return qualities.get(quality, 'Aceptable')
    
    def _generate_specific_points(
        self,
        evaluation: Evaluation,
        answer: str
    ) -> List[str]:
        """Genera puntos específicos de la respuesta"""
        points = []
        
        # Aspectos positivos
        if evaluation.key_concepts_covered:
            points.append(
                f"✓ Bien: Cubriste {', '.join(evaluation.key_concepts_covered[:2])}"
            )
        
        # Aspectos a mejorar
        if evaluation.key_concepts_missing:
            points.append(
                f"⚠ Faltó mencionar: {evaluation.key_concepts_missing[0]}"
            )
        
        # Longitud de respuesta
        if len(answer.split()) < 10:
            points.append(
                "💡 Tip: Respuestas muy cortas pueden omitir detalles importantes"
            )
        elif len(answer.split()) > 200:
            points.append(
                "💡 Tip: Intenta ser más conciso y directo"
            )
        
        return points
    
    def _generate_progressive_hints(
        self,
        evaluation: Evaluation,
        question: str,
        level: str
    ) -> List[str]:
        """Genera pistas progresivas (de menos a más específicas)"""
        if evaluation.is_correct and evaluation.score >= 8:
            return []  # No necesita pistas
        
        hints = []
        
        if evaluation.key_concepts_missing:
            # Pista 1: Concepto general
            hints.append(
                f"🔍 Pista 1: Considera el concepto de '{evaluation.key_concepts_missing[0]}'"
            )
            
            # Pista 2: Pregunta guía
            hints.append(
                f"🤔 Pista 2: ¿Cómo se relaciona {evaluation.key_concepts_missing[0]} con el problema?"
            )
            
            # Pista 3: Ejemplo (solo para juniors o si score < 5)
            if level == 'junior' or evaluation.score < 5:
                hints.append(
                    f"📚 Pista 3: Ejemplo - {evaluation.key_concepts_missing[0]} se usa cuando necesitas..."
                )
        
        return hints
    
    def _get_learning_resources(
        self,
        missing_concepts: List[str],
        role: str
    ) -> List[Dict]:
        """Obtiene recursos de aprendizaje relevantes"""
        resources = []
        
        # Base de datos de recursos (en producción, desde DB)
        resource_db = {
            'OOP': {
                'title': 'Programación Orientada a Objetos - Guía Completa',
                'url': 'https://refactoring.guru/es/design-patterns',
                'type': 'tutorial',
                'duration': '30 min'
            },
            'SOLID': {
                'title': 'Principios SOLID Explicados con Ejemplos',
                'url': 'https://refactoring.guru/es/design-patterns/principles',
                'type': 'article',
                'duration': '20 min'
            },
            # ... más recursos
        }
        
        for concept in missing_concepts[:2]:
            if concept in resource_db:
                resources.append(resource_db[concept])
        
        return resources
    
    def _generate_encouragement(
        self,
        emotion: str,
        is_correct: bool,
        streak: int
    ) -> str:
        """Genera mensaje de ánimo adaptado a la emoción"""
        if emotion in ['sadness', 'fear']:
            return "💪 Respira hondo. Cada error es aprendizaje. ¡Sigue adelante, vas bien!"
        
        elif emotion == 'anger':
            return "😌 Es normal frustrarse. Vamos paso a paso. Tú puedes con esto."
        
        elif is_correct and streak >= 3:
            return f"🔥 ¡Racha de {streak}! Estás imparable. Sigue así, campeón/a."
        
        elif is_correct:
            return "✨ ¡Excelente! Cada acierto te acerca más a tu meta."
        
        else:
            return "🌱 Recuerda: los mejores profesionales también se equivocaron. ¡Ánimo!"
    
    def _generate_next_steps(
        self,
        evaluation: Evaluation,
        user_profile: Dict
    ) -> List[str]:
        """Genera próximos pasos recomendados"""
        steps = []
        
        if not evaluation.is_correct:
            if evaluation.key_concepts_missing:
                steps.append(
                    f"1. Repasa el concepto de {evaluation.key_concepts_missing[0]}"
                )
            steps.append(
                "2. Intenta reformular tu respuesta con lo aprendido"
            )
        
        elif evaluation.score < 8:
            steps.append(
                "1. Profundiza en los detalles de tu respuesta"
            )
            steps.append(
                "2. Considera casos edge y alternativas"
            )
        
        else:
            steps.append(
                "1. Mantén este nivel de detalle en las siguientes preguntas"
            )
        
        return steps
```

---

### CONTINUACIÓN EN SIGUIENTE MENSAJE...

El documento es muy extenso. ¿Deseas que continúe con las secciones restantes:

- **FASE 3: CI/CD Completo** (GitHub Actions pipeline)
- **FASE 4: Tests Exhaustivos** (Unit, Integration, Load)
- **FASE 5: Monitoreo y Observabilidad** (Prometheus, Grafana, Sentry)
- **FASE 6: Performance y Caché** (Redis, optimizaciones)
- **Plan de Implementación con Prioridades**
- **Métricas de Éxito y KPIs**

¿Qué sección te gustaría que detalle primero? 🎯

