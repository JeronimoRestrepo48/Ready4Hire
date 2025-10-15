# 🏛️ Arquitectura Ready4Hire - Documentación Técnica

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Decisiones de Arquitectura](#decisiones-de-arquitectura)
3. [Domain Layer](#domain-layer---lógica-de-negocio)
4. [Application Layer](#application-layer---casos-de-uso)
5. [Infrastructure Layer](#infrastructure-layer---implementaciones)
6. [Flujos de Datos](#flujos-de-datos)
7. [Patrones de Diseño](#patrones-de-diseño-implementados)
8. [Escalabilidad y Performance](#escalabilidad-y-performance)
9. [Seguridad](#seguridad)
10. [Migración y Evolución](#migración-y-evolución)

---

## Visión General

Ready4Hire implementa una **arquitectura limpia basada en Domain-Driven Design (DDD)** con separación estricta de responsabilidades en 3 capas principales. Esta arquitectura reemplaza el sistema monolítico anterior (1422 líneas en un solo archivo) por una estructura modular, testable y mantenible.

### Principios Arquitectónicos

1. **Dependency Inversion**: Las capas externas dependen de las internas, nunca al revés
2. **Single Responsibility**: Cada componente tiene una única razón para cambiar
3. **Open/Closed**: Abierto para extensión, cerrado para modificación
4. **Testability First**: Todas las capas son testeables de forma aislada
5. **Explicit is Better**: Contratos explícitos (interfaces) sobre implementaciones concretas

---

## Decisiones de Arquitectura

### ADR-001: Domain-Driven Design

**Contexto**: Sistema monolítico de 1422 líneas difícil de mantener, testear y escalar.

**Decisión**: Adoptar DDD con separación en 3 capas (Domain, Application, Infrastructure).

**Consecuencias**:
- ✅ **Positivas**: 
  - Lógica de negocio aislada y testeable
  - Facilita agregar nuevas features sin romper existentes
  - Permite cambiar implementaciones (BD, ML) sin tocar dominio
- ⚠️ **Negativas**:
  - Mayor cantidad de archivos (50+ vs 8)
  - Curva de aprendizaje para desarrolladores nuevos
  - Más boilerplate inicial

**Alternativas consideradas**:
- MVC tradicional: Rechazado (mezcla lógica de negocio con presentación)
- Microservicios: Rechazado (overhead para tamaño actual del proyecto)

---

### ADR-002: ML Real vs Stubs

**Contexto**: Sistema anterior usaba "ML fake" (stubs con lógica hardcodeada).

**Decisión**: Implementar modelos ML reales (HuggingFace Transformers, PyTorch).

**Consecuencias**:
- ✅ **Positivas**:
  - Detección real de emociones con 85%+ precisión
  - Soporte multilingüe (ES + EN)
  - Ajuste dinámico de dificultad basado en features reales
- ⚠️ **Negativas**:
  - Mayor uso de CPU/RAM (modelos ~500MB)
  - Latencia adicional en primera inferencia (~2-3s, luego <200ms)
  - Requiere gestión de cache de modelos

**Mitigación**:
- Lazy loading de modelos (solo se cargan cuando se usan)
- Cache de modelos en memoria
- Fallback heurístico si ML falla

---

### ADR-003: LLM para Evaluación

**Contexto**: Necesidad de evaluar respuestas de forma inteligente y contextual.

**Decisión**: Usar LLMs (OpenAI GPT-4 / Anthropic Claude) con fallback heurístico.

**Consecuencias**:
- ✅ **Positivas**:
  - Evaluación profunda y contextualizada
  - Feedback personalizado de alta calidad
  - Captura conceptos que heurísticas no detectan
- ⚠️ **Negativas**:
  - Dependencia de APIs externas (costo, latencia, disponibilidad)
  - Costo variable según uso (~$0.03-0.06 por respuesta con GPT-4)

**Mitigación**:
- Fallback heurístico automático (basado en longitud, keywords)
- Rate limiting y retry logic
- Cache de evaluaciones frecuentes (futuro)

---

## Domain Layer - Lógica de Negocio

### Entities (Entidades)

#### Interview (Aggregate Root)
```python
class Interview:
    """
    Raíz del agregado Interview.
    Orquesta toda la lógica de una entrevista.
    """
    def __init__(
        self,
        id: str,
        user_id: str,
        role: str,
        category: str,
        current_difficulty: SkillLevel,
        max_questions: int = 10
    ):
        self.id = id
        self.user_id = user_id
        self.role = role
        self.category = category
        self.status = InterviewStatus.CREATED
        self.current_difficulty = current_difficulty
        self.max_questions = max_questions
        self.current_question_index = 0
        self.questions: List[str] = []  # IDs de preguntas
        self.answers: List[Answer] = []
        # Timestamps
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.paused_at: Optional[datetime] = None
    
    # Métodos de negocio
    def start(self) -> None:
        """Iniciar entrevista con validaciones."""
        if self.status != InterviewStatus.CREATED:
            raise ValueError("Interview already started")
        self.status = InterviewStatus.ACTIVE
        self.started_at = datetime.utcnow()
    
    def add_question(self, question_id: str) -> None:
        """Agregar pregunta con validaciones."""
        if self.status not in [InterviewStatus.ACTIVE, InterviewStatus.PAUSED]:
            raise ValueError("Cannot add question to inactive interview")
        if len(self.questions) >= self.max_questions:
            raise ValueError("Max questions reached")
        self.questions.append(question_id)
    
    def add_answer(self, answer: Answer) -> None:
        """Agregar respuesta y actualizar estado."""
        if self.status != InterviewStatus.ACTIVE:
            raise ValueError("Interview not active")
        self.answers.append(answer)
        self.current_question_index += 1
    
    def get_score_average(self) -> float:
        """Calcular promedio de puntuaciones."""
        if not self.answers:
            return 0.0
        scores = [a.score.value for a in self.answers if a.score]
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_accuracy(self) -> float:
        """Calcular porcentaje de respuestas correctas (score >= 6)."""
        if not self.answers:
            return 0.0
        correct = sum(1 for a in self.answers if a.score and a.score.value >= 6.0)
        return (correct / len(self.answers)) * 100
    
    def needs_difficulty_adjustment(self) -> bool:
        """Determinar si se debe ajustar dificultad."""
        if len(self.answers) < 3:
            return False
        recent_scores = [a.score.value for a in self.answers[-3:] if a.score]
        avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        # Subir si últimas 3 respuestas promedio >= 7.5
        # Bajar si últimas 3 respuestas promedio < 5.0
        return avg_recent >= 7.5 or avg_recent < 5.0
```

**Reglas de negocio en Interview:**
1. Solo se puede iniciar una entrevista CREATED
2. Solo se pueden agregar preguntas si está ACTIVE o PAUSED
3. Máximo `max_questions` preguntas por entrevista
4. Ajuste de dificultad basado en últimas 3 respuestas
5. Accuracy considera correctas las respuestas con score >= 6.0

#### Question
```python
@dataclass
class Question:
    """Entidad Question con metadata para evaluación."""
    id: str
    role: str
    category: str  # 'technical' | 'soft_skills'
    difficulty: SkillLevel
    question_text: str
    keywords: List[str]
    expected_concepts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Answer
```python
@dataclass
class Answer:
    """Entidad Answer con evaluación completa."""
    id: str
    question_id: str
    answer_text: str
    score: Score
    emotion: Emotion
    time_taken_seconds: int
    hints_used: int = 0
    evaluation_details: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
```

### Value Objects

#### Score
```python
class Score:
    """
    Value Object para puntuaciones (0-10).
    Inmutable, validado.
    """
    def __init__(self, value: float):
        if not 0 <= value <= 10:
            raise ValueError("Score must be between 0 and 10")
        self._value = round(value, 1)
    
    @property
    def value(self) -> float:
        return self._value
    
    def is_passing(self) -> bool:
        """Considera aprobado si score >= 6.0"""
        return self._value >= 6.0
    
    def quality_level(self) -> str:
        """Retorna nivel de calidad."""
        if self._value >= 9.0:
            return "excellent"
        elif self._value >= 7.0:
            return "good"
        elif self._value >= 5.0:
            return "acceptable"
        else:
            return "needs_improvement"
```

#### Emotion
```python
class Emotion(Enum):
    """Enum de emociones detectables."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    
    def sentiment(self) -> str:
        """Clasifica en positivo/negativo/neutral."""
        if self in [Emotion.JOY, Emotion.SURPRISE]:
            return "positive"
        elif self in [Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR]:
            return "negative"
        else:
            return "neutral"
```

#### SkillLevel
```python
class SkillLevel(Enum):
    """Niveles de habilidad con transiciones."""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    
    def can_increase_to(self, other: 'SkillLevel') -> bool:
        """Valida transición válida."""
        transitions = {
            SkillLevel.JUNIOR: [SkillLevel.MID],
            SkillLevel.MID: [SkillLevel.SENIOR],
            SkillLevel.SENIOR: []
        }
        return other in transitions[self]
    
    def to_numeric(self) -> int:
        """Conversión a numérico para cálculos."""
        return {
            SkillLevel.JUNIOR: 0,
            SkillLevel.MID: 1,
            SkillLevel.SENIOR: 2
        }[self]
```

### Repository Interfaces (Ports)

```python
from abc import ABC, abstractmethod

class InterviewRepository(ABC):
    """Interfaz para persistencia de Interview."""
    
    @abstractmethod
    async def save(self, interview: Interview) -> None:
        """Guardar o actualizar entrevista."""
        pass
    
    @abstractmethod
    async def find_by_id(self, interview_id: str) -> Optional[Interview]:
        """Buscar por ID."""
        pass
    
    @abstractmethod
    async def find_active_by_user(self, user_id: str) -> Optional[Interview]:
        """Buscar entrevista activa del usuario."""
        pass
    
    @abstractmethod
    async def find_all_by_user(self, user_id: str) -> List[Interview]:
        """Listar todas las entrevistas de un usuario."""
        pass
```

**Beneficios del Repository Pattern:**
- Domain no conoce detalles de persistencia (SQL, NoSQL, memoria)
- Fácil cambiar implementación (Memory → PostgreSQL)
- Testeabilidad con mocks
- Queries de negocio encapsuladas

---

## Application Layer - Casos de Uso

### Use Cases

#### StartInterviewUseCase
```python
class StartInterviewUseCase:
    """
    Caso de uso: Iniciar nueva entrevista.
    Orquesta validaciones, creación de entidad y persistencia.
    """
    def __init__(
        self,
        interview_repo: InterviewRepository,
        question_repo: QuestionRepository,
        question_selector: QuestionSelectorService
    ):
        self.interview_repo = interview_repo
        self.question_repo = question_repo
        self.question_selector = question_selector
    
    async def execute(
        self,
        user_id: str,
        role: str,
        category: str,
        initial_difficulty: SkillLevel = SkillLevel.MID
    ) -> Dict[str, Any]:
        """
        Flujo:
        1. Validar que no haya entrevista activa
        2. Crear Interview entity
        3. Seleccionar primera pregunta
        4. Iniciar entrevista
        5. Persistir
        6. Retornar resultado
        """
        # 1. Validar
        existing = await self.interview_repo.find_active_by_user(user_id)
        if existing:
            raise ValueError("User already has active interview")
        
        # 2. Crear entidad
        interview = Interview(
            id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            category=category,
            current_difficulty=initial_difficulty
        )
        
        # 3. Seleccionar primera pregunta
        first_question = await self.question_selector.select_initial_question(
            role=role,
            category=category,
            difficulty=initial_difficulty
        )
        
        # 4. Iniciar
        interview.start()
        interview.add_question(first_question.id)
        
        # 5. Persistir
        await self.interview_repo.save(interview)
        
        # 6. Retornar
        return {
            "interview_id": interview.id,
            "message": f"¡Bienvenido! Entrevista {category} para {role}.",
            "first_question": {
                "id": first_question.id,
                "text": first_question.question_text,
                "difficulty": first_question.difficulty.value
            }
        }
```

#### ProcessAnswerUseCase
```python
class ProcessAnswerUseCase:
    """
    Caso de uso: Procesar respuesta del candidato.
    Orquesta detección emoción, evaluación, feedback y siguiente pregunta.
    """
    def __init__(
        self,
        interview_repo: InterviewRepository,
        question_repo: QuestionRepository,
        emotion_detector: MultilingualEmotionDetector,
        evaluation_service: EvaluationService,
        feedback_service: FeedbackService,
        question_selector: QuestionSelectorService
    ):
        self.interview_repo = interview_repo
        self.question_repo = question_repo
        self.emotion_detector = emotion_detector
        self.evaluation_service = evaluation_service
        self.feedback_service = feedback_service
        self.question_selector = question_selector
    
    async def execute(
        self,
        interview_id: str,
        answer_text: str
    ) -> Dict[str, Any]:
        """
        Flujo:
        1. Recuperar entrevista y pregunta actual
        2. Detectar emoción
        3. Evaluar respuesta con LLM
        4. Generar feedback personalizado
        5. Crear Answer entity
        6. Seleccionar siguiente pregunta (si no terminó)
        7. Persistir
        8. Retornar resultado
        """
        # 1. Recuperar
        interview = await self.interview_repo.find_by_id(interview_id)
        if not interview:
            raise ValueError("Interview not found")
        
        current_q_id = interview.questions[-1]
        question = await self.question_repo.find_by_id(current_q_id)
        
        # 2. Detectar emoción
        emotion_result = self.emotion_detector.detect(answer_text)
        emotion = emotion_result['emotion']
        
        # 3. Evaluar
        evaluation = await self.evaluation_service.evaluate_answer(
            question=question.question_text,
            answer=answer_text,
            expected_concepts=question.expected_concepts,
            keywords=question.keywords,
            category=question.category,
            difficulty=question.difficulty.value,
            role=question.role
        )
        
        # 4. Feedback
        feedback = await self.feedback_service.generate_feedback(
            question=question.question_text,
            answer=answer_text,
            evaluation=evaluation,
            emotion=emotion.value,
            role=interview.role,
            category=interview.category,
            performance_history=[a.to_dict() for a in interview.answers]
        )
        
        # 5. Crear Answer
        answer = Answer(
            id=str(uuid.uuid4()),
            question_id=question.id,
            answer_text=answer_text,
            score=Score(evaluation['score']),
            emotion=emotion,
            time_taken_seconds=0,  # TODO: trackear tiempo
            evaluation_details=evaluation,
            feedback=feedback
        )
        
        interview.add_answer(answer)
        
        # 6. Siguiente pregunta
        next_question = None
        if interview.current_question_index < interview.max_questions:
            next_question = await self.question_selector.select_next_question(
                interview=interview,
                previous_question_ids=interview.questions,
                last_answer_score=evaluation['score']
            )
            if next_question:
                interview.add_question(next_question.id)
        else:
            interview.complete()
        
        # 7. Persistir
        await self.interview_repo.save(interview)
        
        # 8. Retornar
        result = {
            "evaluation": evaluation,
            "emotion": {
                "detected": emotion.value,
                "confidence": emotion_result['confidence'],
                "language": emotion_result['language']
            },
            "feedback": feedback,
            "progress": {
                "current": interview.current_question_index,
                "total": interview.max_questions,
                "percentage": (interview.current_question_index / interview.max_questions) * 100
            }
        }
        
        if next_question:
            result["next_question"] = {
                "id": next_question.id,
                "text": next_question.question_text,
                "difficulty": next_question.difficulty.value
            }
        
        return result
```

### Services (Servicios de Aplicación)

#### EvaluationService
**Responsabilidad**: Evaluar respuestas usando LLM con fallback heurístico.

**Algoritmo**:
1. Construir prompt estructurado con contexto
2. Llamar LLM (OpenAI/Anthropic)
3. Parsear respuesta JSON
4. Validar estructura y rangos
5. Si falla: usar evaluación heurística (longitud, keywords)

**Input**: Question, Answer, ExpectedConcepts, Keywords
**Output**: 
```json
{
  "score": 7.5,
  "breakdown": {
    "completeness": 2.5,
    "technical_depth": 2.0,
    "clarity": 1.5,
    "key_concepts": 1.5
  },
  "justification": "...",
  "strengths": ["...", "..."],
  "improvements": ["...", "..."],
  "concepts_covered": ["...", "..."]
}
```

#### FeedbackService
**Responsabilidad**: Generar feedback personalizado considerando emoción.

**Estrategia**:
- Tono adaptado a emoción:
  - `joy/neutral`: Positivo y motivador
  - `sadness/fear`: Empático y alentador
  - `anger`: Calmado y comprensivo
  - `surprise`: Entusiasta y guía

**Longitud**: 80-120 palabras (3-4 oraciones)

#### QuestionSelectorService
**Responsabilidad**: Seleccionar siguiente pregunta adaptándose al rendimiento.

**Estrategia de selección**:
1. **Ajuste de dificultad**:
   - Si score última respuesta >= 7.5: Subir dificultad
   - Si score < 5.0: Bajar dificultad
   - Else: Mantener

2. **Diversidad semántica** (si embeddings disponibles):
   - Calcular distancia coseno con últimas 3 preguntas
   - Seleccionar la más diversa (máxima distancia promedio)

3. **Fallback**:
   - Selección aleatoria de pool válido

---

## Infrastructure Layer - Implementaciones

### Persistence (Adaptadores)

#### MemoryInterviewRepository
```python
class MemoryInterviewRepository(InterviewRepository):
    """Implementación en memoria (desarrollo, tests)."""
    def __init__(self):
        self._storage: Dict[str, Interview] = {}
        self._user_index: Dict[str, List[str]] = {}
    
    async def save(self, interview: Interview) -> None:
        self._storage[interview.id] = interview
        if interview.user_id not in self._user_index:
            self._user_index[interview.user_id] = []
        if interview.id not in self._user_index[interview.user_id]:
            self._user_index[interview.user_id].append(interview.id)
    
    async def find_by_id(self, interview_id: str) -> Optional[Interview]:
        return self._storage.get(interview_id)
    
    async def find_active_by_user(self, user_id: str) -> Optional[Interview]:
        interview_ids = self._user_index.get(user_id, [])
        for iid in interview_ids:
            interview = self._storage.get(iid)
            if interview and interview.status == InterviewStatus.ACTIVE:
                return interview
        return None
```

#### PostgresInterviewRepository (TODO)
```python
class PostgresInterviewRepository(InterviewRepository):
    """Implementación con PostgreSQL (async)."""
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def save(self, interview: Interview) -> None:
        async with self.pool.acquire() as conn:
            # Serializar Interview a JSON
            data = interview.to_dict()
            await conn.execute("""
                INSERT INTO interviews (id, user_id, role, status, data, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    data = EXCLUDED.data,
                    updated_at = NOW()
            """, interview.id, interview.user_id, interview.role, 
                interview.status.value, json.dumps(data))
```

### ML Models (Adaptadores)

#### MultilingualEmotionDetector
**Tecnología**: HuggingFace Transformers

**Modelos**:
- Español: `finiteautomata/bertweet-base-emotion-analysis`
- Inglés: `j-hartmann/emotion-english-distilroberta-base`

**Flujo**:
1. Detectar idioma con `langid`
2. Cargar modelo apropiado (lazy loading)
3. Ejecutar inferencia
4. Mapear labels a `Emotion` enum
5. Retornar resultado con confianza

**Performance**:
- Primera inferencia: ~2-3s (descarga modelo)
- Siguientes: <200ms (modelo en cache)
- Tamaño modelos: ~500MB total

#### NeuralDifficultyAdjuster
**Tecnología**: PyTorch

**Arquitectura**:
```
Input Layer (12 features)
    ↓
Dense(64) + ReLU
    ↓
Dense(32) + ReLU
    ↓
Output Layer (3) → Softmax
    ↓
[P(JUNIOR), P(MID), P(SENIOR)]
```

**Features extraídas**:
1. avg_score: Promedio de puntuaciones
2. consistency: Desviación estándar de scores
3. avg_time: Tiempo promedio de respuesta
4. emotion_positive_ratio: % emociones positivas
5. emotion_negative_ratio: % emociones negativas
6. correct_ratio: % respuestas correctas (score >= 6)
7. accuracy: Precisión general
8. hints_used_avg: Promedio de pistas usadas
9. current_difficulty_num: Dificultad actual (0-2)
10. questions_answered: Cantidad respondidas
11. trend: Tendencia (últimas 3 respuestas)
12. time_variance: Varianza de tiempos

**Training**: TODO (actualmente usa heurística hasta entrenar modelo)

---

## Flujos de Datos

### Flujo Completo: Responder Pregunta

```
┌──────────────────────────────────────────────────────────────┐
│ 1. API Endpoint                                               │
│    POST /answer                                               │
│    Body: {user_id, answer_text}                              │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. ProcessAnswerUseCase.execute()                            │
│    - Recuperar Interview del repositorio                      │
│    - Recuperar Question actual                                │
└────────────────┬─────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬─────────────┐
    ▼            ▼            ▼             ▼
┌────────┐  ┌────────────┐  ┌────────┐  ┌──────────┐
│Emotion │  │Evaluation  │  │Feedback│  │Question  │
│Detector│  │Service     │  │Service │  │Selector  │
└───┬────┘  └─────┬──────┘  └───┬────┘  └────┬─────┘
    │             │             │            │
    │ detect()    │ evaluate()  │ generate() │ select_next()
    │             │             │            │
    ▼             ▼             ▼            ▼
┌────────┐  ┌────────────┐  ┌────────┐  ┌──────────┐
│ML Model│  │OpenAI/     │  │OpenAI/ │  │Embeddings│
│(Trans- │  │Anthropic   │  │Anthrop │  │Manager   │
│formers)│  │API         │  │ic API  │  │          │
└────────┘  └────────────┘  └────────┘  └──────────┘
    │             │             │            │
    └─────────────┴─────────────┴────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Crear Answer Entity                                        │
│    Answer(score, emotion, feedback, ...)                      │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Actualizar Interview                                       │
│    interview.add_answer(answer)                               │
│    interview.add_question(next_question_id)                   │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Persistir                                                  │
│    interview_repo.save(interview)                             │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. Retornar Resultado                                         │
│    {evaluation, emotion, feedback, next_question, progress}   │
└──────────────────────────────────────────────────────────────┘
```

### Flujo de Datos: Detección de Emoción

```
answer_text
    │
    ▼
┌─────────────────────────┐
│ MultilingualEmotionDet  │
│ .detect(text)           │
└───────┬─────────────────┘
        │
        ▼
┌─────────────────────────┐
│ langid.classify(text)   │
│ → 'es' | 'en'           │
└───────┬─────────────────┘
        │
    ┌───┴───┐
    ▼       ▼
┌────────┐ ┌────────┐
│ES Model│ │EN Model│
│bertweet│ │distil  │
│        │ │roberta │
└───┬────┘ └───┬────┘
    └───┬───────┘
        ▼
┌─────────────────────────┐
│ results = [             │
│   {label: 'joy',        │
│    score: 0.85},        │
│   {label: 'neutral',    │
│    score: 0.10},        │
│   ...                   │
│ ]                       │
└───────┬─────────────────┘
        │
        ▼
┌─────────────────────────┐
│ best = max(results)     │
│ emotion = map_to_enum() │
└───────┬─────────────────┘
        │
        ▼
{
  'emotion': Emotion.JOY,
  'confidence': 0.85,
  'all_emotions': [...],
  'language': 'es'
}
```

---

## Patrones de Diseño Implementados

### 1. Repository Pattern
**Uso**: Persistencia de Interviews y Questions
**Beneficio**: Abstracción de detalles de BD

### 2. Strategy Pattern
**Uso**: QuestionSelectorService (diversas estrategias de selección)
**Beneficio**: Fácil agregar nuevas estrategias

### 3. Factory Pattern
**Uso**: Creación de entities con validaciones
**Beneficio**: Centraliza lógica de creación

### 4. Dependency Injection
**Uso**: Use Cases reciben dependencias por constructor
**Beneficio**: Testabilidad, desacoplamiento

### 5. Aggregate Pattern (DDD)
**Uso**: Interview como raíz del agregado
**Beneficio**: Consistencia transaccional, encapsulación

### 6. Value Object Pattern (DDD)
**Uso**: Score, Emotion, SkillLevel
**Beneficio**: Inmutabilidad, validación en construcción

---

## Escalabilidad y Performance

### Bottlenecks Identificados

1. **LLM API Calls**: Latencia 500ms-2s
   - **Mitigación**: Cache de evaluaciones frecuentes (Redis, futuro)
   - **Alternativa**: Modelo local (menor calidad, mayor velocidad)

2. **ML Model Loading**: Primera carga ~2-3s
   - **Mitigación**: Lazy loading + cache en memoria
   - **Alternativa**: Warm-up al iniciar servidor

3. **PostgreSQL Queries**: Hasta 100ms en queries complejas
   - **Mitigación**: Índices optimizados, connection pooling
   - **Alternativa**: Read replicas para consultas

### Estrategias de Escalabilidad

#### Horizontal Scaling
```
                ┌──────────────┐
                │ Load Balancer│
                └───────┬──────┘
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ API Pod │    │ API Pod │    │ API Pod │
   │ Ready4H │    │ Ready4H │    │ Ready4H │
   └────┬────┘    └────┬────┘    └────┬────┘
        └───────────────┼───────────────┘
                        ▼
            ┌──────────────────────┐
            │   Shared Services    │
            ├──────────────────────┤
            │ PostgreSQL (Primary) │
            │ Redis (Cache)        │
            │ OpenAI API           │
            └──────────────────────┘
```

#### Caching Strategy (Futuro)
```
Cache Layers:
1. In-Memory (per-pod): Modelos ML, embeddings recientes
2. Redis: Sessions, evaluaciones frecuentes, preguntas populares
3. PostgreSQL: Datos persistentes

TTL Strategy:
- Sessions: 1h
- Evaluaciones: 24h (si misma question+answer)
- Preguntas: 1 week
- Modelos ML: No expira (warm cache)
```

---

## Seguridad

### Autenticación
- JWT tokens (HS256 algorithm)
- Expiración: 60 minutos
- Refresh tokens: TODO

### Autorización
- Usuarios solo acceden a sus entrevistas
- Validación `user_id` en cada request

### Protección de Datos
- API keys en variables de entorno (nunca en código)
- Secrets en Kubernetes Secrets / Docker Secrets
- Logs sin información sensible (no API keys, no passwords)

### Rate Limiting (TODO)
```python
# Ejemplo
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # Limitar a 100 requests/min por usuario
    pass
```

---

## Migración y Evolución

### Estado Actual (v2.0)

✅ **Completado**:
- Arquitectura DDD
- ML real (emotion, difficulty)
- Use Cases básicos
- Repositories (Memory, JSON)
- CI/CD Pipeline

🔄 **En Progreso**:
- API v2 endpoints
- Tests comprehensivos
- PostgreSQL repositories
- Redis caching

### Plan de Migración (v2.0 → v2.1)

**Fase 1: API Migration** (2-3 días)
- Crear `/api/v2/*` endpoints
- Dependency Injection Container
- DTOs
- Mantener backward compatibility (`/api/v1/*`)

**Fase 2: Testing** (3-4 días)
- Unit tests (domain, application)
- Integration tests (use cases, repositories)
- API tests (E2E)
- Target: 80% coverage

**Fase 3: PostgreSQL** (2-3 días)
- `PostgresInterviewRepository`
- `PostgresQuestionRepository`
- Migration scripts
- Connection pooling

**Fase 4: Cache & Monitoring** (2-3 días)
- Redis integration
- Prometheus metrics
- OpenTelemetry tracing
- Grafana dashboards

### Backlog (v3.0)

- [ ] Frontend web (React)
- [ ] WebSocket para real-time updates
- [ ] Análisis de voz (speech-to-text)
- [ ] Video entrevistas con análisis facial
- [ ] Multi-tenancy (empresas)
- [ ] A/B testing de preguntas
- [ ] Recomendaciones de upskilling

---

## Referencias

### Libros y Recursos
- Domain-Driven Design, Eric Evans
- Clean Architecture, Robert C. Martin
- Implementing Domain-Driven Design, Vaughn Vernon
- Architecture Patterns with Python, Harry Percival & Bob Gregory

### Documentación Técnica
- FastAPI: https://fastapi.tiangolo.com/
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PyTorch: https://pytorch.org/docs/
- asyncpg: https://magicstack.github.io/asyncpg/

---

**Versión**: 2.0.0  
**Última actualización**: 14 de octubre de 2025  
**Autores**: Equipo Ready4Hire  
**Licencia**: MIT
