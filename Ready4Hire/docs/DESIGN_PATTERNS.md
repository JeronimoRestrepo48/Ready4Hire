# 🎨 Patrones de Diseño - Ready4Hire Infrastructure

## 📋 Resumen

Este documento describe los patrones de diseño implementados en la infraestructura de Ready4Hire. Estos patrones mejoran la modularidad, testabilidad, mantenibilidad y extensibilidad del código.

---

## 🏗️ Patrones Implementados

### 1. Factory Pattern

**Ubicación:** `app/infrastructure/patterns/factory.py`

**Propósito:** Crear servicios de forma consistente y desacoplada.

**Componentes:**
- `LLMServiceFactory`: Crea servicios LLM (Ollama, OpenAI, etc.)
- `CacheServiceFactory`: Crea servicios de cache (Memory, Redis, etc.)
- `MLServiceFactory`: Crea servicios ML (Embeddings, Emotion, etc.)
- `RepositoryFactory`: Crea repositorios (Memory, JSON, PostgreSQL, etc.)

**Uso:**
```python
from app.infrastructure.patterns.factory import get_llm_factory

factory = get_llm_factory()
llm_service = factory.create(provider="ollama", model="llama3.2:3b")
```

**Beneficios:**
- Desacoplamiento de implementaciones concretas
- Fácil cambio de proveedores
- Testing simplificado con mocks

---

### 2. Strategy Pattern

**Ubicación:** `app/infrastructure/patterns/strategy.py`

**Propósito:** Estrategias intercambiables para algoritmos.

**Componentes:**
- `EvaluationStrategy`: Estrategias de evaluación (LLM, Heuristic, Hybrid)
- `FeedbackStrategy`: Estrategias de feedback (Personalized, Concise, Detailed)
- `QuestionSelectionStrategy`: Estrategias de selección (Random, Clustering, Adaptive)
- `StrategyRegistry`: Registro centralizado de estrategias

**Uso:**
```python
from app.infrastructure.patterns.strategy import get_strategy_registry

registry = get_strategy_registry()
eval_strategy = registry.get_evaluation_strategy("llm", llm_service=llm_service)
result = eval_strategy.evaluate(question="...", answer="...", ...)
```

**Beneficios:**
- Algoritmos intercambiables en runtime
- Fácil agregar nuevas estrategias
- Separación de lógica de negocio

---

### 3. Repository Pattern

**Ubicación:** `app/infrastructure/patterns/repository.py`

**Propósito:** Abstracción para acceso a datos.

**Componentes:**
- `BaseRepository`: Interfaz común para repositorios
- `RepositoryRegistry`: Registro centralizado de repositorios

**Uso:**
```python
from app.infrastructure.patterns.repository import get_repository_registry

registry = get_repository_registry()
repo = registry.get("json_question", tech_file="...", soft_file="...")
questions = await repo.find_all_technical()
```

**Beneficios:**
- Independencia de implementación de datos
- Testing con repositorios en memoria
- Fácil cambio de almacenamiento

---

### 4. Adapter Pattern

**Ubicación:** `app/infrastructure/patterns/adapter.py`

**Propósito:** Adaptar servicios externos a interfaces comunes.

**Componentes:**
- `ServiceAdapter`: Adaptador base
- `LLMAdapter`: Adapta proveedores LLM
- `CacheAdapter`: Adapta proveedores de cache

**Uso:**
```python
from app.infrastructure.patterns.adapter import LLMAdapter

adapter = LLMAdapter(provider="ollama", base_url="http://localhost:11434")
response = adapter.generate("Hello")
health = adapter.health_check()
```

**Beneficios:**
- Integración consistente de servicios externos
- Fácil cambio de proveedores
- Health checks unificados

---

### 5. Observer Pattern

**Ubicación:** `app/infrastructure/patterns/observer.py`

**Propósito:** Sistema de eventos y notificaciones.

**Componentes:**
- `EventObserver`: Observador abstracto
- `EventPublisher`: Publicador de eventos
- `EventBus`: Bus de eventos global (singleton)
- `Event`: Clase de datos para eventos

**Uso:**
```python
from app.infrastructure.patterns.observer import get_event_bus, EventObserver

class MyObserver(EventObserver):
    def on_event(self, event):
        print(f"Evento recibido: {event.name}")

bus = get_event_bus()
bus.subscribe(MyObserver())
bus.publish("interview_started", {"user_id": "123"})
```

**Beneficios:**
- Comunicación desacoplada entre componentes
- Fácil agregar nuevos observadores
- Sistema de eventos centralizado

---

### 6. Builder Pattern

**Ubicación:** `app/infrastructure/patterns/builder.py`

**Propósito:** Construcción paso a paso de objetos complejos.

**Componentes:**
- `InterviewBuilder`: Builder para Interview
- `EvaluationBuilder`: Builder para Evaluation
- `QuestionSelectorBuilder`: Builder para QuestionSelector

**Uso:**
```python
from app.infrastructure.patterns.builder import InterviewBuilder

interview = (InterviewBuilder()
    .with_user_id("user123")
    .with_role("Software Developer")
    .with_difficulty("mid")
    .with_questions(questions)
    .build())
```

**Beneficios:**
- Construcción clara y legible
- Validación en construcción
- Fluent interface

---

### 7. Decorator Pattern

**Ubicación:** `app/infrastructure/patterns/decorator.py`

**Propósito:** Agregar funcionalidades sin modificar código existente.

**Componentes:**
- `CachedService`: Agrega cache a servicios
- `LoggedService`: Agrega logging
- `RetryService`: Agrega retry logic
- `MetricsService`: Agrega métricas
- `log_method_call`: Decorador para métodos
- `retry_on_failure`: Decorador de retry

**Uso:**
```python
from app.infrastructure.patterns.decorator import CachedService, MetricsService
from app.infrastructure.cache import get_cache_service
from app.infrastructure.monitoring import get_metrics

@CachedService(get_cache_service(), ttl=3600)
@MetricsService(get_metrics())
class MyService:
    def generate(self, prompt):
        return "response"
```

**Beneficios:**
- Funcionalidades adicionales sin modificar código
- Composición flexible
- Separación de responsabilidades

---

### 8. Facade Pattern

**Ubicación:** `app/infrastructure/patterns/facade.py`

**Propósito:** Interfaces simplificadas para sistemas complejos.

**Componentes:**
- `LLMFacade`: Facade para servicios LLM
- `MLFacade`: Facade para servicios ML
- `InfrastructureFacade`: Facade principal

**Uso:**
```python
from app.infrastructure.patterns.facade import InfrastructureFacade

facade = InfrastructureFacade(container)
result = facade.evaluate_interview_answer(
    question="...",
    answer="...",
    role="Software Developer"
)
```

**Beneficios:**
- Interfaces simples y claras
- Oculta complejidad interna
- Facilita uso de servicios

---

### 9. Proxy Pattern

**Ubicación:** `app/infrastructure/patterns/proxy.py`

**Propósito:** Control de acceso y optimización.

**Componentes:**
- `LazyServiceProxy`: Lazy loading de servicios
- `CachedServiceProxy`: Cache transparente
- `ProtectedServiceProxy`: Control de acceso

**Uso:**
```python
from app.infrastructure.patterns.proxy import LazyServiceProxy, CachedServiceProxy

# Lazy loading
lazy_service = LazyServiceProxy(lambda: MyService())
# Servicio solo se inicializa cuando se usa

# Cache transparente
cached_service = CachedServiceProxy(service, cache_service, ttl=3600)
# Resultados se cachean automáticamente
```

**Beneficios:**
- Optimización de recursos (lazy loading)
- Cache transparente
- Control de acceso

---

## 🔄 Integración con Container

Los patrones se integran con el `Container` existente:

```python
from app.container import get_container
from app.infrastructure.patterns.facade import InfrastructureFacade

container = get_container()
facade = InfrastructureFacade(container)

# Usar facade simplificado
result = facade.evaluate_interview_answer(...)
```

---

## 📊 Comparación de Patrones

| Patrón | Propósito | Cuándo Usar |
|--------|-----------|-------------|
| Factory | Creación | Múltiples implementaciones del mismo tipo |
| Strategy | Algoritmos | Diferentes formas de hacer lo mismo |
| Repository | Datos | Acceso a datos independiente de implementación |
| Adapter | Integración | Integrar servicios externos |
| Observer | Eventos | Comunicación desacoplada |
| Builder | Construcción | Objetos complejos con muchos parámetros |
| Decorator | Funcionalidades | Agregar características sin modificar |
| Facade | Simplificación | Ocultar complejidad |
| Proxy | Control | Lazy loading, cache, control de acceso |

---

## 🧪 Testing con Patrones

Los patrones facilitan el testing:

```python
# Factory permite fácil mocking
mock_factory = Mock()
mock_factory.create.return_value = MockService()

# Strategy permite testing de diferentes algoritmos
strategy = HeuristicEvaluationStrategy()
result = strategy.evaluate(...)

# Repository permite testing con datos en memoria
repo = MemoryRepository()
```

---

## 🚀 Extensibilidad

Agregar nuevos componentes es fácil:

```python
# Registrar nueva estrategia
registry = get_strategy_registry()
registry.register_evaluation_strategy("custom", CustomStrategy)

# Registrar nuevo factory
LLMServiceFactory.register("new_provider", NewProviderService)

# Agregar nuevo observer
bus = get_event_bus()
bus.subscribe(MyNewObserver())
```

---

## 📝 Buenas Prácticas

1. **Usar Factories para creación**: No instanciar directamente servicios
2. **Usar Strategies para algoritmos**: Hacer código intercambiable
3. **Usar Repositories para datos**: Mantener lógica de datos separada
4. **Usar Observers para eventos**: Evitar acoplamiento directo
5. **Usar Builders para construcción compleja**: Hacer código más legible
6. **Usar Decorators para funcionalidades**: No modificar código existente
7. **Usar Facades para simplificar**: Ocultar complejidad cuando sea apropiado
8. **Usar Proxies para optimización**: Lazy loading y cache cuando sea necesario

---

## 🔗 Referencias

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://en.wikipedia.org/wiki/Design_Patterns)
- [Python Design Patterns](https://python-patterns.guide/)
- [Architecture Patterns](https://martinfowler.com/architecture/)

---

**Fecha de implementación:** 2025-11-03  
**Versión:** v1.0

