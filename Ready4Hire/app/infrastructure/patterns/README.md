# 🎨 Design Patterns - Ready4Hire

Este módulo contiene implementaciones de patrones de diseño para mejorar la arquitectura de Ready4Hire.

## 📁 Estructura

```
patterns/
├── __init__.py          # Exports principales
├── factory.py           # Factory Pattern
├── strategy.py          # Strategy Pattern
├── repository.py        # Repository Pattern
├── adapter.py           # Adapter Pattern
├── observer.py          # Observer Pattern
├── builder.py           # Builder Pattern
├── decorator.py         # Decorator Pattern
├── facade.py            # Facade Pattern
└── proxy.py             # Proxy Pattern
```

## 🚀 Quick Start

### Factory Pattern
```python
from app.infrastructure.patterns.factory import get_llm_factory

factory = get_llm_factory()
llm = factory.create(provider="ollama", model="llama3.2:3b")
```

### Strategy Pattern
```python
from app.infrastructure.patterns.strategy import get_strategy_registry

registry = get_strategy_registry()
strategy = registry.get_evaluation_strategy("llm", llm_service=llm)
```

### Observer Pattern
```python
from app.infrastructure.patterns.observer import get_event_bus, EventObserver

class MyObserver(EventObserver):
    def on_event(self, event):
        print(f"Event: {event.name}")

bus = get_event_bus()
bus.subscribe(MyObserver())
bus.publish("interview_started", {"user_id": "123"})
```

### Facade Pattern
```python
from app.container import get_container
from app.infrastructure.patterns.facade import InfrastructureFacade

container = get_container()
facade = InfrastructureFacade(container)
result = facade.evaluate_interview_answer(...)
```

## 📚 Documentación Completa

Ver [DESIGN_PATTERNS.md](../../docs/DESIGN_PATTERNS.md) para documentación completa.

