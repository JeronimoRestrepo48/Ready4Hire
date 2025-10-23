# 🧪 Ready4Hire - Test Suite

Este directorio contiene **TODOS** los tests del proyecto en un solo lugar.

## 📁 Estructura

```
tests/
├── unit/                          # Tests unitarios
│   ├── test_interview_entity.py   # Entidad Interview
│   ├── test_value_objects.py      # Value Objects
│   └── test_exceptions.py         # Excepciones de dominio
│
├── test_integration.py            # Tests de integración (wrapper)
└── test_integration_full.py       # Tests de integración completos
```

## 🚀 Ejecutar Tests

### Opción 1: Script consolidado (Recomendado)
```bash
./run_tests.sh
```

### Opción 2: Pytest
```bash
pytest Ready4Hire/tests/ -v
```

### Opción 3: Tests específicos
```bash
# Tests unitarios
python3 Ready4Hire/tests/unit/test_interview_entity.py

# Tests de integración
python3 Ready4Hire/tests/test_integration.py
```

## ✅ Tests Incluidos

### Tests Unitarios (3)
- ✅ `test_interview_entity.py` - Entidad Interview
- ✅ `test_value_objects.py` - Value Objects (SkillLevel, etc.)
- ✅ `test_exceptions.py` - Excepciones de dominio

### Tests de Integración (6)
1. ✅ Async Ollama Client (Mock)
2. ✅ Sanitización Input/Output
3. ✅ Validación DTOs
4. ✅ Métricas Prometheus
5. ✅ Flujo de Entrevista (Mock)
6. ✅ Circuit Breaker + Integración

**Total**: 9 tests

## 📊 Coverage

Los tests cubren:
- ✅ Dominio (Entities, Value Objects, Exceptions)
- ✅ Infraestructura (Circuit Breaker, Ollama Client, Security)
- ✅ Aplicación (DTOs, Validación)
- ✅ Monitoring (Métricas de Prometheus)

## 🔧 Configuración

Los tests usan:
- **unittest.mock** para mocks (no requiere Ollama corriendo)
- **asyncio** para tests asíncronos
- **pydantic** para validación de DTOs

## 📝 Notas

- Los tests de integración usan **mocks** y no requieren servicios externos
- Para simulaciones reales, ver `/simulate_interviews.py`
- Tests antiguos/obsoletos están en `/archive/old_tests/`
