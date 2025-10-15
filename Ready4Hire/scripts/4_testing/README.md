# ✅ Fase 4: Testing

Scripts para validar el modelo fine-tuned.

## Script

### step1_test_model.py
**Ejecuta tests completos del modelo**

```bash
# Test básico
python3 step1_test_model.py

# Con comparación vs modelo base
python3 step1_test_model.py --compare --base-model llama3.2:3b

# Solo tests técnicos
python3 step1_test_model.py --technical-only
```

**¿Qué hace?**
- Ejecuta tests con respuestas buenas y malas
- Verifica que el modelo distinga calidad
- Calcula accuracy y métricas
- Compara con modelo base (opcional)
- Guarda resultados detallados

**Parámetros opcionales**:
- `--model`: Modelo a testear (default: ready4hire:latest)
- `--compare`: Comparar con modelo base
- `--base-model`: Modelo base (default: llama3.2:3b)
- `--technical-only`: Solo tests técnicos

**Output**: `tests/results/test_results.json`

## 📊 Métricas de éxito

✅ **Accuracy**: >80%  
✅ **Separación**: >3.0 puntos (good - bad)  
✅ **Mejora vs Base**: +10% o más  

## Ejemplo de resultados

```
PASO 4.1: TESTING DEL MODELO
═══════════════════════════════════════

Testeando: ready4hire:latest
───────────────────────────────────────
ℹ Test 1/3: Explica la diferencia entre let, const y var...
✓ PASS - Good: 8.5, Bad: 4.2
ℹ Test 2/3: ¿Qué es el Virtual DOM en React?
✓ PASS - Good: 9.1, Bad: 3.8

Resultados
───────────────────────────────────────
ℹ Total tests: 3
✓ Passed: 3
ℹ Accuracy: 100.0%

✅ PASO 4.1 COMPLETADO
✓ Modelo aprobado (accuracy >= 80%)
```

## ⏱️ Tiempo estimado: ~5 minutos

## ✅ Deployment Final

Una vez aprobados los tests:

```bash
# 1. Configurar .env
echo "MODEL_NAME=ready4hire:latest" >> ../../.env

# 2. Iniciar aplicación
cd ../..
python3 -m uvicorn app.main:app --reload

# 3. Abrir navegador
# http://localhost:8000
```
