# Plan de Integración ML - Question Selector Service

**Fecha:** 2025-10-15  
**Objetivo:** Integrar Advanced Clustering y Continuous Learning con Question Selector

---

## 📋 Fase 1: Análisis del Estado Actual

### Componentes Existentes

1. **QuestionSelectorService** (`app/application/services/question_selector_service.py`)
   - ✅ Selección adaptativa de preguntas
   - ✅ Ajuste dinámico de dificultad
   - ⚠️ Embeddings manager deprecated
   - ⚠️ Selección básica (random o similarity)

2. **QuestionEmbeddingsService** (`app/infrastructure/ml/question_embeddings.py`)
   - ✅ SentenceTransformers (all-MiniLM-L6-v2)
   - ✅ RankNet para ranking
   - ✅ Caché de embeddings

3. **Nuevos Módulos ML** (Recién creados)
   - ✅ `AdvancedQuestionClusteringService` - UMAP + HDBSCAN
   - ✅ `ContinuousLearningSystem` - Multi-armed bandits

### Problemas Identificados

1. **Embeddings Manager Deprecated:**
   - El servicio actual referencia un módulo obsoleto
   - Necesita adaptador para nuevo QuestionEmbeddingsService

2. **Selección Básica:**
   - Solo usa random o similarity simple
   - No aprovecha clustering ni aprendizaje continuo

3. **Sin Feedback Loop:**
   - No actualiza modelo con resultados reales
   - No aprende de entrevistas pasadas

---

## 🎯 Fase 2: Objetivos de Integración

### Objetivo 1: Conectar Question Embeddings Service
- Reemplazar embeddings_manager deprecated
- Usar QuestionEmbeddingsService directamente
- Mantener compatibilidad con código existente

### Objetivo 2: Integrar Advanced Clustering
- Usar clusters para diversificación
- Seleccionar preguntas de diferentes tópicos
- Mejorar cobertura de conceptos

### Objetivo 3: Integrar Continuous Learning
- Tracking de performance por pregunta
- Multi-armed bandits para exploración/explotación
- Feedback loop automático

### Objetivo 4: Mantener Retrocompatibilidad
- No romper API existente
- Degradación gradual si ML falla
- Fallback a métodos tradicionales

---

## 🔧 Fase 3: Diseño de la Solución

### Arquitectura Propuesta

```
QuestionSelectorService (Enhanced)
├── QuestionEmbeddingsService (embeddings + RankNet)
├── AdvancedQuestionClusteringService (UMAP + HDBSCAN)
├── ContinuousLearningSystem (MAB + feedback)
└── QuestionRepository (datos)
```

### Flujo de Selección Mejorado

```
1. Determinar dificultad objetivo (ajuste dinámico)
   ↓
2. Obtener preguntas candidatas (filtro básico)
   ↓
3. Cluster candidatas por tópico (UMAP + HDBSCAN)
   ↓
4. Aplicar MAB para exploración/explotación
   ↓
5. Seleccionar pregunta final (diversificación + performance)
   ↓
6. Actualizar sistema de aprendizaje (feedback loop)
```

### Métodos a Modificar

1. **`__init__`**: Agregar nuevos servicios ML
2. **`select_next_question`**: Integrar clustering y MAB
3. **`_select_best_question`**: Usar clusters y performance tracking
4. **`_calculate_relevance_score`**: Incorporar cluster coherence
5. **Nuevo: `update_question_performance`**: Feedback loop

---

## 📝 Fase 4: Plan de Implementación

### Paso 4.1: Crear Enhanced Question Selector Service
**Archivo:** `question_selector_service_enhanced.py`
- Nueva versión con ML integrado
- Mantener interfaz compatible
- Agregar configuración ML

### Paso 4.2: Implementar Métodos de Clustering
**Métodos nuevos:**
- `_cluster_candidates()`: Agrupar preguntas por tópico
- `_select_from_clusters()`: Selección diversificada
- `_calculate_cluster_diversity()`: Métricas de diversidad

### Paso 4.3: Implementar Continuous Learning
**Métodos nuevos:**
- `update_question_performance()`: Actualizar stats
- `_apply_exploration_strategy()`: MAB selection
- `_get_question_performance()`: Obtener histórico

### Paso 4.4: Crear Tests Unitarios
**Archivo:** `test_question_selector_enhanced.py`
- Test selección básica
- Test clustering
- Test MAB strategies
- Test feedback loop

### Paso 4.5: Migración Gradual
1. Desplegar versión enhanced en paralelo
2. A/B testing entre versiones
3. Monitorear métricas
4. Migración completa

---

## 📊 Fase 5: Métricas de Éxito

### Métricas Técnicas
- ✅ Diversidad de preguntas: >0.7 (cosine distance)
- ✅ Cobertura de tópicos: >80% de clusters visitados
- ✅ Latencia de selección: <500ms
- ✅ Accuracy de MAB: >baseline después de 20 entrevistas

### Métricas de Negocio
- ✅ Satisfacción del candidato: >4/5
- ✅ Discriminación de habilidades: >0.6 (correlation)
- ✅ Tiempo de entrevista: 15-25 minutos
- ✅ Tasa de completación: >85%

---

## 🚧 Fase 6: Riesgos y Mitigaciones

### Riesgo 1: Performance Degradation
**Mitigación:**
- Caché agresivo de embeddings y clusters
- Lazy loading de modelos
- Timeout y fallback a selección simple

### Riesgo 2: Cold Start Problem
**Mitigación:**
- Usar datos sintéticos iniciales
- Epsilon-greedy con ε alto al inicio
- Reducir ε gradualmente con datos

### Riesgo 3: Overfitting a Patrones
**Mitigación:**
- Regularización en MAB
- Rotación forzada de clusters
- Monitoring de diversidad

### Riesgo 4: Complejidad Excesiva
**Mitigación:**
- Interfaz simple y clara
- Documentación exhaustiva
- Feature flags para deshabilitar ML

---

## ✅ Fase 7: Checklist de Implementación

- [ ] Crear `question_selector_service_enhanced.py`
- [ ] Implementar integración con QuestionEmbeddingsService
- [ ] Implementar métodos de clustering
- [ ] Implementar continuous learning
- [ ] Crear tests unitarios
- [ ] Crear tests de integración
- [ ] Documentar API y uso
- [ ] Agregar logging y monitoring
- [ ] Configurar feature flags
- [ ] Desplegar en staging
- [ ] A/B testing
- [ ] Desplegar en producción
- [ ] Monitorear métricas

---

## 📅 Timeline

| Fase | Duración | Status |
|------|----------|--------|
| Análisis | 1 hora | ✅ COMPLETADO |
| Diseño | 1 hora | 🔄 EN PROGRESO |
| Implementación | 3-4 horas | ⏳ PENDIENTE |
| Testing | 2 horas | ⏳ PENDIENTE |
| Documentación | 1 hora | ⏳ PENDIENTE |
| Despliegue | 1 hora | ⏳ PENDIENTE |

**Total estimado:** 8-10 horas de desarrollo

---

## 📚 Próximos Documentos

1. **INTEGRATION_GUIDE.md** - Guía detallada de integración
2. **API_REFERENCE.md** - Referencia completa de la API enhanced
3. **MIGRATION_GUIDE.md** - Cómo migrar de versión antigua a nueva
4. **PERFORMANCE_TUNING.md** - Optimización de parámetros ML

---

**Autor:** Jeronimo Restrepo Angel  
**Última actualización:** 2025-10-15
