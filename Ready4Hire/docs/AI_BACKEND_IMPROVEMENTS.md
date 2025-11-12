# Mejoras Sugeridas para el Backend de IA

Basado en el análisis de logs y código, aquí están las mejoras prioritarias:

## 🔴 Problemas Críticos Detectados

### 1. **Timeout Configuración Inconsistente**
**Problema:** 
- `config.py` tiene `OLLAMA_TIMEOUT: int = 30`
- `ollama_client.py` usa `timeout: int = 45` por defecto
- El cliente no está usando la configuración de `Settings`

**Impacto:** Los logs muestran timeouts a 30s cuando deberían ser 45s

**Solución:**
```python
# En container.py, al inicializar OllamaClient:
from app.config import settings

self.llm_client = OllamaClient(
    timeout=settings.OLLAMA_TIMEOUT,  # Usar configuración
    max_retries=settings.OLLAMA_MAX_RETRIES,
    ...
)
```

### 2. **Mapeo de Roles Incompleto**
**Problema:** 
- "Software Developer" no encuentra template (busca "software_developer" pero el template es "software_engineer")
- Varios nombres de roles no mapean correctamente

**Impacto:** Se usa template genérico en vez de específico

**Solución:**
```python
# En advanced_prompts.py, mejorar _get_template_for_role:

ROLE_MAPPING = {
    "software developer": "software_engineer",
    "software engineer": "software_engineer",
    "developer": "software_engineer",
    "programmer": "software_engineer",
    "frontend developer": "frontend_developer",
    "backend developer": "backend_developer",
    # ... más mapeos
}

def _get_template_for_role(self, role: str) -> PromptTemplate:
    role_lower = role.lower().strip()
    
    # Primero intentar mapeo directo
    if role_lower in ROLE_MAPPING:
        mapped_key = ROLE_MAPPING[role_lower]
        if mapped_key in self.templates:
            return self.templates[mapped_key]
    
    # Luego normalizar
    role_normalized = role_lower.replace(" ", "_").replace("-", "_")
    ...
```

### 3. **Error de Importación `get_sync_session`**
**Problema:** 
- `main_v2_improved.py` intenta importar `get_sync_session` que no existe
- Línea 181: `cannot import name 'get_sync_session'`

**Solución:**
```python
# Verificar qué existe en postgres_sync_service.py
# O usar el método correcto de conexión
```

## 🟡 Mejoras de Rendimiento

### 4. **Circuit Breaker Más Resiliente**
**Mejora:** 
- Agregar health check periódico
- Mejor logging cuando el circuito está abierto
- Recovery automático más inteligente

```python
# En ollama_client.py
def _check_health_periodic(self):
    """Verifica salud cada 30s cuando el circuito está abierto"""
    if self.circuit_breaker and self.circuit_breaker.state == CircuitState.OPEN:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("🟢 Ollama recuperado, intentando cerrar circuito")
                # Forzar cierre del circuito
```

### 5. **Cache de Evaluaciones Mejorado**
**Mejora:**
- Cachear también hints generados
- Cachear sanitización de respuestas
- Invalidación inteligente por cambios en prompts

### 6. **Batch Processing para Múltiples Evaluaciones**
**Mejora:**
- Si hay múltiples respuestas pendientes, procesarlas en batch
- Reducir overhead de conexiones

```python
async def batch_evaluate_answers(self, evaluations: List[Dict]) -> List[Dict]:
    """Procesa múltiples evaluaciones en paralelo"""
    tasks = [self.evaluate_answer(**eval_data) for eval_data in evaluations]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

## 🟢 Mejoras de Calidad

### 7. **Mejor Manejo de Errores de Conexión**
**Mejora:**
- Detectar cuando Ollama se desconecta
- Reintentar con backoff exponencial más agresivo
- Fallback a evaluación heurística más rápido

```python
# En ollama_client.py
def generate(self, ...):
    try:
        return self._generate_internal(...)
    except OllamaConnectionError as e:
        # Si es error de conexión, intentar health check
        if self._check_health():
            # Si Ollama está disponible, reintentar
            return self._generate_internal(...)
        else:
            # Ollama está caído, fallback inmediato
            raise OllamaUnavailableError("Ollama service is down")
```

### 8. **Validación de Respuestas LLM Mejorada**
**Mejora:**
- Detectar cuando el LLM no responde JSON válido
- Retry automático con prompt más estricto
- Validación de campos requeridos

```python
def _parse_evaluation_response(self, response: str, retry_on_fail: bool = True) -> Dict:
    """Parsea con retry automático si falla"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        if retry_on_fail:
            # Retry con prompt más estricto
            logger.warning("JSON inválido, reintentando con prompt más estricto")
            return self._retry_with_strict_json_prompt(...)
        raise
```

### 9. **Métricas y Observabilidad**
**Mejora:**
- Tracking de latencia por tipo de evaluación
- Contador de fallbacks heurísticos
- Métricas de éxito/fallo por rol

```python
class EvaluationMetrics:
    def __init__(self):
        self.total_evaluations = 0
        self.llm_success = 0
        self.fallback_count = 0
        self.latency_by_role = defaultdict(list)
    
    def record_evaluation(self, role: str, success: bool, latency: float, used_fallback: bool):
        self.total_evaluations += 1
        if success:
            self.llm_success += 1
        if used_fallback:
            self.fallback_count += 1
        self.latency_by_role[role].append(latency)
```

### 10. **Prompts Más Eficientes**
**Mejora:**
- Reducir tokens innecesarios en prompts
- Usar system prompts más cortos
- Optimizar estructura JSON

### 11. **Sistema de Reintentos Inteligente**
**Mejora:**
- Detectar tipo de error (timeout vs conexión vs parse)
- Ajustar estrategia según el error
- Timeout adaptativo según complejidad de pregunta

## 📊 Priorización

**Alta Prioridad (Crítico):**
1. ✅ Fix timeout configuration
2. ✅ Fix role mapping
3. ✅ Fix get_sync_session import

**Media Prioridad (Importante):**
4. ✅ Circuit breaker mejorado
5. ✅ Mejor manejo de errores de conexión
6. ✅ Validación de respuestas mejorada

**Baja Prioridad (Mejora continua):**
7. ✅ Batch processing
8. ✅ Métricas avanzadas
9. ✅ Prompts más eficientes

## 🚀 Implementación Sugerida

1. **Fase 1 (Urgente - 1 día):**
   - Fix configuración timeout
   - Fix mapeo de roles
   - Fix importación

2. **Fase 2 (Importante - 3 días):**
   - Mejorar circuit breaker
   - Mejorar manejo de errores
   - Validación mejorada

3. **Fase 3 (Mejora continua - 1 semana):**
   - Batch processing
   - Métricas avanzadas
   - Optimizaciones de prompts

