# 🔧 Correcciones Requeridas para Flujo Conversacional

## 📋 Resumen Ejecutivo

Se identificaron **5 issues críticos** que impiden que el flujo conversacional funcione correctamente según los requisitos. Este documento detalla las correcciones necesarias.

---

## 🔴 CRÍTICO 1: Respetar Modo PRÁCTICA vs EXAMEN

### Problema
El código actual no respeta el modo de entrevista. Usa valores hardcodeados en lugar de usar `InterviewMode`.

### Correcciones Necesarias

**Archivo:** `app/main_v2_improved.py`

**Línea 897:** Cambiar de:
```python
MAX_ATTEMPTS = 3
```

A:
```python
MAX_ATTEMPTS = interview.mode.max_attempts_per_question()
```

**Línea 958:** Agregar verificación antes de generar hints:
```python
if not evaluation["is_correct"] and attempts_left > 0 and interview.mode.hints_enabled():
    # Generar hint...
```

**Línea 936:** Usar estilo de feedback según modo:
```python
feedback_style = interview.mode.feedback_style()
# Pasar a feedback_service.generate_feedback(..., feedback_style=feedback_style)
```

**Línea 745:** Validar tiempo límite en modo EXAM:
```python
if interview.mode.is_exam():
    time_limit = interview.mode.time_limit_seconds()
    if answer_request.time_taken and answer_request.time_taken > time_limit:
        return ProcessAnswerResponse(
            ...,
            feedback="⏱️ Tiempo agotado. En modo examen cada pregunta tiene límite de tiempo.",
            ...
        )
```

---

## 🔴 CRÍTICO 2: Feedback Motivacional

### Problema
No se genera feedback motivacional cuando la respuesta es incorrecta en modo PRÁCTICA.

### Correcciones Necesarias

**Archivo:** `app/main_v2_improved.py`

**Después de línea 943:** Agregar generación de feedback motivacional:
```python
# Generar feedback motivacional si respuesta incorrecta (solo modo PRÁCTICA)
motivational_feedback = ""
if not evaluation["is_correct"] and interview.mode.is_practice():
    try:
        from app.infrastructure.llm.advanced_prompts import get_prompt_engine
        prompt_engine = get_prompt_engine()
        motivational_prompt = prompt_engine.get_motivational_feedback_prompt(
            role=interview.role,
            question=interview.current_question.text,
            answer=answer_request.answer,
            evaluation=evaluation,
            attempt=current_attempt,
        )
        motivational_response = c.evaluation_service.llm_service.generate(
            prompt=motivational_prompt,
            temperature=0.8,
            max_tokens=200,
        )
        from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
        sanitizer = ResponseSanitizer()
        motivational_feedback = sanitizer.sanitize_feedback(
            motivational_response, 
            role=interview.role, 
            category=interview.current_question.category
        )
    except Exception as e:
        logger.warning(f"Error generando feedback motivacional: {e}")
        # Fallback
        motivational_feedback = "💪 No te desanimes. Cada intento es una oportunidad de aprender. ¡Sigue adelante!"

# Agregar al feedback principal
if motivational_feedback:
    feedback_result = f"{feedback_result}\n\n{motivational_feedback}"
```

---

## 🔴 CRÍTICO 3: Respuesta Correcta Después de 3 Intentos

### Problema
Después de 3 intentos fallidos, no se proporciona la respuesta correcta con consejos de mejora.

### Correcciones Necesarias

**Archivo:** `app/main_v2_improved.py`

**Después de línea 1017:** Agregar lógica cuando se agotan los intentos:
```python
# Si se agotaron los intentos y respuesta incorrecta, dar respuesta correcta
if not evaluation["is_correct"] and current_attempt >= MAX_ATTEMPTS:
    try:
        from app.infrastructure.llm.advanced_prompts import get_prompt_engine
        prompt_engine = get_prompt_engine()
        correct_answer_prompt = prompt_engine.get_correct_answer_prompt(
            role=interview.role,
            question=interview.current_question.text,
            expected_concepts=interview.current_question.expected_concepts,
        )
        correct_answer_response = c.evaluation_service.llm_service.generate(
            prompt=correct_answer_prompt,
            temperature=0.6,
            max_tokens=300,
        )
        from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
        sanitizer = ResponseSanitizer()
        correct_answer = sanitizer.sanitize_feedback(
            correct_answer_response,
            role=interview.role,
            category=interview.current_question.category
        )
        
        # Generar consejos de mejora
        improvement_prompt = prompt_engine.get_improvement_tips_prompt(
            role=interview.role,
            question=interview.current_question.text,
            answer=answer_request.answer,
            correct_answer=correct_answer,
        )
        improvement_response = c.evaluation_service.llm_service.generate(
            prompt=improvement_prompt,
            temperature=0.7,
            max_tokens=200,
        )
        improvement_tips = sanitizer.sanitize_feedback(
            improvement_response,
            role=interview.role,
            category=interview.current_question.category
        )
        
        feedback_result = f"""{feedback_result}

📚 Respuesta Correcta Esperada:
{correct_answer}

💡 Consejos de Mejora:
{improvement_tips}
"""
    except Exception as e:
        logger.warning(f"Error generando respuesta correcta: {e}")
        # Fallback simple
        feedback_result = f"""{feedback_result}

📚 Conceptos clave que debías mencionar: {', '.join(interview.current_question.expected_concepts)}
💡 Recomendación: Estudia estos conceptos y cómo se relacionan para fortalecer tu comprensión.
"""
```

---

## 🔴 CRÍTICO 4: Feedback Final al Completar

### Problema
No se genera feedback final cuando se completa la entrevista.

### Correcciones Necesarias

**Archivo:** `app/main_v2_improved.py`

**Línea 1099:** Antes de retornar `ProcessAnswerResponse`, agregar:
```python
# Generar feedback final completo
try:
    all_answers = [
        {
            "question": q.text,
            "answer": a.text,
            "score": a.score.value,
            "is_correct": a.is_correct,
            "evaluation_details": a.evaluation_details,
        }
        for q, a in zip(interview.questions_history, interview.answers_history)
    ]
    
    overall_score = sum([a.score.value for a in interview.answers]) / len(interview.answers) if interview.answers else 0
    accuracy = sum([1 for a in interview.answers if a.is_correct]) / len(interview.answers) * 100 if interview.answers else 0
    
    final_feedback = c.feedback_service.generate_final_feedback(
        role=interview.role,
        category=interview.interview_type,
        all_answers=all_answers,
        overall_score=overall_score,
        accuracy=accuracy,
    )
except Exception as e:
    logger.error(f"Error generando feedback final: {e}")
    final_feedback = f"¡Felicidades por completar la entrevista! Tu score promedio fue {overall_score:.2f}/10."

# Agregar feedback final a la respuesta
feedback_result = f"{feedback_result}\n\n🎉 FEEDBACK FINAL:\n{final_feedback}"
```

---

## 🔴 CRÍTICO 5: Generar Reportes con Gráficos

### Problema
No se generan reportes con gráficos al completar la entrevista.

### Correcciones Necesarias

**Archivo:** `app/main_v2_improved.py`

**Después de línea 1097:** Agregar generación de reporte:
```python
# Generar reporte completo con gráficos
try:
    from app.infrastructure.llm.report_generator import get_report_generator
    
    report_generator = get_report_generator()
    
    interview_data = {
        "id": interview.id,
        "role": interview.role,
        "mode": interview.mode.to_string(),
        "completed_at": interview.completed_at.isoformat() if interview.completed_at else datetime.now(timezone.utc).isoformat(),
        "answers_history": [a.to_dict() for a in interview.answers],
        "questions_history": [q.to_dict() for q in interview.questions_history],
    }
    
    user_data = {
        "name": f"User_{interview.user_id}",
        "user_id": interview.user_id,
    }
    
    report = report_generator.generate_report(interview_data, user_data)
    
    # Exportar a JSON
    report_json = report_generator.export_to_json(report)
    
    # Guardar reporte en metadata de entrevista
    interview.metadata["report"] = report_json
    interview.metadata["report_id"] = report.interview_id
    interview.metadata["certificate_eligible"] = report.certificate_eligible
    interview.metadata["certificate_id"] = report.certificate_id
    
    await c.interview_repository.save(interview)
    
    logger.info(f"📊 Reporte generado para entrevista {interview.id}")
    
except Exception as e:
    logger.error(f"Error generando reporte: {e}")
    # Continuar sin reporte (no crítico)
```

**En la respuesta `ProcessAnswerResponse`:** Agregar campo de reporte:
```python
return ProcessAnswerResponse(
    ...,
    interview_completed=True,
    final_report=interview.metadata.get("report"),  # JSON del reporte
    report_url=report.shareable_url if 'report' in locals() else None,
    certificate_eligible=interview.metadata.get("certificate_eligible", False),
    certificate_id=interview.metadata.get("certificate_id"),
)
```

---

## 🟡 IMPORTANTE: Mejoras en Fallbacks

### Agregar Fallback para Evaluación

**Archivo:** `app/main_v2_improved.py`

**Línea 912:** Agregar try-except:
```python
try:
    evaluation = c.evaluation_service.evaluate_answer(...)
except Exception as e:
    logger.error(f"Error en evaluación, usando fallback: {e}")
    # Fallback: evaluación simple basada en keywords
    evaluation = {
        "score": 5.0,  # Score neutro
        "is_correct": False,
        "justification": "No se pudo evaluar la respuesta completamente. Revisa los conceptos clave mencionados.",
        "breakdown": {},
        "strengths": [],
        "improvements": ["Asegúrate de mencionar conceptos clave relacionados con la pregunta"],
        "concepts_covered": [],
        "missing_concepts": interview.current_question.expected_concepts,
    }
```

### Agregar Logging de Fallbacks

**Crear función helper:**
```python
def log_fallback(fallback_name: str, reason: str, context: dict = None):
    """Log cuando se usa un fallback"""
    logger.warning(f"🔄 FALLBACK ACTIVADO: {fallback_name} - Razón: {reason}")
    if context:
        logger.debug(f"Contexto: {context}")
```

---

## 📊 Priorización

1. **CRÍTICO 1** - Respetar Modo PRÁCTICA vs EXAMEN (IMPACTO ALTO)
2. **CRÍTICO 2** - Feedback Motivacional (IMPACTO ALTO)
3. **CRÍTICO 3** - Respuesta Correcta Después de 3 Intentos (IMPACTO MEDIO)
4. **CRÍTICO 4** - Feedback Final (IMPACTO MEDIO)
5. **CRÍTICO 5** - Reportes con Gráficos (IMPACTO MEDIO)
6. **IMPORTANTE** - Mejoras en Fallbacks (IMPACTO BAJO)

---

**Fecha:** 2025-11-03  
**Prioridad:** ALTA  
**Tiempo estimado:** 4-6 horas

