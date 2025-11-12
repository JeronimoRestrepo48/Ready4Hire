"""
Servicio de evaluación de respuestas usando Ollama (LLM local).
Evalúa respuestas del candidato y asigna puntuación detallada.

Mejoras v2.1:
- Caché de 2 niveles (memoria + disco) para 95% reducción de latencia
- Model warm-up para eliminar cold start
- Explicaciones detalladas mejoradas

Mejoras v2.2:
- Recopilación automática de datos para fine-tuning
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import re
import logging

from app.infrastructure.llm.llm_service import OllamaLLMService
from app.infrastructure.cache.evaluation_cache import EvaluationCache
from app.infrastructure.ml.training_data_collector import TrainingDataCollector
from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
from app.infrastructure.monitoring.metrics import get_metrics
from app.domain.value_objects.score import Score

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Servicio para evaluar respuestas de entrevistas usando Ollama local.
    Elimina dependencias de APIs externas (OpenAI, Anthropic).

    Mejoras v2.1:
    - Caché inteligente de 2 niveles (95% reducción de latencia)
    - Model warm-up automático
    - Explicaciones detalladas mejoradas

    Mejoras v2.2:
    - Recopilación automática de datos para fine-tuning
    """

    def __init__(
        self,
        llm_service: Optional[OllamaLLMService] = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.3,
        enable_cache: bool = True,
        cache_ttl_days: int = 7,
        collect_training_data: bool = False,
        training_collector: Optional[TrainingDataCollector] = None,
        use_advanced_prompts: bool = True,
    ):
        """
        Inicializa el servicio de evaluación.

        Args:
            llm_service: Servicio LLM Ollama (se crea uno si no se provee)
            model: Modelo Ollama a usar
            temperature: Temperatura para evaluación (más baja = más consistente)
            enable_cache: Activar caché de evaluaciones (default: True)
            cache_ttl_days: Tiempo de vida del caché en días (default: 7)
            collect_training_data: Recopilar datos para fine-tuning (default: False) (v2.2)
            training_collector: Collector personalizado (se crea uno si collect_training_data=True) (v2.2)
        """
        self.llm_service = llm_service or OllamaLLMService(model=model, temperature=temperature, max_tokens=1024)
        self.model = model
        self.temperature = temperature

        # Caché de evaluaciones (Mejora v2.1)
        self.enable_cache = enable_cache
        self.cache = EvaluationCache(ttl_days=cache_ttl_days) if enable_cache else None

        # Training data collection (Mejora v2.2)
        self.collect_training_data = collect_training_data
        if collect_training_data and training_collector is None:
            self.training_collector = TrainingDataCollector()
            logger.info("TrainingDataCollector inicializado para recopilación automática")
        else:
            self.training_collector = training_collector

        # Advanced prompts (Mejora v3.2)
        self.use_advanced_prompts = use_advanced_prompts
        if use_advanced_prompts:
            try:
                from app.infrastructure.llm.advanced_prompts import get_prompt_engine

                self.prompt_engine = get_prompt_engine()
                logger.info("✅ Advanced prompt engine initialized")
            except Exception as e:
                logger.warning(f"Could not load advanced prompts: {e}, using fallback")
                self.prompt_engine = None
        else:
            self.prompt_engine = None

        # Response sanitizer (Mejora v3.3)
        self.sanitizer = ResponseSanitizer()

        # Métricas avanzadas (Mejora v3.4)
        try:
            self.metrics = get_metrics(enabled=True)
        except Exception as e:
            logger.warning(f"⚠️ Métricas no disponibles: {e}")
            self.metrics = None

        # Model warm-up (Mejora v2.1)
        self._warmup_model()

        logger.info(
            f"EvaluationService inicializado (model={model}, cache={enable_cache}, "
            f"training_collection={collect_training_data}, advanced_prompts={use_advanced_prompts})"
        )

    def _warmup_model(self):
        """
        Pre-carga el modelo para eliminar cold start (~30s → 5s).
        Se ejecuta al inicializar el servicio.
        """
        try:
            logger.info("Iniciando warm-up del modelo...")
            start_time = datetime.now()

            # Generar respuesta corta para cargar modelo en memoria
            self.llm_service.generate(prompt="Hello, ready for interviews!", max_tokens=10)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Warm-up completado en {elapsed:.2f}s")

        except Exception as e:
            logger.warning(f"Warm-up falló (no crítico): {e}")

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        keywords: List[str],
        category: str,
        difficulty: str,
        role: str,
    ) -> Dict[str, Any]:
        """
        Evalúa una respuesta del candidato usando Ollama.

        Mejoras v2.1:
        - Busca en caché antes de llamar al LLM (95% más rápido)
        - Guarda resultado en caché para futuras consultas

        Args:
            question: Texto de la pregunta
            answer: Respuesta del candidato
            expected_concepts: Conceptos esperados en la respuesta
            keywords: Palabras clave relevantes
            category: Categoría (soft_skills, technical)
            difficulty: Dificultad (junior, mid, senior)
            role: Rol/posición

        Returns:
            Dict con score (0-10), puntuación detallada y justificación
        """
        # MEJORA v2.1: Intentar obtener del caché primero
        if self.enable_cache and self.cache:
            cached_result = self.cache.get(
                question=question,
                answer=answer,
                model=self.model,
                temperature=self.temperature,
                expected_concepts=expected_concepts,
                keywords=keywords,
            )
            if cached_result:
                cached_result["from_cache"] = True
                logger.debug("Evaluación obtenida del caché (latencia <10ms)")
                
                # MEJORA v3.4: Registrar métricas de cache hit
                if self.metrics:
                    self.metrics.record_evaluation(
                        success=True,
                        latency_ms=5.0,  # Cache hit es muy rápido
                        cached=True,
                        score=cached_result.get("score", 0.0),
                    )
                    self.metrics.observe_histogram(f"evaluation_cache_hit_by_role_{role.lower().replace(' ', '_')}", 5.0)
                
                return cached_result

        # No está en caché, evaluar con LLM
        try:
            start_time = datetime.now()

            # Use advanced prompts if available (v3.2)
            if self.use_advanced_prompts and self.prompt_engine:
                interview_mode = "practice"  # TODO: Get from context
                prompt = self.prompt_engine.get_evaluation_prompt(
                    role=role,
                    question=question,
                    answer=answer,
                    expected_concepts=expected_concepts,
                    difficulty=difficulty,
                    interview_mode=interview_mode,
                )
                logger.debug("Using advanced profession-specific prompt")
            else:
                prompt = self._build_evaluation_prompt(
                    question=question,
                    answer=answer,
                    expected_concepts=expected_concepts,
                    keywords=keywords,
                    category=category,
                    difficulty=difficulty,
                    role=role,
                )

            # ⚡ Generar evaluación con Ollama OPTIMIZADO para velocidad
            response = self.llm_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=512,  # ⚡ Reducido de 1024 a 512 para respuestas más rápidas
            )

            # Parsear respuesta JSON con retry automático si falla
            try:
                result = self._parse_evaluation_response(response, retry_on_fail=True)
            except ValueError as e:
                if "retry disponible" in str(e):
                    # NUEVO: Retry con prompt más estricto
                    logger.warning(f"⚠️ JSON inválido, reintentando con prompt más estricto...")
                    strict_prompt = self._build_strict_json_prompt(
                        question=question,
                        answer=answer,
                        expected_concepts=expected_concepts,
                        keywords=keywords,
                        category=category,
                        difficulty=difficulty,
                        role=role,
                    )
                    retry_response = self.llm_service.generate(
                        prompt=strict_prompt,
                        temperature=self.temperature * 0.5,  # Temperatura más baja para más consistencia
                        max_tokens=512,
                    )
                    result = self._parse_evaluation_response(retry_response, retry_on_fail=False)
                    logger.info("✅ Retry exitoso con prompt estricto")
                    # MEJORA v3.4: Registrar métricas de retry
                    if self.metrics:
                        self.metrics.inc_counter("evaluation_retries_total")
                else:
                    raise

            # Validar estructura
            validated_result = self._validate_evaluation_result(result)
            
            # MEJORA v3.3: Sanitizar respuesta para que parezca de agente especializado
            validated_result = self.sanitizer.sanitize_evaluation_response(validated_result)
            validated_result["role"] = role
            validated_result["category"] = category

            # MEJORA v2.1: Guardar en caché para futuras consultas
            elapsed = (datetime.now() - start_time).total_seconds()
            elapsed_ms = elapsed * 1000
            validated_result["evaluation_time_seconds"] = round(elapsed, 2)
            validated_result["from_cache"] = False

            # MEJORA v3.4: Registrar métricas avanzadas
            if self.metrics:
                self.metrics.record_evaluation(
                    success=True,
                    latency_ms=elapsed_ms,
                    cached=False,
                )
                # Métricas por rol
                self.metrics.observe_histogram(f"evaluation_duration_by_role_{role.lower().replace(' ', '_')}", elapsed_ms)
                # Métricas por categoría
                self.metrics.observe_histogram(f"evaluation_duration_by_category_{category}", elapsed_ms)
                # Métricas por score
                self.metrics.observe_histogram("evaluation_score_distribution", validated_result["score"])
                # Registrar con score completo
                self.metrics.record_evaluation(
                    success=True,
                    latency_ms=elapsed_ms,
                    cached=False,
                    score=validated_result["score"],
                )

            if self.enable_cache and self.cache:
                self.cache.set(
                    question=question,
                    answer=answer,
                    model=self.model,
                    temperature=self.temperature,
                    expected_concepts=expected_concepts,
                    keywords=keywords,
                    result=validated_result,
                )
                logger.debug(f"Evaluación LLM completada en {elapsed:.2f}s y guardada en caché")

            # MEJORA v2.2: Recopilar datos para fine-tuning (solo evaluaciones LLM exitosas)
            if self.collect_training_data and self.training_collector:
                try:
                    self.training_collector.collect(
                        question=question,
                        answer=answer,
                        evaluation_result=validated_result,
                        expected_concepts=expected_concepts,
                        keywords=keywords,
                        category=category,
                        difficulty=difficulty,
                        role=role,
                    )
                    logger.debug("Datos de entrenamiento recopilados exitosamente")
                except Exception as e:
                    logger.warning(f"Error al recopilar datos de entrenamiento: {e}")

            return validated_result

        except Exception as e:
            error_type = type(e).__name__
            if "Timeout" in error_type or "timeout" in str(e).lower():
                logger.warning(f"⚠️ Timeout en evaluación LLM después de 45s, usando fallback heurístico")
            else:
                logger.error(f"Error en evaluación LLM ({error_type}): {str(e)}, usando fallback heurístico")
            
            # Fallback a evaluación heurística
            elapsed = (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0
            elapsed_ms = elapsed * 1000
            
            # MEJORA v3.4: Registrar métricas de fallback
            if self.metrics:
                self.metrics.record_evaluation(
                    success=False,
                    latency_ms=elapsed_ms,
                    cached=False,
                )
                self.metrics.inc_counter("evaluation_fallbacks_total")
                self.metrics.observe_histogram(f"evaluation_fallback_duration_by_role_{role.lower().replace(' ', '_')}", elapsed_ms)
            
            fallback_result = self._heuristic_evaluation(answer, expected_concepts, keywords)
            fallback_result["from_cache"] = False
            fallback_result["fallback"] = True
            # Sanitizar también el fallback
            fallback_result = self.sanitizer.sanitize_evaluation_response(fallback_result)
            fallback_result["role"] = role
            fallback_result["category"] = category
            return fallback_result

    def evaluate_answer_sync(
        self,
        question_text: str,
        answer_text: str,
        expected_concepts: Optional[List[str]] = None,
        category: str = "technical",
        difficulty: str = "mid",
        role: str = "Software Engineer",
        keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Wrapper síncrono mantenido por compatibilidad con tareas async (Celery).

        Args:
            question_text: Texto de la pregunta.
            answer_text: Respuesta proporcionada por el candidato.
            expected_concepts: Lista de conceptos esperados en la respuesta.
            category: Categoría de la pregunta (technical, soft_skills, etc.).
            difficulty: Nivel de dificultad (junior, mid, senior).
            role: Rol objetivo de la entrevista.
            keywords: Palabras clave relevantes para la evaluación.

        Returns:
            Resultado de evaluación con score y desglose.
        """
        return self.evaluate_answer(
            question=question_text,
            answer=answer_text,
            expected_concepts=expected_concepts or [],
            keywords=keywords or [],
            category=category,
            difficulty=difficulty,
            role=role,
        )

    def _build_evaluation_prompt(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        keywords: List[str],
        category: str,
        difficulty: str,
        role: str,
    ) -> str:
        """⚡ Construye el prompt de evaluación OPTIMIZADO y MULTIPROFESIÓN."""

        # Contexto específico por categoría
        context_type = "técnica" if category == "technical" else "de habilidades blandas"

        # Criterios adaptativos según la categoría
        criteria_guide = ""
        if category == "technical":
            criteria_guide = """
CRITERIOS TÉCNICOS:
- Completitud: Cubre todos los aspectos de la pregunta
- Profundidad: Demuestra comprensión técnica profunda
- Claridad: Explica de forma estructurada y comprensible
- Conceptos: Usa terminología técnica correcta"""
        else:
            criteria_guide = """
CRITERIOS SOFT SKILLS:
- Completitud: Proporciona ejemplo concreto y relevante
- Profundidad: Muestra reflexión y aprendizaje
- Claridad: Estructura clara (situación-acción-resultado)
- Conceptos: Demuestra competencia comportamental"""

        return f"""Evalúa respuesta de entrevista para {role} ({difficulty}).

P: {question}
R: {answer}
Conceptos: {', '.join(expected_concepts[:5]) if expected_concepts else 'Variados'}

{criteria_guide}

Evalúa (0-10):
- Completitud (0-3): ¿Responde todo?
- Profundidad (0-3): ¿Comprensión profunda?
- Claridad (0-2): ¿Bien explicado?
- Conceptos (0-2): ¿Usa términos clave?

JSON (sin texto extra):
{{
  "score": <0-10>,
  "breakdown": {{"completeness": <0-3>, "technical_depth": <0-3>, "clarity": <0-2>, "key_concepts": <0-2>}},
  "justification": "<2 oraciones>",
  "strengths": ["<1>", "<2>"],
  "improvements": ["<1>", "<2>"],
  "concepts_covered": ["<1>", "<2>"],
  "missing_concepts": ["<1>", "<2>"]
}}"""
    
    def _build_strict_json_prompt(
        self,
        question: str,
        answer: str,
        expected_concepts: List[str],
        keywords: List[str],
        category: str,
        difficulty: str,
        role: str,
    ) -> str:
        """
        Construye un prompt más estricto que fuerza JSON válido.
        Usado cuando el primer intento falla.
        """
        context_type = "técnica" if category == "technical" else "de habilidades blandas"
        
        return f"""Evalúa para {role}.

P: {question}
R: {answer}
Conceptos: {', '.join(expected_concepts[:3]) if expected_concepts else 'Variados'}

RESPONDE SOLO JSON (sin texto):
{{
  "score": <0-10>,
  "breakdown": {{"completeness": <0-3>, "technical_depth": <0-3>, "clarity": <0-2>, "key_concepts": <0-2>}},
  "justification": "<2 oraciones>",
  "strengths": ["<1>", "<2>"],
  "improvements": ["<1>", "<2>"],
  "concepts_covered": ["<1>"],
  "missing_concepts": ["<1>"]
}}"""

    def _parse_evaluation_response(self, response: str, retry_on_fail: bool = True) -> Dict[str, Any]:
        """
        Parsea la respuesta del LLM extrayendo el JSON.
        Maneja casos donde el LLM añade texto extra.
        
        NUEVO: Soporte para retry automático con prompt más estricto.
        
        Args:
            response: Respuesta del LLM
            retry_on_fail: Si True, intenta retry con prompt más estricto si falla
            
        Returns:
            Dict con la evaluación parseada
            
        Raises:
            ValueError: Si no se puede parsear y no hay retry disponible
        """
        # Intentar parsear directamente
        try:
            parsed = json.loads(response)
            # Validar que tenga campos mínimos requeridos
            if "score" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Intentar extraer JSON con regex (más robusto)
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if "score" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # NUEVO: Intentar múltiples patrones de extracción
        patterns = [
            r'\{[^{}]*"score"[^{}]*\}',  # Buscar JSON con "score"
            r'\{.*?"score".*?\}',  # Patrón más flexible
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if "score" in parsed:
                        logger.warning(f"⚠️ JSON parseado con patrón alternativo para respuesta inválida")
                        return parsed
                except json.JSONDecodeError:
                    continue

        # Si falla todo y no hay retry, lanzar excepción
        if not retry_on_fail:
            raise ValueError("No se pudo parsear respuesta JSON del LLM")
        
        # Si retry está habilitado, la excepción se manejará en el caller
        raise ValueError("No se pudo parsear respuesta JSON del LLM - retry disponible")

    def _validate_evaluation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y normaliza el resultado de evaluación.

        Mejoras v2.1:
        - Agrega missing_concepts para feedback más claro
        - Incluye confidence score del breakdown
        """
        # Asegurar que score esté entre 0 y 10
        score = float(result.get("score", 5.0))
        score = max(0.0, min(10.0, score))

        # Validar breakdown
        breakdown = result.get("breakdown", {})
        breakdown = {
            "completeness": max(0, min(3, float(breakdown.get("completeness", 1.5)))),
            "technical_depth": max(0, min(3, float(breakdown.get("technical_depth", 1.5)))),
            "clarity": max(0, min(2, float(breakdown.get("clarity", 1.0)))),
            "key_concepts": max(0, min(2, float(breakdown.get("key_concepts", 1.0)))),
        }

        # Calcular confidence score del breakdown (qué tan bien distribuido está)
        breakdown_total = sum(breakdown.values())
        breakdown_confidence = round((breakdown_total / 10) * 100, 1)  # Porcentaje

        # Asegurar listas
        strengths = result.get("strengths", [])
        if not isinstance(strengths, list):
            strengths = [str(strengths)]

        improvements = result.get("improvements", [])
        if not isinstance(improvements, list):
            improvements = [str(improvements)]

        concepts_covered = result.get("concepts_covered", [])
        if not isinstance(concepts_covered, list):
            concepts_covered = [str(concepts_covered)]

        # MEJORA v2.1: Agregar missing_concepts
        missing_concepts = result.get("missing_concepts", [])
        if not isinstance(missing_concepts, list):
            missing_concepts = []

        return {
            "score": round(score, 1),
            "is_correct": score >= 6.0,  # ⚡ Agregar campo is_correct basado en score
            "breakdown": breakdown,
            "breakdown_confidence": breakdown_confidence,
            "justification": result.get("justification", "Evaluación completada"),
            "strengths": strengths[:3],  # Máximo 3
            "improvements": improvements[:3],  # Máximo 3
            "concepts_covered": concepts_covered[:5],  # Máximo 5
            "missing_concepts": missing_concepts[:3],  # Máximo 3 (NUEVO v2.1)
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
        }

    def _heuristic_evaluation(self, answer: str, expected_concepts: List[str], keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluación heurística de respaldo cuando el LLM falla.
        Análisis basado en longitud, palabras clave y conceptos.
        """
        answer_lower = answer.lower()
        answer_words = answer.split()
        answer_length = len(answer_words)

        # 1. Puntuación por longitud
        if answer_length < 10:
            base_score = 2.0
        elif answer_length < 30:
            base_score = 5.0
        elif answer_length < 80:
            base_score = 7.0
        else:
            base_score = 8.0

        # 2. Bonus por conceptos esperados
        concepts_found = 0
        for concept in expected_concepts:
            if concept.lower() in answer_lower:
                concepts_found += 1

        concept_bonus = (concepts_found / max(len(expected_concepts), 1)) * 2.0

        # 3. Bonus por keywords
        keywords_found = 0
        for keyword in keywords:
            if keyword.lower() in answer_lower:
                keywords_found += 1

        keyword_bonus = (keywords_found / max(len(keywords), 1)) * 1.0

        # Score final
        final_score = min(10.0, base_score + concept_bonus + keyword_bonus)

        return {
            "score": round(final_score, 1),
            "is_correct": final_score >= 6.0,  # ⚡ Agregar campo is_correct basado en score
            "breakdown": {
                "completeness": round(final_score * 0.3, 1),
                "technical_depth": round(final_score * 0.3, 1),
                "clarity": round(final_score * 0.2, 1),
                "key_concepts": round(final_score * 0.2, 1),
            },
            "justification": f"Evaluación heurística: {concepts_found}/{len(expected_concepts)} conceptos encontrados, {keywords_found}/{len(keywords)} keywords detectadas.",
            "strengths": ["Respuesta proporcionada"],
            "improvements": ["Considerar agregar más detalles técnicos", "Mencionar conceptos clave"],
            "concepts_covered": [c for c in expected_concepts if c.lower() in answer_lower],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "fallback": True,
            "model": "heuristic",
        }

    def batch_evaluate(self, evaluations: List[Dict[str, Any]], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Evalúa múltiples respuestas en batch con procesamiento paralelo.
        
        NUEVO: Procesamiento asíncrono optimizado para mejor rendimiento.

        Args:
            evaluations: Lista de dicts con question, answer, expected_concepts, etc.
            max_concurrent: Número máximo de evaluaciones concurrentes (default: 3)

        Returns:
            Lista de resultados de evaluación en el mismo orden que las entradas
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(evaluations)  # Pre-allocar para mantener orden
        
        # Usar ThreadPoolExecutor para procesamiento paralelo
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Enviar todas las tareas
            future_to_index = {
                executor.submit(self._evaluate_single, eval_data): i
                for i, eval_data in enumerate(evaluations)
            }
            
            # Procesar resultados conforme completan
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error en batch evaluation [{index}]: {str(e)}")
                    # Fallback heurístico
                    eval_data = evaluations[index]
                    results[index] = self._heuristic_evaluation(
                        answer=eval_data.get("answer", ""),
                        expected_concepts=eval_data.get("expected_concepts", []),
                        keywords=eval_data.get("keywords", []),
                    )
                    results[index]["fallback"] = True

        return results
    
    def _evaluate_single(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa una sola respuesta (usado por batch_evaluate).
        
        Args:
            eval_data: Dict con question, answer, expected_concepts, etc.
            
        Returns:
            Resultado de evaluación
        """
        return self.evaluate_answer(
            question=eval_data.get("question", ""),
            answer=eval_data.get("answer", ""),
            expected_concepts=eval_data.get("expected_concepts", []),
            keywords=eval_data.get("keywords", []),
            category=eval_data.get("category", "technical"),
            difficulty=eval_data.get("difficulty", "mid"),
            role=eval_data.get("role", ""),
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché de evaluaciones.

        Returns:
            Dict con hit_rate, latencia, entradas, etc.
        """
        if not self.enable_cache or not self.cache:
            return {"enabled": False, "message": "Caché deshabilitado"}

        stats = self.cache.get_stats()
        stats["enabled"] = True
        return stats

    def clear_cache(self):
        """Limpia todo el caché de evaluaciones."""
        if self.enable_cache and self.cache:
            self.cache.clear()
            logger.info("Caché de evaluaciones limpiado")

    def clear_expired_cache(self):
        """Limpia solo las entradas expiradas del caché."""
        if self.enable_cache and self.cache:
            self.cache.clear_expired()
            logger.info("Entradas expiradas del caché eliminadas")

    def get_top_cached_evaluations(self, limit: int = 10) -> List[tuple]:
        """
        Obtiene las evaluaciones más frecuentemente cacheadas.

        Returns:
            Lista de (pregunta_snippet, hit_count)
        """
        if not self.enable_cache or not self.cache:
            return []

        return self.cache.get_top_cached(limit=limit)

    # =========================================================================
    # TRAINING DATA COLLECTION (v2.2)
    # =========================================================================

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la recopilación de datos de entrenamiento.

        Returns:
            Dict con total_examples, por categoría, dificultad, etc.
        """
        if not self.collect_training_data or not self.training_collector:
            return {"enabled": False, "message": "Recopilación de datos deshabilitada"}

        stats = self.training_collector.get_stats()
        stats["enabled"] = True
        return stats

    def enable_training_collection(self, min_score_threshold: float = 3.0):
        """
        Habilita la recopilación de datos de entrenamiento.

        Args:
            min_score_threshold: Score mínimo para recopilar (default: 3.0)
        """
        if self.training_collector is None:
            self.training_collector = TrainingDataCollector(min_score_threshold=min_score_threshold)
        self.collect_training_data = True
        logger.info("Recopilación de datos de entrenamiento habilitada")

    def disable_training_collection(self):
        """Deshabilita la recopilación de datos de entrenamiento."""
        self.collect_training_data = False
        logger.info("Recopilación de datos de entrenamiento deshabilitada")

    def export_training_data(self, output_path: Optional[str] = None) -> str:
        """
        Exporta los datos de entrenamiento recopilados.

        Args:
            output_path: Ruta personalizada (opcional)

        Returns:
            Ruta del archivo exportado
        """
        if not self.training_collector:
            raise ValueError("TrainingDataCollector no inicializado")

        if output_path:
            import shutil

            shutil.copy(self.training_collector.storage_path, output_path)
            logger.info(f"Datos de entrenamiento exportados a: {output_path}")
            return output_path

        storage_path_str = str(self.training_collector.storage_path)
        logger.info(f"Datos de entrenamiento en: {storage_path_str}")
        return storage_path_str
