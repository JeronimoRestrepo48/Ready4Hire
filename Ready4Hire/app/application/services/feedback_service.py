"""
Servicio de generación de feedback personalizado usando Ollama.
Genera retroalimentación constructiva basada en el desempeño.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re
import logging

from app.infrastructure.llm.llm_service import OllamaLLMService
from app.infrastructure.llm.response_sanitizer import ResponseSanitizer
from app.domain.value_objects.emotion import Emotion

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Servicio para generar feedback personalizado usando Ollama local.
    Adapta el tono según la emoción detectada y el rendimiento.
    """

    def __init__(
        self, llm_service: Optional[OllamaLLMService] = None, model: str = "llama3.2:3b", temperature: float = 0.7
    ):
        """
        Inicializa el servicio de feedback.

        Args:
            llm_service: Servicio LLM Ollama (se crea uno si no se provee)
            model: Modelo Ollama a usar
            temperature: Temperatura (más alta = más creativo)
        """
        self.llm_service = llm_service or OllamaLLMService(model=model, temperature=temperature, max_tokens=256)
        self.model = model
        self.temperature = temperature
        self.sanitizer = ResponseSanitizer()

    def _get_profession_context(self, role: str, category: str) -> str:
        """Genera contexto específico según la profesión para feedback más relevante."""
        role_lower = role.lower()

        # Contextos específicos por tipo de profesión
        if any(tech in role_lower for tech in ["developer", "engineer", "programmer", "architect"]):
            return """**Enfoque para roles técnicos:**
- Valora la precisión técnica y el uso correcto de terminología
- Reconoce ejemplos de código, arquitecturas o soluciones prácticas
- Sugiere mejoras en profundidad técnica cuando sea necesario"""
        elif any(data in role_lower for data in ["data", "analyst", "scientist"]):
            return """**Enfoque para roles de datos:**
- Aprecia el pensamiento analítico y uso de datos
- Valora menciones de herramientas, metodologías y métricas
- Sugiere mejoras en análisis o visualización cuando aplique"""
        elif any(design in role_lower for design in ["designer", "ux", "ui"]):
            return """**Enfoque para roles de diseño:**
- Valora creatividad, empatía con usuarios y proceso de diseño
- Reconoce menciones de herramientas y principios de diseño
- Sugiere mejoras en UX research o iteración de diseño"""
        elif any(biz in role_lower for biz in ["manager", "product", "business", "analyst"]):
            return """**Enfoque para roles de negocio:**
- Aprecia pensamiento estratégico y orientación a resultados
- Valora ejemplos de liderazgo, toma de decisiones y métricas
- Sugiere mejoras en gestión de stakeholders o impacto de negocio"""
        elif any(mkt in role_lower for mkt in ["marketing", "sales", "content"]):
            return """**Enfoque para roles comerciales/marketing:**
- Valora creatividad, orientación a resultados y conocimiento del cliente
- Reconoce métricas de rendimiento y casos de éxito
- Sugiere mejoras en estrategia o ejecución de campañas"""
        else:
            return f"""**Enfoque para {role}:**
- Valora conocimiento específico del rol y experiencia práctica
- Reconoce habilidades profesionales demostradas
- Sugiere mejoras relevantes para el contexto del puesto"""

    def generate_feedback(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        emotion: Emotion,
        role: str,
        category: str,
        performance_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Genera feedback personalizado para el candidato.

        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            evaluation: Resultado de la evaluación
            emotion: Emoción detectada (Enum)
            role: Rol/posición
            category: Categoría (soft_skills, technical)
            performance_history: Historial de respuestas anteriores

        Returns:
            Feedback personalizado y constructivo
        """
        try:
            prompt = self._build_feedback_prompt(
                question=question,
                answer=answer,
                evaluation=evaluation,
                emotion=emotion,
                role=role,
                category=category,
                performance_history=performance_history,
            )

            # Generar feedback con Ollama
            feedback = self.llm_service.generate(prompt=prompt, temperature=self.temperature, max_tokens=256)

            # Limpiar feedback (eliminar etiquetas, etc.)
            feedback = self._clean_feedback(feedback)
            
            # Sanitizar para que parezca de agente especializado
            feedback = self.sanitizer.sanitize_feedback(feedback, role=role, category=category)

            return feedback

        except Exception as e:
            logger.error(f"Error generando feedback con LLM: {str(e)}, usando fallback")
            # Fallback a feedback genérico
            return self._generate_fallback_feedback(evaluation.get("score", 5.0), emotion)

    def _build_feedback_prompt(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        emotion: Emotion,
        role: str,
        category: str,
        performance_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Construye el prompt para generar feedback personalizado."""

        score = evaluation.get("score", 0)
        breakdown = evaluation.get("breakdown", {})
        strengths = evaluation.get("strengths", [])
        improvements = evaluation.get("improvements", [])

        # Contexto de rendimiento histórico
        history_context = ""
        if performance_history and len(performance_history) > 0:
            avg_score = sum(h.get("score", 0) for h in performance_history) / len(performance_history)
            history_context = f"""
**Contexto de desempeño:**
- Promedio en respuestas anteriores: {avg_score:.1f}/10
- Número de respuestas previas: {len(performance_history)}
- Tendencia: {"📈 Mejorando" if score >= avg_score else "📊 Estable"}
"""

        # Mapear emoción a nombre
        emotion_name = emotion.value if isinstance(emotion, Emotion) else str(emotion)

        # Contexto profesional específico
        profession_context = self._get_profession_context(role, category)

        return f"""Mentor senior {role}. Feedback profesional (3-4 oraciones).

Rol: {role} | Categoría: {category} | Emoción: {emotion_name}
Score: {score}/10 | Comp: {breakdown.get('completeness', 0):.1f}/3 | Prof: {breakdown.get('technical_depth', 0):.1f}/3 | Clar: {breakdown.get('clarity', 0):.1f}/2

P: {question}
R: {answer}

Fortalezas: {', '.join(strengths[:2]) if strengths else 'Respuesta proporcionada'}
Mejoras: {', '.join(improvements[:2]) if improvements else 'Continúa desarrollando'}

{profession_context}

Instrucciones:
- NO uses "como modelo de IA", "no puedo", "como asistente"
- Sé directo, profesional, específico
- Tono según emoción: {emotion_name}
- 80-120 palabras máximo

Feedback:"""

    def _clean_feedback(self, feedback: str) -> str:
        """Limpia el feedback eliminando etiquetas y formato innecesario."""
        # Eliminar etiquetas comunes que los LLMs pueden añadir
        patterns_to_remove = [
            r"^Feedback:\s*",
            r"^Respuesta:\s*",
            r"^Resultado:\s*",
            r"\*\*Feedback:\*\*\s*",
            r"```.*```",
        ]

        for pattern in patterns_to_remove:
            feedback = re.sub(pattern, "", feedback, flags=re.IGNORECASE | re.DOTALL)

        # Limpiar espacios extra
        feedback = re.sub(r"\s+", " ", feedback).strip()

        return feedback

    def _generate_fallback_feedback(self, score: float, emotion: Emotion) -> str:
        """Genera feedback genérico cuando el LLM falla."""

        emotion_name = emotion.value if isinstance(emotion, Emotion) else str(emotion)

        # Feedback según puntuación
        if score >= 8:
            base_feedback = "¡Excelente respuesta! 🎯 Demuestras un sólido conocimiento del tema. Sigue así, tu preparación es evidente."
        elif score >= 6:
            base_feedback = "Buena respuesta. ✓ Cubres los puntos principales correctamente. Considera profundizar más en los detalles técnicos."
        elif score >= 4:
            base_feedback = "Tu respuesta es un buen inicio. 💡 Te recomiendo revisar los conceptos clave y practicar con más ejemplos."
        else:
            base_feedback = (
                "Veo que este tema puede ser un desafío. 📚 No te desanimes, te sugiero estudiar más este tema."
            )

        # Ajustar según emoción
        if emotion in [Emotion.SADNESS, Emotion.FEAR, Emotion.ANGER]:
            emotional_addon = " Recuerda que cada entrevista es una oportunidad para aprender. ¡Ánimo! 💪"
        elif emotion == Emotion.JOY:
            emotional_addon = " ¡Tu entusiasmo es contagioso! 🌟"
        else:
            emotional_addon = " ¡Continuemos con la siguiente pregunta!"

        return base_feedback + emotional_addon

    def generate_final_feedback(
        self, role: str, category: str, all_answers: List[Dict[str, Any]], overall_score: float, accuracy: float, mode: str = "practice"
    ) -> str:
        """
        Genera feedback final al completar la entrevista con MEMORIA CONVERSACIONAL COMPLETA.

        Args:
            role: Rol/posición
            category: Categoría
            all_answers: Todas las respuestas de la entrevista (incluye contexto + técnicas)
            overall_score: Puntuación promedio general
            accuracy: Porcentaje de respuestas correctas
            mode: Modo de entrevista (practice | exam)

        Returns:
            Feedback final completo con análisis profundo
        """
        try:
            prompt = self._build_final_feedback_prompt(
                role=role, category=category, all_answers=all_answers, overall_score=overall_score, accuracy=accuracy, mode=mode
            )

            # Generar con Ollama (más tokens para análisis completo)
            feedback = self.llm_service.generate(prompt=prompt, temperature=0.7, max_tokens=500)

            return self._clean_feedback(feedback)

        except Exception as e:
            logger.error(f"Error generando feedback final: {str(e)}, usando fallback")
            return self._generate_fallback_final_feedback(overall_score, accuracy)

    def _build_final_feedback_prompt(
        self, role: str, category: str, all_answers: List[Dict[str, Any]], overall_score: float, accuracy: float, mode: str = "practice"
    ) -> str:
        """Construye prompt para feedback final con MEMORIA CONVERSACIONAL COMPLETA."""
        
        # Separar respuestas de contexto y técnicas
        context_answers = [a for a in all_answers if a.get("phase") == "context"]
        technical_answers = [a for a in all_answers if a.get("phase") == "technical"]
        
        return f"""Eres un mentor experto proporcionando feedback final COMPLETO, VALIOSO y ACCIONABLE de una entrevista.

**CONTEXTO DE LA ENTREVISTA:**
- Rol/Profesión evaluada: {role}
- Tipo de entrevista: {category}
- Modo: {mode} ({'Modo práctica - aprendizaje interactivo' if mode == 'practice' else 'Modo examen - evaluación formal'})
- Puntuación promedio: {overall_score:.1f}/10
- Precisión: {accuracy:.1f}% ({'Excelente' if accuracy >= 80 else 'Buena' if accuracy >= 60 else 'En desarrollo'})
- Total preguntas técnicas respondidas: {len(technical_answers)}
- Total preguntas de contexto: {len(context_answers)}

**MEMORIA CONVERSACIONAL COMPLETA:**

**1. PERFIL DEL CANDIDATO (preguntas de contexto):**
{self._format_context_history(context_answers)}

**2. RENDIMIENTO TÉCNICO (preguntas principales):**
{self._format_answer_history(technical_answers)}

**TU TAREA - GENERA FEEDBACK FINAL COMPLETO Y VALIOSO:**

Genera un feedback final estructurado (10-15 oraciones, 300-450 palabras) que sea:

**1. RESUMEN EJECUTIVO (2-3 oraciones) - SÉ DIRECTO Y HONESTO:**
   - Evalúa el desempeño general de forma equilibrada y realista
   - Menciona cómo el modo {mode} impactó en el proceso de evaluación
   - Proporciona una visión general clara del nivel del candidato

**2. ANÁLISIS PROFUNDO (4-5 oraciones) - SÉ ESPECÍFICO Y VALIOSO:**
   - Identifica PATRONES claros en las respuestas:
     * ¿Qué fortalezas fueron CONSISTENTES a lo largo de la entrevista?
     * ¿Qué áreas de mejora aparecieron REPETIDAMENTE?
   - RELACIONA el perfil de contexto con el rendimiento técnico:
     * ¿Cómo se reflejaron las habilidades mencionadas en las respuestas técnicas?
     * ¿Hay desconexiones entre lo que dijo y lo que demostró?
   - DESTACA insights específicos:
     * Menciona ejemplos concretos de respuestas destacables o problemáticas
     * Identifica temas o conceptos donde el candidato mostró mayor/menor dominio

**3. RECOMENDACIONES CONCRETAS (3-4 oraciones) - SÉ ACCIONABLE:**
   - Prioriza 2-3 áreas de mejora MÁS IMPORTANTES para {role}
   - Proporciona pasos ESPECÍFICOS y ACCIONABLES:
     * "Estudia [tema específico] enfocándote en [aspecto concreto]"
     * "Practica [tipo de ejercicio o proyecto] para mejorar [habilidad específica]"
   - Sugiere recursos o enfoques de estudio RELEVANTES:
     * Menciona tipos de proyectos, áreas de práctica, o recursos específicos
     * Conecta con el contexto de {role} y la industria

**4. MENSAJE MOTIVACIONAL (2-3 oraciones) - SÉ GENUINO Y DINÁMICO:**
   - Reconoce el esfuerzo y el aprendizaje logrado de forma específica
   - Motiva con un mensaje positivo pero realista
   - Proporciona perspectiva sobre el progreso y próximos pasos
   - Usa emojis estratégicamente: 🏆 💪 📈 🎯 ⭐ 🚀

**ESTILO Y TONO:**
- Profesional pero cercano y empático
- Específico y concreto - evita generalidades
- Valioso - el candidato debe sentir que aprendió algo útil
- Dinámico - mantén el engagement con estructura clara y lenguaje vivo
- Adaptado al contexto de {role} y la industria

**ESTRUCTURA SUGERIDA:**
"[Resumen ejecutivo con evaluación general]. [Análisis profundo con patrones identificados y relación contexto-rendimiento]. [Recomendaciones concretas priorizadas y accionables]. [Mensaje motivacional genuino y orientado al futuro]."

**IMPORTANTE:**
- NO uses frases genéricas como "sigue practicando" o "estudia más"
- NO repitas información que ya está en las respuestas individuales
- SÉ ESPECÍFICO - menciona conceptos, temas o habilidades concretas
- PROPORCIONA VALOR - el candidato debe salir con insights claros y acciones concretas
- MANTÉN EL FOCO - prioriza lo más importante, no intentes cubrir todo

Responde SOLO el feedback en español (sin JSON, sin etiquetas), listo para mostrar directamente al candidato."""

    def _format_answer_history(self, answers: List[Dict[str, Any]]) -> str:
        """Formatea historial de respuestas técnicas."""
        lines = []
        for i, answer in enumerate(answers, 1):
            question = answer.get("question", "N/A")
            answer_text = answer.get("answer", "N/A")
            score = answer.get("score", 0)
            is_correct = answer.get("is_correct", False)
            evaluation = answer.get("evaluation_details", {})
            
            # Truncar textos largos
            question_short = question[:100] + "..." if len(question) > 100 else question
            answer_short = answer_text[:80] + "..." if len(answer_text) > 80 else answer_text
            
            status = "✅ Correcta" if is_correct else "❌ Incorrecta"
            lines.append(
                f"Pregunta {i}: {question_short}\n"
                f"  Respuesta: {answer_short}\n"
                f"  Score: {score:.1f}/10 | {status}"
            )
        return "\n\n".join(lines[:10])  # Máximo 10 para no saturar el prompt
    
    def _format_context_history(self, answers: List[Dict[str, Any]]) -> str:
        """Formatea historial de respuestas de contexto."""
        if not answers:
            return "No hay preguntas de contexto registradas."
        
        lines = []
        for i, answer in enumerate(answers, 1):
            question = answer.get("question", "N/A")
            answer_text = answer.get("answer", "N/A")
            
            # Truncar textos largos
            question_short = question[:100] + "..." if len(question) > 100 else question
            answer_short = answer_text[:120] + "..." if len(answer_text) > 120 else answer_text
            
            lines.append(
                f"Pregunta Contexto {i}: {question_short}\n"
                f"  Respuesta: {answer_short}"
            )
        return "\n\n".join(lines)

    def _generate_fallback_final_feedback(self, overall_score: float, accuracy: float) -> str:
        """Genera feedback final genérico."""

        if overall_score >= 8:
            performance_msg = "¡Excelente desempeño! 🏆 Has demostrado un sólido dominio de los temas."
        elif overall_score >= 6:
            performance_msg = "Buen desempeño general. ✓ Tienes una base sólida que puedes seguir desarrollando."
        elif overall_score >= 4:
            performance_msg = "Desempeño moderado. 📈 Hay áreas claras donde puedes mejorar con práctica."
        else:
            performance_msg = (
                "Hay mucho espacio para crecer. 📚 No te desanimes, esto es una oportunidad de aprendizaje."
            )

        accuracy_msg = f"Tu precisión fue del {accuracy:.1f}%."

        recommendation = "Te recomiendo: revisar los conceptos donde tuviste más dificultad, practicar con más ejemplos reales, y volver a intentarlo en unos días. ¡Cada intento te acerca más al éxito! 💪"

        return f"{performance_msg} {accuracy_msg} {recommendation}"
