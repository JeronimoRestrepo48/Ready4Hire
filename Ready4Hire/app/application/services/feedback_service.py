"""
Servicio de generación de feedback personalizado usando Ollama.
Genera retroalimentación constructiva basada en el desempeño.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
import logging

from app.infrastructure.llm.llm_service import OllamaLLMService
from app.domain.value_objects.emotion import Emotion

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Servicio para generar feedback personalizado usando Ollama local.
    Adapta el tono según la emoción detectada y el rendimiento.
    """
    
    def __init__(
        self,
        llm_service: Optional[OllamaLLMService] = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.7
    ):
        """
        Inicializa el servicio de feedback.
        
        Args:
            llm_service: Servicio LLM Ollama (se crea uno si no se provee)
            model: Modelo Ollama a usar
            temperature: Temperatura (más alta = más creativo)
        """
        self.llm_service = llm_service or OllamaLLMService(
            model=model,
            temperature=temperature,
            max_tokens=256
        )
        self.model = model
        self.temperature = temperature
    
    def generate_feedback(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        emotion: Emotion,
        role: str,
        category: str,
        performance_history: Optional[List[Dict[str, Any]]] = None
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
                performance_history=performance_history
            )
            
            # Generar feedback con Ollama
            feedback = self.llm_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=256
            )
            
            # Limpiar feedback (eliminar etiquetas, etc.)
            feedback = self._clean_feedback(feedback)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error generando feedback con LLM: {str(e)}, usando fallback")
            # Fallback a feedback genérico
            return self._generate_fallback_feedback(
                evaluation.get("score", 5.0),
                emotion
            )
    
    def _build_feedback_prompt(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        emotion: Emotion,
        role: str,
        category: str,
        performance_history: Optional[List[Dict[str, Any]]] = None
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
        
        return f"""Eres un mentor experto y empático que proporciona feedback constructivo a candidatos.

**Contexto:**
- Rol: {role}
- Categoría: {category}
- Emoción detectada: {emotion_name}

**Pregunta:**
{question}

**Respuesta del candidato:**
{answer}

**Evaluación:**
- Puntuación: {score}/10
- Completitud: {breakdown.get('completeness', 0):.1f}/3
- Profundidad: {breakdown.get('technical_depth', 0):.1f}/3
- Claridad: {breakdown.get('clarity', 0):.1f}/2
- Conceptos clave: {breakdown.get('key_concepts', 0):.1f}/2

**Fortalezas:**
{chr(10).join(f'✓ {s}' for s in strengths) if strengths else '✓ Respuesta proporcionada'}

**Mejoras:**
{chr(10).join(f'→ {i}' for i in improvements) if improvements else '→ Ninguna identificada'}

{history_context}

**Genera feedback que:**
1. Sea empático (ajusta tono según emoción: {emotion_name})
2. Destaque fortalezas específicas
3. Dé mejoras concretas y accionables
4. Motive al candidato
5. Sea breve: 3-4 oraciones (80-120 palabras)

**Tono según emoción:**
- joy/neutral: Positivo y motivador 🎉
- sadness/fear: Empático y alentador 💪
- anger: Calmado y comprensivo 🤝
- surprise: Entusiasta y guía ⭐

Responde SOLO el feedback en español, sin etiquetas ni formato adicional."""
    
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
        feedback = re.sub(r'\s+', ' ', feedback).strip()
        
        return feedback
    
    def _generate_fallback_feedback(
        self,
        score: float,
        emotion: Emotion
    ) -> str:
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
            base_feedback = "Veo que este tema puede ser un desafío. 📚 No te desanimes, te sugiero estudiar más este tema."
        
        # Ajustar según emoción
        if emotion in [Emotion.SADNESS, Emotion.FEAR, Emotion.ANGER]:
            emotional_addon = " Recuerda que cada entrevista es una oportunidad para aprender. ¡Ánimo! 💪"
        elif emotion == Emotion.JOY:
            emotional_addon = " ¡Tu entusiasmo es contagioso! 🌟"
        else:
            emotional_addon = " ¡Continuemos con la siguiente pregunta!"
        
        return base_feedback + emotional_addon
    
    def generate_final_feedback(
        self,
        role: str,
        category: str,
        all_answers: List[Dict[str, Any]],
        overall_score: float,
        accuracy: float
    ) -> str:
        """
        Genera feedback final al completar la entrevista.
        
        Args:
            role: Rol/posición
            category: Categoría
            all_answers: Todas las respuestas de la entrevista
            overall_score: Puntuación promedio general
            accuracy: Porcentaje de respuestas correctas
        
        Returns:
            Feedback final completo
        """
        try:
            prompt = self._build_final_feedback_prompt(
                role=role,
                category=category,
                all_answers=all_answers,
                overall_score=overall_score,
                accuracy=accuracy
            )
            
            # Generar con Ollama
            feedback = self.llm_service.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=384
            )
            
            return self._clean_feedback(feedback)
            
        except Exception as e:
            logger.error(f"Error generando feedback final: {str(e)}, usando fallback")
            return self._generate_fallback_final_feedback(overall_score, accuracy)
    
    def _build_final_feedback_prompt(
        self,
        role: str,
        category: str,
        all_answers: List[Dict[str, Any]],
        overall_score: float,
        accuracy: float
    ) -> str:
        """Construye prompt para feedback final."""
        return f"""Eres un mentor experto proporcionando feedback final de entrevista.

**Contexto:**
- Rol: {role}
- Categoría: {category}
- Puntuación promedio: {overall_score:.1f}/10
- Precisión: {accuracy:.1f}%
- Total de preguntas: {len(all_answers)}

**Rendimiento por pregunta:**
{self._format_answer_history(all_answers)}

**Genera feedback final que:**
1. Resume desempeño general (honesto y equilibrado)
2. Identifica patrones (fortalezas y mejoras consistentes)
3. Da recomendaciones concretas (próximos pasos)
4. Motiva al candidato (mensaje positivo y realista)

**Longitud:** 5-7 oraciones (150-200 palabras)

Responde SOLO el feedback en español, sin etiquetas."""
    
    def _format_answer_history(self, answers: List[Dict[str, Any]]) -> str:
        """Formatea historial de respuestas."""
        lines = []
        for i, answer in enumerate(answers, 1):
            score = answer.get("score", 0)
            emotion = answer.get("emotion", "neutral")
            lines.append(f"{i}. Score: {score:.1f}/10, Emoción: {emotion}")
        return "\n".join(lines[:10])  # Máximo 10 para no saturar el prompt
    
    def _generate_fallback_final_feedback(
        self,
        overall_score: float,
        accuracy: float
    ) -> str:
        """Genera feedback final genérico."""
        
        if overall_score >= 8:
            performance_msg = "¡Excelente desempeño! 🏆 Has demostrado un sólido dominio de los temas."
        elif overall_score >= 6:
            performance_msg = "Buen desempeño general. ✓ Tienes una base sólida que puedes seguir desarrollando."
        elif overall_score >= 4:
            performance_msg = "Desempeño moderado. 📈 Hay áreas claras donde puedes mejorar con práctica."
        else:
            performance_msg = "Hay mucho espacio para crecer. 📚 No te desanimes, esto es una oportunidad de aprendizaje."
        
        accuracy_msg = f"Tu precisión fue del {accuracy:.1f}%."
        
        recommendation = "Te recomiendo: revisar los conceptos donde tuviste más dificultad, practicar con más ejemplos reales, y volver a intentarlo en unos días. ¡Cada intento te acerca más al éxito! 💪"
        
        return f"{performance_msg} {accuracy_msg} {recommendation}"
