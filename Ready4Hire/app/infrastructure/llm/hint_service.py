"""
Servicio de Pistas Inteligentes (Hints)
Sistema progresivo de ayuda para modo práctica
"""

from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HintRequest:
    """Request para generar una pista"""

    question: str
    answer: str
    expected_concepts: List[str]
    attempt_number: int
    role: str
    score: float


@dataclass
class HintResponse:
    """Respuesta con pista generada"""

    hint_text: str
    hint_level: int  # 1=sutil, 2=medio, 3=directo
    remaining_attempts: int
    should_reveal_answer: bool
    emoji: str


class HintService:
    """
    Servicio de pistas inteligentes progresivas.

    Estrategia:
    - Intento 1 (score < 6): Hint sutil - dirección general
    - Intento 2 (score < 6): Hint medio - concepto específico
    - Intento 3 (score < 6): Hint directo - casi la respuesta
    - Intento 4+: Revelar respuesta correcta

    Solo funciona en MODO PRÁCTICA
    """

    MAX_ATTEMPTS = 3

    def __init__(self, llm_service=None, prompt_engine=None):
        from app.infrastructure.llm.ollama_service import OllamaLLMService
        from app.infrastructure.llm.advanced_prompts import get_prompt_engine

        self.llm_service = llm_service or OllamaLLMService(
            model="llama3.2:3b",
            temperature=0.7,  # Más creativo para hints
            max_tokens=256,
        )
        self.prompt_engine = prompt_engine or get_prompt_engine()

    async def generate_hint(self, request: HintRequest) -> HintResponse:
        """
        Genera una pista progresiva basada en el intento.

        Args:
            request: Datos de la pregunta y respuesta

        Returns:
            HintResponse con la pista generada
        """
        # Si ya agotó los intentos, revelar respuesta
        if request.attempt_number > self.MAX_ATTEMPTS:
            return self._create_final_reveal(request)

        # Si el score es bueno, no dar hint
        if request.score >= 6.0:
            return HintResponse(
                hint_text="¡Vas por buen camino! Sigue desarrollando tu respuesta. 💪",
                hint_level=0,
                remaining_attempts=self.MAX_ATTEMPTS - request.attempt_number + 1,
                should_reveal_answer=False,
                emoji="💪",
            )

        # Generar hint según el nivel de intento
        hint_text = await self._generate_progressive_hint(request)

        remaining = self.MAX_ATTEMPTS - request.attempt_number

        return HintResponse(
            hint_text=hint_text,
            hint_level=request.attempt_number,
            remaining_attempts=max(0, remaining),
            should_reveal_answer=False,
            emoji=self._get_emoji_for_level(request.attempt_number),
        )

    async def _generate_progressive_hint(self, request: HintRequest) -> str:
        """Genera hint usando el LLM con prompt específico"""
        try:
            prompt = self.prompt_engine.get_hint_prompt(
                role=request.role,
                question=request.question,
                answer=request.answer,
                expected_concepts=request.expected_concepts,
                attempts=request.attempt_number,
            )

            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=256,
            )

            hint_text = response.strip()

            # Agregar motivación según el intento
            motivation = self._get_motivation_for_attempt(request.attempt_number, request.remaining_attempts)

            return f"{hint_text}\n\n{motivation}"

        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return self._get_fallback_hint(request)

    def _create_final_reveal(self, request: HintRequest) -> HintResponse:
        """Crea respuesta revelando la solución completa"""
        concepts_str = ", ".join(request.expected_concepts)

        reveal_text = f"""💡 **Respuesta Completa:**

Has usado tus {self.MAX_ATTEMPTS} intentos. Aquí está lo que deberías haber mencionado:

**Conceptos Clave:**
{self._format_concepts(request.expected_concepts)}

💪 **¡No te desanimes!** Esto es parte del aprendizaje. Revisa estos conceptos y vuelve a intentarlo en otra pregunta similar.

📚 **Consejo:** Practica explicando estos conceptos en voz alta o escríbelos en tus propias palabras para internalizarlos mejor.
"""

        return HintResponse(
            hint_text=reveal_text, hint_level=4, remaining_attempts=0, should_reveal_answer=True, emoji="💡"
        )

    def _format_concepts(self, concepts: List[str]) -> str:
        """Formatea lista de conceptos para mostrar"""
        if not concepts:
            return "• (No hay conceptos específicos definidos)"

        return "\n".join(f"• **{concept}**" for concept in concepts)

    def _get_emoji_for_level(self, level: int) -> str:
        """Retorna emoji apropiado según nivel de hint"""
        emojis = {
            1: "💡",  # Hint sutil
            2: "🤔",  # Hint medio
            3: "⚡",  # Hint directo
        }
        return emojis.get(level, "💭")

    def _get_motivation_for_attempt(self, attempt: int, remaining: int) -> str:
        """Mensajes motivacionales según intento"""
        if attempt == 1:
            return f"💪 Tienes {remaining} intentos más. ¡Puedes hacerlo!"
        elif attempt == 2:
            return f"⚡ ¡Casi ahí! Te queda {remaining} intento. Piensa en los conceptos clave."
        elif attempt == 3:
            return "🎯 ¡Último intento! Esta pista es muy específica, úsala bien."
        else:
            return "📚 ¡Sigue aprendiendo! Cada error es una oportunidad de crecimiento."

    def _get_fallback_hint(self, request: HintRequest) -> str:
        """Pista de respaldo si falla la generación"""
        if request.attempt_number == 1:
            return f"💡 **Pista:** Piensa en conceptos relacionados con: {', '.join(request.expected_concepts[:2])}. ¿Qué sabes sobre ellos?"
        elif request.attempt_number == 2:
            return f"🤔 **Pista más directa:** Tu respuesta debería cubrir específicamente: **{request.expected_concepts[0]}**. ¿Puedes explicar este concepto?"
        else:
            concepts_str = ", ".join(f"**{c}**" for c in request.expected_concepts[:3])
            return f"⚡ **Pista muy directa:** Asegúrate de mencionar: {concepts_str}. Explica cómo se relacionan con la pregunta."

    def should_offer_hint(self, score: float, attempt: int) -> bool:
        """Determina si se debe ofrecer una pista"""
        return score < 6.0 and attempt <= self.MAX_ATTEMPTS

    def get_hint_button_text(self, attempt: int) -> str:
        """Texto para el botón de hint en el frontend"""
        if attempt == 1:
            return "💡 Necesito una pista"
        elif attempt == 2:
            return "🤔 Otra pista por favor"
        elif attempt == 3:
            return "⚡ Dame la última pista"
        else:
            return "📖 Mostrar respuesta"


# Factory function
_hint_service_instance = None


def get_hint_service() -> HintService:
    """Obtiene instancia singleton del servicio de hints"""
    global _hint_service_instance
    if _hint_service_instance is None:
        _hint_service_instance = HintService()
    return _hint_service_instance
