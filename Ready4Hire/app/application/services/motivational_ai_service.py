"""
Motivational AI Service
Servicio de IA que genera mensajes motivacionales personalizados para gamificación
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from app.infrastructure.llm.llm_service import OllamaLLMService

logger = logging.getLogger(__name__)


class MotivationalAIService:
    """
    Servicio de IA motivacional para gamificación.
    Genera mensajes personalizados, emocionantes y motivadores.
    """

    def __init__(
        self,
        llm_service: Optional[OllamaLLMService] = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.8,  # Alta para creatividad
    ):
        """Inicializa el servicio motivacional."""
        self.llm_service = llm_service or OllamaLLMService(model=model, temperature=temperature, max_tokens=150)
        self.model = model
        self.temperature = temperature

    def generate_game_intro(self, game_type: str, difficulty: str, user_stats: Dict) -> str:
        """Genera introducción motivadora para un juego."""
        try:
            prompt = f"""Eres un coach motivacional entusiasta para juegos de entrenamiento profesional.

**Contexto:**
- Juego: {game_type}
- Dificultad: {difficulty}
- Nivel del usuario: {user_stats.get('level', 1)}
- Racha actual: {user_stats.get('streak_days', 0)} días

Genera un mensaje corto (2-3 oraciones) que:
1. Sea EMOCIONANTE y MOTIVADOR
2. Anime al usuario a dar lo mejor
3. Reconozca su progreso si tiene racha activa
4. Use un tono enérgico y positivo

NO uses formato markdown. Solo texto directo.
"""

            message = self.llm_service.generate(prompt=prompt, max_tokens=100)
            return self._clean_message(message)

        except Exception as e:
            logger.error(f"Error generating game intro: {e}")
            return self._fallback_game_intro(game_type)

    def generate_victory_message(self, score: int, max_score: int, performance_data: Dict) -> str:
        """Genera mensaje de victoria personalizado."""
        try:
            percentage = (score / max_score * 100) if max_score > 0 else 0
            performance_level = "excelente" if percentage >= 80 else "bueno" if percentage >= 60 else "aceptable"

            prompt = f"""Eres un coach motivacional celebrando el logro de un jugador.

**Resultado:**
- Puntuación: {score}/{max_score} ({percentage:.0f}%)
- Desempeño: {performance_level}
- Tiempo: {performance_data.get('time_taken', 0)} segundos
- Precisión: {performance_data.get('accuracy', 0):.0f}%

Genera un mensaje de celebración (2-3 oraciones) que:
1. CELEBRE el logro con entusiasmo
2. Destaque aspectos positivos específicos
3. Motive a seguir mejorando
4. Use emojis apropiados (máximo 2-3)

NO uses formato markdown. Solo texto directo.
"""

            message = self.llm_service.generate(prompt=prompt, max_tokens=120)
            return self._clean_message(message)

        except Exception as e:
            logger.error(f"Error generating victory message: {e}")
            return self._fallback_victory_message(score, max_score)

    def generate_encouragement(self, performance: str, user_context: Dict) -> str:
        """Genera mensaje de aliento cuando el desempeño no es óptimo."""
        try:
            prompt = f"""Eres un mentor empático y motivador.

**Contexto:**
- Desempeño actual: {performance}
- Juegos completados: {user_context.get('games_played', 0)}
- Nivel: {user_context.get('level', 1)}

El usuario no obtuvo el resultado esperado. Genera un mensaje (2-3 oraciones) que:
1. Sea EMPÁTICO pero MOTIVADOR
2. Enfatice que el error es parte del aprendizaje
3. Anime a intentarlo de nuevo
4. Sea breve y genuino

NO uses formato markdown. Solo texto directo.
"""

            message = self.llm_service.generate(prompt=prompt, max_tokens=100)
            return self._clean_message(message)

        except Exception as e:
            logger.error(f"Error generating encouragement: {e}")
            return "¡No te desanimes! Cada intento te hace más fuerte. ¡Sigue adelante! 💪"

    def generate_streak_motivation(self, streak_days: int) -> str:
        """Genera mensaje motivacional basado en racha de días."""
        try:
            milestone = ""
            if streak_days >= 30:
                milestone = "¡UN MES COMPLETO!"
            elif streak_days >= 14:
                milestone = "¡DOS SEMANAS!"
            elif streak_days >= 7:
                milestone = "¡UNA SEMANA!"
            elif streak_days >= 3:
                milestone = "¡Tres días seguidos!"

            prompt = f"""Eres un coach celebrando la consistencia de un estudiante.

**Racha actual:** {streak_days} días consecutivos {milestone}

Genera un mensaje emocionante (1-2 oraciones) que:
1. CELEBRE la racha con entusiasmo
2. Reconozca la disciplina y constancia
3. Motive a mantener el impulso
4. Use 1-2 emojis de fuego/celebración

NO uses formato markdown. Solo texto directo.
"""

            message = self.llm_service.generate(prompt=prompt, max_tokens=80)
            return self._clean_message(message)

        except Exception as e:
            logger.error(f"Error generating streak motivation: {e}")
            return f"🔥 ¡{streak_days} días de racha! ¡Imparable! 🚀"

    def generate_level_up_message(self, new_level: int, achievements_unlocked: List[str]) -> str:
        """Genera mensaje celebrando subida de nivel."""
        try:
            achievements_text = (
                f"Desbloqueaste: {', '.join(achievements_unlocked[:2])}"
                if achievements_unlocked
                else "¡Nuevas recompensas disponibles!"
            )

            prompt = f"""Eres un narrador épico celebrando un logro importante.

**¡NIVEL AUMENTADO!**
- Nuevo nivel: {new_level}
- {achievements_text}

Genera un mensaje ÉPICO y EMOCIONANTE (2-3 oraciones) que:
1. Celebre el nivel alcanzado dramáticamente
2. Mencione los logros desbloqueados
3. Motive a seguir creciendo
4. Use lenguaje energético y positivo

NO uses formato markdown. Solo texto directo.
"""

            message = self.llm_service.generate(prompt=prompt, max_tokens=120)
            return self._clean_message(message)

        except Exception as e:
            logger.error(f"Error generating level up message: {e}")
            return f"🎉 ¡NIVEL {new_level} ALCANZADO! ¡Eres imparable! ¡Sigue conquistando! 🏆"

    def generate_daily_challenge(self, user_stats: Dict, profession: str) -> str:
        """Genera desafío diario motivador."""
        try:
            prompt = f"""Eres un entrenador creando un desafío diario emocionante.

**Usuario:**
- Profesión: {profession}
- Nivel: {user_stats.get('level', 1)}
- Racha: {user_stats.get('streak_days', 0)} días

Genera un mensaje de desafío diario (2-3 oraciones) que:
1. Proponga un objetivo claro y alcanzable para hoy
2. Sea MOTIVADOR y RETADOR
3. Esté relacionado con su profesión
4. Anime a superarse

NO uses formato markdown. Solo texto directo.
"""

            message = self.llm_service.generate(prompt=prompt, max_tokens=100)
            return self._clean_message(message)

        except Exception as e:
            logger.error(f"Error generating daily challenge: {e}")
            return f"🎯 Desafío de hoy: ¡Completa 3 juegos y mejora tu puntuación! ¡Tú puedes! 💪"

    def generate_comeback_message(self, days_inactive: int) -> str:
        """Genera mensaje de bienvenida de regreso motivador."""
        try:
            prompt = f"""Eres un amigo entusiasta dando la bienvenida de vuelta.

El usuario estuvo inactivo {days_inactive} días.

Genera un mensaje de bienvenida (2 oraciones) que:
1. Le dé la bienvenida con calidez
2. Lo motive a retomar sin presión
3. Sea positivo y alentador

NO uses formato markdown. Solo texto directo.
"""

            message = self.llm_service.generate(prompt=prompt, max_tokens=80)
            return self._clean_message(message)

        except Exception as e:
            logger.error(f"Error generating comeback message: {e}")
            return "¡Qué bueno verte de vuelta! ¿Listo para seguir mejorando? ¡Vamos! 🚀"

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _clean_message(self, message: str) -> str:
        """Limpia el mensaje generado."""
        # Remover markdown, asteriscos, etc.
        message = message.replace("**", "").replace("*", "").replace("#", "").strip()
        # Remover comillas si están al inicio y final
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]
        return message

    def _fallback_game_intro(self, game_type: str) -> str:
        """Mensaje de introducción de respaldo."""
        messages = {
            "code_challenge": "💻 ¡Hora de programar! Demuestra tus habilidades técnicas. ¡Tú puedes!",
            "quick_quiz": "⚡ ¡Quiz rápido! Pon a prueba tu conocimiento. ¡Vamos!",
            "scenario_simulator": "🎯 Enfrenta un escenario real. ¿Cómo actuarías?",
            "speed_round": "⏱️ ¡Ronda rápida! Velocidad y precisión. ¡Adelante!",
            "skill_builder": "🛠️ Construye tu habilidad paso a paso. ¡Vamos a mejorar!",
        }
        return messages.get(game_type, "🎮 ¡Prepárate para el desafío! ¡Mucha suerte!")

    def _fallback_victory_message(self, score: int, max_score: int) -> str:
        """Mensaje de victoria de respaldo."""
        percentage = (score / max_score * 100) if max_score > 0 else 0
        if percentage >= 90:
            return f"🎉 ¡INCREÍBLE! {score}/{max_score} puntos. ¡Eres una estrella! ⭐"
        elif percentage >= 70:
            return f"👏 ¡Muy bien! {score}/{max_score} puntos. ¡Excelente trabajo!"
        else:
            return f"💪 {score}/{max_score} puntos. ¡Buen esfuerzo! ¡Sigue practicando!"
