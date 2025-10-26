"""
Badge Service - Manejo de insignias y logros
Sistema completo de badges para gamificación
"""

from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BadgeService:
    """
    Servicio para gestión de badges e insignias.
    Maneja la lógica de desbloqueo, progreso y recompensas.
    """

    def __init__(self):
        # En producción, esto vendría de la base de datos
        self.badge_definitions = self._get_badge_definitions()

    def _get_badge_definitions(self) -> List[Dict]:
        """Definiciones de todos los badges disponibles"""
        return [
            # ============= COMMON BADGES =============
            {
                "id": "first_step",
                "name": "🎬 Primer Paso",
                "description": "Completa tu primera entrevista",
                "icon": "🎬",
                "category": "milestone",
                "rarity": "common",
                "requirement_type": "interviews_completed",
                "requirement_value": 1,
                "reward_points": 50,
                "reward_xp": 100,
            },
            {
                "id": "novice_player",
                "name": "🎮 Jugador Novato",
                "description": "Juega tu primer juego de práctica",
                "icon": "🎮",
                "category": "general",
                "rarity": "common",
                "requirement_type": "games_played",
                "requirement_value": 1,
                "reward_points": 30,
                "reward_xp": 50,
            },
            {
                "id": "early_bird",
                "name": "🌅 Madrugador",
                "description": "Completa una entrevista antes de las 8 AM",
                "icon": "🌅",
                "category": "general",
                "rarity": "common",
                "requirement_type": "early_interviews",
                "requirement_value": 1,
                "reward_points": 40,
                "reward_xp": 60,
            },
            {
                "id": "night_owl",
                "name": "🦉 Búho Nocturno",
                "description": "Completa una entrevista después de las 10 PM",
                "icon": "🦉",
                "category": "general",
                "rarity": "common",
                "requirement_type": "night_interviews",
                "requirement_value": 1,
                "reward_points": 40,
                "reward_xp": 60,
            },
            {
                "id": "quick_learner",
                "name": "⚡ Aprendiz Rápido",
                "description": "Completa 3 entrevistas en un día",
                "icon": "⚡",
                "category": "general",
                "rarity": "common",
                "requirement_type": "interviews_in_day",
                "requirement_value": 3,
                "reward_points": 60,
                "reward_xp": 100,
            },
            {
                "id": "explorer",
                "name": "🧭 Explorador",
                "description": "Prueba 3 tipos diferentes de juegos",
                "icon": "🧭",
                "category": "general",
                "rarity": "common",
                "requirement_type": "game_types_tried",
                "requirement_value": 3,
                "reward_points": 50,
                "reward_xp": 80,
            },
            
            # ============= RARE BADGES =============
            {
                "id": "code_master",
                "name": "💻 Código Maestro",
                "description": "Gana 10 desafíos de código",
                "icon": "💻",
                "category": "technical",
                "rarity": "rare",
                "requirement_type": "code_challenges_won",
                "requirement_value": 10,
                "reward_points": 300,
                "reward_xp": 500,
            },
            {
                "id": "quiz_champion",
                "name": "🧠 Campeón de Quiz",
                "description": "Gana 15 quizzes rápidos",
                "icon": "🧠",
                "category": "technical",
                "rarity": "rare",
                "requirement_type": "quiz_won",
                "requirement_value": 15,
                "reward_points": 250,
                "reward_xp": 450,
            },
            {
                "id": "scenario_expert",
                "name": "🎯 Experto en Escenarios",
                "description": "Completa 10 simuladores de escenario",
                "icon": "🎯",
                "category": "technical",
                "rarity": "rare",
                "requirement_type": "scenarios_completed",
                "requirement_value": 10,
                "reward_points": 280,
                "reward_xp": 480,
            },
            {
                "id": "speed_demon",
                "name": "⏱️ Demonio de Velocidad",
                "description": "Completa 10 rondas de velocidad",
                "icon": "⏱️",
                "category": "technical",
                "rarity": "rare",
                "requirement_type": "speed_rounds_won",
                "requirement_value": 10,
                "reward_points": 270,
                "reward_xp": 470,
            },
            {
                "id": "problem_solver",
                "name": "🧩 Solucionador",
                "description": "Resuelve 20 problemas complejos",
                "icon": "🧩",
                "category": "technical",
                "rarity": "rare",
                "requirement_type": "problems_solved",
                "requirement_value": 20,
                "reward_points": 350,
                "reward_xp": 550,
            },
            {
                "id": "interview_veteran",
                "name": "🎓 Veterano",
                "description": "Completa 10 entrevistas",
                "icon": "🎓",
                "category": "milestone",
                "rarity": "rare",
                "requirement_type": "interviews_completed",
                "requirement_value": 10,
                "reward_points": 300,
                "reward_xp": 500,
            },
            {
                "id": "perfectionist",
                "name": "💯 Perfeccionista",
                "description": "Obtén puntuación perfecta en 5 juegos",
                "icon": "💯",
                "category": "achievement",
                "rarity": "rare",
                "requirement_type": "perfect_scores",
                "requirement_value": 5,
                "reward_points": 320,
                "reward_xp": 520,
            },
            {
                "id": "streak_starter",
                "name": "🔥 Iniciador de Racha",
                "description": "Mantén una racha de 3 días",
                "icon": "🔥",
                "category": "achievement",
                "rarity": "rare",
                "requirement_type": "streak_days",
                "requirement_value": 3,
                "reward_points": 200,
                "reward_xp": 350,
            },
            
            # ============= EPIC BADGES =============
            {
                "id": "unstoppable_streak",
                "name": "🔥 Racha Imparable",
                "description": "Mantén una racha de 7 días consecutivos",
                "icon": "🔥",
                "category": "achievement",
                "rarity": "epic",
                "requirement_type": "streak_days",
                "requirement_value": 7,
                "reward_points": 500,
                "reward_xp": 800,
            },
            {
                "id": "champion",
                "name": "🏆 Campeón",
                "description": "Gana 50 juegos en total",
                "icon": "🏆",
                "category": "achievement",
                "rarity": "epic",
                "requirement_type": "games_won",
                "requirement_value": 50,
                "reward_points": 600,
                "reward_xp": 1000,
            },
            {
                "id": "ai_master",
                "name": "🤖 Maestro de IA",
                "description": "Completa 25 desafíos con IA",
                "icon": "🤖",
                "category": "technical",
                "rarity": "epic",
                "requirement_type": "ai_challenges_won",
                "requirement_value": 25,
                "reward_points": 650,
                "reward_xp": 1100,
            },
            {
                "id": "full_stack_hero",
                "name": "⚙️ Héroe Full Stack",
                "description": "Completa desafíos en todas las categorías técnicas",
                "icon": "⚙️",
                "category": "technical",
                "rarity": "epic",
                "requirement_type": "all_categories_completed",
                "requirement_value": 1,
                "reward_points": 700,
                "reward_xp": 1200,
            },
            {
                "id": "knowledge_seeker",
                "name": "📚 Buscador de Conocimiento",
                "description": "Completa 30 entrevistas",
                "icon": "📚",
                "category": "milestone",
                "rarity": "epic",
                "requirement_type": "interviews_completed",
                "requirement_value": 30,
                "reward_points": 550,
                "reward_xp": 950,
            },
            {
                "id": "ace",
                "name": "🎖️ As",
                "description": "Obtén puntuación perfecta en 15 juegos",
                "icon": "🎖️",
                "category": "achievement",
                "rarity": "epic",
                "requirement_type": "perfect_scores",
                "requirement_value": 15,
                "reward_points": 620,
                "reward_xp": 1050,
            },
            {
                "id": "marathon_runner",
                "name": "🏃 Maratonista",
                "description": "Mantén una racha de 14 días",
                "icon": "🏃",
                "category": "achievement",
                "rarity": "epic",
                "requirement_type": "streak_days",
                "requirement_value": 14,
                "reward_points": 750,
                "reward_xp": 1300,
            },
            {
                "id": "rising_star",
                "name": "⭐ Estrella Ascendente",
                "description": "Alcanza el nivel 10",
                "icon": "⭐",
                "category": "milestone",
                "rarity": "epic",
                "requirement_type": "level_reached",
                "requirement_value": 10,
                "reward_points": 600,
                "reward_xp": 1000,
            },
            {
                "id": "point_collector",
                "name": "💰 Coleccionista de Puntos",
                "description": "Acumula 5,000 puntos totales",
                "icon": "💰",
                "category": "achievement",
                "rarity": "epic",
                "requirement_type": "total_points",
                "requirement_value": 5000,
                "reward_points": 580,
                "reward_xp": 980,
            },
            
            # ============= LEGENDARY BADGES =============
            {
                "id": "supreme_master",
                "name": "👑 Maestro Supremo",
                "description": "Alcanza el nivel 25",
                "icon": "👑",
                "category": "milestone",
                "rarity": "legendary",
                "requirement_type": "level_reached",
                "requirement_value": 25,
                "reward_points": 2500,
                "reward_xp": 5000,
            },
            {
                "id": "diamond",
                "name": "💎 Diamante",
                "description": "Acumula 10,000 puntos totales",
                "icon": "💎",
                "category": "achievement",
                "rarity": "legendary",
                "requirement_type": "total_points",
                "requirement_value": 10000,
                "reward_points": 3000,
                "reward_xp": 6000,
            },
            {
                "id": "legendary_streak",
                "name": "⚡ Racha Legendaria",
                "description": "Mantén una racha de 30 días consecutivos",
                "icon": "⚡",
                "category": "achievement",
                "rarity": "legendary",
                "requirement_type": "streak_days",
                "requirement_value": 30,
                "reward_points": 2800,
                "reward_xp": 5500,
            },
            {
                "id": "grand_champion",
                "name": "🥇 Gran Campeón",
                "description": "Gana 100 juegos en total",
                "icon": "🥇",
                "category": "achievement",
                "rarity": "legendary",
                "requirement_type": "games_won",
                "requirement_value": 100,
                "reward_points": 3200,
                "reward_xp": 6500,
            },
            {
                "id": "interview_master",
                "name": "🎯 Maestro de Entrevistas",
                "description": "Completa 50 entrevistas con éxito",
                "icon": "🎯",
                "category": "milestone",
                "rarity": "legendary",
                "requirement_type": "interviews_completed",
                "requirement_value": 50,
                "reward_points": 2600,
                "reward_xp": 5200,
            },
            {
                "id": "flawless_victory",
                "name": "✨ Victoria Perfecta",
                "description": "Obtén puntuación perfecta en 30 juegos",
                "icon": "✨",
                "category": "achievement",
                "rarity": "legendary",
                "requirement_type": "perfect_scores",
                "requirement_value": 30,
                "reward_points": 3500,
                "reward_xp": 7000,
            },
            {
                "id": "hall_of_fame",
                "name": "🏛️ Salón de la Fama",
                "description": "Alcanza el Top 10 del ranking global",
                "icon": "🏛️",
                "category": "achievement",
                "rarity": "legendary",
                "requirement_type": "ranking_position",
                "requirement_value": 10,
                "reward_points": 4000,
                "reward_xp": 8000,
            },
            {
                "id": "platinum",
                "name": "🌟 Platino",
                "description": "Acumula 25,000 puntos totales",
                "icon": "🌟",
                "category": "achievement",
                "rarity": "legendary",
                "requirement_type": "total_points",
                "requirement_value": 25000,
                "reward_points": 5000,
                "reward_xp": 10000,
            },
            {
                "id": "ultimate_master",
                "name": "🔱 Maestro Definitivo",
                "description": "Alcanza el nivel 50",
                "icon": "🔱",
                "category": "milestone",
                "rarity": "legendary",
                "requirement_type": "level_reached",
                "requirement_value": 50,
                "reward_points": 6000,
                "reward_xp": 12000,
            },
        ]

    def check_badge_unlock(self, user_stats: Dict, badge_id: str) -> Dict:
        """
        Verifica si un usuario ha desbloqueado un badge.

        Returns:
            {
                "unlocked": bool,
                "progress": float (0-1),
                "badge": Dict
            }
        """
        badge = next((b for b in self.badge_definitions if b["id"] == badge_id), None)

        if not badge:
            return {"unlocked": False, "progress": 0, "badge": None}

        requirement_type = badge["requirement_type"]
        requirement_value = badge["requirement_value"]

        # Obtener valor actual del usuario
        current_value = user_stats.get(requirement_type, 0)

        # Calcular progreso
        progress = min(current_value / requirement_value, 1.0)
        unlocked = progress >= 1.0

        return {"unlocked": unlocked, "progress": progress, "badge": badge}

    def check_all_badges(self, user_stats: Dict) -> List[Dict]:
        """
        Verifica todos los badges para un usuario.

        Returns:
            Lista de badges con su estado de desbloqueo y progreso
        """
        results = []

        for badge in self.badge_definitions:
            result = self.check_badge_unlock(user_stats, badge["id"])
            results.append(result)

        return results

    def get_newly_unlocked_badges(self, old_stats: Dict, new_stats: Dict, current_badges: List[str]) -> List[Dict]:
        """
        Determina qué badges se desbloquearon con la última acción.

        Args:
            old_stats: Estadísticas antes de la acción
            new_stats: Estadísticas después de la acción
            current_badges: IDs de badges ya desbloqueados

        Returns:
            Lista de badges recién desbloqueados
        """
        newly_unlocked = []

        for badge in self.badge_definitions:
            badge_id = badge["id"]

            # Si ya está desbloqueado, ignorar
            if badge_id in current_badges:
                continue

            # Verificar con estadísticas viejas
            old_result = self.check_badge_unlock(old_stats, badge_id)

            # Verificar con estadísticas nuevas
            new_result = self.check_badge_unlock(new_stats, badge_id)

            # Si se desbloqueó con esta acción
            if not old_result["unlocked"] and new_result["unlocked"]:
                newly_unlocked.append(badge)

        return newly_unlocked

    def get_badges_by_rarity(self, rarity: str) -> List[Dict]:
        """Obtiene todos los badges de una rareza específica"""
        return [b for b in self.badge_definitions if b["rarity"] == rarity]

    def get_badges_by_category(self, category: str) -> List[Dict]:
        """Obtiene todos los badges de una categoría específica"""
        return [b for b in self.badge_definitions if b["category"] == category]

    def get_badge_by_id(self, badge_id: str) -> Optional[Dict]:
        """Obtiene un badge específico por ID"""
        return next((b for b in self.badge_definitions if b["id"] == badge_id), None)

    def calculate_badge_score(self, badges: List[str]) -> int:
        """
        Calcula puntuación total basada en badges desbloqueados.

        Los badges más raros valen más.
        """
        rarity_multipliers = {"common": 1, "rare": 2, "epic": 5, "legendary": 10}

        total_score = 0

        for badge_id in badges:
            badge = self.get_badge_by_id(badge_id)
            if badge:
                multiplier = rarity_multipliers.get(badge["rarity"], 1)
                total_score += badge["reward_points"] * multiplier

        return total_score
