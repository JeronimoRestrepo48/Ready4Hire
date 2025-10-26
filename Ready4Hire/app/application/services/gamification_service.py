"""
Gamification Service
Servicio de gamificación con sistema de logros, puntos y rankings
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from app.domain.entities.gamification import (
    Achievement,
    AchievementType,
    UserGameStats,
    LeaderboardEntry,
    HintSystem,
)

logger = logging.getLogger(__name__)


class GamificationService:
    """Servicio de gamificación"""

    def __init__(self):
        self.achievements_db: Dict[str, Achievement] = self._initialize_achievements()
        self.user_stats_db: Dict[str, UserGameStats] = {}

    def _initialize_achievements(self) -> Dict[str, Achievement]:
        """Inicializa catálogo de logros"""
        achievements = {
            "first_interview": Achievement(
                id="first_interview",
                name="¡Primera Entrevista!",
                description="Completa tu primera entrevista",
                icon="🎯",
                type=AchievementType.INTERVIEW_COMPLETE,
                points=50,
                requirement={"interviews_completed": 1},
            ),
            "perfect_10": Achievement(
                id="perfect_10",
                name="Perfección Total",
                description="Obtén un puntaje perfecto (10/10)",
                icon="💯",
                type=AchievementType.PERFECT_SCORE,
                points=200,
                requirement={"perfect_scores": 1},
            ),
            "streak_7": Achievement(
                id="streak_7",
                name="Semana Activa",
                description="Practica 7 días consecutivos",
                icon="🔥",
                type=AchievementType.STREAK,
                points=150,
                requirement={"streak_days": 7},
            ),
            "game_master": Achievement(
                id="game_master",
                name="Maestro del Juego",
                description="Completa 50 juegos",
                icon="🎮",
                type=AchievementType.GAME_MASTER,
                points=500,
                requirement={"games_played": 50},
            ),
            "skill_expert": Achievement(
                id="skill_expert",
                name="Experto en Habilidad",
                description="Domina una habilidad específica",
                icon="⭐",
                type=AchievementType.SKILL_EXPERT,
                points=300,
                requirement={"skill_mastery": 1},
            ),
            "fast_learner": Achievement(
                id="fast_learner",
                name="Aprendiz Rápido",
                description="Completa 10 entrevistas en 7 días",
                icon="⚡",
                type=AchievementType.FAST_LEARNER,
                points=250,
                requirement={"interviews_in_week": 10},
            ),
            "level_10": Achievement(
                id="level_10",
                name="Nivel 10 Alcanzado",
                description="Alcanza el nivel 10",
                icon="🏆",
                type=AchievementType.CONSISTENT,
                points=400,
                requirement={"level": 10},
            ),
            "challenger": Achievement(
                id="challenger",
                name="Desafiante",
                description="Completa 5 entrevistas en modo examen",
                icon="💪",
                type=AchievementType.CHALLENGER,
                points=350,
                requirement={"exam_mode_interviews": 5},
            ),
        }
        return achievements

    def get_user_stats(self, user_id: str) -> UserGameStats:
        """Obtiene estadísticas del usuario"""
        if user_id not in self.user_stats_db:
            self.user_stats_db[user_id] = UserGameStats(user_id=user_id)
        return self.user_stats_db[user_id]

    def update_stats(self, user_id: str, game_type: str, won: bool, points: int):
        """Actualiza estadísticas después de un juego"""
        stats = self.get_user_stats(user_id)
        stats.add_game_result(game_type, won, points)

        # Check for achievements
        self._check_achievements(user_id, stats)

        logger.info(f"Stats updated for {user_id}: Level {stats.level}, XP {stats.experience}")

    def _check_achievements(self, user_id: str, stats: UserGameStats):
        """Verifica y desbloquea logros"""
        unlocked = []

        for achievement_id, achievement in self.achievements_db.items():
            if achievement.unlocked:
                continue

            # Check requirements
            if achievement.type == AchievementType.GAME_MASTER:
                if stats.total_games_played >= achievement.requirement["games_played"]:
                    achievement.unlocked = True
                    achievement.unlocked_at = datetime.now(timezone.utc)
                    unlocked.append(achievement)

            elif achievement.type == AchievementType.STREAK:
                if stats.streak_days >= achievement.requirement["streak_days"]:
                    achievement.unlocked = True
                    achievement.unlocked_at = datetime.now(timezone.utc)
                    unlocked.append(achievement)

            elif achievement.type == AchievementType.CONSISTENT:
                if stats.level >= achievement.requirement["level"]:
                    achievement.unlocked = True
                    achievement.unlocked_at = datetime.now(timezone.utc)
                    unlocked.append(achievement)

        if unlocked:
            logger.info(f"Achievements unlocked for {user_id}: {[a.name for a in unlocked]}")

        return unlocked

    def get_user_achievements(self, user_id: str) -> List[Achievement]:
        """Obtiene logros del usuario"""
        stats = self.get_user_stats(user_id)
        self._check_achievements(user_id, stats)
        return list(self.achievements_db.values())

    def get_leaderboard(self, profession: Optional[str] = None, limit: int = 100) -> List[LeaderboardEntry]:
        """Obtiene ranking global o por profesión"""
        entries = []

        for user_id, stats in self.user_stats_db.items():
            # TODO: Get user info from database
            entry = LeaderboardEntry(
                user_id=user_id,
                username=f"User {user_id[:8]}",
                total_points=stats.total_points,
                level=stats.level,
                games_won=stats.total_games_won,
                achievements_count=sum(1 for a in self.achievements_db.values() if a.unlocked),
                rank=0,
                profession=profession or "General",
            )
            entries.append(entry)

        # Sort by points
        entries.sort(key=lambda x: x.total_points, reverse=True)

        # Assign ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        return entries[:limit]

    def get_user_rank(self, user_id: str, profession: Optional[str] = None) -> int:
        """Obtiene ranking del usuario"""
        leaderboard = self.get_leaderboard(profession)
        for entry in leaderboard:
            if entry.user_id == user_id:
                return entry.rank
        return -1

    def award_points(self, user_id: str, points: int, reason: str):
        """Otorga puntos al usuario"""
        stats = self.get_user_stats(user_id)
        stats.total_points += points
        stats.experience += points
        stats._update_level()

        logger.info(f"Awarded {points} points to {user_id} for {reason}")


class HintServiceEnhanced:
    """Servicio de pistas mejorado con penalizaciones"""

    def __init__(self):
        self.hints_cache: Dict[str, HintSystem] = {}

    def create_hints_for_question(
        self, question_id: str, question_text: str, expected_concepts: List[str], llm_service
    ) -> HintSystem:
        """Genera pistas inteligentes con IA para una pregunta"""
        if question_id in self.hints_cache:
            return self.hints_cache[question_id]

        # Generate hints with LLM
        prompt = f"""Genera 3 pistas progresivas para ayudar a responder esta pregunta de entrevista.
Las pistas deben ser sutiles y progresivamente más específicas.

Pregunta: {question_text}

Conceptos esperados: {', '.join(expected_concepts)}

Formato:
Pista 1: [Pista general/conceptual]
Pista 2: [Pista más específica]
Pista 3: [Pista casi directa pero sin dar la respuesta completa]

Responde solo con las 3 pistas, una por línea."""

        try:
            response = llm_service.generate(prompt, max_tokens=200)
            hints = []
            for line in response.split("\n"):
                if line.strip() and ("Pista" in line or len(hints) > 0):
                    hint_text = line.split(":", 1)[-1].strip()
                    if hint_text:
                        hints.append(hint_text)

            if len(hints) < 3:
                # Fallback hints
                hints = [
                    "Piensa en los conceptos fundamentales relacionados con este tema.",
                    f"Considera estos aspectos: {', '.join(expected_concepts[:2])}",
                    f"La respuesta debería incluir: {', '.join(expected_concepts)}",
                ]

        except Exception as e:
            logger.error(f"Error generating hints: {e}")
            hints = [
                "Revisa los conceptos básicos del tema.",
                "Piensa en ejemplos prácticos.",
                "Considera la mejor práctica más común.",
            ]

        hint_system = HintSystem(question_id=question_id, hints=hints[:3])
        self.hints_cache[question_id] = hint_system
        return hint_system

    def get_hint(self, question_id: str, hint_index: int) -> Optional[str]:
        """Obtiene una pista específica"""
        if question_id not in self.hints_cache:
            return None

        hint_system = self.hints_cache[question_id]
        if hint_index < len(hint_system.hints) and hint_index < hint_system.max_hints:
            return hint_system.hints[hint_index]

        return None

    def use_hint(self, question_id: str) -> Optional[Dict]:
        """Usa una pista y retorna información"""
        if question_id not in self.hints_cache:
            return None

        hint_system = self.hints_cache[question_id]
        hint_text = hint_system.get_next_hint()

        if hint_text:
            return {
                "hint": hint_text,
                "hints_used": hint_system.hints_used,
                "hints_remaining": min(hint_system.max_hints, len(hint_system.hints)) - hint_system.hints_used,
                "score_penalty": hint_system.score_penalty_per_hint,
                "total_penalty": hint_system.hints_used * hint_system.score_penalty_per_hint,
            }

        return None
