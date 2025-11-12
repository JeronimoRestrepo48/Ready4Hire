"""
Gamification Domain Entities
Entidades para sistema de gamificación
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional
from enum import Enum


class AchievementType(Enum):
    """Tipos de logros"""

    INTERVIEW_COMPLETE = "interview_complete"
    PERFECT_SCORE = "perfect_score"
    STREAK = "streak"
    GAME_MASTER = "game_master"
    SKILL_EXPERT = "skill_expert"
    FAST_LEARNER = "fast_learner"
    CONSISTENT = "consistent"
    CHALLENGER = "challenger"


class GameType(Enum):
    """Tipos de juegos"""

    # Juegos originales
    CODE_CHALLENGE = "code_challenge"
    QUICK_QUIZ = "quick_quiz"
    SCENARIO_SIMULATOR = "scenario_simulator"
    SKILL_BUILDER = "skill_builder"
    SPEED_ROUND = "speed_round"
    PROBLEM_SOLVER = "problem_solver"
    DEBUGGING_CHALLENGE = "debugging_challenge"
    SYSTEM_DESIGN = "system_design"
    ALGORITHM_RACE = "algorithm_race"
    CODE_REVIEW = "code_review"
    API_BUILDER = "api_builder"
    DATABASE_QUEST = "database_quest"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    
    # 10 nuevos juegos universales
    MEMORY_CHALLENGE = "memory_challenge"
    LOGIC_PUZZLES = "logic_puzzles"
    TIME_MANAGEMENT = "time_management"
    COMMUNICATION_SKILLS = "communication_skills"
    PROBLEM_SOLVING_RACE = "problem_solving_race"
    DECISION_TREES = "decision_trees"
    PATTERN_RECOGNITION = "pattern_recognition"
    CRITICAL_THINKING = "critical_thinking"
    INNOVATION_LAB = "innovation_lab"
    STRESS_TEST = "stress_test"


@dataclass
class Achievement:
    """Logro/Achievement del usuario"""

    id: str
    name: str
    description: str
    icon: str
    type: AchievementType
    points: int
    requirement: Dict[str, any]  # Criterios para desbloquear
    unlocked: bool = False
    unlocked_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 a 1.0

    def check_unlock(self, user_stats: Dict) -> bool:
        """Verifica si el logro debe desbloquearse"""
        # Implementación específica según el tipo
        return False


@dataclass
class UserGameStats:
    """Estadísticas de juegos del usuario"""

    user_id: str
    total_games_played: int = 0
    total_games_won: int = 0
    total_points: int = 0
    level: int = 1
    experience: int = 0
    streak_days: int = 0
    last_play_date: Optional[datetime] = None
    games_by_type: Dict[str, int] = field(default_factory=dict)
    best_scores: Dict[str, int] = field(default_factory=dict)

    def add_game_result(self, game_type: str, won: bool, points: int):
        """Agrega resultado de un juego"""
        self.total_games_played += 1
        if won:
            self.total_games_won += 1
        self.total_points += points
        self.experience += points

        # Update games by type
        self.games_by_type[game_type] = self.games_by_type.get(game_type, 0) + 1

        # Update best score
        if game_type not in self.best_scores or points > self.best_scores[game_type]:
            self.best_scores[game_type] = points

        # Update level
        self._update_level()

        # Update streak
        self._update_streak()

    def _update_level(self):
        """Actualiza nivel basado en experiencia"""
        # Cada 1000 XP = 1 nivel
        self.level = 1 + (self.experience // 1000)

    def _update_streak(self):
        """Actualiza racha de días consecutivos"""
        today = datetime.now(timezone.utc).date()
        if self.last_play_date:
            last_date = self.last_play_date.date()
            if last_date == today:
                return  # Ya jugó hoy
            elif (today - last_date).days == 1:
                self.streak_days += 1
            else:
                self.streak_days = 1
        else:
            self.streak_days = 1

        self.last_play_date = datetime.now(timezone.utc)


@dataclass
class Game:
    """Juego de gamificación"""

    id: str
    name: str
    description: str
    type: GameType
    profession: str
    difficulty: str  # junior, mid, senior
    duration_minutes: int
    points_reward: int
    ai_powered: bool = True
    metadata: Dict = field(default_factory=dict)

    def generate_challenge(self, user_profile: Dict) -> Dict:
        """Genera desafío personalizado con IA"""
        return {
            "challenge_id": f"{self.id}_{datetime.now(timezone.utc).timestamp()}",
            "instructions": "",
            "content": {},
            "hints": [],
            "time_limit": self.duration_minutes * 60,
        }


@dataclass
class GameSession:
    """Sesión de juego activa"""

    id: str
    user_id: str
    game_id: str
    game_type: GameType
    started_at: datetime
    status: str = "active"  # active, completed, abandoned
    score: int = 0
    completed: bool = False
    completed_at: Optional[datetime] = None
    time_taken: int = 0
    hints_used: int = 0
    answers: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def complete(self, final_score: int):
        """Completa la sesión de juego"""
        self.completed = True
        self.status = "completed"
        self.score = final_score
        self.completed_at = datetime.now(timezone.utc)
        self.time_taken = int((self.completed_at - self.started_at).total_seconds())


@dataclass
class LeaderboardEntry:
    """Entrada en el ranking/leaderboard"""

    user_id: str
    username: str
    total_points: int
    level: int
    games_won: int
    achievements_count: int
    rank: int
    profession: str
    avatar_url: Optional[str] = None


@dataclass
class HintSystem:
    """Sistema de pistas para preguntas"""

    question_id: str
    hints: List[str] = field(default_factory=list)
    max_hints: int = 3
    hints_used: int = 0
    score_penalty_per_hint: int = 10  # Penalización por pista

    def get_next_hint(self) -> Optional[str]:
        """Obtiene la siguiente pista disponible"""
        if self.hints_used >= min(len(self.hints), self.max_hints):
            return None

        hint = self.hints[self.hints_used] if self.hints_used < len(self.hints) else None
        if hint:
            self.hints_used += 1
        return hint

    def calculate_final_score(self, base_score: int) -> int:
        """Calcula score final considerando pistas usadas"""
        penalty = self.hints_used * self.score_penalty_per_hint
        return max(0, base_score - penalty)
