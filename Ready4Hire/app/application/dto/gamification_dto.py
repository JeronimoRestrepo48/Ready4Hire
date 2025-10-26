"""
Gamification DTOs
Data Transfer Objects para el sistema de gamificación
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# ============================================================================
# User Stats & Progress
# ============================================================================


class UserStatsResponse(BaseModel):
    """Estadísticas del usuario"""

    user_id: str
    level: int
    experience: int
    total_points: int
    total_games_played: int
    total_games_won: int
    streak_days: int
    rank: int
    games_by_type: Dict[str, int]
    best_scores: Dict[str, int]


class AchievementDTO(BaseModel):
    """Logro/Achievement"""

    id: str
    name: str
    description: str
    icon: str
    points: int
    unlocked: bool
    unlocked_at: Optional[datetime] = None
    progress: float = Field(ge=0.0, le=1.0)


class LeaderboardEntryDTO(BaseModel):
    """Entrada en el ranking"""

    rank: int
    user_id: str
    username: str
    total_points: int
    level: int
    games_won: int
    achievements_count: int
    profession: str
    avatar_url: Optional[str] = None


# ============================================================================
# Games
# ============================================================================


class GameDTO(BaseModel):
    """Información de juego"""

    id: str
    name: str
    description: str
    type: str
    profession: str
    difficulty: str
    duration_minutes: int
    points_reward: int
    ai_powered: bool = True


class StartGameRequest(BaseModel):
    """Request para iniciar juego"""

    user_id: str
    game_id: str
    profession: str
    difficulty: str = "mid"
    metadata: Optional[Dict] = None


class StartGameResponse(BaseModel):
    """Response al iniciar juego"""

    session_id: str
    game_id: str
    game_type: str
    challenge: Dict
    time_limit: int
    points_reward: int
    started_at: datetime


class SubmitAnswerRequest(BaseModel):
    """Request para enviar respuesta de juego"""

    session_id: str
    answer: Dict
    time_taken: Optional[int] = None


class SubmitAnswerResponse(BaseModel):
    """Response al enviar respuesta"""

    correct: bool
    points_earned: int
    total_score: int
    feedback: str
    next_challenge: Optional[Dict] = None
    completed: bool = False


class CompleteGameRequest(BaseModel):
    """Request para completar juego"""

    session_id: str


class CompleteGameResponse(BaseModel):
    """Response al completar juego"""

    session_id: str
    final_score: int
    time_taken: int
    achievements_unlocked: List[AchievementDTO]
    xp_gained: int
    new_level: int
    rank_change: int
    summary: Dict


# ============================================================================
# Hints
# ============================================================================


class RequestHintRequest(BaseModel):
    """Request para solicitar pista"""

    interview_id: str
    question_id: str


class RequestHintResponse(BaseModel):
    """Response con pista"""

    hint: str
    hints_used: int
    hints_remaining: int
    score_penalty: int
    total_penalty: int


# ============================================================================
# Professions
# ============================================================================


class ProfessionDTO(BaseModel):
    """Información de profesión"""

    id: str
    name: str
    category: str
    description: str
    technical_skills: List[str]
    soft_skills: List[str]
    common_tools: List[str]
    difficulty_level: str
    remote_friendly: bool


class SkillDTO(BaseModel):
    """Información de habilidad"""

    id: str
    name: str
    category: str
    description: str
    level: str


class UpdateUserProfileRequest(BaseModel):
    """Request para actualizar perfil de usuario"""

    user_id: str
    profession_id: str
    technical_skills: List[str]
    soft_skills: List[str]
    interests: List[str]
    goals: Optional[str] = None


# ============================================================================
# Interview Mode Enhancements
# ============================================================================


class InterviewModeConfig(BaseModel):
    """Configuración de modo de entrevista"""

    mode: str = Field(..., pattern="^(practice|exam)$")
    hints_enabled: bool = True
    feedback_immediate: bool = True
    can_retry: bool = True
    time_limit: Optional[int] = None
    show_score: bool = True


class InterviewWithHintsRequest(BaseModel):
    """Request mejorado para iniciar entrevista con config de modo"""

    user_id: str
    role: str
    category: str
    difficulty: str
    mode_config: InterviewModeConfig
