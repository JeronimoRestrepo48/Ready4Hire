"""
GraphQL Schema for Mobile App
Definición completa del esquema GraphQL
"""

from typing import List, Optional
from datetime import datetime
import strawberry
from enum import Enum


# ============================================================================
# Enums
# ============================================================================


@strawberry.enum
class InterviewMode(Enum):
    PRACTICE = "practice"
    EXAM = "exam"


@strawberry.enum
class SkillLevel(Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"


@strawberry.enum
class InterviewStatus(Enum):
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ============================================================================
# Types
# ============================================================================


@strawberry.type
class User:
    id: int
    email: str
    name: str
    last_name: str
    country: Optional[str]
    profession: Optional[str]
    level: int
    experience: int
    total_points: int
    streak_days: int


@strawberry.type
class Question:
    id: str
    text: str
    category: str
    difficulty: str
    topic: str
    expected_concepts: List[str]


@strawberry.type
class Answer:
    id: int
    answer_text: str
    score: float
    is_correct: bool
    emotion: str
    feedback: str
    time_taken_seconds: int
    hints_used: int


@strawberry.type
class Interview:
    id: int
    interview_id: str
    user_id: int
    role: str
    interview_type: str
    mode: str
    skill_level: str
    status: str
    current_phase: str
    created_at: datetime
    completed_at: Optional[datetime]
    average_score: float
    total_questions: int
    correct_answers: int


@strawberry.type
class InterviewWithDetails:
    interview: Interview
    questions: List[Question]
    answers: List[Answer]


@strawberry.type
class Badge:
    id: int
    name: str
    description: str
    icon: str
    category: str
    rarity: str
    xp_reward: int


@strawberry.type
class UserBadge:
    badge: Badge
    earned_at: datetime


@strawberry.type
class GameResult:
    game_id: str
    score: int
    completed: bool
    xp_earned: int
    time_taken_seconds: int


@strawberry.type
class Report:
    id: int
    interview_id: int
    average_score: float
    success_rate: float
    percentile: int
    strengths: List[str]
    improvements: List[str]


@strawberry.type
class Certificate:
    id: int
    certificate_id: str
    candidate_name: str
    role: str
    score: float
    certification_level: str
    issued_at: datetime
    validation_url: str


@strawberry.type
class UserProgress:
    id: int
    skill_or_topic: str
    current_level: float
    mastery_level: float
    times_encountered: int
    times_successful: int


# ============================================================================
# Inputs
# ============================================================================


@strawberry.input
class StartInterviewInput:
    user_id: int
    role: str
    interview_type: str = "technical"
    mode: str = "practice"
    skill_level: str = "junior"


@strawberry.input
class ProcessAnswerInput:
    interview_id: str
    answer_text: str
    time_taken_seconds: int = 0


@strawberry.input
class LoginInput:
    email: str
    password: str


@strawberry.input
class RegisterInput:
    email: str
    password: str
    name: str
    last_name: str
    country: Optional[str] = None


# ============================================================================
# Query (with resolvers)
# ============================================================================


@strawberry.type
class Query:
    """GraphQL Queries"""

    @strawberry.field
    async def user(self, user_id: int) -> Optional[User]:
        """Obtiene usuario por ID"""
        from app.api.graphql_resolvers import resolve_user

        return await resolve_user(user_id)

    @strawberry.field
    async def interviews(self, user_id: int, status: Optional[str] = None, limit: int = 10) -> List[Interview]:
        """Lista entrevistas del usuario"""
        from app.api.graphql_resolvers import resolve_interviews

        return await resolve_interviews(user_id, status, limit)

    @strawberry.field
    async def interview_details(self, interview_id: str) -> Optional[InterviewWithDetails]:
        """Obtiene detalles completos de una entrevista"""
        # TODO: Implementar con PostgreSQL
        return None

    @strawberry.field
    async def user_badges(self, user_id: int) -> List[UserBadge]:
        """Obtiene badges del usuario"""
        # TODO: Implementar con PostgreSQL
        return []

    @strawberry.field
    async def leaderboard(self, limit: int = 10) -> List[User]:
        """Obtiene leaderboard global"""
        # TODO: Implementar con PostgreSQL
        return []

    @strawberry.field
    async def reports(self, user_id: int) -> List[Report]:
        """Obtiene reportes del usuario"""
        # TODO: Implementar con PostgreSQL
        return []

    @strawberry.field
    async def user_progress(self, user_id: int) -> List[UserProgress]:
        """Obtiene progreso del usuario por skills"""
        # TODO: Implementar con PostgreSQL
        return []


# ============================================================================
# Mutation
# ============================================================================


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def login(self, input: LoginInput) -> Optional[str]:
        """Login y retorna JWT token"""
        from app.api.graphql_resolvers import resolve_login

        return await resolve_login(input)

    @strawberry.mutation
    async def register(self, input: RegisterInput) -> Optional[User]:
        """Registra nuevo usuario"""
        # TODO: Implementar registro
        return None

    @strawberry.mutation
    async def start_interview(self, input: StartInterviewInput) -> Optional[Interview]:
        """Inicia una nueva entrevista"""
        # TODO: Implementar con servicio de entrevistas
        return None

    @strawberry.mutation
    async def process_answer(self, input: ProcessAnswerInput) -> Optional[Answer]:
        """Procesa respuesta del candidato"""
        # TODO: Implementar con evaluation service
        return None

    @strawberry.mutation
    async def complete_interview(self, interview_id: str) -> Optional[Report]:
        """Completa entrevista y genera reporte"""
        # TODO: Implementar con report generator
        return None

    @strawberry.mutation
    async def complete_game(self, user_id: int, game_type: str, score: int) -> Optional[GameResult]:
        """Completa un juego de gamificación"""
        # TODO: Implementar con gamification service
        return None


# ============================================================================
# Subscription
# ============================================================================


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def interview_updates(self, interview_id: str) -> Interview:
        """Suscripción a actualizaciones de entrevista en tiempo real"""
        # TODO: Implementar con WebSockets
        yield None

    @strawberry.subscription
    async def badge_earned(self, user_id: int) -> Badge:
        """Suscripción a badges ganados"""
        # TODO: Implementar con WebSockets
        yield None


# ============================================================================
# Schema
# ============================================================================

schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
