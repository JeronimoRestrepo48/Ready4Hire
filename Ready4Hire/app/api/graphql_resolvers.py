"""
GraphQL Resolvers
Implementación de todas las queries, mutations y subscriptions
"""

from typing import Optional, List
import strawberry
from datetime import datetime
import asyncio

from app.infrastructure.persistence.postgres_sync_service import get_postgres_sync
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.value_objects.interview_mode import InterviewMode
from app.infrastructure.security.auth_service import AuthService
from app.infrastructure.llm.report_generator import ReportGenerator
from app.infrastructure.llm.certificate_generator import CertificateGenerator
from app.api.graphql_schema import (
    User,
    Interview,
    InterviewWithDetails,
    Question,
    Answer,
    Badge,
    UserBadge,
    GameResult,
    Report,
    Certificate,
    UserProgress,
    StartInterviewInput,
    ProcessAnswerInput,
    LoginInput,
    RegisterInput,
)


# ============================================================================
# Query Resolvers
# ============================================================================


async def resolve_user(user_id: int) -> Optional[User]:
    """Obtiene usuario por ID"""
    try:
        pg_sync = await get_postgres_sync()
        user_data = await pg_sync.get_user_by_id(user_id)

        if user_data:
            return User(**user_data)
        return None
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None


async def resolve_interviews(user_id: int, status: Optional[str] = None, limit: int = 10) -> List[Interview]:
    """Lista entrevistas del usuario"""
    try:
        pg_sync = await get_postgres_sync()
        interviews = await pg_sync.get_user_interviews(user_id, status, limit)

        return [Interview(**interview) for interview in interviews]
    except Exception as e:
        print(f"Error fetching interviews: {e}")
        return []


async def resolve_interview_details(interview_id: str) -> Optional[InterviewWithDetails]:
    """Obtiene detalles completos de una entrevista"""
    try:
        pg_sync = await get_postgres_sync()
        interview_data = await pg_sync.get_interview_by_id(interview_id)

        if not interview_data:
            return None

        interview = Interview(**interview_data["interview"])
        questions = [Question(**q) for q in interview_data["questions"]]
        answers = [Answer(**a) for a in interview_data["answers"]]

        return InterviewWithDetails(interview=interview, questions=questions, answers=answers)
    except Exception as e:
        print(f"Error fetching interview details: {e}")
        return None


async def resolve_user_badges(user_id: int) -> List[UserBadge]:
    """Obtiene badges del usuario"""
    try:
        pg_sync = await get_postgres_sync()
        badges = await pg_sync.get_user_badges(user_id)

        return [UserBadge(badge=Badge(**b["badge"]), earned_at=b["earned_at"]) for b in badges]
    except Exception as e:
        print(f"Error fetching user badges: {e}")
        return []


async def resolve_leaderboard(limit: int = 10) -> List[User]:
    """Obtiene leaderboard global"""
    try:
        pg_sync = await get_postgres_sync()
        leaderboard = await pg_sync.get_leaderboard(limit)

        return [User(**user) for user in leaderboard]
    except Exception as e:
        print(f"Error fetching leaderboard: {e}")
        return []


async def resolve_reports(user_id: int) -> List[Report]:
    """Obtiene reportes del usuario"""
    try:
        pg_sync = await get_postgres_sync()
        reports = await pg_sync.get_user_reports(user_id)

        return [Report(**report) for report in reports]
    except Exception as e:
        print(f"Error fetching reports: {e}")
        return []


async def resolve_user_progress(user_id: int) -> List[UserProgress]:
    """Obtiene progreso del usuario por skills"""
    try:
        pg_sync = await get_postgres_sync()
        progress = await pg_sync.get_user_progress(user_id)

        return [UserProgress(**p) for p in progress]
    except Exception as e:
        print(f"Error fetching user progress: {e}")
        return []


# ============================================================================
# Mutation Resolvers
# ============================================================================


async def resolve_login(input: LoginInput) -> Optional[dict]:
    """Login y retorna JWT token + user"""
    try:
        auth_service = AuthService()
        result = await auth_service.authenticate(input.email, input.password)

        if result:
            return {"token": result["token"], "user": User(**result["user"])}
        return None
    except Exception as e:
        print(f"Error during login: {e}")
        return None


async def resolve_register(input: RegisterInput) -> Optional[User]:
    """Registra nuevo usuario"""
    try:
        pg_sync = await get_postgres_sync()

        # Hash password
        from app.infrastructure.security.argon2_hasher import Argon2Hasher

        hasher = Argon2Hasher()
        hashed_password = hasher.hash_password(input.password)

        # Create user
        user_data = {
            "email": input.email,
            "password_hash": hashed_password,
            "name": input.name,
            "last_name": input.lastName,
            "country": input.country,
            "level": 1,
            "experience": 0,
            "total_points": 0,
            "streak_days": 0,
        }

        user_id = await pg_sync.create_user(user_data)
        user_data["id"] = user_id

        return User(**user_data)
    except Exception as e:
        print(f"Error during registration: {e}")
        return None


async def resolve_start_interview(input: StartInterviewInput) -> Optional[Interview]:
    """Inicia una nueva entrevista"""
    try:
        from app.container import get_container

        container = get_container()

        # Crear entrevista usando el caso de uso existente
        interview_service = container.interview_service()

        interview = await interview_service.start_interview(
            user_id=str(input.userId),
            role=input.role,
            interview_type=input.interviewType,
            skill_level=SkillLevel.from_string(input.skillLevel),
            mode=InterviewMode.from_string(input.mode),
        )

        # Sincronizar con PostgreSQL
        pg_sync = await get_postgres_sync()
        await pg_sync.sync_interview(interview)

        return Interview(
            id=0,  # Will be set by DB
            interviewId=interview.interview_id,
            userId=input.userId,
            role=interview.role,
            interviewType=interview.interview_type,
            mode=interview.mode.to_string(),
            skillLevel=interview.skill_level.to_string(),
            status=interview.status.value,
            currentPhase=interview.current_phase,
            createdAt=interview.created_at,
            completedAt=None,
            averageScore=0.0,
            totalQuestions=len(interview.questions),
            correctAnswers=0,
        )
    except Exception as e:
        print(f"Error starting interview: {e}")
        return None


async def resolve_process_answer(input: ProcessAnswerInput) -> Optional[Answer]:
    """Procesa respuesta del candidato"""
    try:
        from app.container import get_container

        container = get_container()

        evaluation_service = container.evaluation_service()
        interview_service = container.interview_service()

        # Get interview
        interview = await interview_service.get_interview(input.interviewId)
        if not interview:
            return None

        # Get current question
        current_question = interview.get_current_question()
        if not current_question:
            return None

        # Evaluate answer
        evaluation = await evaluation_service.evaluate(
            question=current_question, answer_text=input.answerText, interview=interview
        )

        # Sync with PostgreSQL
        pg_sync = await get_postgres_sync()
        answer_id = await pg_sync.sync_answer(interview.interview_id, evaluation)

        return Answer(
            id=answer_id,
            answerText=input.answerText,
            score=evaluation.score.value,
            isCorrect=evaluation.is_correct,
            emotion=evaluation.emotion.value if evaluation.emotion else "neutral",
            feedback=evaluation.feedback,
            timeTakenSeconds=input.timeTakenSeconds,
            hintsUsed=0,
        )
    except Exception as e:
        print(f"Error processing answer: {e}")
        return None


async def resolve_complete_interview(interview_id: str) -> Optional[Report]:
    """Completa entrevista y genera reporte"""
    try:
        from app.container import get_container

        container = get_container()

        interview_service = container.interview_service()
        interview = await interview_service.get_interview(interview_id)

        if not interview:
            return None

        # Generate report
        report_generator = ReportGenerator()
        report_data = await report_generator.generate_report_data(interview)

        # Sync with PostgreSQL
        pg_sync = await get_postgres_sync()
        await pg_sync.create_report(interview_id, report_data)

        return Report(**report_data)
    except Exception as e:
        print(f"Error completing interview: {e}")
        return None


async def resolve_complete_game(user_id: int, game_type: str, score: int) -> Optional[GameResult]:
    """Completa un juego de gamificación"""
    try:
        from app.container import get_container

        container = get_container()

        gamification_service = container.gamification_service()

        # Process game completion
        result = await gamification_service.complete_game(user_id=str(user_id), game_type=game_type, score=score)

        return GameResult(
            gameId=game_type,
            score=score,
            completed=True,
            xpEarned=result.get("xp_earned", 0),
            timeTakenSeconds=result.get("time_taken", 0),
        )
    except Exception as e:
        print(f"Error completing game: {e}")
        return None


# ============================================================================
# Subscription Resolvers
# ============================================================================


async def resolve_interview_updates(interview_id: str):
    """Suscripción a actualizaciones de entrevista en tiempo real"""
    try:
        # TODO: Implementar con Redis pub/sub o WebSocket channels
        while True:
            pg_sync = await get_postgres_sync()
            interview_data = await pg_sync.get_interview_by_id(interview_id)

            if interview_data:
                yield Interview(**interview_data["interview"])

            await asyncio.sleep(2)  # Poll every 2 seconds
    except Exception as e:
        print(f"Error in interview updates subscription: {e}")


async def resolve_badge_earned(user_id: int):
    """Suscripción a badges ganados"""
    try:
        # TODO: Implementar con Redis pub/sub
        while True:
            pg_sync = await get_postgres_sync()
            latest_badge = await pg_sync.get_latest_badge(user_id)

            if latest_badge:
                yield Badge(**latest_badge)

            await asyncio.sleep(5)  # Poll every 5 seconds
    except Exception as e:
        print(f"Error in badge earned subscription: {e}")
