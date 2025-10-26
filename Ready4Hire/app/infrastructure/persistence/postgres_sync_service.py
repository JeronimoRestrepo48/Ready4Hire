"""
PostgreSQL Sync Service
Sincroniza resultados de IA y entrevistas con PostgreSQL de WebApp
"""

import asyncpg
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class InterviewData:
    """Datos de entrevista para sincronizar"""

    interview_id: str
    user_id: int
    role: str
    interview_type: str
    mode: str
    skill_level: str
    status: str
    current_phase: str
    context_question_index: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    average_score: float = 0.0
    total_questions: int = 0
    correct_answers: int = 0
    total_hints_used: int = 0
    total_time_seconds: int = 0
    current_streak: int = 0
    max_streak: int = 0
    context_answers: Optional[List[str]] = None
    metadata: Optional[Dict] = None


@dataclass
class QuestionData:
    """Datos de pregunta para sincronizar"""

    interview_id: int  # ID en PostgreSQL
    question_id: str
    text: str
    category: str
    difficulty: str
    topic: str
    expected_concepts: List[str]
    keywords: List[str]
    order_index: int
    asked_at: datetime


@dataclass
class AnswerData:
    """Datos de respuesta con evaluación para sincronizar"""

    interview_id: int  # ID en PostgreSQL
    question_id: int  # ID en PostgreSQL
    answer_text: str
    score: float
    is_correct: bool
    emotion: str
    emotion_confidence: float
    feedback: str
    evaluation_details: Dict
    time_taken_seconds: int
    hints_used: int
    attempts_count: int
    answered_at: datetime


class PostgresSyncService:
    """
    Servicio para sincronizar datos de Python con PostgreSQL.

    Flujo:
    1. Python genera evaluación
    2. Sync service guarda en PostgreSQL
    3. WebApp puede consultar datos persistidos

    Beneficios:
    - Persistencia completa de entrevistas
    - Reportes históricos
    - Análisis de progreso
    - Integración con gamificación
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "ready4hire_db",
        user: str = "postgres",
        password: str = "password",
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Inicializa el pool de conexiones"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=5,
                max_size=20,
            )
            logger.info(f"✅ PostgreSQL sync pool initialized ({self.host}:{self.port}/{self.database})")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise

    async def close(self):
        """Cierra el pool de conexiones"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL sync pool closed")

    async def sync_interview(self, interview_data: InterviewData) -> int:
        """
        Sincroniza o actualiza una entrevista en PostgreSQL.

        Args:
            interview_data: Datos de la entrevista

        Returns:
            ID de la entrevista en PostgreSQL
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self.pool.acquire() as conn:
            # Check if interview exists
            existing = await conn.fetchrow(
                'SELECT "Id" FROM "Interviews" WHERE "InterviewId" = $1', interview_data.interview_id
            )

            if existing:
                # Update existing
                await conn.execute(
                    """
                    UPDATE "Interviews"
                    SET "Status" = $2,
                        "CurrentPhase" = $3,
                        "ContextQuestionIndex" = $4,
                        "StartedAt" = $5,
                        "CompletedAt" = $6,
                        "AverageScore" = $7,
                        "TotalQuestions" = $8,
                        "CorrectAnswers" = $9,
                        "TotalHintsUsed" = $10,
                        "TotalTimeSeconds" = $11,
                        "CurrentStreak" = $12,
                        "MaxStreak" = $13,
                        "ContextAnswers" = $14,
                        "Metadata" = $15
                    WHERE "InterviewId" = $1
                """,
                    interview_data.interview_id,
                    interview_data.status,
                    interview_data.current_phase,
                    interview_data.context_question_index,
                    interview_data.started_at,
                    interview_data.completed_at,
                    interview_data.average_score,
                    interview_data.total_questions,
                    interview_data.correct_answers,
                    interview_data.total_hints_used,
                    interview_data.total_time_seconds,
                    interview_data.current_streak,
                    interview_data.max_streak,
                    json.dumps(interview_data.context_answers) if interview_data.context_answers else None,
                    json.dumps(interview_data.metadata) if interview_data.metadata else None,
                )

                logger.info(f"✅ Interview updated: {interview_data.interview_id}")
                return existing["Id"]

            else:
                # Insert new
                row = await conn.fetchrow(
                    """
                    INSERT INTO "Interviews" (
                        "InterviewId", "UserId", "Role", "InterviewType",
                        "Mode", "SkillLevel", "Status", "CurrentPhase",
                        "ContextQuestionIndex", "CreatedAt", "StartedAt",
                        "CompletedAt", "AverageScore", "TotalQuestions",
                        "CorrectAnswers", "TotalHintsUsed", "TotalTimeSeconds",
                        "CurrentStreak", "MaxStreak", "ContextAnswers", "Metadata"
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                    RETURNING "Id"
                """,
                    interview_data.interview_id,
                    interview_data.user_id,
                    interview_data.role,
                    interview_data.interview_type,
                    interview_data.mode,
                    interview_data.skill_level,
                    interview_data.status,
                    interview_data.current_phase,
                    interview_data.context_question_index,
                    interview_data.created_at,
                    interview_data.started_at,
                    interview_data.completed_at,
                    interview_data.average_score,
                    interview_data.total_questions,
                    interview_data.correct_answers,
                    interview_data.total_hints_used,
                    interview_data.total_time_seconds,
                    interview_data.current_streak,
                    interview_data.max_streak,
                    json.dumps(interview_data.context_answers) if interview_data.context_answers else None,
                    json.dumps(interview_data.metadata) if interview_data.metadata else None,
                )

                logger.info(f"✅ Interview inserted: {interview_data.interview_id}")
                return row["Id"]

    async def sync_question(self, question_data: QuestionData) -> int:
        """
        Sincroniza una pregunta en PostgreSQL.

        Returns:
            ID de la pregunta en PostgreSQL
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO "InterviewQuestions" (
                    "InterviewId", "QuestionId", "Text", "Category",
                    "Difficulty", "Topic", "ExpectedConcepts", "Keywords",
                    "OrderIndex", "AskedAt"
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING "Id"
            """,
                question_data.interview_id,
                question_data.question_id,
                question_data.text,
                question_data.category,
                question_data.difficulty,
                question_data.topic,
                json.dumps(question_data.expected_concepts),
                json.dumps(question_data.keywords),
                question_data.order_index,
                question_data.asked_at,
            )

            logger.info(f"✅ Question inserted: {question_data.question_id}")
            return row["Id"]

    async def sync_answer(self, answer_data: AnswerData) -> int:
        """
        Sincroniza una respuesta con evaluación en PostgreSQL.

        Returns:
            ID de la respuesta en PostgreSQL
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO "InterviewAnswers" (
                    "InterviewId", "QuestionId", "AnswerText", "Score",
                    "IsCorrect", "Emotion", "EmotionConfidence", "Feedback",
                    "EvaluationDetails", "TimeTakenSeconds", "HintsUsed",
                    "AttemptsCount", "AnsweredAt"
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING "Id"
            """,
                answer_data.interview_id,
                answer_data.question_id,
                answer_data.answer_text,
                answer_data.score,
                answer_data.is_correct,
                answer_data.emotion,
                answer_data.emotion_confidence,
                answer_data.feedback,
                json.dumps(answer_data.evaluation_details),
                answer_data.time_taken_seconds,
                answer_data.hints_used,
                answer_data.attempts_count,
                answer_data.answered_at,
            )

            logger.info(f"✅ Answer inserted for question {answer_data.question_id}")
            return row["Id"]

    async def sync_user_progress(
        self,
        user_id: int,
        skill_or_topic: str,
        type_: str,
        current_level: float,
        mastery_level: float,
        score_history: List[float],
    ):
        """Actualiza el progreso del usuario en un skill o topic"""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self.pool.acquire() as conn:
            # Upsert
            await conn.execute(
                """
                INSERT INTO "UserProgress" (
                    "UserId", "SkillOrTopic", "Type", "CurrentLevel",
                    "MasteryLevel", "TimesEncountered", "TimesSuccessful",
                    "ScoreHistory", "FirstEncountered", "LastUpdated"
                )
                VALUES ($1, $2, $3, $4, $5, 1, $6, $7, $8, $8)
                ON CONFLICT ("UserId", "SkillOrTopic", "Type")
                DO UPDATE SET
                    "CurrentLevel" = $4,
                    "MasteryLevel" = $5,
                    "TimesEncountered" = "UserProgress"."TimesEncountered" + 1,
                    "TimesSuccessful" = "UserProgress"."TimesSuccessful" + $6,
                    "ScoreHistory" = $7,
                    "LastUpdated" = $8
            """,
                user_id,
                skill_or_topic,
                type_,
                current_level,
                mastery_level,
                1 if current_level >= 7.0 else 0,
                json.dumps(score_history),
                datetime.now(timezone.utc),
            )

            logger.info(f"✅ User progress updated: {user_id} - {skill_or_topic}")

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Obtiene usuario por email"""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT * FROM "Users" WHERE "Email" = $1', email)

            if row:
                return dict(row)
            return None

    async def get_interview_by_id(self, interview_id: str) -> Optional[Dict]:
        """Obtiene entrevista por ID"""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT * FROM "Interviews" WHERE "InterviewId" = $1', interview_id)

            if row:
                return dict(row)
            return None


# Factory singleton
_postgres_sync = None


async def get_postgres_sync() -> PostgresSyncService:
    """Obtiene instancia singleton del servicio de sincronización"""
    global _postgres_sync

    if _postgres_sync is None:
        from app.config import settings

        _postgres_sync = PostgresSyncService(
            host=settings.DATABASE_HOST if hasattr(settings, "DATABASE_HOST") else "localhost",
            port=settings.DATABASE_PORT if hasattr(settings, "DATABASE_PORT") else 5432,
            database=settings.DATABASE_NAME if hasattr(settings, "DATABASE_NAME") else "ready4hire_db",
            user=settings.DATABASE_USER if hasattr(settings, "DATABASE_USER") else "postgres",
            password=settings.DATABASE_PASSWORD if hasattr(settings, "DATABASE_PASSWORD") else "password",
        )

        await _postgres_sync.initialize()

    return _postgres_sync
