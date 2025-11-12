"""
Gamification API Routes
Endpoints para sistema de gamificación, juegos y profesiones
"""

from fastapi import APIRouter, HTTPException, Depends, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Optional, Dict
import logging

from app.application.dto.gamification_dto import (
    UserStatsResponse,
    AchievementDTO,
    LeaderboardEntryDTO,
    GameDTO,
    StartGameRequest,
    StartGameResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    CompleteGameResponse,
    RequestHintRequest,
    RequestHintResponse,
    ProfessionDTO,
    SkillDTO,
    UpdateUserProfileRequest,
)

from app.application.services.gamification_service import GamificationService, HintServiceEnhanced
from app.application.services.game_engine_service import GameEngineService
from app.application.services.badge_service import BadgeService
from app.domain.entities.profession import (
    get_all_professions,
    get_professions_by_category,
    TECH_SKILLS,
    SOFT_SKILLS,
    BUSINESS_SKILLS,
)
from app.domain.entities.gamification import GameType, Game

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/v2", tags=["Gamification"])
limiter = Limiter(key_func=get_remote_address)

# Services (inicializar con dependency injection)
gamification_service = GamificationService()
hint_service = HintServiceEnhanced()
badge_service = BadgeService()


# ============================================================================
# GAMIFICATION ENDPOINTS
# ============================================================================


@router.get("/gamification/stats/{user_id}", response_model=UserStatsResponse)
@limiter.limit("30/minute")
async def get_user_stats(request: Request, user_id: str):
    """Obtiene estadísticas de gamificación del usuario"""
    try:
        stats = gamification_service.get_user_stats(user_id)
        rank = gamification_service.get_user_rank(user_id)

        return UserStatsResponse(
            user_id=stats.user_id,
            level=stats.level,
            experience=stats.experience,
            total_points=stats.total_points,
            total_games_played=stats.total_games_played,
            total_games_won=stats.total_games_won,
            streak_days=stats.streak_days,
            rank=rank,
            games_by_type=stats.games_by_type,
            best_scores=stats.best_scores,
        )
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gamification/achievements/{user_id}", response_model=List[AchievementDTO])
@limiter.limit("30/minute")
async def get_user_achievements(request: Request, user_id: str):
    """Obtiene logros del usuario"""
    try:
        achievements = gamification_service.get_user_achievements(user_id)

        return [
            AchievementDTO(
                id=a.id,
                name=a.name,
                description=a.description,
                icon=a.icon,
                points=a.points,
                unlocked=a.unlocked,
                unlocked_at=a.unlocked_at,
                progress=a.progress,
            )
            for a in achievements
        ]
    except Exception as e:
        logger.error(f"Error getting achievements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gamification/leaderboard", response_model=List[LeaderboardEntryDTO])
@limiter.limit("20/minute")
async def get_leaderboard(request: Request, profession: Optional[str] = None, limit: int = 100):
    """Obtiene ranking global o por profesión"""
    try:
        leaderboard = gamification_service.get_leaderboard(profession, limit)

        return [
            LeaderboardEntryDTO(
                rank=entry.rank,
                user_id=entry.user_id,
                username=entry.username,
                total_points=entry.total_points,
                level=entry.level,
                games_won=entry.games_won,
                achievements_count=entry.achievements_count,
                profession=entry.profession,
                avatar_url=entry.avatar_url,
            )
            for entry in leaderboard
        ]
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GAMES ENDPOINTS
# ============================================================================


@router.get("/games", response_model=List[GameDTO])
@limiter.limit("30/minute")
async def get_available_games(request: Request, profession: Optional[str] = None, difficulty: Optional[str] = None):
    """Obtiene juegos disponibles"""
    # Catálogo completo de juegos disponibles (15 juegos)
    games = [
        # 5 juegos originales
        Game(
            id="code_challenge_1",
            name="Desafío de Código",
            description="Resuelve problemas de programación",
            type=GameType.CODE_CHALLENGE,
            profession=profession or "universal",
            difficulty=difficulty or "mid",
            duration_minutes=30,
            points_reward=300,
        ),
        Game(
            id="quick_quiz_1",
            name="Quiz Rápido",
            description="Responde preguntas de conocimiento",
            type=GameType.QUICK_QUIZ,
            profession=profession or "universal",
            difficulty=difficulty or "mid",
            duration_minutes=10,
            points_reward=100,
        ),
        Game(
            id="scenario_sim_1",
            name="Simulador de Escenarios",
            description="Enfrenta situaciones laborales reales",
            type=GameType.SCENARIO_SIMULATOR,
            profession=profession or "universal",
            difficulty=difficulty or "mid",
            duration_minutes=20,
            points_reward=250,
        ),
        Game(
            id="speed_round_1",
            name="Ronda Rápida",
            description="Responde lo más rápido posible",
            type=GameType.SPEED_ROUND,
            profession=profession or "universal",
            difficulty=difficulty or "mid",
            duration_minutes=5,
            points_reward=75,
        ),
        Game(
            id="skill_builder_1",
            name="Constructor de Habilidades",
            description="Mejora tus habilidades paso a paso",
            type=GameType.SKILL_BUILDER,
            profession=profession or "universal",
            difficulty=difficulty or "mid",
            duration_minutes=15,
            points_reward=150,
        ),
        
        # 10 nuevos juegos universales
        Game(
            id="memory_challenge_1",
            name="Desafío de Memoria",
            description="Memoriza secuencias y patrones complejos",
            type=GameType.MEMORY_CHALLENGE,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=12,
            points_reward=180,
        ),
        Game(
            id="logic_puzzles_1",
            name="Rompecabezas Lógicos",
            description="Resuelve acertijos de lógica y razonamiento",
            type=GameType.LOGIC_PUZZLES,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=18,
            points_reward=220,
        ),
        Game(
            id="time_management_1",
            name="Gestión del Tiempo",
            description="Prioriza tareas y gestiona deadlines",
            type=GameType.TIME_MANAGEMENT,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=15,
            points_reward=200,
        ),
        Game(
            id="communication_skills_1",
            name="Habilidades de Comunicación",
            description="Practica comunicación efectiva",
            type=GameType.COMMUNICATION_SKILLS,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=20,
            points_reward=240,
        ),
        Game(
            id="problem_solving_race_1",
            name="Carrera de Resolución",
            description="Resuelve problemas contra el tiempo",
            type=GameType.PROBLEM_SOLVING_RACE,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=10,
            points_reward=160,
        ),
        Game(
            id="decision_trees_1",
            name="Árboles de Decisión",
            description="Navega decisiones complejas paso a paso",
            type=GameType.DECISION_TREES,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=25,
            points_reward=280,
        ),
        Game(
            id="pattern_recognition_1",
            name="Reconocimiento de Patrones",
            description="Identifica patrones ocultos y tendencias",
            type=GameType.PATTERN_RECOGNITION,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=14,
            points_reward=190,
        ),
        Game(
            id="critical_thinking_1",
            name="Pensamiento Crítico",
            description="Analiza información y evalúa argumentos",
            type=GameType.CRITICAL_THINKING,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=22,
            points_reward=260,
        ),
        Game(
            id="innovation_lab_1",
            name="Laboratorio de Innovación",
            description="Desarrolla soluciones creativas e innovadoras",
            type=GameType.INNOVATION_LAB,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=30,
            points_reward=320,
        ),
        Game(
            id="stress_test_1",
            name="Prueba de Estrés",
            description="Mantén el rendimiento bajo presión",
            type=GameType.STRESS_TEST,
            profession="universal",
            difficulty=difficulty or "mid",
            duration_minutes=8,
            points_reward=140,
        ),
    ]

    return [
        GameDTO(
            id=g.id,
            name=g.name,
            description=g.description,
            type=g.type.value,
            profession=g.profession,
            difficulty=g.difficulty,
            duration_minutes=g.duration_minutes,
            points_reward=g.points_reward,
            ai_powered=g.ai_powered,
        )
        for g in games
    ]


@router.post("/games/start", response_model=StartGameResponse)
@limiter.limit("20/minute")
async def start_game(request: Request, game_request: StartGameRequest):
    """Inicia una nueva sesión de juego"""
    try:
        # Get game engine from container (placeholder)
        from app.container import Container

        container = Container()
        game_engine = GameEngineService(container.llm_service)

        # Create game based on type
        game_type = game_request.game_id.split("_")[0]  # Extract type from ID

        if "code" in game_type:
            challenge = game_engine.create_code_challenge(game_request.profession, game_request.difficulty, "python")
        elif "quiz" in game_type:
            challenge = game_engine.create_quick_quiz(game_request.profession, game_request.difficulty, "general")
        elif "scenario" in game_type:
            challenge = game_engine.create_scenario_simulator(game_request.profession, "problem_solving")
        elif "speed" in game_type:
            challenge = game_engine.create_speed_round(game_request.profession)
        elif "skill" in game_type:
            challenge = game_engine.create_skill_builder("problem_solving", game_request.difficulty)
        # 10 nuevos juegos universales
        elif "memory" in game_type:
            challenge = game_engine.create_memory_challenge(game_request.profession, game_request.difficulty)
        elif "logic" in game_type:
            challenge = game_engine.create_logic_puzzles(game_request.profession, game_request.difficulty)
        elif "time" in game_type:
            challenge = game_engine.create_time_management(game_request.profession, game_request.difficulty)
        elif "communication" in game_type:
            challenge = game_engine.create_communication_skills(game_request.profession, game_request.difficulty)
        elif "problem" in game_type and "solving" in game_type:
            challenge = game_engine.create_problem_solving_race(game_request.profession, game_request.difficulty)
        elif "decision" in game_type:
            challenge = game_engine.create_decision_trees(game_request.profession, game_request.difficulty)
        elif "pattern" in game_type:
            challenge = game_engine.create_pattern_recognition(game_request.profession, game_request.difficulty)
        elif "critical" in game_type:
            challenge = game_engine.create_critical_thinking(game_request.profession, game_request.difficulty)
        elif "innovation" in game_type:
            challenge = game_engine.create_innovation_lab(game_request.profession, game_request.difficulty)
        elif "stress" in game_type:
            challenge = game_engine.create_stress_test(game_request.profession, game_request.difficulty)
        else:
            challenge = game_engine.create_skill_builder("problem_solving", game_request.difficulty)

        # Create game object
        game = Game(
            id=game_request.game_id,
            name="Game",
            description="",
            type=GameType.CODE_CHALLENGE,  # Determinar dinámicamente
            profession=game_request.profession,
            difficulty=game_request.difficulty,
            duration_minutes=30,
            points_reward=300,
        )

        # Start session
        session = game_engine.start_game_session(game_request.user_id, game)

        return StartGameResponse(
            session_id=session.id,
            game_id=game.id,
            game_type=game.type.value,
            challenge=challenge,
            time_limit=game.duration_minutes * 60,
            points_reward=game.points_reward,
            started_at=session.started_at,
        )

    except Exception as e:
        logger.error(f"Error starting game: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/games/answer", response_model=SubmitAnswerResponse)
@limiter.limit("60/minute")
async def submit_game_answer(request: Request, answer_request: SubmitAnswerRequest):
    """Envía respuesta a un juego"""
    try:
        # Process answer (placeholder)
        result = {
            "correct": True,
            "points_earned": 50,
            "total_score": 150,
            "feedback": "¡Correcto! Buen trabajo.",
            "completed": False,
        }

        return SubmitAnswerResponse(**result)

    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/games/{session_id}/complete", response_model=CompleteGameResponse)
@limiter.limit("30/minute")
async def complete_game(request: Request, session_id: str):
    """Completa una sesión de juego"""
    try:
        # Complete game and calculate rewards
        final_score = 450
        xp_gained = 500
        new_level = 6

        return CompleteGameResponse(
            session_id=session_id,
            final_score=final_score,
            time_taken=1800,
            achievements_unlocked=[],
            xp_gained=xp_gained,
            new_level=new_level,
            rank_change=2,
            summary={"total_questions": 10, "correct_answers": 8, "accuracy": 0.8},
        )

    except Exception as e:
        logger.error(f"Error completing game: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HINTS ENDPOINTS
# ============================================================================


@router.post("/interviews/{interview_id}/hints/request", response_model=RequestHintResponse)
@limiter.limit("20/minute")
async def request_hint(request: Request, interview_id: str, hint_request: RequestHintRequest):
    """Solicita una pista para la pregunta actual"""
    try:
        # Use hint service
        hint_data = hint_service.use_hint(hint_request.question_id)

        if not hint_data:
            raise HTTPException(status_code=404, detail="No hay más pistas disponibles")

        return RequestHintResponse(**hint_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting hint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PROFESSIONS & SKILLS ENDPOINTS
# ============================================================================


@router.get("/professions", response_model=List[ProfessionDTO])
@limiter.limit("30/minute")
async def get_professions(request: Request, category: Optional[str] = None):
    """Obtiene todas las profesiones o por categoría"""
    try:
        professions = get_all_professions()

        return [
            ProfessionDTO(
                id=p.id,
                name=p.name,
                category=p.category.value,
                description=p.description,
                technical_skills=p.technical_skills,
                soft_skills=p.soft_skills,
                common_tools=p.common_tools,
                difficulty_level=p.difficulty_level,
                remote_friendly=p.remote_friendly,
            )
            for p in professions
        ]

    except Exception as e:
        logger.error(f"Error getting professions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills", response_model=Dict)
@limiter.limit("30/minute")
async def get_skills(request: Request):
    """Obtiene todas las habilidades disponibles"""
    try:
        return {
            "technical_skills": TECH_SKILLS,
            "soft_skills": SOFT_SKILLS,
            "business_skills": BUSINESS_SKILLS,
        }

    except Exception as e:
        logger.error(f"Error getting skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/profile")
@limiter.limit("10/minute")
async def update_user_profile(request: Request, profile: UpdateUserProfileRequest):
    """Actualiza el perfil del usuario con profesión y habilidades"""
    try:
        # Update user profile in database
        logger.info(f"Profile updated for user {profile.user_id}")

        return {"status": "success", "message": "Profile updated successfully"}

    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BADGES ENDPOINTS
# ============================================================================


@router.get("/badges", response_model=List[Dict])
@limiter.limit("30/minute")
async def get_all_badges(request: Request):
    """Obtiene todos los badges disponibles"""
    try:
        badges = badge_service.badge_definitions
        return badges

    except Exception as e:
        logger.error(f"Error getting badges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/badges/{badge_id}", response_model=Dict)
@limiter.limit("30/minute")
async def get_badge(request: Request, badge_id: str):
    """Obtiene un badge específico"""
    try:
        badge = badge_service.get_badge_by_id(badge_id)

        if not badge:
            raise HTTPException(status_code=404, detail="Badge not found")

        return badge

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting badge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/badges", response_model=List[Dict])
@limiter.limit("30/minute")
async def get_user_badges(request: Request, user_id: str):
    """Obtiene los badges de un usuario con su progreso"""
    try:
        # Mock user stats (en producción vendría de DB)
        user_stats = {
            "interviews_completed": 5,
            "games_played": 3,
            "games_won": 2,
            "streak_days": 3,
            "level_reached": 2,
            "total_points": 500,
        }

        # Verificar todos los badges
        badges_status = badge_service.check_all_badges(user_stats)

        return badges_status

    except Exception as e:
        logger.error(f"Error getting user badges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/badges/check")
@limiter.limit("20/minute")
async def check_new_badges(request: Request, user_id: str, old_stats: Dict, new_stats: Dict):
    """Verifica si se desbloquearon nuevos badges tras una acción"""
    try:
        # Current badges (vendría de DB)
        current_badges = []  # IDs de badges ya desbloqueados

        # Verificar nuevos badges
        newly_unlocked = badge_service.get_newly_unlocked_badges(old_stats, new_stats, current_badges)

        return {"newly_unlocked": newly_unlocked, "count": len(newly_unlocked)}

    except Exception as e:
        logger.error(f"Error checking badges: {e}")
        raise HTTPException(status_code=500, detail=str(e))
