"""
Data Transfer Objects (DTOs) for Ready4Hire API.
Separates API layer from domain layer.
"""
from .interview_dto import (
    StartInterviewRequest,
    StartInterviewResponse,
    ProcessAnswerRequest,
    ProcessAnswerResponse,
    EndInterviewResponse,
    InterviewSummaryDTO,
    QuestionDTO,
    EvaluationDTO,
    EmotionDTO,
    ProgressDTO
)
from .health_dto import HealthResponse, ComponentHealth
from .auth_dto import LoginRequest, TokenResponse, UserDTO

__all__ = [
    # Interview DTOs
    "StartInterviewRequest",
    "StartInterviewResponse",
    "ProcessAnswerRequest",
    "ProcessAnswerResponse",
    "EndInterviewResponse",
    "InterviewSummaryDTO",
    "QuestionDTO",
    "EvaluationDTO",
    "EmotionDTO",
    "ProgressDTO",
    # Health DTOs
    "HealthResponse",
    "ComponentHealth",
    # Auth DTOs
    "LoginRequest",
    "TokenResponse",
    "UserDTO",
]

