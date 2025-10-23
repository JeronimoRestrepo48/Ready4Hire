"""
DTOs for Interview API endpoints.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class StartInterviewRequest(BaseModel):
    """Request to start a new interview."""
    
    user_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Unique user identifier",
        example="user-12345"
    )
    role: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Job role/position",
        example="Backend Developer"
    )
    category: str = Field(
        ...,
        pattern="^(technical|soft_skills)$",
        description="Interview category: technical or soft_skills",
        example="technical"
    )
    difficulty: str = Field(
        ...,
        pattern="^(junior|mid|senior)$",
        description="Difficulty level: junior, mid, or senior",
        example="mid"
    )
    
    @validator("user_id")
    def validate_user_id(cls, v):
        """Validate user_id format."""
        if not v or v.strip() == "":
            raise ValueError("user_id cannot be empty")
        
        stripped = v.strip()
        
        # 🔐 Solo alfanuméricos, guiones y guiones bajos
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', stripped):
            raise ValueError("user_id contains invalid characters")
        
        # 🔐 No permitir IDs demasiado largos (DoS)
        if len(stripped) > 100:
            raise ValueError("user_id too long (max 100 characters)")
        
        return stripped
    
    @validator("role")
    def validate_role(cls, v):
        """Validate role."""
        stripped = v.strip()
        
        # 🔐 Sanitizar rol
        import re
        # Solo permitir letras, números, espacios y algunos caracteres especiales
        if not re.match(r'^[a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s./-]+$', stripped):
            raise ValueError("role contains invalid characters")
        
        # Detectar patrones peligrosos
        dangerous = ['<script', 'javascript:', 'onerror=', 'onload=']
        for pattern in dangerous:
            if pattern.lower() in stripped.lower():
                raise ValueError(f"Contenido no permitido detectado en rol")
        
        return stripped
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-12345",
                "role": "Backend Developer",
                "category": "technical",
                "difficulty": "mid"
            }
        }


class QuestionDTO(BaseModel):
    """Question data transfer object."""
    
    id: str
    text: str
    category: str
    difficulty: str
    topic: Optional[str] = None
    expected_concepts: List[str] = Field(default_factory=list)


class StartInterviewResponse(BaseModel):
    """Response when starting an interview."""
    
    interview_id: str = Field(..., description="Unique interview identifier")
    first_question: QuestionDTO = Field(..., description="First question")
    status: str = Field(..., description="Current interview phase")
    message: Optional[str] = Field(None, description="Welcome message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "interview_id": "interview_user-12345_1729012345.67",
                "first_question": {
                    "id": "context_0",
                    "text": "¿Cuántos años de experiencia tienes?",
                    "category": "context",
                    "difficulty": "context",
                    "topic": "context",
                    "expected_concepts": []
                },
                "status": "context",
                "message": "¡Bienvenido! Responde 5 preguntas de contexto."
            }
        }


class ProcessAnswerRequest(BaseModel):
    """Request to process a user's answer."""
    
    answer: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User's answer to the question"
    )
    time_taken: Optional[int] = Field(
        None,
        ge=0,
        le=3600,
        description="Time taken to answer in seconds"
    )
    
    @validator("answer")
    def validate_answer(cls, v):
        """Validate answer content."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Answer cannot be empty or only whitespace")
        
        # 🔐 Detección de prompt injection
        import re
        injection_patterns = [
            r'ignore\s+(all\s+)?(previous|all)\s+instructions',  # Más flexible
            r'system:\s*you\s+are',
            r'</s>',
            r'<\|endoftext\|>',
            r'{{\s*system',
            r'\[\[system\]\]',
            r'<script',
            r'javascript:',
            r'onerror\s*=',
            r'onload\s*='
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                raise ValueError(f"Contenido no permitido detectado: patrón sospechoso")
        
        # 🔐 Verificar repetición excesiva (spam)
        words = stripped.split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = len(words) / max(len(unique_words), 1)
            if repetition_ratio > 15:  # >15x la misma palabra
                raise ValueError("Respuesta con repetición excesiva detectada")
        
        # 🔐 Verificar caracteres de control peligrosos
        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', stripped):
            raise ValueError("Caracteres de control no permitidos detectados")
        
        return stripped
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Tengo 3 años de experiencia en desarrollo backend con Python y FastAPI.",
                "time_taken": 45
            }
        }


class EvaluationDTO(BaseModel):
    """Evaluation result DTO."""
    
    score: float = Field(..., ge=0, le=10)
    is_correct: bool
    feedback: str
    breakdown: Optional[Dict[str, float]] = None
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    concepts_covered: List[str] = Field(default_factory=list)
    missing_concepts: List[str] = Field(default_factory=list)


class EmotionDTO(BaseModel):
    """Emotion detection result DTO."""
    
    emotion: str
    confidence: float = Field(..., ge=0, le=1)
    language: Optional[str] = None


class ProgressDTO(BaseModel):
    """Interview progress DTO."""
    
    context_completed: int
    questions_completed: int
    total_questions: int = 10
    percentage: Optional[float] = None


class ProcessAnswerResponse(BaseModel):
    """Response when processing an answer."""
    
    evaluation: EvaluationDTO
    feedback: str
    emotion: EmotionDTO
    next_question: Optional[QuestionDTO] = None
    motivation: Optional[str] = None
    phase: str = Field(..., description="Current interview phase")
    progress: ProgressDTO
    attempts_left: Optional[int] = None
    interview_status: str
    interview_completed: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "evaluation": {
                    "score": 7.5,
                    "is_correct": True,
                    "feedback": "Buena respuesta",
                    "breakdown": {
                        "completeness": 2.5,
                        "technical_depth": 2.0,
                        "clarity": 1.5,
                        "key_concepts": 1.5
                    },
                    "strengths": ["Claridad", "Estructura"],
                    "improvements": ["Profundizar en detalles"],
                    "concepts_covered": ["Python", "FastAPI"],
                    "missing_concepts": []
                },
                "feedback": "Excelente descripción de tu experiencia...",
                "emotion": {
                    "emotion": "confident",
                    "confidence": 0.85,
                    "language": "es"
                },
                "next_question": {
                    "id": "tech_42",
                    "text": "¿Qué es REST?",
                    "category": "technical",
                    "difficulty": "mid",
                    "topic": "api_design"
                },
                "motivation": "¡Vas muy bien! Continúa así.",
                "phase": "questions",
                "progress": {
                    "context_completed": 5,
                    "questions_completed": 3,
                    "total_questions": 10,
                    "percentage": 30.0
                },
                "attempts_left": 3,
                "interview_status": "active",
                "interview_completed": False
            }
        }


class InterviewSummaryDTO(BaseModel):
    """Summary of completed interview."""
    
    total_score: float = Field(..., ge=0, le=10)
    questions_answered: int
    correct_answers: int
    accuracy: float = Field(..., ge=0, le=100)
    time_taken_seconds: int
    strengths: List[str] = Field(default_factory=list)
    areas_to_improve: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class EndInterviewResponse(BaseModel):
    """Response when ending an interview."""
    
    interview_id: str
    status: str
    summary: InterviewSummaryDTO
    completed_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "interview_id": "interview_user-12345_1729012345.67",
                "status": "completed",
                "summary": {
                    "total_score": 8.2,
                    "questions_answered": 10,
                    "correct_answers": 8,
                    "accuracy": 80.0,
                    "time_taken_seconds": 1200,
                    "strengths": ["API Design", "Python"],
                    "areas_to_improve": ["Testing", "Security"],
                    "recommendations": ["Estudiar pytest", "Revisar OWASP"]
                },
                "completed_at": "2025-10-21T10:30:00Z"
            }
        }

