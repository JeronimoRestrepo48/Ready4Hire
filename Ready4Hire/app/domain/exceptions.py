"""
Domain Exceptions for Ready4Hire.
Custom exceptions for business logic errors.
"""
from typing import Optional, Dict, Any


class Ready4HireException(Exception):
    """Base exception for all Ready4Hire errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


# ============================================================================
# Interview Domain Exceptions
# ============================================================================

class InterviewDomainException(Ready4HireException):
    """Base exception for Interview-related errors."""
    pass


class InterviewNotFound(InterviewDomainException):
    """Interview with given ID not found."""
    
    def __init__(self, interview_id: str):
        super().__init__(
            message=f"Interview with ID '{interview_id}' not found",
            error_code="INTERVIEW_NOT_FOUND",
            details={"interview_id": interview_id}
        )


class InterviewAlreadyExists(InterviewDomainException):
    """User already has an active interview."""
    
    def __init__(self, user_id: str, existing_interview_id: str):
        super().__init__(
            message=f"User '{user_id}' already has an active interview",
            error_code="INTERVIEW_ALREADY_ACTIVE",
            details={
                "user_id": user_id,
                "existing_interview_id": existing_interview_id
            }
        )


class InterviewAlreadyCompleted(InterviewDomainException):
    """Cannot modify a completed interview."""
    
    def __init__(self, interview_id: str):
        super().__init__(
            message=f"Interview '{interview_id}' is already completed",
            error_code="INTERVIEW_ALREADY_COMPLETED",
            details={"interview_id": interview_id}
        )


class InterviewNotActive(InterviewDomainException):
    """Interview is not in active status."""
    
    def __init__(self, interview_id: str, current_status: str):
        super().__init__(
            message=f"Interview '{interview_id}' is not active (current: {current_status})",
            error_code="INTERVIEW_NOT_ACTIVE",
            details={
                "interview_id": interview_id,
                "current_status": current_status
            }
        )


class MaxQuestionsReached(InterviewDomainException):
    """Maximum number of questions reached."""
    
    def __init__(self, interview_id: str, max_questions: int):
        super().__init__(
            message=f"Interview '{interview_id}' has reached maximum questions ({max_questions})",
            error_code="MAX_QUESTIONS_REACHED",
            details={
                "interview_id": interview_id,
                "max_questions": max_questions
            }
        )


class NoActiveQuestion(InterviewDomainException):
    """No active question to answer."""
    
    def __init__(self, interview_id: str):
        super().__init__(
            message=f"Interview '{interview_id}' has no active question",
            error_code="NO_ACTIVE_QUESTION",
            details={"interview_id": interview_id}
        )


# ============================================================================
# Question Domain Exceptions
# ============================================================================

class QuestionDomainException(Ready4HireException):
    """Base exception for Question-related errors."""
    pass


class QuestionNotFound(QuestionDomainException):
    """Question with given ID not found."""
    
    def __init__(self, question_id: str):
        super().__init__(
            message=f"Question with ID '{question_id}' not found",
            error_code="QUESTION_NOT_FOUND",
            details={"question_id": question_id}
        )


class NoQuestionsAvailable(QuestionDomainException):
    """No questions available for given criteria."""
    
    def __init__(self, category: str, difficulty: str):
        super().__init__(
            message=f"No questions available for category='{category}', difficulty='{difficulty}'",
            error_code="NO_QUESTIONS_AVAILABLE",
            details={
                "category": category,
                "difficulty": difficulty
            }
        )


# ============================================================================
# Validation Exceptions
# ============================================================================

class ValidationException(Ready4HireException):
    """Base exception for validation errors."""
    pass


class InvalidUserId(ValidationException):
    """User ID is invalid."""
    
    def __init__(self, user_id: str, reason: str):
        super().__init__(
            message=f"Invalid user_id '{user_id}': {reason}",
            error_code="INVALID_USER_ID",
            details={"user_id": user_id, "reason": reason}
        )


class InvalidSkillLevel(ValidationException):
    """Skill level is invalid."""
    
    def __init__(self, skill_level: str):
        super().__init__(
            message=f"Invalid skill level '{skill_level}'. Must be: junior, mid, or senior",
            error_code="INVALID_SKILL_LEVEL",
            details={"skill_level": skill_level}
        )


class InvalidCategory(ValidationException):
    """Category is invalid."""
    
    def __init__(self, category: str):
        super().__init__(
            message=f"Invalid category '{category}'. Must be: technical or soft_skills",
            error_code="INVALID_CATEGORY",
            details={"category": category}
        )


class InvalidAnswerLength(ValidationException):
    """Answer is too short or too long."""
    
    def __init__(self, length: int, min_length: int = 1, max_length: int = 10000):
        super().__init__(
            message=f"Answer length {length} is invalid. Must be between {min_length} and {max_length} characters",
            error_code="INVALID_ANSWER_LENGTH",
            details={
                "length": length,
                "min_length": min_length,
                "max_length": max_length
            }
        )


# ============================================================================
# Service Exceptions
# ============================================================================

class ServiceException(Ready4HireException):
    """Base exception for service layer errors."""
    pass


class LLMServiceError(ServiceException):
    """Error calling LLM service."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message=f"LLM Service Error: {message}",
            error_code="LLM_SERVICE_ERROR",
            details={
                "original_error": str(original_error) if original_error else None
            }
        )


class EvaluationError(ServiceException):
    """Error evaluating answer."""
    
    def __init__(self, message: str):
        super().__init__(
            message=f"Evaluation Error: {message}",
            error_code="EVALUATION_ERROR"
        )


class EmbeddingsError(ServiceException):
    """Error generating or loading embeddings."""
    
    def __init__(self, message: str):
        super().__init__(
            message=f"Embeddings Error: {message}",
            error_code="EMBEDDINGS_ERROR"
        )


# ============================================================================
# Authentication & Authorization Exceptions
# ============================================================================

class AuthenticationException(Ready4HireException):
    """Base exception for authentication errors."""
    pass


class InvalidToken(AuthenticationException):
    """JWT token is invalid or expired."""
    
    def __init__(self, reason: str = "Invalid or expired token"):
        super().__init__(
            message=reason,
            error_code="INVALID_TOKEN"
        )


class MissingToken(AuthenticationException):
    """JWT token is missing from request."""
    
    def __init__(self):
        super().__init__(
            message="Authentication token is required",
            error_code="MISSING_TOKEN"
        )


class UnauthorizedAccess(AuthenticationException):
    """User is not authorized to access resource."""
    
    def __init__(self, user_id: str, resource: str):
        super().__init__(
            message=f"User '{user_id}' is not authorized to access '{resource}'",
            error_code="UNAUTHORIZED_ACCESS",
            details={
                "user_id": user_id,
                "resource": resource
            }
        )


# ============================================================================
# Rate Limiting Exceptions
# ============================================================================

class RateLimitExceeded(Ready4HireException):
    """Rate limit has been exceeded."""
    
    def __init__(self, limit: int, window: str):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}",
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "limit": limit,
                "window": window
            }
        )

