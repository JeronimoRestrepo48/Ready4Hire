"""
Unit tests for domain exceptions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest

from app.domain.exceptions import (
    Ready4HireException,
    InterviewNotFound,
    InterviewAlreadyExists,
    QuestionNotFound,
    InvalidUserId,
    LLMServiceError,
    InvalidToken
)


class TestDomainExceptions:
    """Tests for domain exceptions."""
    
    def test_base_exception(self):
        """Test base Ready4HireException."""
        exc = Ready4HireException(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.details == {"key": "value"}
        assert str(exc) == "Test error"
    
    def test_exception_to_dict(self):
        """Test exception to_dict method."""
        exc = Ready4HireException(
            message="Test error",
            error_code="TEST_ERROR",
            details={"foo": "bar"}
        )
        
        result = exc.to_dict()
        
        assert result == {
            "error": "TEST_ERROR",
            "message": "Test error",
            "details": {"foo": "bar"}
        }
    
    def test_interview_not_found(self):
        """Test InterviewNotFound exception."""
        exc = InterviewNotFound("interview-123")
        
        assert "interview-123" in exc.message
        assert exc.error_code == "INTERVIEW_NOT_FOUND"
        assert exc.details["interview_id"] == "interview-123"
    
    def test_interview_already_exists(self):
        """Test InterviewAlreadyExists exception."""
        exc = InterviewAlreadyExists("user-1", "existing-interview-id")
        
        assert "user-1" in exc.message
        assert exc.error_code == "INTERVIEW_ALREADY_ACTIVE"
        assert exc.details["user_id"] == "user-1"
        assert exc.details["existing_interview_id"] == "existing-interview-id"
    
    def test_question_not_found(self):
        """Test QuestionNotFound exception."""
        exc = QuestionNotFound("question-99")
        
        assert "question-99" in exc.message
        assert exc.error_code == "QUESTION_NOT_FOUND"
    
    def test_invalid_user_id(self):
        """Test InvalidUserId exception."""
        exc = InvalidUserId("user@123", "contains invalid characters")
        
        assert "user@123" in exc.message
        assert "invalid characters" in exc.message
        assert exc.error_code == "INVALID_USER_ID"
    
    def test_llm_service_error(self):
        """Test LLMServiceError exception."""
        original = ValueError("API timeout")
        exc = LLMServiceError("API call failed", original)
        
        assert "API call failed" in exc.message
        assert exc.error_code == "LLM_SERVICE_ERROR"
        assert "API timeout" in exc.details["original_error"]
    
    def test_invalid_token(self):
        """Test InvalidToken exception."""
        exc = InvalidToken("Token has expired")
        
        assert "expired" in exc.message
        assert exc.error_code == "INVALID_TOKEN"

