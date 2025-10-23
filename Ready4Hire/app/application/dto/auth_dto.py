"""
DTOs for Authentication endpoints.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


class LoginRequest(BaseModel):
    """Login request with credentials."""
    
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "testuser",
                "password": "SecurePassword123"
            }
        }


class TokenResponse(BaseModel):
    """JWT token response."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class UserDTO(BaseModel):
    """User information DTO."""
    
    user_id: str
    username: str
    email: Optional[EmailStr] = None
    created_at: datetime
    is_active: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-12345",
                "username": "testuser",
                "email": "test@example.com",
                "created_at": "2025-10-21T10:00:00Z",
                "is_active": True
            }
        }

