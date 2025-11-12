"""
Authentication routes for JWT token management.
Includes login, refresh token, and token validation endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional

from app.config import settings
from app.infrastructure.security.auth import (
    authenticate_user,
    create_token_response,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user_id,
    get_optional_user_id,
    verify_password,
    hash_password,
)
from app.application.dto.auth_dto import (
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    UserDTO,
)
from app.domain.exceptions import InvalidToken, MissingToken
from app.infrastructure.security.rate_limiting import get_rate_limit_key

# Rate limiter (usa user-based cuando está disponible)
limiter = Limiter(key_func=get_rate_limit_key)

# Router
router = APIRouter(prefix="/api/v2/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, login_data: LoginRequest):
    """
    Authenticate user and return access + refresh tokens.
    
    Rate limited to 5 attempts per minute to prevent brute force.
    """
    user = authenticate_user(login_data.username, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    token_response = create_token_response(user)
    
    return TokenResponse(
        access_token=token_response["access_token"],
        token_type=token_response["token_type"],
        expires_in=token_response["expires_in"],
        refresh_token=token_response["refresh_token"],
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("20/minute")
async def refresh_token(request: Request, refresh_data: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    
    Validates refresh token and issues new access + refresh tokens.
    """
    try:
        # Decode and validate refresh token
        payload = decode_token(refresh_data.refresh_token)
        
        # Verify it's a refresh token
        if payload.get("type") != "refresh":
            raise InvalidToken("Invalid token type. Expected refresh token.")
        
        user_id = payload.get("sub")
        if not user_id:
            raise InvalidToken("Token missing user identifier")
        
        # Create new token pair
        access_token = create_access_token(data={"sub": user_id})
        refresh_token_new = create_refresh_token(user_id)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token_new,
        )
        
    except InvalidToken as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.get("/me", response_model=UserDTO)
@limiter.limit("100/minute")
async def get_current_user(request: Request, user_id: str = Depends(get_current_user_id)):
    """
    Get current authenticated user information.
    """
    # In production, fetch from database
    # For now, return basic info from token
    return UserDTO(
        user_id=user_id,
        username=user_id,  # Default, should come from DB
        email=None,
    )


@router.post("/logout")
@limiter.limit("100/minute")
async def logout(request: Request, user_id: Optional[str] = Depends(get_optional_user_id)):
    """
    Logout endpoint.
    
    Note: With JWT, logout is client-side (delete token).
    This endpoint exists for consistency and can be extended to blacklist tokens.
    """
    return {"message": "Logged out successfully"}


@router.get("/validate")
@limiter.limit("100/minute")
async def validate_token(request: Request, user_id: Optional[str] = Depends(get_optional_user_id)):
    """
    Validate if current token is still valid.
    """
    if user_id:
        return {"valid": True, "user_id": user_id}
    else:
        return {"valid": False}

