"""
Authentication and JWT token management.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.domain.exceptions import InvalidToken, MissingToken

# Password hashing context
# Migrado de bcrypt a argon2 por compatibilidad con Python 3.13
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)  # auto_error=False para permitir endpoints opcionales


def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against one provided by user."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload data to encode in token
        expires_delta: Token expiration time (default: from settings)

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc), "type": "access"})

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return encoded_jwt


def create_refresh_token(user_id: str) -> str:
    """
    Create JWT refresh token.

    Args:
        user_id: User identifier

    Returns:
        Encoded JWT refresh token
    """
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {"sub": user_id, "exp": expire, "iat": datetime.now(timezone.utc), "type": "refresh"}

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token to decode

    Returns:
        Decoded token payload

    Raises:
        InvalidToken: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError as e:
        raise InvalidToken(f"Token validation failed: {str(e)}")


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Dependency to get current user ID from JWT token.

    Args:
        credentials: HTTP Authorization credentials

    Returns:
        User ID from token

    Raises:
        MissingToken: If token is missing
        InvalidToken: If token is invalid
    """
    if not credentials:
        raise MissingToken()

    token = credentials.credentials
    payload = decode_token(token)

    # Validate token type
    if payload.get("type") != "access":
        raise InvalidToken("Invalid token type. Expected access token.")

    # Extract user_id
    user_id: str = payload.get("sub")
    if not user_id:
        raise InvalidToken("Token missing user identifier")

    return user_id


def get_optional_user_id(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """
    Optional dependency to get current user ID.
    Returns None if no token provided.

    Args:
        credentials: HTTP Authorization credentials (optional)

    Returns:
        User ID from token or None
    """
    if not credentials:
        return None

    try:
        return get_current_user_id(credentials)
    except (MissingToken, InvalidToken):
        return None


# Simple in-memory user store (for demo purposes)
# In production, this should be a database
# NOTE: Los passwords son hasheados lazy para evitar errores al importar
USERS_DB: Dict[str, Dict[str, Any]] = {}


def _initialize_demo_users():
    """Inicializa usuarios demo (lazy loading)."""
    global USERS_DB
    if not USERS_DB:
        USERS_DB.update(
            {
                "testuser": {
                    "user_id": "user-test-001",
                    "username": "testuser",
                    "hashed_password": hash_password("TestPassword123"),
                    "email": "test@ready4hire.com",
                    "is_active": True,
                },
                "demo": {
                    "user_id": "user-demo-001",
                    "username": "demo",
                    "hashed_password": hash_password("DemoPassword123"),
                    "email": "demo@ready4hire.com",
                    "is_active": True,
                },
            }
        )


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user with username and password.

    Args:
        username: Username
        password: Plain text password

    Returns:
        User data if authentication succeeds, None otherwise
    """
    user = USERS_DB.get(username)
    if not user:
        return None

    if not verify_password(password, user["hashed_password"]):
        return None

    if not user.get("is_active", True):
        return None

    return user


def create_token_response(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create token response for authenticated user.

    Args:
        user: User data

    Returns:
        Token response with access and refresh tokens
    """
    access_token = create_access_token(data={"sub": user["user_id"], "username": user["username"]})

    refresh_token = create_refresh_token(user["user_id"])

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "refresh_token": refresh_token,
        "user": {"user_id": user["user_id"], "username": user["username"], "email": user.get("email")},
    }
