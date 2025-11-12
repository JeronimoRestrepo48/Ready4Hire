"""
Rate limiting utilities with user-based and IP-based limiting.
"""

from slowapi.util import get_remote_address
from fastapi import Request
from typing import Optional
from app.infrastructure.security.auth import get_optional_user_id
from app.config import settings

# Rate limit keys
RATE_LIMIT_KEY_USER = "user"
RATE_LIMIT_KEY_IP = "ip"
RATE_LIMIT_KEY_ANONYMOUS = "anonymous"


def get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key based on authentication status.
    
    Priority:
    1. If user is authenticated -> use user_id
    2. Otherwise -> use IP address
    
    This allows:
    - Authenticated users: Higher limits per user_id
    - Anonymous users: Lower limits per IP
    """
    try:
        # Try to get user_id from token
        # We need to extract it from the request
        from app.infrastructure.security.auth import security
        from fastapi import Depends
        
        # Check if Authorization header exists
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from app.infrastructure.security.auth import decode_token
                token = auth_header.split(" ")[1]
                payload = decode_token(token)
                user_id = payload.get("sub")
                if user_id and payload.get("type") == "access":
                    return f"{RATE_LIMIT_KEY_USER}:{user_id}"
            except Exception:
                # If token is invalid, fall back to IP
                pass
        
        # Fall back to IP address
        ip_address = get_remote_address(request)
        return f"{RATE_LIMIT_KEY_IP}:{ip_address}"
    except Exception:
        # Ultimate fallback
        ip_address = get_remote_address(request)
        return f"{RATE_LIMIT_KEY_ANONYMOUS}:{ip_address}"


def get_user_rate_limit_key(user_id: str) -> str:
    """Get rate limit key for a specific user."""
    return f"{RATE_LIMIT_KEY_USER}:{user_id}"


def get_ip_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on IP address."""
    ip_address = get_remote_address(request)
    return f"{RATE_LIMIT_KEY_IP}:{ip_address}"


# Rate limit configurations
RATE_LIMITS = {
    "authenticated": {
        "per_minute": settings.RATE_LIMIT_PER_MINUTE * 2,  # 2x for authenticated users
        "burst": settings.RATE_LIMIT_BURST * 2,
    },
    "anonymous": {
        "per_minute": settings.RATE_LIMIT_PER_MINUTE,
        "burst": settings.RATE_LIMIT_BURST,
    },
    "strict": {
        "per_minute": 10,  # Very strict for sensitive endpoints
        "burst": 2,
    },
}

