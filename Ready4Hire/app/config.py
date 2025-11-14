"""
Configuration module for Ready4Hire.
Loads settings from environment variables with validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Ready4Hire"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = True

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_WORKERS: int = 4
    API_RELOAD: bool = True

    # CORS (Security Critical!)
    CORS_ORIGINS: str = "http://localhost:5214,http://localhost:3000"
    CORS_ALLOW_CREDENTIALS: bool = True

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v):
        """Validate and parse CORS origins."""
        if v == "*":
            raise ValueError("CORS_ORIGINS='*' is not allowed for security reasons. Specify explicit origins.")
        return v

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # Security
    JWT_SECRET_KEY: str = Field(default="insecure_default_change_me_but_32_chars_long_for_dev_testing")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    API_KEY_ENABLED: bool = False
    API_KEY: Optional[str] = None

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        """Ensure JWT secret is changed in production."""
        # En Pydantic V2, info.data contiene los valores validados
        env = info.data.get("ENVIRONMENT", "development") if info.data else "development"

        # En producción, secret DEBE ser seguro y largo
        if env == "production":
            if "insecure_default" in v.lower() or "change_me" in v.lower():
                raise ValueError("JWT_SECRET_KEY must be changed in production!")
            if len(v) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")

        # Verificar longitud mínima
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")

        return v

    # LLM
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TIMEOUT: int = 120  # Timeout aumentado para feedback completo, hints y respuestas largas
    OLLAMA_MAX_RETRIES: int = 3
    LLM_TEMPERATURE_EVALUATION: float = 0.3
    LLM_TEMPERATURE_FEEDBACK: float = 0.7
    LLM_MAX_TOKENS_EVALUATION: int = 512
    LLM_MAX_TOKENS_FEEDBACK: int = 1024

    # ML
    USE_ML_SELECTOR: bool = False
    USE_ML_CLUSTERING: bool = True
    USE_ML_LEARNING: bool = True
    ML_EXPLORATION_STRATEGY: str = "balanced"
    ENABLE_ML_FALLBACK: bool = True
    EMBEDDINGS_CACHE_ENABLED: bool = True
    EMBEDDINGS_CACHE_PATH: str = "app/datasets/embeddings_cache.pkl"

    # PostgreSQL Settings (for sync with WebApp)
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "ready4hire_db"
    DATABASE_USER: str = "ready4hire_user"
    DATABASE_PASSWORD: str = "ready4hire2024!"
    DATABASE_POOL_MIN_SIZE: int = 5
    DATABASE_POOL_MAX_SIZE: int = 20

    # Database (Backend usa solo MEMORY para estado temporal de entrevistas)
    # WebApp tiene su propia base de datos PostgreSQL para persistencia permanente
    DATABASE_TYPE: str = "memory"  # Fijo - Backend es stateless

    # Redis
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL_SECONDS: int = 3600
    REDIS_PASSWORD: str = "Ready4Hire2024!"
    REDIS_MAX_CONNECTIONS: int = 10

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20

    # Logging
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_FORMAT: str = Field(default="json", pattern="^(json|text)$")
    LOG_FILE: str = "logs/ready4hire_api.log"
    LOG_MAX_SIZE_MB: int = 100
    LOG_BACKUP_COUNT: int = 10
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_FILE: str = "logs/audit_log.jsonl"

    # Monitoring
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    TRACING_ENABLED: bool = False
    TRACING_ENDPOINT: Optional[str] = "http://localhost:14268/api/traces"
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENVIRONMENT: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1

    # Audio
    STT_ENABLED: bool = True
    STT_MODEL: str = "base"
    TTS_ENABLED: bool = True

    # Frontend
    FRONTEND_URL: str = "http://localhost:5214"
    STATIC_FILES_PATH: str = "app/static"

    # Data
    QUESTIONS_PATH: str = "app/datasets"
    TRAINING_DATA_COLLECTION_ENABLED: bool = False
    TRAINING_DATA_PATH: str = "data/training"

    # Testing
    TEST_DATABASE_URL: Optional[str] = None
    TEST_OLLAMA_MODEL: str = "llama3.2:3b"

    # Feature Flags
    ENABLE_SWAGGER_DOCS: bool = True
    ENABLE_REDOC: bool = True
    ENABLE_HEALTH_CHECK: bool = True
    ENABLE_METRICS_ENDPOINT: bool = True
    ENABLE_STREAMING: bool = True  # LLM streaming para mejor UX

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Convenience function
settings = get_settings()
