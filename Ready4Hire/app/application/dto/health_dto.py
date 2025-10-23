"""
DTOs for Health Check endpoints.
"""
from pydantic import BaseModel, Field
from typing import Dict
from datetime import datetime


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    message: str = ""
    latency_ms: float = Field(default=0, ge=0)


class HealthResponse(BaseModel):
    """Overall health check response."""
    
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    timestamp: datetime
    components: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": "2025-10-21T10:30:00Z",
                "components": {
                    "llm_service": "healthy",
                    "repositories": "healthy",
                    "ml": "healthy",
                    "audio": "STT: ✅ TTS: ✅"
                }
            }
        }

