"""
API Layer for Ready4Hire
========================

This module contains the FastAPI routes and endpoints for the Ready4Hire system.
It provides a clean REST API interface that delegates to the core business logic.
"""

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional
import os
from pathlib import Path

from backend.core.services import InterviewService, QuestionService, AudioService
from backend.data.repository import create_repository_from_env
from backend.llm.providers import create_llm_manager_from_env


class APIRouter:
    """Main API router for Ready4Hire."""
    
    def __init__(self):
        """Initialize API router with dependencies."""
        self.app = FastAPI(
            title="Ready4Hire API",
            description="Modular AI Interview Simulation System",
            version="2.0.0"
        )
        
        # Initialize services
        self.repository = create_repository_from_env()
        self.llm_manager = create_llm_manager_from_env()
        self.interview_service = InterviewService(self.repository, self.llm_manager)
        self.question_service = QuestionService(self.repository)
        self.audio_service = AudioService()
        
        # Setup routes
        self._setup_routes()
        self._setup_static_files()
        self._setup_middleware()
    
    def _setup_routes(self):
        """Setup all API routes."""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            llm_info = self.llm_manager.get_active_provider_info()
            return {
                "status": "healthy",
                "version": "2.0.0",
                "llm_provider": llm_info
            }
        
        # Interview endpoints
        @self.app.post("/start_interview")
        async def start_interview(payload: dict = Body(...)):
            """Start a new interview session."""
            try:
                user_id = payload.get("user_id")
                role = payload.get("role")
                interview_type = payload.get("type")
                mode = payload.get("mode", "practice")
                
                if not user_id or not role or not interview_type:
                    raise HTTPException(
                        status_code=400,
                        detail="user_id, role, and type are required"
                    )
                
                result = self.interview_service.start_interview(
                    str(user_id), role, interview_type, mode
                )
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/next_question")
        async def next_question(payload: dict = Body(...)):
            """Get the next question for the user."""
            try:
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=400, detail="user_id is required")
                
                result = self.interview_service.next_question(str(user_id))
                
                if "error" in result:
                    raise HTTPException(status_code=404, detail=result["error"])
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/answer")
        async def process_answer(request: Request):
            """Process user's answer."""
            try:
                data = await request.json()
                user_id = data.get("user_id")
                answer = data.get("answer")
                
                if not user_id or not answer:
                    raise HTTPException(
                        status_code=400,
                        detail="user_id and answer are required"
                    )
                
                # Basic input sanitization
                answer = self._sanitize_input(answer)
                
                result = self.interview_service.process_answer(str(user_id), answer)
                
                if "error" in result:
                    raise HTTPException(status_code=404, detail=result["error"])
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/end_interview")
        async def end_interview(payload: dict = Body(...)):
            """End the interview and get summary."""
            try:
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=400, detail="user_id is required")
                
                result = self.interview_service.end_interview(str(user_id))
                
                if "error" in result:
                    raise HTTPException(status_code=404, detail=result["error"])
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Question management endpoints
        @self.app.get("/get_roles")
        def get_roles():
            """Get available roles."""
            try:
                roles = self.question_service.get_available_roles()
                return {"roles": roles}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/get_levels")
        def get_levels():
            """Get available levels."""
            try:
                levels = self.question_service.get_available_levels()
                return {"levels": levels}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/get_question_bank")
        def get_question_bank(role: Optional[str] = None, level: Optional[str] = None):
            """Get question bank with optional filtering."""
            try:
                questions = self.question_service.get_question_bank(role, level)
                return {"questions": questions}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Session management endpoints
        @self.app.get("/interview_history")
        def get_interview_history(user_id: str):
            """Get interview history for a user."""
            try:
                if not user_id:
                    raise HTTPException(status_code=400, detail="user_id is required")
                
                result = self.interview_service.get_interview_history(user_id)
                
                if "error" in result:
                    raise HTTPException(status_code=404, detail=result["error"])
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/reset_interview")
        def reset_interview(payload: dict = Body(...)):
            """Reset interview session for a user."""
            try:
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=400, detail="user_id is required")
                
                result = self.interview_service.reset_interview(str(user_id))
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Audio endpoints (placeholder - will integrate with existing audio_utils)
        @self.app.post("/stt")
        async def speech_to_text(audio: UploadFile = File(...), lang: str = Form('es')):
            """Convert speech to text."""
            try:
                # TODO: Integrate with existing audio_utils.py or implement in AudioService
                # For now, return a placeholder response
                return {"text": "Audio transcription not yet implemented in modular architecture"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tts")
        async def text_to_speech(text: str = Form(...), lang: str = Form('es')):
            """Convert text to speech."""
            try:
                # TODO: Integrate with existing audio_utils.py or implement in AudioService
                # For now, return a placeholder response
                raise HTTPException(status_code=501, detail="Text-to-speech not yet implemented in modular architecture")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # LLM management endpoints
        @self.app.get("/llm/status")
        def get_llm_status():
            """Get LLM provider status."""
            try:
                info = self.llm_manager.get_active_provider_info()
                return info
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/llm/test")
        async def test_llm(payload: dict = Body(...)):
            """Test LLM with a simple prompt."""
            try:
                prompt = payload.get("prompt", "Hello, how are you?")
                response = self.llm_manager.generate_text(prompt)
                return {"prompt": prompt, "response": response}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_static_files(self):
        """Setup static file serving for frontend."""
        # Get the static directory from the original app structure
        current_dir = Path(__file__).parent.parent.parent
        static_dir = current_dir / "app" / "static"
        
        if static_dir.exists():
            self.app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')
            
            # Serve frontend at root
            @self.app.get("/")
            def serve_frontend():
                """Serve the frontend HTML."""
                index_file = static_dir / 'index.html'
                if index_file.exists():
                    return FileResponse(str(index_file))
                else:
                    return {"message": "Ready4Hire API", "version": "2.0.0", "docs": "/docs"}
    
    def _setup_middleware(self):
        """Setup middleware for security and logging."""
        
        @self.app.middleware("http")
        async def security_middleware(request: Request, call_next):
            """Basic security middleware."""
            # Add basic security headers
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            return response
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler."""
            print(f"Unhandled exception: {str(exc)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    def _sanitize_input(self, text: str) -> str:
        """Basic input sanitization."""
        import re
        
        # Remove potential dangerous patterns
        forbidden_patterns = [
            r"(?i)ignore previous instructions",
            r"(?i)forget all previous",
            r"(?i)system prompt",
            r"(?i)you are now",
            r"(?i)act as",
        ]
        
        for pattern in forbidden_patterns:
            text = re.sub(pattern, "", text)
        
        # Basic cleanup
        text = re.sub(r'[<>"\'`]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# Create the application instance
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    router = APIRouter()
    return router.get_app()


# For backward compatibility and direct running
app = create_app()