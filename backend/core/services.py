"""
Core Business Logic for Ready4Hire
==================================

This module contains the core business logic for the interview simulation system.
It orchestrates the interaction between the data layer and LLM layer.
"""

from typing import Dict, Any, List, Optional
import time
import uuid
from datetime import datetime

from backend.data.repository import DataRepository, Question, UserSession, InteractionLog
from backend.llm.providers import LLMManager


class InterviewService:
    """Core service for managing interview sessions."""
    
    def __init__(self, repository: DataRepository, llm_manager: LLMManager):
        """
        Initialize interview service.
        
        Args:
            repository: Data repository instance
            llm_manager: LLM manager instance
        """
        self.repository = repository
        self.llm_manager = llm_manager
        self.sessions = {}  # In-memory session cache
    
    def start_interview(self, user_id: str, role: str, interview_type: str, mode: str = "practice") -> Dict[str, Any]:
        """
        Start a new interview session.
        
        Args:
            user_id: Unique user identifier
            role: Target role (e.g., "ia", "devops")
            interview_type: Type of interview ("technical" or "soft")
            mode: Interview mode ("practice" or "exam")
            
        Returns:
            Dict containing the first question and session state
        """
        # Create new session
        session = UserSession(
            user_id=user_id,
            role=role,
            level="",  # Will be determined through context questions
            interview_type=interview_type,
            mode=mode,
            stage="context",
            context={},
            history=[],
            answered_questions=[]
        )
        
        # Save session
        self.sessions[user_id] = session
        self.repository.save_session(session)
        
        # Start with context questions
        context_questions = [
            "¿Cuál es tu rol objetivo o especialización?",
            "¿Cuál es tu nivel de experiencia? (junior, semi-senior, senior)",
            "¿Cuántos años de experiencia tienes?",
            "¿Qué conocimientos o tecnologías dominas? (separados por comas)",
            "¿Qué herramientas utilizas frecuentemente? (separados por comas)",
            "¿Tienes alguna expectativa específica para esta entrevista?"
        ]
        
        session.context["questions"] = context_questions
        session.context["asked"] = 0
        
        return {
            "question": context_questions[0],
            "session_state": {
                "stage": session.stage,
                "progress": f"1/{len(context_questions)}"
            }
        }
    
    def next_question(self, user_id: str) -> Dict[str, Any]:
        """
        Get the next question for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing the next question or interview completion info
        """
        session = self._get_session(user_id)
        if not session:
            return {"error": "No active session found"}
        
        if session.stage == "context":
            return self._handle_context_stage(session)
        elif session.stage == "interview":
            return self._get_next_interview_question(session)
        elif session.stage == "completed":
            return {"message": "Interview completed", "stage": "completed"}
        else:
            return {"error": "Invalid session stage"}
    
    def process_answer(self, user_id: str, answer: str) -> Dict[str, Any]:
        """
        Process user's answer and provide feedback.
        
        Args:
            user_id: User identifier
            answer: User's answer
            
        Returns:
            Dict containing feedback and next steps
        """
        session = self._get_session(user_id)
        if not session:
            return {"error": "No active session found"}
        
        if session.stage == "context":
            return self._process_context_answer(session, answer)
        elif session.stage == "interview":
            return self._process_interview_answer(session, answer)
        else:
            return {"error": "Invalid session stage"}
    
    def end_interview(self, user_id: str) -> Dict[str, Any]:
        """
        End the interview and provide summary.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing interview summary
        """
        session = self._get_session(user_id)
        if not session:
            return {"error": "No active session found"}
        
        session.stage = "completed"
        self.repository.save_session(session)
        
        # Generate summary
        summary = self._generate_interview_summary(session)
        
        return {
            "summary": summary,
            "score": session.score,
            "questions_answered": len(session.answered_questions),
            "session_state": {"stage": "completed"}
        }
    
    def get_interview_history(self, user_id: str) -> Dict[str, Any]:
        """Get interview history for a user."""
        session = self._get_session(user_id)
        if not session:
            return {"error": "No active session found"}
        
        return {"history": session.history}
    
    def reset_interview(self, user_id: str) -> Dict[str, Any]:
        """Reset user's interview session."""
        if user_id in self.sessions:
            del self.sessions[user_id]
        
        self.repository.delete_session(user_id)
        return {"message": "Interview reset successfully"}
    
    def _get_session(self, user_id: str) -> Optional[UserSession]:
        """Get session from cache or repository."""
        if user_id in self.sessions:
            return self.sessions[user_id]
        
        session = self.repository.get_session(user_id)
        if session:
            self.sessions[user_id] = session
        
        return session
    
    def _handle_context_stage(self, session: UserSession) -> Dict[str, Any]:
        """Handle context gathering stage."""
        questions = session.context.get("questions", [])
        asked = session.context.get("asked", 0)
        
        if asked >= len(questions):
            # Context gathering complete, start interview
            session.stage = "interview"
            self.repository.save_session(session)
            return self._get_next_interview_question(session)
        
        return {
            "question": questions[asked],
            "session_state": {
                "stage": session.stage,
                "progress": f"{asked + 1}/{len(questions)}"
            }
        }
    
    def _process_context_answer(self, session: UserSession, answer: str) -> Dict[str, Any]:
        """Process context question answer."""
        questions = session.context.get("questions", [])
        asked = session.context.get("asked", 0)
        
        # Store context answer
        if asked == 0:
            session.role = answer.strip()
        elif asked == 1:
            session.level = answer.strip()
        elif asked == 2:
            try:
                session.context["years"] = int(answer.strip())
            except ValueError:
                session.context["years"] = answer.strip()
        elif asked == 3:
            session.context["knowledge"] = [k.strip() for k in answer.split(",") if k.strip()]
        elif asked == 4:
            session.context["tools"] = [t.strip() for t in answer.split(",") if t.strip()]
        elif asked == 5:
            session.context["expectations"] = answer.strip()
        
        # Move to next context question
        session.context["asked"] = asked + 1
        self.repository.save_session(session)
        
        # Get next question
        return self.next_question(session.user_id)
    
    def _get_next_interview_question(self, session: UserSession) -> Dict[str, Any]:
        """Get the next interview question based on session context."""
        # Filter questions based on session context
        filters = {
            "type": session.interview_type,
            "role": session.role,
            "level": session.level
        }
        
        available_questions = self.repository.get_questions(filters)
        
        # Remove already answered questions
        available_questions = [
            q for q in available_questions 
            if q.id not in session.answered_questions
        ]
        
        if not available_questions:
            # No more questions, end interview
            session.stage = "completed"
            self.repository.save_session(session)
            return {"message": "No more questions available", "stage": "completed"}
        
        # Select next question (simple selection for now)
        next_question = available_questions[0]
        session.current_question = next_question
        self.repository.save_session(session)
        
        return {
            "question": next_question.text,
            "question_id": next_question.id,
            "type": next_question.type,
            "session_state": {
                "stage": session.stage,
                "questions_remaining": len(available_questions) - 1
            }
        }
    
    def _process_interview_answer(self, session: UserSession, answer: str) -> Dict[str, Any]:
        """Process interview question answer."""
        if not session.current_question:
            return {"error": "No current question"}
        
        question = session.current_question
        
        # Evaluate answer using LLM
        feedback, score, is_correct = self._evaluate_answer(question, answer)
        
        # Log interaction
        interaction = InteractionLog(
            user_id=session.user_id,
            session_id=session.user_id,  # Using user_id as session_id for now
            timestamp=datetime.now().isoformat(),
            question_id=question.id,
            question_text=question.text,
            user_answer=answer,
            feedback=feedback,
            score=score,
            correct=is_correct
        )
        self.repository.log_interaction(interaction)
        
        # Update session
        session.answered_questions.append(question.id)
        session.score += score
        session.history.append({
            "question": question.text,
            "answer": answer,
            "feedback": feedback,
            "score": score,
            "timestamp": interaction.timestamp
        })
        
        # Generate hint if answer is incorrect and in practice mode
        hint = None
        if not is_correct and session.mode == "practice":
            hint = self._generate_hint(question, answer)
        
        self.repository.save_session(session)
        
        response = {
            "feedback": feedback,
            "score": score,
            "correct": is_correct,
            "session_state": {
                "stage": session.stage,
                "total_score": session.score,
                "questions_answered": len(session.answered_questions)
            }
        }
        
        if hint:
            response["hint"] = hint
        
        return response
    
    def _evaluate_answer(self, question: Question, answer: str) -> tuple[str, int, bool]:
        """Evaluate user's answer using LLM."""
        prompt = f"""
        Evalúa la siguiente respuesta a una pregunta de entrevista:
        
        Pregunta: {question.text}
        Respuesta del usuario: {answer}
        Respuesta esperada: {question.expected or question.answer or "No especificada"}
        
        Proporciona:
        1. Feedback constructivo y motivador (máximo 200 palabras)
        2. Puntuación de 0-10
        3. ¿Es correcta? (Sí/No)
        
        Formato de respuesta:
        FEEDBACK: [tu feedback aquí]
        PUNTUACION: [0-10]
        CORRECTA: [Si/No]
        """
        
        try:
            response = self.llm_manager.generate_text(prompt)
            
            # Parse LLM response
            feedback = self._extract_section(response, "FEEDBACK")
            score_str = self._extract_section(response, "PUNTUACION")
            correct_str = self._extract_section(response, "CORRECTA")
            
            score = int(score_str) if score_str.isdigit() else 5
            is_correct = correct_str.lower() in ["si", "sí", "yes", "true"]
            
            return feedback, score, is_correct
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            # Fallback evaluation
            return "Respuesta recibida. Continúa con la siguiente pregunta.", 5, True
    
    def _generate_hint(self, question: Question, answer: str) -> str:
        """Generate a hint for incorrect answers."""
        if question.hints:
            return question.hints[0]  # Return first available hint
        
        prompt = f"""
        Genera una pista útil para esta pregunta de entrevista:
        
        Pregunta: {question.text}
        Respuesta incorrecta del usuario: {answer}
        Respuesta esperada: {question.expected or question.answer or "No especificada"}
        
        La pista debe:
        - Ser educativa pero no dar la respuesta directamente
        - Guiar al usuario hacia la respuesta correcta
        - Ser motivadora y constructiva
        - Máximo 100 palabras
        
        Pista:
        """
        
        try:
            hint = self.llm_manager.generate_text(prompt)
            return hint.strip()
        except Exception as e:
            print(f"Error generating hint: {e}")
            return "Piensa en los conceptos fundamentales relacionados con esta pregunta."
    
    def _generate_interview_summary(self, session: UserSession) -> str:
        """Generate interview summary using LLM."""
        history_text = "\n".join([
            f"P: {item['question']}\nR: {item['answer']}\nFeedback: {item['feedback']}\nPuntuación: {item['score']}"
            for item in session.history[-5:]  # Last 5 questions
        ])
        
        prompt = f"""
        Genera un resumen de entrevista para un candidato:
        
        Rol objetivo: {session.role}
        Nivel: {session.level}
        Tipo de entrevista: {session.interview_type}
        Preguntas respondidas: {len(session.answered_questions)}
        Puntuación total: {session.score}
        
        Últimas interacciones:
        {history_text}
        
        Proporciona un resumen que incluya:
        1. Fortalezas identificadas
        2. Áreas de mejora
        3. Recomendaciones específicas
        4. Próximos pasos sugeridos
        
        Máximo 300 palabras, tono profesional y constructivo.
        """
        
        try:
            summary = self.llm_manager.generate_text(prompt)
            return summary.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Entrevista completada. Preguntas respondidas: {len(session.answered_questions)}, Puntuación total: {session.score}"
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from LLM response."""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(f"{section_name}:"):
                return line.split(':', 1)[1].strip()
        return ""


class QuestionService:
    """Service for managing questions."""
    
    def __init__(self, repository: DataRepository):
        """
        Initialize question service.
        
        Args:
            repository: Data repository instance
        """
        self.repository = repository
    
    def get_available_roles(self) -> List[str]:
        """Get list of available roles."""
        questions = self.repository.get_questions()
        roles = list(set(q.role for q in questions if q.role))
        return sorted(roles)
    
    def get_available_levels(self) -> List[str]:
        """Get list of available levels."""
        questions = self.repository.get_questions()
        levels = list(set(q.level for q in questions if q.level))
        return sorted(levels)
    
    def get_question_bank(self, role: str = None, level: str = None) -> List[Dict[str, Any]]:
        """Get question bank with optional filtering."""
        filters = {}
        if role:
            filters["role"] = role
        if level:
            filters["level"] = level
        
        questions = self.repository.get_questions(filters)
        
        return [
            {
                "id": q.id,
                "text": q.text,
                "role": q.role,
                "level": q.level,
                "type": q.type,
                "tags": q.tags
            }
            for q in questions
        ]


class AudioService:
    """Service for audio processing (placeholder for future implementation)."""
    
    def __init__(self):
        """Initialize audio service."""
        pass
    
    def transcribe_audio(self, audio_file, language: str = "es") -> str:
        """Transcribe audio to text."""
        # TODO: Implement audio transcription
        # This would integrate with the existing audio_utils.py
        raise NotImplementedError("Audio transcription not yet implemented in modular architecture")
    
    def synthesize_text(self, text: str, language: str = "es") -> str:
        """Synthesize text to audio."""
        # TODO: Implement text-to-speech
        # This would integrate with the existing audio_utils.py
        raise NotImplementedError("Text-to-speech not yet implemented in modular architecture")