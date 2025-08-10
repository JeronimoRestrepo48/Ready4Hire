"""
Data Layer for Ready4Hire
=========================

This module provides data access and model definitions for the Ready4Hire system.
It abstracts data operations and provides a clean interface for the core business logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os


@dataclass
class Question:
    """Data model for interview questions."""
    id: str
    text: str
    role: str
    level: str
    type: str  # "technical" or "soft"
    answer: Optional[str] = None
    expected: Optional[str] = None
    hints: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    tags: Optional[List[str]] = None


@dataclass
class UserSession:
    """Data model for user interview sessions."""
    user_id: str
    role: str
    level: str
    interview_type: str
    mode: str
    stage: str
    context: Dict[str, Any]
    history: List[Dict[str, Any]]
    current_question: Optional[Question] = None
    score: int = 0
    answered_questions: List[str] = None
    
    def __post_init__(self):
        if self.answered_questions is None:
            self.answered_questions = []


@dataclass
class InteractionLog:
    """Data model for interaction logging."""
    user_id: str
    session_id: str
    timestamp: str
    question_id: str
    question_text: str
    user_answer: str
    feedback: str
    score: int
    correct: bool


class DataRepository(ABC):
    """Abstract base class for data repositories."""
    
    @abstractmethod
    def get_questions(self, filters: Dict[str, Any] = None) -> List[Question]:
        """Get questions based on filters."""
        pass
    
    @abstractmethod
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        pass
    
    @abstractmethod
    def save_session(self, session: UserSession) -> bool:
        """Save user session."""
        pass
    
    @abstractmethod
    def get_session(self, user_id: str) -> Optional[UserSession]:
        """Get user session."""
        pass
    
    @abstractmethod
    def delete_session(self, user_id: str) -> bool:
        """Delete user session."""
        pass
    
    @abstractmethod
    def log_interaction(self, interaction: InteractionLog) -> bool:
        """Log user interaction."""
        pass


class FileSystemRepository(DataRepository):
    """File system-based data repository implementation."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize file system repository.
        
        Args:
            data_path: Path to data directory. Defaults to app/datasets if None.
        """
        if data_path is None:
            # Default to the existing app/datasets directory
            current_dir = Path(__file__).parent.parent.parent
            data_path = current_dir / "app" / "datasets"
        
        self.data_path = Path(data_path)
        self.sessions_path = self.data_path / "sessions"
        self.logs_path = self.data_path / "logs"
        
        # Create directories if they don't exist
        self.sessions_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        self._questions_cache = None
    
    def _load_questions_from_file(self, file_path: Path) -> List[Question]:
        """Load questions from a JSONL file."""
        questions = []
        
        if not file_path.exists():
            return questions
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        question = Question(
                            id=data.get('id', f"{file_path.stem}_{line_num}"),
                            text=data.get('question', data.get('text', '')),
                            role=data.get('role', ''),
                            level=data.get('level', ''),
                            type=data.get('type', 'technical'),
                            answer=data.get('answer'),
                            expected=data.get('expected'),
                            hints=data.get('hints', []),
                            resources=data.get('resources', []),
                            tags=data.get('tags', [])
                        )
                        questions.append(question)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error loading questions from {file_path}: {e}")
        
        return questions
    
    def _load_all_questions(self) -> List[Question]:
        """Load all questions from data files."""
        if self._questions_cache is not None:
            return self._questions_cache
        
        questions = []
        
        # Load technical questions
        tech_file = self.data_path / "tech_questions.jsonl"
        questions.extend(self._load_questions_from_file(tech_file))
        
        # Load soft skills questions
        soft_file = self.data_path / "soft_skills.jsonl"
        soft_questions = self._load_questions_from_file(soft_file)
        for q in soft_questions:
            q.type = "soft"
        questions.extend(soft_questions)
        
        self._questions_cache = questions
        return questions
    
    def get_questions(self, filters: Dict[str, Any] = None) -> List[Question]:
        """Get questions based on filters."""
        all_questions = self._load_all_questions()
        
        if not filters:
            return all_questions
        
        filtered_questions = []
        for question in all_questions:
            match = True
            
            for key, value in filters.items():
                if key == "role" and question.role.lower() != value.lower():
                    match = False
                    break
                elif key == "level" and question.level.lower() != value.lower():
                    match = False
                    break
                elif key == "type" and question.type.lower() != value.lower():
                    match = False
                    break
                elif key == "tags" and value not in (question.tags or []):
                    match = False
                    break
            
            if match:
                filtered_questions.append(question)
        
        return filtered_questions
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        all_questions = self._load_all_questions()
        
        for question in all_questions:
            if question.id == question_id:
                return question
        
        return None
    
    def save_session(self, session: UserSession) -> bool:
        """Save user session to file."""
        try:
            session_file = self.sessions_path / f"{session.user_id}.json"
            
            # Convert dataclass to dict
            session_data = asdict(session)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving session for {session.user_id}: {e}")
            return False
    
    def get_session(self, user_id: str) -> Optional[UserSession]:
        """Get user session from file."""
        try:
            session_file = self.sessions_path / f"{user_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Convert dict back to dataclass
            session = UserSession(**session_data)
            return session
            
        except Exception as e:
            print(f"Error loading session for {user_id}: {e}")
            return None
    
    def delete_session(self, user_id: str) -> bool:
        """Delete user session file."""
        try:
            session_file = self.sessions_path / f"{user_id}.json"
            
            if session_file.exists():
                session_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error deleting session for {user_id}: {e}")
            return False
    
    def log_interaction(self, interaction: InteractionLog) -> bool:
        """Log user interaction to file."""
        try:
            log_file = self.logs_path / "interactions.jsonl"
            
            # Convert dataclass to dict
            interaction_data = asdict(interaction)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(interaction_data, f, ensure_ascii=False)
                f.write('\n')
            
            return True
        except Exception as e:
            print(f"Error logging interaction: {e}")
            return False


class DatabaseRepository(DataRepository):
    """Database-based data repository implementation (placeholder for future SQL support)."""
    
    def __init__(self, connection_string: str):
        """
        Initialize database repository.
        
        Args:
            connection_string: Database connection string.
        """
        self.connection_string = connection_string
        # TODO: Implement database connection and operations
        raise NotImplementedError("Database repository not yet implemented")
    
    def get_questions(self, filters: Dict[str, Any] = None) -> List[Question]:
        # TODO: Implement database query
        raise NotImplementedError("Database repository not yet implemented")
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        # TODO: Implement database query
        raise NotImplementedError("Database repository not yet implemented")
    
    def save_session(self, session: UserSession) -> bool:
        # TODO: Implement database save
        raise NotImplementedError("Database repository not yet implemented")
    
    def get_session(self, user_id: str) -> Optional[UserSession]:
        # TODO: Implement database query
        raise NotImplementedError("Database repository not yet implemented")
    
    def delete_session(self, user_id: str) -> bool:
        # TODO: Implement database delete
        raise NotImplementedError("Database repository not yet implemented")
    
    def log_interaction(self, interaction: InteractionLog) -> bool:
        # TODO: Implement database insert
        raise NotImplementedError("Database repository not yet implemented")


def create_repository_from_env() -> DataRepository:
    """Create data repository based on environment configuration."""
    repo_type = os.getenv("DATA_REPOSITORY", "filesystem").lower()
    
    if repo_type == "filesystem":
        data_path = os.getenv("DATA_PATH")
        return FileSystemRepository(data_path)
    elif repo_type == "database":
        connection_string = os.getenv("DATABASE_URL")
        if not connection_string:
            raise ValueError("DATABASE_URL environment variable is required for database repository")
        return DatabaseRepository(connection_string)
    else:
        raise ValueError(f"Unsupported repository type: {repo_type}")