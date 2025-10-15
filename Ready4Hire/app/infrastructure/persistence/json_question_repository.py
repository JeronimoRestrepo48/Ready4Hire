"""
JSON Implementation: Question Repository
Lee preguntas desde archivos JSONL (compatible con estructura actual)
"""
import json
from typing import List, Optional
from pathlib import Path

from app.domain.entities.question import Question
from app.domain.repositories.question_repository import QuestionRepository


class JsonQuestionRepository(QuestionRepository):
    """
    Implementación que lee preguntas desde archivos JSONL.
    Compatible con los archivos actuales tech_questions.jsonl y soft_skills.jsonl
    """
    
    def __init__(self, tech_file: str = "app/datasets/tech_questions.jsonl", 
                 soft_file: str = "app/datasets/soft_skills.jsonl"):
        """
        Inicializa el repositorio con archivos JSONL.
        
        Args:
            tech_file: Ruta al archivo de preguntas técnicas
            soft_file: Ruta al archivo de preguntas de soft skills
        """
        self.tech_file = Path(tech_file)
        self.soft_file = Path(soft_file)
        self._tech_cache: List[Question] = []
        self._soft_cache: List[Question] = []
        self._load_questions()
    
    def _load_questions(self) -> None:
        """Carga preguntas desde archivos JSONL"""
        # Cargar preguntas técnicas
        if self.tech_file.exists():
            with open(self.tech_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        data['category'] = 'technical'
                        self._tech_cache.append(Question.from_dict(data))
        
        # Cargar preguntas de soft skills
        if self.soft_file.exists():
            with open(self.soft_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        data['category'] = 'soft_skills'
                        self._soft_cache.append(Question.from_dict(data))
    
    async def find_by_id(self, question_id: str) -> Optional[Question]:
        """Busca pregunta por ID"""
        all_questions = self._tech_cache + self._soft_cache
        
        for question in all_questions:
            if question.id == question_id:
                return question
        
        return None
    
    async def find_by_role(self, role: str, category: str = "technical") -> List[Question]:
        """Busca preguntas por rol y categoría"""
        cache = self._tech_cache if category == "technical" else self._soft_cache
        role_lower = role.lower()
        
        return [
            q for q in cache 
            if role_lower in q.role.lower() or q.role.lower() in role_lower
        ]
    
    async def find_by_difficulty(self, difficulty: str, category: str = "technical") -> List[Question]:
        """Busca preguntas por dificultad"""
        cache = self._tech_cache if category == "technical" else self._soft_cache
        
        return [q for q in cache if q.difficulty == difficulty]
    
    async def find_all_technical(self) -> List[Question]:
        """Obtiene todas las preguntas técnicas"""
        return self._tech_cache.copy()
    
    async def find_all_soft_skills(self) -> List[Question]:
        """Obtiene todas las preguntas de soft skills"""
        return self._soft_cache.copy()
    
    async def search(self, query: str, category: Optional[str] = None) -> List[Question]:
        """Busca preguntas por texto"""
        query_lower = query.lower()
        
        if category == "technical":
            cache = self._tech_cache
        elif category == "soft_skills":
            cache = self._soft_cache
        else:
            cache = self._tech_cache + self._soft_cache
        
        return [
            q for q in cache 
            if (query_lower in q.text.lower() or 
                query_lower in q.topic.lower() or
                any(query_lower in kw.lower() for kw in q.keywords))
        ]
    
    def reload(self) -> None:
        """Recarga las preguntas desde los archivos"""
        self._tech_cache.clear()
        self._soft_cache.clear()
        self._load_questions()
