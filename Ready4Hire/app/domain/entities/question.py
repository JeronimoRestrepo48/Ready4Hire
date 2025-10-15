"""
Entity: Pregunta
"""
from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4


@dataclass
class Question:
    """
    Entidad: Pregunta de entrevista
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    text: str = ""
    category: str = ""  # technical, soft_skills
    difficulty: str = "medium"  # easy, medium, hard
    role: str = ""
    topic: str = ""
    keywords: List[str] = field(default_factory=list)
    expected_concepts: List[str] = field(default_factory=list)
    good_answer_example: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def is_technical(self) -> bool:
        """Verifica si es una pregunta técnica"""
        return self.category == "technical"
    
    def is_soft_skills(self) -> bool:
        """Verifica si es una pregunta de soft skills"""
        return self.category == "soft_skills"
    
    def matches_role(self, role: str) -> bool:
        """Verifica si la pregunta es apropiada para el rol"""
        return self.role.lower() in role.lower() or role.lower() in self.role.lower()
    
    def to_dict(self) -> dict:
        """Convierte a diccionario"""
        return {
            'id': self.id,
            'text': self.text,
            'category': self.category,
            'difficulty': self.difficulty,
            'role': self.role,
            'topic': self.topic,
            'keywords': self.keywords,
            'expected_concepts': self.expected_concepts,
            'good_answer_example': self.good_answer_example,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Question':
        """Crea desde diccionario"""
        return cls(
            id=data.get('id', str(uuid4())),
            text=data.get('question', data.get('text', '')),
            category=data.get('category', 'technical'),
            difficulty=data.get('difficulty', 'medium'),
            role=data.get('role', ''),
            topic=data.get('topic', ''),
            keywords=data.get('keywords', []),
            expected_concepts=data.get('expected_concepts', []),
            good_answer_example=data.get('good_answer', None),
            metadata=data.get('metadata', {})
        )
