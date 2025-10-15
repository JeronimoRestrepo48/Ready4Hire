"""
Entity: Respuesta
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

from app.domain.value_objects.score import Score
from app.domain.value_objects.emotion import Emotion


@dataclass
class Answer:
    """
    Entidad: Respuesta del candidato
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    question_id: str = ""
    text: str = ""
    score: float = 0.0
    is_correct: bool = False
    emotion: Emotion = Emotion.NEUTRAL
    time_taken: int = 0  # segundos
    hints_used: int = 0
    evaluation_details: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_score_object(self) -> Score:
        """Retorna el score como objeto de valor"""
        return Score(self.score)
    
    def was_quick(self, threshold: int = 30) -> bool:
        """Verifica si fue una respuesta rápida"""
        return self.time_taken < threshold
    
    def used_hints(self) -> bool:
        """Verifica si usó pistas"""
        return self.hints_used > 0
    
    def to_dict(self) -> dict:
        """Convierte a diccionario"""
        return {
            'id': self.id,
            'question_id': self.question_id,
            'text': self.text,
            'score': self.score,
            'is_correct': self.is_correct,
            'emotion': self.emotion.value,
            'time_taken': self.time_taken,
            'hints_used': self.hints_used,
            'evaluation_details': self.evaluation_details,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Answer':
        """Crea desde diccionario"""
        return cls(
            id=data.get('id', str(uuid4())),
            question_id=data.get('question_id', ''),
            text=data.get('text', ''),
            score=data.get('score', 0.0),
            is_correct=data.get('is_correct', False),
            emotion=Emotion(data.get('emotion', 'neutral')),
            time_taken=data.get('time_taken', 0),
            hints_used=data.get('hints_used', 0),
            evaluation_details=data.get('evaluation_details', {}),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()
        )
