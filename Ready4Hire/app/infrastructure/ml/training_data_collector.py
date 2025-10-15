"""
Training Data Collector - Recopila evaluaciones para crear dataset de fine-tuning.

Este módulo captura cada evaluación realizada y la almacena en formato
apropiado para entrenar el modelo LLM.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """
    Ejemplo de entrenamiento para fine-tuning.
    
    Representa una evaluación completa que puede usarse para entrenar el modelo.
    """
    # Input
    question: str
    answer: str
    expected_concepts: List[str]
    keywords: List[str]
    category: str  # technical, soft_skills
    difficulty: str  # junior, mid, senior
    role: str
    
    # Output (evaluación correcta/esperada)
    score: float
    breakdown: Dict[str, float]
    justification: str
    strengths: List[str]
    improvements: List[str]
    concepts_covered: List[str]
    missing_concepts: List[str]
    
    # Metadata
    example_id: str
    created_at: str
    model_used: str
    evaluation_source: str  # "llm", "human", "heuristic"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        """Crea desde diccionario."""
        return cls(**data)


class TrainingDataCollector:
    """
    Recopila datos de evaluaciones para crear dataset de fine-tuning.
    
    Características:
    - Almacena evaluaciones en JSONL
    - Filtra ejemplos de baja calidad
    - Detecta duplicados
    - Balancea por categoría/dificultad
    """
    
    def __init__(
        self,
        storage_path: str = "data/training/evaluations.jsonl",
        min_score_threshold: float = 0.0,  # Aceptar todos los scores
        collect_human_labels: bool = True,
        collect_llm_labels: bool = True,
        collect_heuristic_labels: bool = False  # Heurísticos suelen ser de baja calidad
    ):
        """
        Inicializa el collector.
        
        Args:
            storage_path: Ruta donde almacenar datos de entrenamiento
            min_score_threshold: Score mínimo para incluir ejemplo (0-10)
            collect_human_labels: Recopilar evaluaciones humanas
            collect_llm_labels: Recopilar evaluaciones de LLM
            collect_heuristic_labels: Recopilar evaluaciones heurísticas
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.min_score_threshold = min_score_threshold
        self.collect_human_labels = collect_human_labels
        self.collect_llm_labels = collect_llm_labels
        self.collect_heuristic_labels = collect_heuristic_labels
        
        # Cache de IDs vistos (para detectar duplicados)
        self._seen_ids = self._load_seen_ids()
        
        logger.info(f"TrainingDataCollector initialized: {storage_path}")
    
    def _load_seen_ids(self) -> set:
        """Carga IDs de ejemplos ya almacenados para detectar duplicados."""
        seen = set()
        if self.storage_path.exists():
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        example = json.loads(line)
                        seen.add(example['example_id'])
                    except Exception as e:
                        logger.warning(f"Error loading seen ID: {e}")
        return seen
    
    def _generate_example_id(
        self,
        question: str,
        answer: str,
        model: str
    ) -> str:
        """
        Genera ID único para un ejemplo.
        
        Usa hash de pregunta + respuesta + modelo para detectar duplicados.
        """
        content = f"{question}::{answer}::{model}".lower()
        return hashlib.md5(content.encode()).hexdigest()
    
    def should_collect(self, evaluation_result: Dict[str, Any]) -> bool:
        """
        Determina si una evaluación debe ser recopilada.
        
        Criterios:
        - Score >= min_threshold
        - Fuente permitida (human/llm/heuristic)
        - No es duplicado
        """
        # Verificar fuente
        source = evaluation_result.get('evaluation_source', 'llm')
        if source == 'human' and not self.collect_human_labels:
            return False
        if source == 'llm' and not self.collect_llm_labels:
            return False
        if source == 'heuristic' and not self.collect_heuristic_labels:
            return False
        
        # Verificar score
        score = evaluation_result.get('score', 0.0)
        if score < self.min_score_threshold:
            return False
        
        # Verificar duplicados
        example_id = self._generate_example_id(
            evaluation_result.get('question', ''),
            evaluation_result.get('answer', ''),
            evaluation_result.get('model', 'unknown')
        )
        if example_id in self._seen_ids:
            return False
        
        return True
    
    def collect(
        self,
        question: str,
        answer: str,
        evaluation_result: Dict[str, Any],
        expected_concepts: List[str],
        keywords: List[str],
        category: str,
        difficulty: str,
        role: str,
        evaluation_source: str = "llm"
    ) -> Optional[TrainingExample]:
        """
        Recopila una evaluación como ejemplo de entrenamiento.
        
        Args:
            question: Pregunta realizada
            answer: Respuesta del candidato
            evaluation_result: Resultado de la evaluación (score, breakdown, etc.)
            expected_concepts: Conceptos esperados
            keywords: Palabras clave
            category: Categoría (technical, soft_skills)
            difficulty: Dificultad (junior, mid, senior)
            role: Rol del candidato
            evaluation_source: Fuente ("human", "llm", "heuristic")
        
        Returns:
            TrainingExample si fue recopilado, None si fue rechazado
        """
        # Agregar source al resultado
        evaluation_result['evaluation_source'] = evaluation_source
        evaluation_result['question'] = question
        evaluation_result['answer'] = answer
        evaluation_result['model'] = evaluation_result.get('model', 'unknown')
        
        # Verificar si debe ser recopilado
        if not self.should_collect(evaluation_result):
            return None
        
        # Crear ejemplo
        example_id = self._generate_example_id(question, answer, evaluation_result['model'])
        
        example = TrainingExample(
            # Input
            question=question,
            answer=answer,
            expected_concepts=expected_concepts,
            keywords=keywords,
            category=category,
            difficulty=difficulty,
            role=role,
            
            # Output
            score=evaluation_result.get('score', 0.0),
            breakdown=evaluation_result.get('breakdown', {}),
            justification=evaluation_result.get('justification', ''),
            strengths=evaluation_result.get('strengths', []),
            improvements=evaluation_result.get('improvements', []),
            concepts_covered=evaluation_result.get('concepts_covered', []),
            missing_concepts=evaluation_result.get('missing_concepts', []),
            
            # Metadata
            example_id=example_id,
            created_at=datetime.now().isoformat(),
            model_used=evaluation_result['model'],
            evaluation_source=evaluation_source
        )
        
        # Almacenar
        self._store_example(example)
        self._seen_ids.add(example_id)
        
        logger.debug(f"Collected training example: {example_id[:8]}...")
        return example
    
    def _store_example(self, example: TrainingExample):
        """Almacena ejemplo en JSONL."""
        try:
            with open(self.storage_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error storing training example: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del dataset recopilado.
        
        Returns:
            Dict con total, por categoría, por dificultad, etc.
        """
        stats = {
            'total_examples': 0,
            'by_category': {},
            'by_difficulty': {},
            'by_source': {},
            'by_role': {},
            'score_distribution': {
                '0-3': 0,
                '3-5': 0,
                '5-7': 0,
                '7-9': 0,
                '9-10': 0
            }
        }
        
        if not self.storage_path.exists():
            return stats
        
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    stats['total_examples'] += 1
                    
                    # Por categoría
                    cat = example.get('category', 'unknown')
                    stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
                    
                    # Por dificultad
                    diff = example.get('difficulty', 'unknown')
                    stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1
                    
                    # Por fuente
                    source = example.get('evaluation_source', 'unknown')
                    stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
                    
                    # Por rol
                    role = example.get('role', 'unknown')
                    stats['by_role'][role] = stats['by_role'].get(role, 0) + 1
                    
                    # Distribución de scores
                    score = example.get('score', 0.0)
                    if score < 3:
                        stats['score_distribution']['0-3'] += 1
                    elif score < 5:
                        stats['score_distribution']['3-5'] += 1
                    elif score < 7:
                        stats['score_distribution']['5-7'] += 1
                    elif score < 9:
                        stats['score_distribution']['7-9'] += 1
                    else:
                        stats['score_distribution']['9-10'] += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing example for stats: {e}")
        
        return stats
    
    def load_all_examples(self) -> List[TrainingExample]:
        """
        Carga todos los ejemplos almacenados.
        
        Returns:
            Lista de TrainingExample
        """
        examples = []
        
        if not self.storage_path.exists():
            return examples
        
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    example = TrainingExample.from_dict(data)
                    examples.append(example)
                except Exception as e:
                    logger.warning(f"Error loading example: {e}")
        
        return examples
    
    def clear(self):
        """Limpia todos los datos recopilados."""
        if self.storage_path.exists():
            self.storage_path.unlink()
        self._seen_ids.clear()
        logger.info("Training data cleared")
