"""
Dataset Generator - Convierte datos de entrenamiento a formato Ollama/Alpaca.

Genera archivos JSONL listos para fine-tuning con Ollama.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from .training_data_collector import TrainingExample, TrainingDataCollector

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Genera datasets en formato Alpaca/Ollama para fine-tuning.
    
    Formato Alpaca:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """
    
    def __init__(self, training_data_collector: TrainingDataCollector):
        """
        Inicializa el generador.
        
        Args:
            training_data_collector: Collector con datos de entrenamiento
        """
        self.collector = training_data_collector
    
    def _build_instruction(self, example: TrainingExample) -> str:
        """
        Construye la instrucción (system prompt) para el ejemplo.
        """
        return f"""Eres un experto evaluador de entrevistas técnicas y de habilidades blandas.

Tu tarea es evaluar la respuesta de un candidato considerando:
1. Completitud (0-3): ¿Aborda todos los aspectos?
2. Profundidad técnica (0-3): ¿Demuestra comprensión profunda?
3. Claridad (0-2): ¿Explicación clara y estructurada?
4. Conceptos clave (0-2): ¿Menciona conceptos esperados?

Contexto:
- Rol: {example.role}
- Categoría: {example.category}
- Dificultad: {example.difficulty}"""
    
    def _build_input(self, example: TrainingExample) -> str:
        """
        Construye el input (pregunta + respuesta + contexto).
        """
        return f"""**Pregunta:**
{example.question}

**Respuesta del candidato:**
{example.answer}

**Conceptos esperados:**
{', '.join(example.expected_concepts) if example.expected_concepts else 'No especificados'}

**Palabras clave relevantes:**
{', '.join(example.keywords) if example.keywords else 'No especificadas'}

Evalúa esta respuesta en formato JSON."""
    
    def _build_output(self, example: TrainingExample) -> str:
        """
        Construye el output esperado (evaluación en JSON).
        """
        output = {
            "score": example.score,
            "breakdown": example.breakdown,
            "justification": example.justification,
            "strengths": example.strengths,
            "improvements": example.improvements,
            "concepts_covered": example.concepts_covered,
            "missing_concepts": example.missing_concepts
        }
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def example_to_alpaca(self, example: TrainingExample) -> Dict[str, str]:
        """
        Convierte un TrainingExample al formato Alpaca.
        
        Returns:
            Dict con keys: instruction, input, output
        """
        return {
            "instruction": self._build_instruction(example),
            "input": self._build_input(example),
            "output": self._build_output(example)
        }
    
    def generate_dataset(
        self,
        output_path: str | Path = "data/training/ready4hire_dataset.jsonl",
        train_split: float = 0.8,
        filter_low_quality: bool = True,
        min_score_quality: float = 3.0,
        balance_categories: bool = True
    ) -> Dict[str, Any]:
        """
        Genera dataset completo para fine-tuning.
        
        Args:
            output_path: Ruta de salida del dataset
            train_split: Fracción para train (resto será validation)
            filter_low_quality: Filtrar ejemplos de baja calidad
            min_score_quality: Score mínimo para considerar alta calidad
            balance_categories: Balancear ejemplos por categoría
        
        Returns:
            Dict con estadísticas del dataset generado
        """
        logger.info("Generating dataset...")
        
        # Cargar todos los ejemplos
        examples = self.collector.load_all_examples()
        
        if not examples:
            logger.warning("No training examples found!")
            return {
                "total_examples": 0,
                "train_size": 0,
                "val_size": 0,
                "filtered_out": 0,
                "train_path": None,
                "validation_path": None
            }
        
        # Filtrar baja calidad si se solicita
        if filter_low_quality:
            examples = [e for e in examples if e.score >= min_score_quality]
            logger.info(f"Filtered to {len(examples)} high-quality examples (score >= {min_score_quality})")
        
        # Balancear categorías si se solicita
        if balance_categories:
            examples = self._balance_by_category(examples)
            logger.info(f"Balanced to {len(examples)} examples across categories")
        
        # Convertir a formato Alpaca
        alpaca_examples = [self.example_to_alpaca(e) for e in examples]
        
        # Split train/validation
        split_idx = int(len(alpaca_examples) * train_split)
        train_examples = alpaca_examples[:split_idx]
        val_examples = alpaca_examples[split_idx:]
        
        # Guardar dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Train set
        train_path = output_path.parent / f"{output_path.stem}_train.jsonl"
        self._save_jsonl(train_examples, train_path)
        
        # Validation set
        val_path = output_path.parent / f"{output_path.stem}_val.jsonl"
        self._save_jsonl(val_examples, val_path)
        
        # Estadísticas
        original_count = len(self.collector.load_all_examples())
        filtered_out = original_count - len(examples)
        
        stats = {
            "total_examples": len(examples),
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "filtered_out": filtered_out,
            "train_path": str(train_path),
            "validation_path": str(val_path),
            "train_split": train_split,
            "filtered_low_quality": filter_low_quality,
            "min_score_quality": min_score_quality if filter_low_quality else None,
            "balanced_categories": balance_categories
        }
        
        logger.info(f"Dataset generated: {stats['total_examples']} examples")
        logger.info(f"  Train: {stats['train_size']} examples ({train_path})")
        logger.info(f"  Val: {stats['val_size']} examples ({val_path})")
        
        return stats
    
    def _balance_by_category(
        self,
        examples: List[TrainingExample],
        max_per_category: Optional[int] = None
    ) -> List[TrainingExample]:
        """
        Balancea ejemplos por categoría.
        
        Si una categoría tiene muchos más ejemplos que otra, los limita
        para evitar bias del modelo hacia esa categoría.
        """
        # Agrupar por categoría
        by_category = {}
        for example in examples:
            cat = example.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(example)
        
        # Determinar tamaño target (mínimo o max_per_category)
        if max_per_category is None:
            target_size = min(len(examples) for examples in by_category.values())
        else:
            target_size = max_per_category
        
        # Limitar cada categoría al target size
        balanced = []
        for cat, cat_examples in by_category.items():
            balanced.extend(cat_examples[:target_size])
            logger.info(f"Category '{cat}': {len(cat_examples)} → {min(len(cat_examples), target_size)} examples")
        
        return balanced
    
    def _save_jsonl(self, examples: List[Dict[str, str]], path: Path):
        """Guarda ejemplos en formato JSONL."""
        with open(path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    def preview_example(self, example_idx: int = 0) -> Optional[str]:
        """
        Muestra un ejemplo del dataset en formato human-readable.
        
        Args:
            example_idx: Índice del ejemplo a mostrar
        
        Returns:
            String formateado con el ejemplo
        """
        examples = self.collector.load_all_examples()
        
        if not examples or example_idx >= len(examples):
            return None
        
        example = examples[example_idx]
        alpaca = self.example_to_alpaca(example)
        
        preview = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      DATASET EXAMPLE PREVIEW                         ║
╚══════════════════════════════════════════════════════════════════════╝

📋 METADATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example ID: {example.example_id}
Category: {example.category}
Difficulty: {example.difficulty}
Role: {example.role}
Score: {example.score}/10
Source: {example.evaluation_source}

📝 INSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{alpaca['instruction']}

💬 INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{alpaca['input']}

✅ OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{alpaca['output']}

"""
        return preview
