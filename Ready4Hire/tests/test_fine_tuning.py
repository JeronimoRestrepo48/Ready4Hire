"""
Tests para el sistema de fine-tuning (v2.2).

Prueba los tres componentes principales:
1. TrainingDataCollector - Recopilación de datos de evaluaciones
2. DatasetGenerator - Conversión a formato Alpaca para Ollama
3. ModelFineTuner - Gestión del proceso de fine-tuning

Tests diseñados para validar:
- Recopilación automática de datos desde EvaluationService
- Generación de datasets en formato JSONL/Alpaca
- Creación de Modelfiles para Ollama
- Filtrado de calidad y balanceo de categorías
"""
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from app.infrastructure.ml.training_data_collector import (
    TrainingDataCollector,
    TrainingExample
)
from app.infrastructure.ml.dataset_generator import DatasetGenerator
from app.infrastructure.ml.model_finetuner import ModelFineTuner
from app.application.services.evaluation_service import EvaluationService


# =========================================================================
# FIXTURES
# =========================================================================

@pytest.fixture
def temp_storage():
    """Crea directorio temporal para tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def training_collector(temp_storage):
    """Collector de datos con almacenamiento temporal."""
    storage_path = temp_storage / "evaluations.jsonl"
    return TrainingDataCollector(
        storage_path=storage_path,
        min_score_threshold=3.0
    )


@pytest.fixture
def sample_evaluation() -> Dict[str, Any]:
    """Evaluación de ejemplo para tests."""
    return {
        "score": 8.5,
        "breakdown": {
            "completeness": 2.8,
            "technical_depth": 2.7,
            "clarity": 1.8,
            "key_concepts": 1.2
        },
        "breakdown_confidence": 85.0,
        "justification": "Excelente explicación de conceptos de POO con ejemplos concretos. Menciona herencia, polimorfismo y encapsulación correctamente.",
        "strengths": [
            "Explica herencia con ejemplo claro de clases Persona -> Empleado",
            "Menciona polimorfismo y su utilidad práctica"
        ],
        "improvements": [
            "Podría profundizar en el concepto de abstracción",
            "Agregar ejemplo de composición vs herencia"
        ],
        "concepts_covered": ["herencia", "polimorfismo", "encapsulación"],
        "missing_concepts": ["abstracción"],
        "evaluated_at": "2024-01-15T10:30:00",
        "model": "llama3.2:3b"
    }


@pytest.fixture
def sample_question_data():
    """Datos de pregunta para tests."""
    return {
        "question": "¿Qué es la Programación Orientada a Objetos y cuáles son sus pilares fundamentales?",
        "answer": "POO es un paradigma que organiza el código en objetos. Los pilares son herencia (reutilización), polimorfismo (múltiples formas), encapsulación (ocultar detalles) y abstracción (simplificación).",
        "expected_concepts": ["herencia", "polimorfismo", "encapsulación", "abstracción"],
        "keywords": ["POO", "objetos", "clases"],
        "category": "technical",
        "difficulty": "mid",
        "role": "Backend Developer"
    }


# =========================================================================
# TESTS: TrainingDataCollector
# =========================================================================

class TestTrainingDataCollector:
    """Tests para TrainingDataCollector."""
    
    def test_collect_valid_example(self, training_collector, sample_evaluation, sample_question_data):
        """Test: Recopila ejemplo válido exitosamente."""
        example = training_collector.collect(
            question=sample_question_data["question"],
            answer=sample_question_data["answer"],
            evaluation_result=sample_evaluation,
            expected_concepts=sample_question_data["expected_concepts"],
            keywords=sample_question_data["keywords"],
            category=sample_question_data["category"],
            difficulty=sample_question_data["difficulty"],
            role=sample_question_data["role"]
        )
        
        assert example is not None
        assert isinstance(example, TrainingExample)
        assert example.score == 8.5
        assert example.category == "technical"
        assert example.difficulty == "mid"
        assert len(example.concepts_covered) == 3
        assert len(example.missing_concepts) == 1
    
    def test_reject_low_score_example(self, training_collector, sample_evaluation, sample_question_data):
        """Test: Rechaza ejemplos con score bajo (< min_threshold)."""
        low_score_eval = sample_evaluation.copy()
        low_score_eval["score"] = 2.0  # Por debajo del threshold (3.0)
        
        example = training_collector.collect(
            question=sample_question_data["question"],
            answer=sample_question_data["answer"],
            evaluation_result=low_score_eval,
            expected_concepts=sample_question_data["expected_concepts"],
            keywords=sample_question_data["keywords"],
            category=sample_question_data["category"],
            difficulty=sample_question_data["difficulty"],
            role=sample_question_data["role"]
        )
        
        assert example is None  # No se recopila
    
    def test_detect_duplicate_examples(self, training_collector, sample_evaluation, sample_question_data):
        """Test: Detecta y rechaza ejemplos duplicados."""
        # Recopilar primera vez
        example1 = training_collector.collect(
            question=sample_question_data["question"],
            answer=sample_question_data["answer"],
            evaluation_result=sample_evaluation,
            expected_concepts=sample_question_data["expected_concepts"],
            keywords=sample_question_data["keywords"],
            category=sample_question_data["category"],
            difficulty=sample_question_data["difficulty"],
            role=sample_question_data["role"]
        )
        assert example1 is not None
        
        # Intentar recopilar duplicado
        example2 = training_collector.collect(
            question=sample_question_data["question"],
            answer=sample_question_data["answer"],  # Misma respuesta
            evaluation_result=sample_evaluation,
            expected_concepts=sample_question_data["expected_concepts"],
            keywords=sample_question_data["keywords"],
            category=sample_question_data["category"],
            difficulty=sample_question_data["difficulty"],
            role=sample_question_data["role"]
        )
        assert example2 is None  # Rechazado por duplicado
    
    def test_get_statistics(self, training_collector, sample_evaluation):
        """Test: Genera estadísticas correctas de datos recopilados."""
        # Recopilar varios ejemplos
        for i in range(5):
            training_collector.collect(
                question=f"Pregunta técnica {i}",
                answer=f"Respuesta detallada {i}",
                evaluation_result=sample_evaluation,
                expected_concepts=["concepto1", "concepto2"],
                keywords=["keyword1"],
                category="technical" if i < 3 else "soft_skills",
                difficulty="mid",
                role="Backend Developer"
            )
        
        stats = training_collector.get_stats()
        
        assert stats["total_examples"] == 5
        assert stats["by_category"]["technical"] == 3
        assert stats["by_category"]["soft_skills"] == 2
        assert stats["by_difficulty"]["mid"] == 5
        assert "Backend Developer" in stats["by_role"]
    
    def test_load_all_examples(self, training_collector, sample_evaluation, sample_question_data):
        """Test: Carga todos los ejemplos almacenados."""
        # Recopilar 3 ejemplos
        for i in range(3):
            training_collector.collect(
                question=f"Pregunta {i}",
                answer=f"Respuesta {i}",
                evaluation_result=sample_evaluation,
                expected_concepts=sample_question_data["expected_concepts"],
                keywords=sample_question_data["keywords"],
                category="technical",
                difficulty="mid",
                role="Backend Developer"
            )
        
        examples = training_collector.load_all_examples()
        
        assert len(examples) == 3
        assert all(isinstance(ex, TrainingExample) for ex in examples)
        assert examples[0].question == "Pregunta 0"


# =========================================================================
# TESTS: DatasetGenerator
# =========================================================================

class TestDatasetGenerator:
    """Tests para DatasetGenerator."""
    
    def test_example_to_alpaca_conversion(self, training_collector, sample_evaluation, sample_question_data):
        """Test: Convierte TrainingExample a formato Alpaca correctamente."""
        # Recopilar ejemplo
        example = training_collector.collect(
            question=sample_question_data["question"],
            answer=sample_question_data["answer"],
            evaluation_result=sample_evaluation,
            expected_concepts=sample_question_data["expected_concepts"],
            keywords=sample_question_data["keywords"],
            category=sample_question_data["category"],
            difficulty=sample_question_data["difficulty"],
            role=sample_question_data["role"]
        )
        
        generator = DatasetGenerator(training_collector)
        alpaca_example = generator.example_to_alpaca(example)
        
        # Validar estructura Alpaca
        assert "instruction" in alpaca_example
        assert "input" in alpaca_example
        assert "output" in alpaca_example
        
        # Validar contenido
        assert "Backend Developer" in alpaca_example["instruction"]
        assert "technical" in alpaca_example["instruction"]
        assert sample_question_data["question"] in alpaca_example["input"]
        
        # Validar output es JSON válido
        output_json = json.loads(alpaca_example["output"])
        assert output_json["score"] == 8.5
        assert "breakdown" in output_json
    
    def test_generate_dataset_train_val_split(self, training_collector, sample_evaluation, temp_storage):
        """Test: Genera dataset con split train/val correcto."""
        # Recopilar 10 ejemplos
        for i in range(10):
            training_collector.collect(
                question=f"Pregunta técnica {i}",
                answer=f"Respuesta detallada {i}",
                evaluation_result=sample_evaluation,
                expected_concepts=["concepto1", "concepto2"],
                keywords=["keyword1"],
                category="technical",
                difficulty="mid",
                role="Backend Developer"
            )
        
        generator = DatasetGenerator(training_collector)
        output_path = temp_storage / "dataset.jsonl"
        
        stats = generator.generate_dataset(
            output_path=output_path,
            train_split=0.8
        )
        
        # Validar estadísticas
        assert stats["total_examples"] == 10
        assert stats["train_examples"] == 8
        assert stats["validation_examples"] == 2
        
        # Validar archivos creados
        train_file = temp_storage / "dataset_train.jsonl"
        val_file = temp_storage / "dataset_val.jsonl"
        assert train_file.exists()
        assert val_file.exists()
        
        # Validar contenido de archivos
        with open(train_file) as f:
            train_lines = f.readlines()
            assert len(train_lines) == 8
            # Validar formato JSONL
            first_example = json.loads(train_lines[0])
            assert "instruction" in first_example
            assert "input" in first_example
            assert "output" in first_example
    
    def test_quality_filtering(self, training_collector, sample_evaluation, temp_storage):
        """Test: Filtra ejemplos por score mínimo."""
        # Recopilar ejemplos con scores variados
        for i in range(5):
            eval_copy = sample_evaluation.copy()
            eval_copy["score"] = 5.0 + i  # Scores: 5.0, 6.0, 7.0, 8.0, 9.0
            
            training_collector.collect(
                question=f"Pregunta {i}",
                answer=f"Respuesta {i}",
                evaluation_result=eval_copy,
                expected_concepts=["concepto1"],
                keywords=["keyword1"],
                category="technical",
                difficulty="mid",
                role="Backend Developer"
            )
        
        generator = DatasetGenerator(training_collector)
        output_path = temp_storage / "dataset.jsonl"
        
        # Generar dataset con min_score_quality=7.0
        stats = generator.generate_dataset(
            output_path=output_path,
            min_score_quality=7.0
        )
        
        # Solo 3 ejemplos deberían pasar el filtro (scores 7.0, 8.0, 9.0)
        assert stats["total_examples"] == 3


# =========================================================================
# TESTS: ModelFineTuner
# =========================================================================

class TestModelFineTuner:
    """Tests para ModelFineTuner."""
    
    def test_create_modelfile(self, temp_storage):
        """Test: Crea Modelfile válido para Ollama."""
        finetuner = ModelFineTuner(
            base_model="llama3.2:3b",
            finetuned_model_name="ready4hire-llama3.2:3b"
        )
        
        dataset_path = temp_storage / "dataset_train.jsonl"
        dataset_path.write_text('{"instruction": "test", "input": "test", "output": "test"}')
        
        modelfile_path = finetuner.create_modelfile(
            dataset_path=dataset_path,
            output_path=temp_storage / "Modelfile"
        )
        
        modelfile = Path(modelfile_path)
        assert modelfile.exists()
        
        # Validar contenido del Modelfile
        content = modelfile.read_text()
        assert "FROM llama3.2:3b" in content
        assert "SYSTEM" in content
        assert "PARAMETER temperature" in content
        assert "Ready4Hire" in content
    
    def test_training_guide_generation(self):
        """Test: Genera guía de entrenamiento completa."""
        finetuner = ModelFineTuner(
            base_model="llama3.2:3b",
            finetuned_model_name="ready4hire-llama3.2:3b"
        )
        
        guide = finetuner.get_training_guide()
        
        # Validar contenido de la guía
        assert "Unsloth" in guide
        assert "pip install" in guide
        assert "python" in guide
        assert "ollama create" in guide
        assert "ready4hire-llama3.2:3b" in guide


# =========================================================================
# TESTS: Integración EvaluationService
# =========================================================================

class TestEvaluationServiceIntegration:
    """Tests de integración con EvaluationService."""
    
    def test_automatic_training_data_collection(self, temp_storage, monkeypatch):
        """Test: EvaluationService recopila datos automáticamente."""
        # Mock response para el LLM
        mock_response = json.dumps({
            "score": 8.5,
            "breakdown": {
                "completeness": 2.8,
                "technical_depth": 2.7,
                "clarity": 1.8,
                "key_concepts": 1.2
            },
            "justification": "Excelente respuesta técnica.",
            "strengths": ["Explicación clara", "Buenos ejemplos"],
            "improvements": ["Agregar más detalles"],
            "concepts_covered": ["herencia", "polimorfismo"],
            "missing_concepts": ["abstracción"]
        })
        
        # Crear collector con almacenamiento temporal
        collector = TrainingDataCollector(
            storage_path=temp_storage / "evaluations.jsonl"
        )
        
        # Crear servicio con training collection habilitado
        service = EvaluationService(
            model="llama3.2:3b",
            enable_cache=False,  # Deshabilitar caché para test
            collect_training_data=True,
            training_collector=collector
        )
        
        # Mock del método generate del LLM service
        monkeypatch.setattr(service.llm_service, "generate", lambda prompt, **kwargs: mock_response)
        
        # Evaluar respuesta
        result = service.evaluate_answer(
            question="¿Qué es POO?",
            answer="POO es programación orientada a objetos con herencia y polimorfismo.",
            expected_concepts=["herencia", "polimorfismo", "encapsulación"],
            keywords=["POO", "objetos"],
            category="technical",
            difficulty="mid",
            role="Backend Developer"
        )
        
        # Validar que la evaluación fue exitosa
        assert result["score"] == 8.5
        
        # Validar que se recopiló el ejemplo
        stats = collector.get_stats()
        assert stats["total_examples"] == 1
        assert stats["by_category"]["technical"] == 1
    
    def test_training_stats_retrieval(self, temp_storage):
        """Test: Obtiene estadísticas de training data correctamente."""
        collector = TrainingDataCollector(
            storage_path=temp_storage / "evaluations.jsonl"
        )
        
        service = EvaluationService(
            model="llama3.2:3b",
            enable_cache=False,
            collect_training_data=True,
            training_collector=collector
        )
        
        # Obtener stats (debería estar vacío)
        stats = service.get_training_stats()
        assert stats["enabled"] is True
        assert stats["total_examples"] == 0
    
    def test_enable_disable_training_collection(self):
        """Test: Habilita/deshabilita recopilación dinámicamente."""
        service = EvaluationService(
            model="llama3.2:3b",
            collect_training_data=False  # Inicialmente deshabilitado
        )
        
        assert service.collect_training_data is False
        
        # Habilitar
        service.enable_training_collection()
        assert service.collect_training_data is True
        assert service.training_collector is not None
        
        # Deshabilitar
        service.disable_training_collection()
        assert service.collect_training_data is False


# =========================================================================
# TESTS: End-to-End
# =========================================================================

class TestEndToEndFineTuning:
    """Tests end-to-end del flujo completo de fine-tuning."""
    
    def test_complete_workflow(self, temp_storage, monkeypatch):
        """Test: Flujo completo desde evaluación hasta dataset generado."""
        # Mock response para el LLM
        mock_response = json.dumps({
            "score": 7.5,
            "breakdown": {
                "completeness": 2.5,
                "technical_depth": 2.3,
                "clarity": 1.5,
                "key_concepts": 1.2
            },
            "justification": "Buena explicación de microservicios.",
            "strengths": ["Define correctamente", "Menciona ventajas"],
            "improvements": ["Agregar desventajas"],
            "concepts_covered": ["escalabilidad", "independencia"],
            "missing_concepts": ["comunicación"]
        })
        
        # 1. Setup: Crear collector y service
        collector = TrainingDataCollector(
            storage_path=temp_storage / "evaluations.jsonl",
            min_score_threshold=5.0
        )
        
        service = EvaluationService(
            enable_cache=False,
            collect_training_data=True,
            training_collector=collector
        )
        
        # Mock del método generate del LLM service
        monkeypatch.setattr(service.llm_service, "generate", lambda prompt, **kwargs: mock_response)
        
        # 2. Recopilar datos: Simular 5 evaluaciones
        questions = [
            "¿Qué son microservicios?",
            "Explica SOLID",
            "¿Qué es Clean Code?",
            "Define RESTful API",
            "¿Qué es CI/CD?"
        ]
        
        for question in questions:
            service.evaluate_answer(
                question=question,
                answer=f"Respuesta técnica sobre {question}",
                expected_concepts=["concepto1", "concepto2"],
                keywords=["keyword1"],
                category="technical",
                difficulty="mid",
                role="Backend Developer"
            )
        
        # 3. Generar dataset
        generator = DatasetGenerator(collector)
        output_path = temp_storage / "dataset.jsonl"
        
        stats = generator.generate_dataset(
            output_path=output_path,
            train_split=0.8
        )
        
        # 4. Validar resultado final
        assert stats["total_examples"] == 5
        assert stats["train_examples"] == 4
        assert stats["validation_examples"] == 1
        
        train_file = temp_storage / "dataset_train.jsonl"
        assert train_file.exists()
        
        # Validar formato Alpaca
        with open(train_file) as f:
            first_line = json.loads(f.readline())
            assert "instruction" in first_line
            assert "input" in first_line
            assert "output" in first_line
            
            # Validar output es JSON válido de evaluación
            output_json = json.loads(first_line["output"])
            assert "score" in output_json
            assert "breakdown" in output_json
            assert "justification" in output_json


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
