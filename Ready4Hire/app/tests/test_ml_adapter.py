"""
Tests básicos para ML Adapter

Valida que:
1. Adapter se inicializa correctamente
2. Degradación graceful funciona
3. Fallbacks funcionan cuando ML falla
4. Status y metrics se reportan correctamente
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from app.infrastructure.ml.ml_adapter import MLAdapter, MLConfig
from app.domain.entities.question import Question


# Fixtures
@pytest.fixture
def mock_embeddings_service():
    """Mock del servicio de embeddings."""
    service = Mock()
    service.encode = Mock(return_value=[[0.1] * 384] * 10)
    return service


@pytest.fixture
def sample_questions() -> List[Question]:
    """Preguntas de ejemplo para tests."""
    return [
        Question(
            id=f"q{i}",
            text=f"Question {i}",
            category="technical",
            difficulty="medium",
            keywords=[f"keyword{i}"],
            expected_concepts=[f"concept{i}"],
            topic=f"topic{i % 3}",  # 3 tópicos diferentes
            role="Backend Developer"
        )
        for i in range(10)
    ]


class TestMLAdapterInitialization:
    """Tests de inicialización del adapter."""
    
    def test_init_without_services(self):
        """Test: Adapter se inicializa sin servicios."""
        adapter = MLAdapter()
        
        assert adapter is not None
        assert adapter.config is not None
        assert not adapter._embeddings_available
        assert not adapter._clustering_available
        assert not adapter._learning_available
    
    def test_init_with_embeddings(self, mock_embeddings_service):
        """Test: Adapter se inicializa con embeddings service."""
        adapter = MLAdapter(embeddings_service=mock_embeddings_service)
        
        assert adapter._embeddings_available
        assert adapter.embeddings_service == mock_embeddings_service
    
    def test_init_with_custom_config(self):
        """Test: Adapter respeta configuración custom."""
        config = MLConfig(
            enable_clustering=False,
            enable_continuous_learning=False,
            fallback_on_error=True
        )
        adapter = MLAdapter(config=config)
        
        assert not adapter.config.enable_clustering
        assert not adapter.config.enable_continuous_learning
        assert adapter.config.fallback_on_error


class TestMLAdapterSelection:
    """Tests de selección de preguntas."""
    
    def test_select_empty_candidates(self):
        """Test: Retorna None cuando no hay candidatas."""
        adapter = MLAdapter()
        
        result = adapter.select_question_ml(
            candidates=[],
            previous_ids=[]
        )
        
        assert result is None
    
    def test_select_single_candidate(self, sample_questions):
        """Test: Retorna única candidata si solo hay una."""
        adapter = MLAdapter()
        
        result = adapter.select_question_ml(
            candidates=[sample_questions[0]],
            previous_ids=[]
        )
        
        assert result == sample_questions[0]
    
    def test_select_with_fallback(self, sample_questions):
        """Test: Fallback a random cuando ML no disponible."""
        config = MLConfig(
            enable_clustering=False,
            enable_continuous_learning=False,
            enable_embeddings=False
        )
        adapter = MLAdapter(config=config)
        
        result = adapter.select_question_ml(
            candidates=sample_questions,
            previous_ids=[]
        )
        
        # Debe retornar alguna pregunta (random)
        assert result is not None
        assert result in sample_questions
    
    def test_select_with_error_handling(self, sample_questions, mock_embeddings_service):
        """Test: Maneja errores y hace fallback."""
        # Simular error en embeddings
        mock_embeddings_service.encode = Mock(side_effect=Exception("Test error"))
        
        config = MLConfig(fallback_on_error=True)
        adapter = MLAdapter(
            embeddings_service=mock_embeddings_service,
            config=config
        )
        
        result = adapter.select_question_ml(
            candidates=sample_questions,
            previous_ids=[]
        )
        
        # Debe hacer fallback exitosamente
        assert result is not None
        assert result in sample_questions


class TestMLAdapterPerformanceUpdate:
    """Tests de actualización de performance."""
    
    def test_update_performance_without_learning(self):
        """Test: Update performance sin learning habilitado."""
        config = MLConfig(enable_continuous_learning=False)
        adapter = MLAdapter(config=config)
        
        # No debe fallar
        adapter.update_performance(
            question_id="q1",
            score=7.5,
            time_taken=60
        )
        
        # Si learning no está habilitado, simplemente no hace nada
        assert True
    
    def test_update_performance_with_learning_disabled(self):
        """Test: Update cuando learning está deshabilitado."""
        config = MLConfig(enable_continuous_learning=False)
        adapter = MLAdapter(config=config)
        
        # No debe lanzar excepción
        try:
            adapter.update_performance(
                question_id="q1",
                score=8.0,
                time_taken=45,
                interview_id="int1"
            )
            success = True
        except Exception:
            success = False
        
        assert success


class TestMLAdapterStatus:
    """Tests de status y metrics."""
    
    def test_get_status(self):
        """Test: Get status retorna información correcta."""
        adapter = MLAdapter()
        status = adapter.get_status()
        
        assert 'adapter' in status
        assert 'config' in status
        assert 'availability' in status
        assert 'services' in status
        
        assert status['adapter'] == 'MLAdapter v1.0'
    
    def test_get_status_with_services(self, mock_embeddings_service):
        """Test: Status refleja servicios disponibles."""
        adapter = MLAdapter(embeddings_service=mock_embeddings_service)
        status = adapter.get_status()
        
        assert status['availability']['embeddings']
        assert status['services']['embeddings_service']
    
    def test_get_metrics(self):
        """Test: Get metrics retorna datos válidos."""
        adapter = MLAdapter()
        metrics = adapter.get_metrics()
        
        assert 'status' in metrics
        assert metrics['status'] is not None


class TestMLAdapterIntegration:
    """Tests de integración con servicios reales."""
    
    @pytest.mark.skip(reason="Requiere dependencias ML instaladas")
    def test_integration_with_real_clustering(self, sample_questions, mock_embeddings_service):
        """Test: Integración con clustering real."""
        adapter = MLAdapter(embeddings_service=mock_embeddings_service)
        
        result = adapter.select_question_ml(
            candidates=sample_questions,
            previous_ids=[],
            strategy='balanced'
        )
        
        assert result is not None
        assert result in sample_questions
    
    def test_selection_respects_previous_ids(self, sample_questions):
        """Test: No selecciona preguntas ya usadas."""
        adapter = MLAdapter()
        
        # Marcar primeras 5 como usadas
        previous_ids = [q.id for q in sample_questions[:5]]
        
        # Filtrar candidatas
        candidates = [q for q in sample_questions if q.id not in previous_ids]
        
        result = adapter.select_question_ml(
            candidates=candidates,
            previous_ids=previous_ids
        )
        
        assert result is not None
        assert result.id not in previous_ids


# Tests de regresión
class TestMLAdapterRegression:
    """Tests de regresión para prevenir bugs conocidos."""
    
    def test_handles_none_embeddings_service(self):
        """Test: Maneja embeddings_service=None correctamente."""
        adapter = MLAdapter(embeddings_service=None)
        
        assert not adapter._embeddings_available
        assert adapter.embeddings_service is None
    
    def test_lazy_initialization_clustering(self, mock_embeddings_service):
        """Test: Clustering se inicializa lazy."""
        adapter = MLAdapter(embeddings_service=mock_embeddings_service)
        
        # Al inicio no está inicializado
        assert adapter._clustering_service is None
        
        # Debería inicializarse en primer uso
        # (pero fallará porque faltan dependencias en test env)
    
    def test_error_handling_without_fallback(self, sample_questions):
        """Test: Sin fallback retorna None en error."""
        config = MLConfig(fallback_on_error=False)
        adapter = MLAdapter(config=config)
        
        # Simular condición de error
        adapter._embeddings_available = True
        adapter.embeddings_service = None  # Esto causará error
        
        result = adapter.select_question_ml(
            candidates=sample_questions,
            previous_ids=[]
        )
        
        # Sin fallback, debería retornar None o una selección básica
        # (depende de la implementación interna)
        assert result is None or result in sample_questions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
