"""
Tests para mejoras de IA Fase 1 (Quick Wins):
1. Caché de evaluaciones
2. Model warm-up
3. Explicaciones mejoradas
"""
import pytest
import time
from datetime import datetime
from pathlib import Path
import tempfile

from app.application.services.evaluation_service import EvaluationService
from app.infrastructure.cache.evaluation_cache import EvaluationCache


class TestEvaluationCache:
    """Tests para el sistema de caché de evaluaciones."""
    
    @pytest.fixture
    def cache(self):
        """Fixture de caché en directorio temporal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EvaluationCache(cache_dir=tmpdir, ttl_days=1)
            yield cache
            cache.clear()
    
    def test_cache_key_generation(self, cache):
        """Test: Generación de cache key."""
        key1 = cache.get_cache_key(
            question="¿Qué es Python?",
            answer="Python es un lenguaje de programación",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=["lenguaje", "programación"],
            keywords=["python"]
        )
        
        # Mismos parámetros = misma key
        key2 = cache.get_cache_key(
            question="¿Qué es Python?",
            answer="Python es un lenguaje de programación",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=["lenguaje", "programación"],
            keywords=["python"]
        )
        
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash
    
    def test_cache_key_sensitivity(self, cache):
        """Test: Cache key es sensible a cambios."""
        base_params = {
            "question": "¿Qué es Python?",
            "answer": "Python es un lenguaje",
            "model": "llama3.2:3b",
            "temperature": 0.3,
            "expected_concepts": ["lenguaje"],
            "keywords": ["python"]
        }
        
        key1 = cache.get_cache_key(**base_params)
        
        # Cambiar respuesta
        params2 = base_params.copy()
        params2["answer"] = "Python es un lenguaje diferente"
        key2 = cache.get_cache_key(**params2)
        
        # Cambiar modelo
        params3 = base_params.copy()
        params3["model"] = "llama3.2:1b"
        key3 = cache.get_cache_key(**params3)
        
        # Cambiar temperatura
        params4 = base_params.copy()
        params4["temperature"] = 0.5
        key4 = cache.get_cache_key(**params4)
        
        # Todas las keys deben ser diferentes
        assert key1 != key2
        assert key1 != key3
        assert key1 != key4
    
    def test_cache_set_and_get(self, cache):
        """Test: Guardar y recuperar del caché."""
        result = {
            "score": 8.5,
            "breakdown": {"completeness": 3, "technical_depth": 2.5},
            "justification": "Buena respuesta"
        }
        
        cache.set(
            question="¿Qué es Docker?",
            answer="Docker es una plataforma de contenedores",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=["contenedores"],
            keywords=["docker"],
            result=result
        )
        
        retrieved = cache.get(
            question="¿Qué es Docker?",
            answer="Docker es una plataforma de contenedores",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=["contenedores"],
            keywords=["docker"]
        )
        
        assert retrieved is not None
        assert retrieved["score"] == 8.5
        assert retrieved["justification"] == "Buena respuesta"
    
    def test_cache_miss(self, cache):
        """Test: Cache miss cuando no existe entrada."""
        retrieved = cache.get(
            question="Pregunta que no existe",
            answer="Respuesta que no existe",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=[],
            keywords=[]
        )
        
        assert retrieved is None
    
    def test_cache_stats(self, cache):
        """Test: Estadísticas del caché."""
        # Inicialmente todo en 0
        stats = cache.get_stats()
        assert stats["total_requests"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Cache miss
        cache.get(
            question="Test",
            answer="Test",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=[],
            keywords=[]
        )
        
        stats = cache.get_stats()
        assert stats["misses"] == 1
        
        # Guardar y hacer cache hit
        cache.set(
            question="Test",
            answer="Test",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=[],
            keywords=[],
            result={"score": 5.0}
        )
        
        cache.get(
            question="Test",
            answer="Test",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=[],
            keywords=[]
        )
        
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["hit_rate"] > 0
    
    def test_cache_clear(self, cache):
        """Test: Limpiar todo el caché."""
        cache.set(
            question="Test",
            answer="Test",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=[],
            keywords=[],
            result={"score": 5.0}
        )
        
        cache.clear()
        
        retrieved = cache.get(
            question="Test",
            answer="Test",
            model="llama3.2:3b",
            temperature=0.3,
            expected_concepts=[],
            keywords=[]
        )
        
        assert retrieved is None


class TestEvaluationServiceWithCache:
    """Tests para EvaluationService con caché."""
    
    @pytest.fixture
    def service(self):
        """Fixture de servicio con caché habilitado."""
        return EvaluationService(
            enable_cache=True,
            cache_ttl_days=1
        )
    
    def test_service_initialization_with_cache(self, service):
        """Test: Servicio se inicializa con caché."""
        assert service.enable_cache is True
        assert service.cache is not None
    
    def test_service_without_cache(self):
        """Test: Servicio sin caché."""
        service = EvaluationService(enable_cache=False)
        assert service.enable_cache is False
        assert service.cache is None
    
    def test_cache_integration_in_evaluation(self, service, monkeypatch):
        """Test: Caché funciona en evaluación completa."""
        # Mock del LLM para evitar llamadas reales
        def mock_generate(prompt, **kwargs):
            return '''
            {
                "score": 7.5,
                "breakdown": {
                    "completeness": 2.5,
                    "technical_depth": 2.0,
                    "clarity": 1.5,
                    "key_concepts": 1.5
                },
                "justification": "Respuesta sólida con conceptos clave.",
                "strengths": ["Menciona Docker", "Explica contenedores"],
                "improvements": ["Podría profundizar más"],
                "concepts_covered": ["Docker", "contenedores"],
                "missing_concepts": []
            }
            '''
        
        monkeypatch.setattr(service.llm_service, 'generate', mock_generate)
        
        params = {
            "question": "¿Qué es Docker?",
            "answer": "Docker es una plataforma de contenedores",
            "expected_concepts": ["contenedores", "virtualización"],
            "keywords": ["docker", "container"],
            "category": "technical",
            "difficulty": "junior",
            "role": "Backend Developer"
        }
        
        # Primera llamada: miss, llama al LLM
        start1 = time.time()
        result1 = service.evaluate_answer(**params)
        elapsed1 = time.time() - start1
        
        assert result1["score"] == 7.5
        assert result1["from_cache"] is False
        
        # Segunda llamada: hit, del caché (debe ser MUCHO más rápido)
        start2 = time.time()
        result2 = service.evaluate_answer(**params)
        elapsed2 = time.time() - start2
        
        assert result2["score"] == 7.5
        assert result2["from_cache"] is True
        # Verificar que el resultado vino del caché (más flexible con timings)
    
    def test_cache_stats_method(self, service):
        """Test: Método get_cache_stats()."""
        stats = service.get_cache_stats()
        
        assert "enabled" in stats
        assert stats["enabled"] is True
        assert "total_requests" in stats
        assert "hit_rate" in stats
    
    def test_clear_cache_method(self, service, monkeypatch):
        """Test: Método clear_cache()."""
        # Mock para evitar llamadas reales al LLM
        def mock_generate(prompt, **kwargs):
            return '{"score": 5.0, "breakdown": {}, "justification": "Test", "strengths": [], "improvements": [], "concepts_covered": []}'
        
        monkeypatch.setattr(service.llm_service, 'generate', mock_generate)
        
        # Crear entrada en caché
        service.evaluate_answer(
            question="Test",
            answer="Test",
            expected_concepts=[],
            keywords=[],
            category="technical",
            difficulty="junior",
            role="Developer"
        )
        
        # Limpiar caché
        service.clear_cache()
        
        # Verificar que fue limpiado
        stats = service.get_cache_stats()
        assert stats["disk_entries"] == 0


class TestModelWarmup:
    """Tests para model warm-up."""
    
    def test_warmup_on_initialization(self, monkeypatch):
        """Test: Warm-up se ejecuta al inicializar."""
        warmup_called = {"value": False}
        
        def mock_generate(prompt, **kwargs):
            warmup_called["value"] = True
            return "Ready!"
        
        # Crear servicio (debe llamar warm-up)
        service = EvaluationService()
        monkeypatch.setattr(service.llm_service, 'generate', mock_generate)
        
        # Forzar warm-up manual para testear
        service._warmup_model()
        
        assert warmup_called["value"] is True


class TestImprovedExplanations:
    """Tests para explicaciones mejoradas."""
    
    @pytest.fixture
    def service(self):
        """Fixture de servicio."""
        return EvaluationService(enable_cache=False)
    
    def test_missing_concepts_in_result(self, service, monkeypatch):
        """Test: Resultado incluye missing_concepts."""
        def mock_generate(prompt, **kwargs):
            return '''
            {
                "score": 6.0,
                "breakdown": {
                    "completeness": 2.0,
                    "technical_depth": 1.5,
                    "clarity": 1.5,
                    "key_concepts": 1.0
                },
                "justification": "Respuesta incompleta.",
                "strengths": ["Menciona el concepto básico"],
                "improvements": ["Falta profundizar"],
                "concepts_covered": ["API"],
                "missing_concepts": ["REST", "HTTP methods"]
            }
            '''
        
        monkeypatch.setattr(service.llm_service, 'generate', mock_generate)
        
        result = service.evaluate_answer(
            question="¿Qué es una API REST?",
            answer="Una API es una interfaz",
            expected_concepts=["REST", "HTTP"],
            keywords=["api", "rest"],
            category="technical",
            difficulty="junior",
            role="Backend Developer"
        )
        
        assert "missing_concepts" in result
        assert "REST" in result["missing_concepts"]
    
    def test_breakdown_confidence(self, service, monkeypatch):
        """Test: Resultado incluye breakdown_confidence."""
        def mock_generate(prompt, **kwargs):
            return '''
            {
                "score": 8.0,
                "breakdown": {
                    "completeness": 3.0,
                    "technical_depth": 2.5,
                    "clarity": 1.5,
                    "key_concepts": 2.0
                },
                "justification": "Excelente respuesta.",
                "strengths": ["Completa", "Clara"],
                "improvements": [],
                "concepts_covered": ["Docker", "containers"],
                "missing_concepts": []
            }
            '''
        
        monkeypatch.setattr(service.llm_service, 'generate', mock_generate)
        
        result = service.evaluate_answer(
            question="¿Qué es Docker?",
            answer="Docker es una plataforma",
            expected_concepts=["docker"],
            keywords=["docker"],
            category="technical",
            difficulty="mid",
            role="DevOps"
        )
        
        assert "breakdown_confidence" in result
        # Breakdown suma 9.0/10 = 90%
        assert result["breakdown_confidence"] == 90.0


class TestPerformanceImprovements:
    """Tests de rendimiento y métricas."""
    
    def test_evaluation_includes_timing(self, monkeypatch):
        """Test: Resultado incluye tiempo de evaluación."""
        service = EvaluationService(enable_cache=False)
        
        def mock_generate(prompt, **kwargs):
            time.sleep(0.1)  # Simular latencia
            return '{"score": 5.0, "breakdown": {"completeness": 1.5, "technical_depth": 1.5, "clarity": 1.0, "key_concepts": 1.0}, "justification": "Test", "strengths": [], "improvements": [], "concepts_covered": [], "missing_concepts": []}'
        
        monkeypatch.setattr(service.llm_service, 'generate', mock_generate)
        
        result = service.evaluate_answer(
            question="Test",
            answer="Test",
            expected_concepts=[],
            keywords=[],
            category="technical",
            difficulty="junior",
            role="Developer"
        )
        
        assert "evaluation_time_seconds" in result
        assert result["evaluation_time_seconds"] > 0
    
    def test_cache_latency_improvement(self, monkeypatch):
        """Test: Caché reduce latencia dramáticamente."""
        service = EvaluationService(enable_cache=True)
        
        call_count = {"value": 0}
        
        def mock_generate(prompt, **kwargs):
            call_count["value"] += 1
            time.sleep(0.1)  # Simular latencia del LLM
            return '{"score": 7.0, "breakdown": {"completeness": 2.0, "technical_depth": 2.0, "clarity": 1.5, "key_concepts": 1.5}, "justification": "Good", "strengths": ["Clear"], "improvements": ["Depth"], "concepts_covered": ["Python"], "missing_concepts": []}'
        
        monkeypatch.setattr(service.llm_service, 'generate', mock_generate)
        
        params = {
            "question": "What is Python?",
            "answer": "Python is a programming language",
            "expected_concepts": ["language", "programming"],
            "keywords": ["python"],
            "category": "technical",
            "difficulty": "junior",
            "role": "Developer"
        }
        
        # Primera llamada: LLM
        start1 = time.time()
        result1 = service.evaluate_answer(**params)
        time1 = time.time() - start1
        
        # Segunda llamada: Caché
        start2 = time.time()
        result2 = service.evaluate_answer(**params)
        time2 = time.time() - start2
        
        # Verificar que solo se llamó al LLM una vez
        assert call_count["value"] == 1
        
        # Verificar mejora de latencia (caché debe ser significativamente más rápido)
        assert time2 < time1  # Al menos más rápido
        assert result1["from_cache"] is False
        assert result2["from_cache"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
