"""
Basic performance tests for API endpoints
Tests that responses are within acceptable time limits
"""
import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def fast_mock():
    """Mock that returns quickly"""
    with patch('app.infrastructure.llm.ollama_client.OllamaClient') as mock:
        instance = MagicMock()
        instance.is_available.return_value = True
        instance.list_models.return_value = ["llama3.2:3b"]
        mock.return_value = instance
        yield mock


class TestPerformance:
    """Basic performance tests"""

    def test_health_endpoint_response_time(self, fast_mock):
        """Test that health check responds quickly"""
        from app.main_v2_improved import app
        
        client = TestClient(app)
        
        start_time = time.time()
        response = client.get("/api/v2/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond in less than 2 seconds
        assert response_time < 2.0
        assert response.status_code in [200, 503]

    def test_root_endpoint_response_time(self, fast_mock):
        """Test that root endpoint responds quickly"""
        from app.main_v2_improved import app
        
        client = TestClient(app)
        
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond in less than 1 second
        assert response_time < 1.0
        assert response.status_code == 200

    def test_concurrent_health_checks(self, fast_mock):
        """Test multiple concurrent requests"""
        from app.main_v2_improved import app
        import concurrent.futures
        
        client = TestClient(app)
        
        def make_request():
            return client.get("/api/v2/health")
        
        start_time = time.time()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should complete
        assert len(responses) == 10
        
        # Should handle 10 concurrent requests in less than 5 seconds
        assert total_time < 5.0

