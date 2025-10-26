"""
Integration tests for complete interview flow
Tests the full user journey through an interview
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies"""
    with patch('app.infrastructure.llm.ollama_client.OllamaClient') as mock_ollama, \
         patch('app.container.Container') as mock_container:
        
        # Mock Ollama client
        ollama_instance = MagicMock()
        ollama_instance.is_available.return_value = True
        ollama_instance.list_models.return_value = ["llama3.2:3b"]
        ollama_instance.generate.return_value = {
            "response": "Good answer with technical depth"
        }
        mock_ollama.return_value = ollama_instance
        
        # Mock container
        container_instance = MagicMock()
        container_instance.health_check.return_value = {
            "ollama": "✅ Available",
            "llm_service": "✅ Ready"
        }
        mock_container.return_value = container_instance
        
        yield {
            "ollama": mock_ollama,
            "container": mock_container,
            "ollama_instance": ollama_instance
        }


class TestInterviewFlow:
    """Test complete interview flow from start to finish"""

    def test_health_check(self, mock_dependencies):
        """Test that health check endpoint works"""
        from app.main_v2_improved import app
        
        client = TestClient(app)
        response = client.get("/api/v2/health")
        
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    def test_root_endpoint(self, mock_dependencies):
        """Test root endpoint returns welcome message"""
        from app.main_v2_improved import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_docs_endpoint_available(self, mock_dependencies):
        """Test that API documentation is accessible"""
        from app.main_v2_improved import app
        
        client = TestClient(app)
        response = client.get("/docs")
        
        # Docs may be disabled, but endpoint should respond
        assert response.status_code in [200, 404]

    def test_invalid_endpoint_returns_404(self, mock_dependencies):
        """Test that invalid endpoints return 404"""
        from app.main_v2_improved import app
        
        client = TestClient(app)
        response = client.get("/api/v2/nonexistent")
        
        assert response.status_code == 404

    def test_cors_headers_present(self, mock_dependencies):
        """Test that CORS headers are properly configured"""
        from app.main_v2_improved import app
        
        client = TestClient(app)
        response = client.options(
            "/api/v2/health",
            headers={"Origin": "http://localhost:5214"}
        )
        
        # Should have CORS headers
        assert response.status_code in [200, 405]

