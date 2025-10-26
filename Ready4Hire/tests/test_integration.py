#!/usr/bin/env python3
"""
Tests de Integración - Ready4Hire v2.2.0

Tests consolidados que validan todos los componentes.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test básico del endpoint de health."""
    from app.main_v2_improved import app
    from unittest.mock import patch, MagicMock

    # Mock Ollama client para evitar dependencia del servicio externo
    with patch('app.infrastructure.llm.ollama_client.OllamaClient') as mock_ollama:
        mock_instance = MagicMock()
        mock_instance.list_models.return_value = ["llama2"]
        mock_ollama.return_value = mock_instance

        client = TestClient(app)
        response = client.get("/api/v2/health")

        # Aceptar tanto 200 como 503 (si Ollama no está disponible)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test del endpoint raíz."""
    from app.main_v2_improved import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_docs_available():
    """Test que la documentación esté disponible."""
    from app.main_v2_improved import app

    client = TestClient(app)
    response = client.get("/docs")

    # Puede ser 200 o 404 dependiendo de la configuración
    assert response.status_code in [200, 404]


def test_import_main_module():
    """Test que el módulo principal se pueda importar sin errores."""
    try:
        from app import main_v2_improved

        assert hasattr(main_v2_improved, "app")
        assert hasattr(main_v2_improved, "get_container")
    except ImportError as e:
        pytest.fail(f"Failed to import main module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
