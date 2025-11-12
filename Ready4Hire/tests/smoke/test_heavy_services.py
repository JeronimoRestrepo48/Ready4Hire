import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application.services.evaluation_service import EvaluationService
from app.application.services.gamification_service import GamificationService
from app.container import MockLLMService
from app.infrastructure.llm.certificate_generator import CertificateData, CertificateGenerator
from app.infrastructure.tasks import evaluation_tasks


class _DummySentenceTransformer:
    """Ligero stub de SentenceTransformer para pruebas de RAG."""

    def __init__(self, dimension: int = 3):
        self._dimension = dimension

    def encode(self, texts: List[str], show_progress_bar: bool = False, convert_to_numpy: bool = True):
        vectors = []
        for index, text in enumerate(texts):
            seed = index + 1
            base = seed * np.ones(self._dimension, dtype=np.float32)
            base[0] += len(text) % 5  # variar ligeramente según texto
            vectors.append(base)
        return np.stack(vectors, axis=0)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


class _DummyFaissIndex:
    """Implementación mínima compatible con la API usada en RAGService."""

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._vectors: List[np.ndarray] = []

    def add(self, embeddings: np.ndarray) -> None:
        for vector in embeddings:
            self._vectors.append(vector.astype(np.float32))

    def search(self, query: np.ndarray, k: int):
        if not self._vectors:
            distances = np.zeros((1, 0), dtype=np.float32)
            indices = np.zeros((1, 0), dtype=np.int64)
            return distances, indices

        query_vec = query[0]
        scored: List[tuple[float, int]] = []
        for idx, vector in enumerate(self._vectors):
            dist = float(np.linalg.norm(query_vec - vector))
            scored.append((dist, idx))

        scored.sort(key=lambda item: item[0])
        scored = scored[:k]

        distances = np.array([[item[0] for item in scored]], dtype=np.float32)
        indices = np.array([[item[1] for item in scored]], dtype=np.int64)
        return distances, indices


class _DummyFaissModule:
    """Módulo simulado con la interfaz necesaria."""

    def IndexFlatL2(self, dimension: int) -> _DummyFaissIndex:
        return _DummyFaissIndex(dimension)

    def write_index(self, *args, **kwargs):
        return None

    def read_index(self, *args, **kwargs):
        raise RuntimeError("Dummy FAISS does not persist indices")


@pytest.fixture(autouse=True)
def _set_mock_ollama() -> None:
    """Asegura que las pruebas usen el Mock LLM cuando sea necesario."""
    os.environ.setdefault("MOCK_OLLAMA", "true")


def test_gamification_service_smoke():
    service = GamificationService()

    for _ in range(50):
        service.update_stats(user_id="user-123", game_type="code_challenge", won=True, points=100)

    stats = service.get_user_stats("user-123")
    assert stats.total_games_played == 50
    assert stats.total_games_won == 50
    assert stats.total_points >= 5000

    unlocked = [achievement for achievement in service.get_user_achievements("user-123") if achievement.unlocked]
    assert any(achievement.id == "game_master" for achievement in unlocked)

    leaderboard = service.get_leaderboard(limit=1)
    assert leaderboard
    assert leaderboard[0].user_id == "user-123"


def test_evaluation_service_with_mock_llm():
    service = EvaluationService(
        llm_service=MockLLMService(),
        enable_cache=False,
        collect_training_data=False,
        use_advanced_prompts=False,
    )

    result = service.evaluate_answer(
        question="Describe el principio de responsabilidad única.",
        answer="El principio establece que cada módulo debe tener una única razón de cambio.",
        expected_concepts=["responsabilidad única", "cohesión"],
        keywords=["principio"],
        category="technical",
        difficulty="mid",
        role="Software Engineer",
    )

    assert "score" in result
    assert "breakdown" in result
    assert result["role"] == "Software Engineer"
    assert result["category"] == "technical"

    # Con MockLLM, el servicio recurre al fallback heurístico.
    assert result.get("fallback") is True or result["score"] >= 0


def test_certificate_generator_preview_and_validation():
    generator = CertificateGenerator(base_url="https://ready4hire.test")

    certificate = CertificateData(
        certificate_id="R4H-TESTCERT123",
        candidate_name="Ada Lovelace",
        role="Data Scientist",
        completion_date=datetime.now(timezone.utc),
        score=8.7,
        percentile=10,
        interview_id="interview-abc",
        validation_url="https://ready4hire.test/verify/R4H-TESTCERT123",
    )

    preview = generator.generate_preview_svg(certificate)
    assert "Ada Lovelace" in preview
    assert "Data Scientist" in preview

    assert generator.validate_certificate("R4H-ABCDEFG12345")
    assert not generator.validate_certificate("INVALID")


def test_rag_service_with_stubbed_dependencies(monkeypatch):
    from app.infrastructure.embeddings import rag_service as rag_module

    monkeypatch.setattr(rag_module, "SentenceTransformer", lambda *args, **kwargs: _DummySentenceTransformer())
    monkeypatch.setattr(rag_module, "faiss", _DummyFaissModule())

    rag = rag_module.RAGService(model_name="dummy-model", knowledge_base_path=None, index_path=None)
    documents = [
        rag_module.KnowledgeDocument(
            id="doc-1",
            text="Principios SOLID aplicados a arquitectura de software.",
            role="Software Engineer",
            topic="Design Patterns",
            difficulty="mid",
            metadata={"source": "kb"},
        ),
        rag_module.KnowledgeDocument(
            id="doc-2",
            text="Buenas prácticas de testing y TDD.",
            role="Software Engineer",
            topic="Testing",
            difficulty="mid",
            metadata={"source": "kb"},
        ),
    ]
    rag.add_documents(documents)

    contexts = rag.retrieve_context("Cuáles son los principios SOLID?", top_k=1, role_filter="Software Engineer")
    assert contexts
    assert "SOLID" in contexts[0].text
    assert contexts[0].source == "kb"


def test_evaluate_answer_async_task_smoke(monkeypatch):
    captured: Dict[str, Any] = {}

    class _StubEvaluationService:
        def evaluate_answer_sync(self, **kwargs):
            return {"score": 8.5, "evaluated_at": datetime.now(timezone.utc).isoformat()}

    class _StubContainer:
        evaluation_service = _StubEvaluationService()

    class _StubCache:
        def set(self, namespace: str, key: str, value: Dict[str, Any], ttl: Any = None):
            captured["cache_key"] = (namespace, key)
            captured["cache_value"] = value

    class _StubWebsocketManager:
        def send_evaluation_result(self, interview_id: str, payload: Dict[str, Any]):
            captured["ws_payload"] = payload

    monkeypatch.setattr("app.container.get_container", lambda: _StubContainer(), raising=False)
    monkeypatch.setattr(evaluation_tasks, "get_redis_cache", lambda: _StubCache())
    monkeypatch.setattr(evaluation_tasks, "get_websocket_manager", lambda: _StubWebsocketManager())

    task = evaluation_tasks.evaluate_answer_async._get_current_object()
    task.push_request(request=SimpleNamespace(id="task-123", retries=0))
    try:
        result = task.run(
            interview_id="interview-1",
            question_id="q-1",
            answer_text="Respuesta de ejemplo",
            user_id="user-1",
            question_data={
                "text": "Pregunta de ejemplo",
                "expected_concepts": ["concepto"],
                "category": "technical",
                "difficulty": "mid",
                "keywords": ["concepto"],
                "role": "Software Engineer",
            },
        )
    finally:
        task.pop_request()

    assert result["status"] == "success"
    assert captured["cache_key"][0] == "evaluation"
    assert captured["ws_payload"]["question_id"] == "q-1"


def test_cleanup_old_evaluations_task(monkeypatch):
    class _StubCache:
        def clear_pattern(self, namespace: str, pattern: str) -> int:
            assert namespace == "evaluation"
            assert pattern == "*"
            return 42

    monkeypatch.setattr(evaluation_tasks, "get_redis_cache", lambda: _StubCache())

    result = evaluation_tasks.cleanup_old_evaluations.run(days=15)
    assert result["status"] == "success"
    assert result["deleted_count"] == 42
    assert result["days"] == 15


