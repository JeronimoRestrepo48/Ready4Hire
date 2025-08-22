import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.parametrize("mode", ["practice", "exam"])
def test_soft_skills_interview_modes(mode):
    """
    Test de entrevista de habilidades blandas alternando entre modo práctica y examen.
    """
    # Iniciar entrevista de soft skills
    start = client.post("/start_interview", json={
        "user_id": "test_soft_1",
        "role": "General",
        "type": "soft",
        "mode": mode
    })
    assert start.status_code == 200
    # Obtener pregunta
    q = client.post("/next_question", json={"user_id": "test_soft_1"})
    assert q.status_code == 200
    data = q.json()
    assert "question" in data
    # Enviar respuesta
    ans = client.post("/answer", json={
        "user_id": "test_soft_1",
        "answer": "Respuesta ejemplo soft skill"
    })
    assert ans.status_code == 200
    # Finalizar entrevista
    end = client.post("/end_interview", params={"user_id": "test_soft_1"})
    assert end.status_code == 200

def test_technical_exam_mode():
    """
    Test de entrevista técnica en modo examen.
    """
    # Iniciar entrevista técnica modo examen
    start = client.post("/start_interview", json={
        "user_id": "test_tech_exam_1",
        "role": "Backend",
        "type": "technical",
        "mode": "exam"
    })
    assert start.status_code == 200
    # Obtener pregunta
    q = client.post("/next_question", json={"user_id": "test_tech_exam_1"})
    assert q.status_code == 200
    data = q.json()
    assert "question" in data
    # Enviar respuesta
    ans = client.post("/answer", json={
        "user_id": "test_tech_exam_1",
        "answer": "Respuesta ejemplo técnica"
    })
    assert ans.status_code == 200
    # Finalizar entrevista
    end = client.post("/end_interview", params={"user_id": "test_tech_exam_1"})
    assert end.status_code == 200
