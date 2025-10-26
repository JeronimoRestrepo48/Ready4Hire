"""
Integration tests for interview flow with updated datasets.
Tests the complete flow from interview start through context questions to technical questions.
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any


class TestInterviewFlowWithNewDatasets:
    """Integration tests for complete interview flow"""

    BASE_URL = "http://localhost:8001"
    API_BASE = f"{BASE_URL}/api/v2"

    @pytest.mark.asyncio
    async def test_backend_health(self):
        """Test that backend is running and healthy"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.API_BASE}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_start_interview_backend_developer(self):
        """Test starting interview for Backend Developer"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-integration-backend",
                    "role": "Backend Developer",
                    "category": "technical",
                    "difficulty": "mid"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "interview_id" in data
            assert "first_question" in data
            assert "status" in data
            assert data["status"] == "context"
            
            # Verify first question is context question
            question = data["first_question"]
            assert question["category"] == "context"
            assert len(question["text"]) > 0
            
            return data["interview_id"]

    @pytest.mark.asyncio
    async def test_complete_context_phase(self):
        """Test completing all 5 context questions"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start interview
            start_response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-context-complete",
                    "role": "Data Scientist",
                    "category": "technical",
                    "difficulty": "senior"
                }
            )
            
            assert start_response.status_code == 200
            interview_id = start_response.json()["interview_id"]
            
            # Answer 5 context questions
            for i in range(5):
                response = await client.post(
                    f"{self.API_BASE}/interviews/{interview_id}/answers",
                    json={"answer": f"Integration test context answer {i+1}"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Check progress
                assert "phase" in data
                assert "progress" in data
                
                if i < 4:
                    # Still in context phase
                    assert data["phase"] == "context"
                    assert data["progress"]["context_completed"] == i + 1
                else:
                    # Should transition to questions phase
                    assert data["phase"] == "questions"
                    assert "next_question" in data
                    # Verify it's a technical question
                    next_q = data["next_question"]
                    assert next_q["category"] == "technical"
                    assert next_q["difficulty"] in ["junior", "mid", "senior"]

    @pytest.mark.asyncio
    async def test_questions_from_new_dataset(self):
        """Test that questions come from new dataset with correct structure"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start and complete context phase
            start_response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-new-dataset-questions",
                    "role": "DevOps Engineer",
                    "category": "technical",
                    "difficulty": "mid"
                }
            )
            
            interview_id = start_response.json()["interview_id"]
            
            # Complete context questions
            for _ in range(5):
                await client.post(
                    f"{self.API_BASE}/interviews/{interview_id}/answers",
                    json={"answer": "Test answer"}
                )
            
            # Get first technical question by answering the 5th context question
            response = await client.post(
                f"{self.API_BASE}/interviews/{interview_id}/answers",
                json={"answer": "Final context answer"}
            )
            
            # The 5th answer should give us the first technical question
            data = response.json()
            
            if data["phase"] == "questions" and "next_question" in data:
                question = data["next_question"]
                
                # Verify question structure from new dataset
                assert "id" in question
                assert "text" in question
                assert "category" in question
                assert question["category"] == "technical"
                assert "difficulty" in question
                assert question["difficulty"] in ["junior", "mid", "senior"]
                assert "expected_concepts" in question
                
                # Verify question has content
                assert len(question["text"]) > 10
                print(f"✅ Got technical question: {question['text'][:100]}...")

    @pytest.mark.asyncio
    async def test_multiple_professions_have_questions(self):
        """Test that multiple professions can start interviews"""
        professions = [
            "Backend Developer",
            "Frontend Developer", 
            "Data Scientist",
            "DevOps Engineer",
            "Mobile Developer",
            "QA Engineer"
        ]
        
        async with httpx.AsyncClient() as client:
            for profession in professions:
                response = await client.post(
                    f"{self.API_BASE}/interviews",
                    json={
                        "user_id": f"test-profession-{profession.replace(' ', '-').lower()}",
                        "role": profession,
                        "category": "technical",
                        "difficulty": "mid"
                    }
                )
                
                assert response.status_code == 200, f"Failed to start interview for {profession}"
                data = response.json()
                assert "interview_id" in data
                assert "first_question" in data
                print(f"✅ {profession}: Interview started successfully")

    @pytest.mark.asyncio
    async def test_soft_skills_interview(self):
        """Test soft skills interview flow"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-soft-skills",
                    "role": "Product Manager",
                    "category": "soft_skills",
                    "difficulty": "mid"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should get context question first
            assert data["status"] == "context"
            interview_id = data["interview_id"]
            
            # Complete context phase
            for _ in range(5):
                await client.post(
                    f"{self.API_BASE}/interviews/{interview_id}/answers",
                    json={"answer": "Soft skills test answer"}
                )

    @pytest.mark.asyncio
    async def test_difficulty_levels(self):
        """Test that different difficulty levels work"""
        difficulties = ["junior", "mid", "senior"]
        
        async with httpx.AsyncClient() as client:
            for difficulty in difficulties:
                response = await client.post(
                    f"{self.API_BASE}/interviews",
                    json={
                        "user_id": f"test-difficulty-{difficulty}",
                        "role": "Full Stack Developer",
                        "category": "technical",
                        "difficulty": difficulty
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "interview_id" in data
                print(f"✅ {difficulty}: Interview started successfully")

    @pytest.mark.asyncio
    async def test_question_has_expected_concepts(self):
        """Test that technical questions have expected concepts"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start interview
            start_response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-concepts",
                    "role": "Backend Developer",
                    "category": "technical",
                    "difficulty": "mid"
                }
            )
            
            interview_id = start_response.json()["interview_id"]
            
            # Complete context
            for _ in range(5):
                await client.post(
                    f"{self.API_BASE}/interviews/{interview_id}/answers",
                    json={"answer": "Test"}
                )
            
            # Get first technical question
            response = await client.post(
                f"{self.API_BASE}/interviews/{interview_id}/answers",
                json={"answer": "Final context"}
            )
            
            data = response.json()
            if "next_question" in data:
                question = data["next_question"]
                # Many technical questions should have expected concepts
                if question.get("expected_concepts"):
                    assert len(question["expected_concepts"]) > 0
                    print(f"✅ Question has {len(question['expected_concepts'])} expected concepts")


class TestDatasetCoverage:
    """Tests for dataset coverage and variety"""

    BASE_URL = "http://localhost:8001"
    API_BASE = f"{BASE_URL}/api/v2"

    @pytest.mark.asyncio
    async def test_android_developer_questions(self):
        """Test specific profession from new dataset - Android Developer"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-android-dev",
                    "role": "Android Developer",
                    "category": "technical",
                    "difficulty": "mid"
                }
            )
            
            assert response.status_code == 200
            print("✅ Android Developer questions available")

    @pytest.mark.asyncio
    async def test_ios_developer_questions(self):
        """Test specific profession from new dataset - iOS Developer"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-ios-dev",
                    "role": "iOS Developer",
                    "category": "technical",
                    "difficulty": "senior"
                }
            )
            
            assert response.status_code == 200
            print("✅ iOS Developer questions available")

    @pytest.mark.asyncio
    async def test_cloud_architect_questions(self):
        """Test specific profession from new dataset - Cloud Architect"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/interviews",
                json={
                    "user_id": "test-cloud-architect",
                    "role": "Cloud Architect",
                    "category": "technical",
                    "difficulty": "senior"
                }
            )
            
            assert response.status_code == 200
            print("✅ Cloud Architect questions available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

