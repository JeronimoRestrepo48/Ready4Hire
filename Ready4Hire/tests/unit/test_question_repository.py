"""
Unit tests for JsonQuestionRepository with updated datasets.
Tests that the repository correctly loads technical_questions_by_profession_v3.jsonl
and soft_skills.jsonl.
"""

import pytest
import os
from pathlib import Path
from app.infrastructure.persistence.json_question_repository import JsonQuestionRepository


class TestJsonQuestionRepository:
    """Test suite for JsonQuestionRepository with new datasets"""

    @pytest.fixture
    def repository(self):
        """Create repository instance with actual dataset files"""
        base_path = Path(__file__).parent.parent.parent / "app" / "datasets"
        tech_file = str(base_path / "technical_questions_by_profession_v3.jsonl")
        soft_file = str(base_path / "soft_skills.jsonl")
        
        return JsonQuestionRepository(tech_file=tech_file, soft_file=soft_file)

    def test_repository_initialization(self, repository):
        """Test that repository initializes and loads questions"""
        assert repository is not None
        assert len(repository._tech_cache) > 0, "Technical questions should be loaded"
        assert len(repository._soft_cache) > 0, "Soft skills questions should be loaded"

    def test_technical_questions_count(self, repository):
        """Test that correct number of technical questions are loaded"""
        # Should have ~810 technical questions
        tech_count = len(repository._tech_cache)
        assert tech_count >= 800, f"Expected at least 800 technical questions, got {tech_count}"
        assert tech_count <= 900, f"Expected at most 900 technical questions, got {tech_count}"

    def test_soft_skills_questions_count(self, repository):
        """Test that correct number of soft skills questions are loaded"""
        # Should have ~101 soft skills questions
        soft_count = len(repository._soft_cache)
        assert soft_count >= 100, f"Expected at least 100 soft skills questions, got {soft_count}"
        assert soft_count <= 110, f"Expected at most 110 soft skills questions, got {soft_count}"

    @pytest.mark.asyncio
    async def test_find_by_role(self, repository):
        """Test finding questions by role"""
        # Test with Backend Developer
        backend_questions = await repository.find_by_role("Backend Developer", category="technical")
        assert len(backend_questions) > 0, "Should find Backend Developer questions"
        
        # Verify questions are for correct role
        for q in backend_questions[:5]:  # Check first 5
            assert "backend" in q.role.lower() or "developer" in q.role.lower()

    @pytest.mark.asyncio
    async def test_find_by_difficulty(self, repository):
        """Test finding questions by difficulty level"""
        # Test each difficulty level
        for difficulty in ["junior", "mid", "senior"]:
            questions = await repository.find_by_difficulty(difficulty, category="technical")
            assert len(questions) > 0, f"Should find {difficulty} questions"
            
            # Verify difficulty matches
            for q in questions[:3]:  # Check first 3
                assert q.difficulty == difficulty

    @pytest.mark.asyncio
    async def test_find_all_technical(self, repository):
        """Test retrieving all technical questions"""
        all_tech = await repository.find_all_technical()
        assert len(all_tech) > 0
        assert len(all_tech) == len(repository._tech_cache)

    @pytest.mark.asyncio
    async def test_find_all_soft_skills(self, repository):
        """Test retrieving all soft skills questions"""
        all_soft = await repository.find_all_soft_skills()
        assert len(all_soft) > 0
        assert len(all_soft) == len(repository._soft_cache)

    @pytest.mark.asyncio
    async def test_search_by_query(self, repository):
        """Test searching questions by text query"""
        # Search for API-related questions
        api_questions = await repository.search("API", category="technical")
        assert len(api_questions) > 0, "Should find API-related questions"

    @pytest.mark.asyncio
    async def test_find_by_id(self, repository):
        """Test finding a specific question by ID"""
        # Get first technical question
        all_tech = await repository.find_all_technical()
        if all_tech:
            first_question = all_tech[0]
            found = await repository.find_by_id(first_question.id)
            assert found is not None
            assert found.id == first_question.id
            assert found.text == first_question.text

    @pytest.mark.asyncio
    async def test_questions_have_required_fields(self, repository):
        """Test that loaded questions have all required fields"""
        all_questions = await repository.find_all_technical()
        
        # Check first 10 questions
        for q in all_questions[:10]:
            assert q.id is not None, "Question should have ID"
            assert q.text is not None, "Question should have text"
            assert len(q.text) > 0, "Question text should not be empty"
            assert q.category in ["technical", "soft_skills"], "Category should be valid"
            assert q.difficulty in ["junior", "mid", "senior", "context"], "Difficulty should be valid"

    def test_multiple_professions_loaded(self, repository):
        """Test that questions for multiple professions are loaded"""
        all_tech = repository._tech_cache
        roles = set(q.role for q in all_tech)
        
        # Should have questions for multiple roles
        assert len(roles) >= 20, f"Expected at least 20 different roles, got {len(roles)}"
        
        # Check for some specific roles
        role_names = [r.lower() for r in roles]
        expected_roles = ["backend developer", "frontend developer", "data scientist", "devops engineer"]
        
        for expected in expected_roles:
            assert any(expected in role for role in role_names), f"Should have questions for {expected}"

    @pytest.mark.asyncio
    async def test_reload_functionality(self, repository):
        """Test that reload() reloads questions from files"""
        initial_count = len(repository._tech_cache)
        repository.reload()
        reloaded_count = len(repository._tech_cache)
        
        assert initial_count == reloaded_count, "Reload should maintain same question count"

    def test_no_obsolete_datasets_loaded(self, repository):
        """Test that obsolete datasets are not being loaded"""
        # Verify no references to old files
        assert "tech_questions.jsonl" not in str(repository.tech_file)
        assert "multiprofession_questions.jsonl" not in str(repository.tech_file)
        assert "technical_questions_by_profession_v3.jsonl" in str(repository.tech_file)


class TestQuestionQuality:
    """Tests for question data quality"""

    @pytest.fixture
    def repository(self):
        base_path = Path(__file__).parent.parent.parent / "app" / "datasets"
        tech_file = str(base_path / "technical_questions_by_profession_v3.jsonl")
        soft_file = str(base_path / "soft_skills.jsonl")
        return JsonQuestionRepository(tech_file=tech_file, soft_file=soft_file)

    @pytest.mark.asyncio
    async def test_questions_not_empty(self, repository):
        """Test that question texts are not empty"""
        all_questions = await repository.find_all_technical()
        for q in all_questions:
            assert len(q.text.strip()) > 10, f"Question text too short: {q.text}"

    @pytest.mark.asyncio
    async def test_expected_concepts_present(self, repository):
        """Test that technical questions have expected concepts"""
        tech_questions = await repository.find_all_technical()
        
        questions_with_concepts = [q for q in tech_questions if q.expected_concepts]
        assert len(questions_with_concepts) > 0, "Some questions should have expected concepts"

    @pytest.mark.asyncio
    async def test_difficulty_distribution(self, repository):
        """Test that questions are distributed across difficulty levels"""
        all_tech = await repository.find_all_technical()
        
        junior = [q for q in all_tech if q.difficulty == "junior"]
        mid = [q for q in all_tech if q.difficulty == "mid"]
        senior = [q for q in all_tech if q.difficulty == "senior"]
        
        # Should have questions at each level
        assert len(junior) > 0, "Should have junior questions"
        assert len(mid) > 0, "Should have mid-level questions"
        assert len(senior) > 0, "Should have senior questions"
        
        # Distribution should be relatively balanced (within 50% of each other)
        counts = [len(junior), len(mid), len(senior)]
        max_count = max(counts)
        min_count = min(counts)
        assert max_count / min_count < 2.0, "Difficulty distribution should be relatively balanced"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

