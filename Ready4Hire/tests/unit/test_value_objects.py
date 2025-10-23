"""
Unit tests for Value Objects.
Tests immutability and validation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest

from app.domain.value_objects.score import Score
from app.domain.value_objects.emotion import Emotion
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.value_objects.interview_status import InterviewStatus


class TestScore:
    """Tests for Score value object."""
    
    def test_create_valid_score(self):
        """Test creating a valid score."""
        score = Score(7.5)
        assert score.value == 7.5
    
    def test_score_rounds_to_one_decimal(self):
        """Test that score rounds to 1 decimal place."""
        score = Score(7.555)
        assert score.value == 7.6
    
    def test_score_out_of_range_raises_error(self):
        """Test that out of range scores raise ValueError."""
        with pytest.raises(ValueError, match="Score must be between 0 and 10"):
            Score(11.0)
        
        with pytest.raises(ValueError, match="Score must be between 0 and 10"):
            Score(-1.0)
    
    def test_is_passing(self):
        """Test is_passing method."""
        assert Score(8.0).is_passing() is True
        assert Score(6.0).is_passing() is True
        assert Score(5.9).is_passing() is False
        assert Score(3.0).is_passing() is False
    
    def test_quality_level(self):
        """Test quality_level method."""
        assert Score(9.5).quality_level() == "excellent"
        assert Score(8.0).quality_level() == "good"
        assert Score(6.0).quality_level() == "acceptable"
        assert Score(4.0).quality_level() == "needs_improvement"
    
    def test_score_equality(self):
        """Test score equality comparison."""
        assert Score(7.5) == Score(7.5)
        assert Score(7.5) != Score(7.6)
    
    def test_score_is_immutable(self):
        """Test that score value cannot be changed."""
        score = Score(7.5)
        with pytest.raises(AttributeError):
            score.value = 8.0  # Should raise AttributeError


class TestEmotion:
    """Tests for Emotion enum."""
    
    def test_emotion_values(self):
        """Test all emotion values exist."""
        assert Emotion.JOY.value == "joy"
        assert Emotion.SADNESS.value == "sadness"
        assert Emotion.ANGER.value == "anger"
        assert Emotion.FEAR.value == "fear"
        assert Emotion.SURPRISE.value == "surprise"
        assert Emotion.NEUTRAL.value == "neutral"
    
    def test_is_positive(self):
        """Test is_positive method."""
        assert Emotion.JOY.is_positive() is True
        assert Emotion.SURPRISE.is_positive() is True
        assert Emotion.SADNESS.is_positive() is False
        assert Emotion.ANGER.is_positive() is False
        assert Emotion.NEUTRAL.is_positive() is False
    
    def test_is_negative(self):
        """Test is_negative method."""
        assert Emotion.SADNESS.is_negative() is True
        assert Emotion.ANGER.is_negative() is True
        assert Emotion.FEAR.is_negative() is True
        assert Emotion.JOY.is_negative() is False
        assert Emotion.NEUTRAL.is_negative() is False


class TestSkillLevel:
    """Tests for SkillLevel enum."""
    
    def test_skill_level_values(self):
        """Test all skill level values exist."""
        assert SkillLevel.JUNIOR.value == "junior"
        assert SkillLevel.MID.value == "mid"
        assert SkillLevel.SENIOR.value == "senior"
    
    def test_from_string(self):
        """Test creating SkillLevel from string."""
        assert SkillLevel.from_string("junior") == SkillLevel.JUNIOR
        assert SkillLevel.from_string("mid") == SkillLevel.MID
        assert SkillLevel.from_string("senior") == SkillLevel.SENIOR
    
    def test_from_string_invalid_raises_error(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            SkillLevel.from_string("expert")
    
    def test_can_increase_to(self):
        """Test skill level transitions."""
        assert SkillLevel.JUNIOR.can_increase_to(SkillLevel.MID) is True
        assert SkillLevel.JUNIOR.can_increase_to(SkillLevel.SENIOR) is False
        
        assert SkillLevel.MID.can_increase_to(SkillLevel.SENIOR) is True
        assert SkillLevel.MID.can_increase_to(SkillLevel.JUNIOR) is False
        
        assert SkillLevel.SENIOR.can_increase_to(SkillLevel.JUNIOR) is False
    
    def test_to_numeric(self):
        """Test converting to numeric value."""
        assert SkillLevel.JUNIOR.to_numeric() == 0
        assert SkillLevel.MID.to_numeric() == 1
        assert SkillLevel.SENIOR.to_numeric() == 2


class TestInterviewStatus:
    """Tests for InterviewStatus enum."""
    
    def test_interview_status_values(self):
        """Test all interview status values exist."""
        assert InterviewStatus.CREATED.value == "created"
        assert InterviewStatus.ACTIVE.value == "active"
        assert InterviewStatus.PAUSED.value == "paused"
        assert InterviewStatus.COMPLETED.value == "completed"
        assert InterviewStatus.CANCELLED.value == "cancelled"
    
    def test_can_add_questions(self):
        """Test can_add_questions method."""
        assert InterviewStatus.CREATED.can_add_questions() is True
        assert InterviewStatus.ACTIVE.can_add_questions() is True
        assert InterviewStatus.PAUSED.can_add_questions() is True
        assert InterviewStatus.COMPLETED.can_add_questions() is False
        assert InterviewStatus.CANCELLED.can_add_questions() is False
    
    def test_is_terminal(self):
        """Test is_terminal method."""
        assert InterviewStatus.COMPLETED.is_terminal() is True
        assert InterviewStatus.CANCELLED.is_terminal() is True
        assert InterviewStatus.ACTIVE.is_terminal() is False
        assert InterviewStatus.PAUSED.is_terminal() is False

