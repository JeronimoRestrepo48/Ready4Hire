"""
Unit tests for Interview entity.
Tests business logic and invariants.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
from datetime import datetime

from app.domain.entities.interview import Interview
from app.domain.entities.question import Question
from app.domain.entities.answer import Answer
from app.domain.value_objects.interview_status import InterviewStatus
from app.domain.value_objects.skill_level import SkillLevel
from app.domain.value_objects.score import Score
from app.domain.value_objects.emotion import Emotion


class TestInterviewEntity:
    """Tests for Interview aggregate root."""
    
    def test_create_interview(self):
        """Test creating a new interview."""
        interview = Interview(
            id="test-interview-1",
            user_id="user-123",
            role="Backend Developer",
            interview_type="technical",
            skill_level=SkillLevel.MID,
            mode="practice"
        )
        
        assert interview.id == "test-interview-1"
        assert interview.user_id == "user-123"
        assert interview.role == "Backend Developer"
        assert interview.status == InterviewStatus.CREATED
        assert interview.skill_level == SkillLevel.MID
        assert interview.interview_type == "technical"
        assert interview.mode == "practice"
        assert interview.current_phase == "context"
        assert len(interview.context_answers) == 0
        assert interview.current_question is None
    
    def test_start_interview(self):
        """Test starting an interview."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        
        interview.start()
        
        assert interview.status == InterviewStatus.ACTIVE
        assert interview.started_at is not None
    
    def test_cannot_start_already_active_interview(self):
        """Test that we can't start an already active interview."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        
        interview.start()
        
        with pytest.raises(ValueError, match="Cannot start interview"):
            interview.start()
    
    def test_add_question(self):
        """Test adding a question to an interview."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        interview.start()
        
        question = Question(
            id="q-1",
            text="What is Python?",
            category="technical",
            difficulty="junior",
            topic="programming",
            keywords=["python"],
            expected_concepts=["programming language"]
        )
        
        interview.add_question(question)
        
        assert interview.current_question == question
        assert question in interview.questions_history
    
    def test_cannot_add_question_to_completed_interview(self):
        """Test that we can't add questions to a completed interview."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        interview.start()
        interview.complete()
        
        question = Question(
            id="q-1",
            text="Test question",
            category="technical",
            difficulty="junior"
        )
        
        with pytest.raises(ValueError, match="Cannot add questions"):
            interview.add_question(question)
    
    def test_add_answer(self):
        """Test adding an answer to an interview."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        interview.start()
        
        question = Question(
            id="q-1",
            text="What is Python?",
            category="technical",
            difficulty="junior"
        )
        interview.add_question(question)
        
        answer = Answer(
            question_id="q-1",
            text="Python is a programming language",
            score=Score(8.0),
            is_correct=True,
            emotion=Emotion.NEUTRAL,
            time_taken=45
        )
        
        interview.add_answer(answer)
        
        assert answer in interview.answers_history
        assert interview.current_question is None  # Cleared after answering
    
    def test_complete_interview(self):
        """Test completing an interview."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        interview.start()
        interview.complete()
        
        assert interview.status == InterviewStatus.COMPLETED
        assert interview.completed_at is not None
    
    def test_get_score_average(self):
        """Test calculating average score."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        interview.start()
        
        # Add question and answer
        question = Question(id="q-1", text="Test", category="technical", difficulty="junior")
        interview.add_question(question)
        
        answer1 = Answer(
            question_id="q-1",
            text="Answer 1",
            score=Score(8.0),
            is_correct=True,
            emotion=Emotion.NEUTRAL,
            time_taken=30
        )
        interview.add_answer(answer1)
        
        # Add another
        question2 = Question(id="q-2", text="Test 2", category="technical", difficulty="junior")
        interview.add_question(question2)
        
        answer2 = Answer(
            question_id="q-2",
            text="Answer 2",
            score=Score(6.0),
            is_correct=True,
            emotion=Emotion.NEUTRAL,
            time_taken=40
        )
        interview.add_answer(answer2)
        
        avg = interview.get_score_average()
        assert avg == 7.0
    
    def test_get_accuracy(self):
        """Test calculating accuracy percentage."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        interview.start()
        
        # Add correct answer
        q1 = Question(id="q-1", text="Test", category="technical", difficulty="junior")
        interview.add_question(q1)
        a1 = Answer(
            question_id="q-1",
            text="Good answer",
            score=Score(8.0),
            is_correct=True,
            emotion=Emotion.NEUTRAL,
            time_taken=30
        )
        interview.add_answer(a1)
        
        # Add incorrect answer
        q2 = Question(id="q-2", text="Test 2", category="technical", difficulty="junior")
        interview.add_question(q2)
        a2 = Answer(
            question_id="q-2",
            text="Bad answer",
            score=Score(3.0),
            is_correct=False,
            emotion=Emotion.NEUTRAL,
            time_taken=20
        )
        interview.add_answer(a2)
        
        accuracy = interview.get_accuracy()
        assert accuracy == 50.0  # 1 out of 2 correct
    
    def test_context_answers(self):
        """Test managing context answers."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        
        interview.add_context_answer("3 years of experience")
        interview.add_context_answer("Python, FastAPI, PostgreSQL")
        interview.advance_context_question()
        
        assert len(interview.context_answers) == 2
        assert interview.context_question_index == 1
    
    def test_transition_to_questions_phase(self):
        """Test transitioning from context to questions phase."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR,
            current_phase="context"
        )
        
        interview.transition_to_questions_phase()
        
        assert interview.current_phase == "questions"
        assert interview.is_in_questions_phase()
        assert not interview.is_in_context_phase()
    
    def test_streak_tracking(self):
        """Test streak tracking for consecutive correct answers."""
        interview = Interview(
            id="test-1",
            user_id="user-1",
            role="Developer",
            skill_level=SkillLevel.JUNIOR
        )
        interview.start()
        
        # Add 3 correct answers
        for i in range(3):
            q = Question(id=f"q-{i}", text=f"Test {i}", category="technical", difficulty="junior")
            interview.add_question(q)
            a = Answer(
                question_id=f"q-{i}",
                text=f"Answer {i}",
                score=Score(8.0),
                is_correct=True,
                emotion=Emotion.NEUTRAL,
                time_taken=30
            )
            interview.add_answer(a)
        
        assert interview.current_streak == 3
        assert interview.max_streak == 3
        
        # Add incorrect answer - breaks streak
        q = Question(id="q-break", text="Break", category="technical", difficulty="junior")
        interview.add_question(q)
        a = Answer(
            question_id="q-break",
            text="Wrong",
            score=Score(3.0),
            is_correct=False,
            emotion=Emotion.NEUTRAL,
            time_taken=20
        )
        interview.add_answer(a)
        
        assert interview.current_streak == 0
        assert interview.max_streak == 3  # Max stays at 3

