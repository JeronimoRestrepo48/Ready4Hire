import time
from app.interview_agent import InterviewAgent


def make_session(history=None, level='junior'):
    return {
        'history': history or [],
        'level': level,
        'exam_start_time': None,
        'exam_answers': [],
    }


def test_get_adaptive_level_up():
    agent = InterviewAgent.__new__(InterviewAgent)
    # two correct in a row
    session = make_session(history=[{'is_correct': True}, {'is_correct': True}], level='junior')
    level = agent._get_adaptive_level(session)
    assert level in ['mid', 'senior'] and level != 'junior'


def test_get_adaptive_level_down():
    agent = InterviewAgent.__new__(InterviewAgent)
    # two incorrect in a row
    session = make_session(history=[{'is_correct': False}, {'is_correct': False}], level='mid')
    level = agent._get_adaptive_level(session)
    assert level in ['junior', 'mid'] and level != 'senior'


def test_get_example_python():
    agent = InterviewAgent.__new__(InterviewAgent)
    ex = agent._get_example('Python class example')
    assert 'def ' in ex or 'clase' in ex.lower() or 'función' in ex.lower()


def test_get_learning_resources_returns_list():
    agent = InterviewAgent.__new__(InterviewAgent)
    resources = agent._get_learning_resources('python')
    assert isinstance(resources, list) and len(resources) > 0


def test_detect_bias_emotional_sequence():
    agent = InterviewAgent.__new__(InterviewAgent)
    session = {'history': [{'emotion': 'sadness'}, {'emotion': 'fear'}, {'emotion': 'anger'}]}
    assert agent._detect_bias(session) is True


def test_detect_bias_thematic_sequence():
    agent = InterviewAgent.__new__(InterviewAgent)
    session = {'history': [{'cluster': 5}, {'cluster': 5}, {'cluster': 5}]}
    assert agent._detect_bias(session) is True
