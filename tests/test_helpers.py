import pytest
from app.interview_agent import filter_by_level, penalize_covered_topics


def make_q(text, level='junior', cluster=None):
    return {'question': text, 'level': level, 'cluster': cluster}


def test_filter_by_level_no_level_returns_copy():
    pool = [make_q('q1','junior'), make_q('q2','mid')]
    out = filter_by_level(pool, None)
    assert out == pool and out is not pool


def test_filter_by_level_with_level():
    pool = [make_q('q1','junior'), make_q('q2','mid')]
    out = filter_by_level(pool, 'mid')
    assert len(out) == 1 and out[0]['level'] == 'mid'


def test_penalize_covered_topics_prefers_not_recent():
    pool = [make_q('q1', cluster=1), make_q('q2', cluster=2), make_q('q3', cluster=1)]
    history = [{'agent': 'q1', 'cluster': 1}, {'agent': 'q4', 'cluster': 3}]
    out = penalize_covered_topics(pool, history)
    # first of reordered should not have cluster 1 (which is recent)
    assert out[0]['cluster'] != 1
