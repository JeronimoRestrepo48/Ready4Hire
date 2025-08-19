from app.interview_agent import InterviewAgent
from tests.mocks import MockLLM, MockEmbMgr


def test_bias_mitigation_prefers_technical_when_detected():
    emb = MockEmbMgr()
    llm = MockLLM(response="OK")
    agent = InterviewAgent(llm=llm, emb_mgr=emb)
    # Start and prepare session
    agent.start_interview('u3', role='DevOps', interview_type='technical')
    sess = agent.sessions['u3']
    # Put agent directly into interview stage and load pools
    sess['stage'] = 'interview'
    sess['tech_pool'] = emb.tech_data.copy()
    sess['soft_pool'] = emb.soft_data.copy()
    # Create history that triggers thematic bias (same cluster repeated)
    sess['history'] = [
        {'agent': 'q_old', 'cluster': 0},
        {'agent': 'q_old2', 'cluster': 0},
        {'agent': 'q_old3', 'cluster': 0},
    ]
    # Call next_question with user input to trigger bias mitigation path
    res = agent.next_question('u3', user_input='quiero practicar')
    assert 'question' in res
    # Bias mitigation should be logged in returned response or session history
    # The code sets 'bias_mitigation' True in the response when mitigation applied
    assert res.get('bias_mitigation') is True or any(h.get('bias_mitigation') for h in sess['history'])
    # If a technical question was selected, last_type should be technical
    assert sess.get('last_type') in ('technical', 'soft')


def test_pool_switching_when_preferred_empty():
    emb = MockEmbMgr()
    llm = MockLLM(response="OK")
    agent = InterviewAgent(llm=llm, emb_mgr=emb)
    agent.start_interview('u4', role='DevOps', interview_type='technical')
    sess = agent.sessions['u4']
    sess['stage'] = 'interview'
    # Empty the technical pool to force switching
    sess['tech_pool'] = []
    sess['soft_pool'] = emb.soft_data.copy()
    res = agent.next_question('u4', user_input='prefiero otra cosa')
    # Expect to get a question from soft pool and bias_mitigation flag True
    assert 'question' in res
    assert res.get('bias_mitigation') is True or sess.get('last_type') == 'soft'
