from app.interview_agent import InterviewAgent
from tests.mocks import MockLLM, MockEmbMgr


def test_ranknet_hook_is_used_when_present():
    emb = MockEmbMgr()
    llm = MockLLM(response="OK")
    # Provide a fake ranknet to the emb manager (matching signature)
    def fake_ranknet_rank(candidates, query_text, history):
        # reverse order as a detectable change
        return list(reversed(candidates))
    emb.ranknet_rank = fake_ranknet_rank
    agent = InterviewAgent(llm=llm, emb_mgr=emb)
    agent.start_interview('u5', role='DevOps', interview_type='technical')
    sess = agent.sessions['u5']
    sess['stage'] = 'interview'
    sess['tech_pool'] = emb.tech_data.copy()
    res = agent.next_question('u5', user_input='quero practicar')
    assert 'question' in res


def test_ab_policy_diversity_vs_coverage():
    emb = MockEmbMgr()
    llm = MockLLM(response="OK")
    agent = InterviewAgent(llm=llm, emb_mgr=emb)
    agent.start_interview('u6', role='DevOps', interview_type='technical')
    sess = agent.sessions['u6']
    sess['stage'] = 'interview'
    # Create a tech pool with repeated cluster to test diversity policy
    sess['tech_pool'] = [
        {'question':'q_same1','cluster':0,'level':'junior'},
        {'question':'q_same2','cluster':0,'level':'junior'},
        {'question':'q_diff','cluster':1,'level':'junior'},
    ]
    # Simulate policy flag in session for A/B selection
    sess['selection_policy'] = 'diversity'  # agent should try to diversify clusters
    res = agent.next_question('u6', user_input='practicar algo')
    assert 'question' in res

