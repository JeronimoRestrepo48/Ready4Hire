from app.interview_agent import InterviewAgent
from tests.mocks import MockLLM, MockEmbMgr
from app.interview_agent import InterviewAgent
from tests.mocks import MockLLM, MockEmbMgr


def test_next_question_with_mocked_emb_mgr_and_llm():
    emb = MockEmbMgr()
    llm = MockLLM(response="Feedback from mock llm")
    agent = InterviewAgent(llm=llm, emb_mgr=emb)
    # start interview
    res = agent.start_interview('u1', role='DevOps', interview_type='technical')
    assert 'question' in res
    # simulate user input to trigger selection
    nextq = agent.next_question('u1', user_input='mi respuesta')
    assert 'question' in nextq


def test_process_answer_with_mocked_components():
    emb = MockEmbMgr()
    llm = MockLLM(response="Good job!")
    agent = InterviewAgent(llm=llm, emb_mgr=emb)
    agent.start_interview('u2', role='DevOps', interview_type='technical')
    # ask a question
    q = agent.next_question('u2')
    # process an answer
    res = agent.process_answer('u2', 'resp1')
    assert 'feedback' in res or 'error' in res
