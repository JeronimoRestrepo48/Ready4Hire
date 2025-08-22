from app.interview_agent import InterviewAgent
from app.embeddings.embeddings_manager import EmbeddingsManager

# Mock user/session for test
user_id = 'testuser'
role = 'Backend'

def simulate_wrong_answer():
    agent = InterviewAgent(llm=None, emb_mgr=None)  # Use default LLM (should be llama3 or fallback)
    agent.start_interview(user_id, role=role, interview_type='technical')
    # Avanza contexto
    for _ in range(6):
        agent.next_question(user_id, user_input='Python')
    # Lanza una pregunta técnica real
    q = agent.next_question(user_id, user_input='Python')
    print('Pregunta:', q)
    # Responde mal varias veces para forzar pistas
    for i in range(1, 5):
        resp = agent.process_answer(user_id, 'Respuesta incorrecta')
        print(f'Intento {i}:')
        print(resp['feedback'])
        if not resp.get('retry', True):
            break

if __name__ == '__main__':
    simulate_wrong_answer()
