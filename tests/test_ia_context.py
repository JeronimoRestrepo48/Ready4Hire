from app.interview_agent import InterviewAgent

user_id = 'test_ia'
role = 'IA'
level = 'senior'
years = 5
knowledge = ['machine learning', 'deep learning', 'NLP']
tools = ['TensorFlow', 'PyTorch', 'scikit-learn']

def simulate_ia_context():
    agent = InterviewAgent()
    # Iniciar entrevista con contexto IA
    agent.start_interview(user_id, role=role, interview_type='technical')
    # Responder preguntas de contexto
    context_questions = agent.sessions[user_id]['context_questions']
    # Simular respuestas de contexto
    for i, q in enumerate(context_questions):
        if 'rol' in q.lower():
            agent.next_question(user_id, user_input=role)
        elif 'nivel' in q.lower():
            agent.next_question(user_id, user_input=level)
        elif 'años' in q.lower():
            agent.next_question(user_id, user_input=str(years))
        elif 'fortalezas' in q.lower():
            agent.next_question(user_id, user_input=', '.join(knowledge))
        elif 'herramientas' in q.lower():
            agent.next_question(user_id, user_input=', '.join(tools))
        else:
            agent.next_question(user_id, user_input='IA')
    # Lanzar la primera pregunta técnica
    q = agent.next_question(user_id, user_input='')
    print('Primera pregunta técnica sugerida para IA:', q)

if __name__ == '__main__':
    simulate_ia_context()
