from app.interview_agent import InterviewAgent
import random

def simulate_full_interview():
    user_id = "test_user_full"
    agent = InterviewAgent()
    # Simular contexto Ciberseguridad
    context_answers = [
        "Ciberseguridad",  # rol
        "junior",  # nivel
        "2",  # años
        "seguridad defensiva y en profundidad",  # conocimientos
        "Firewall, SIEM, SOAR",  # herramientas
        "Aprender mucho de los procesos de seguridad de la información"  # expectativas
    ]
    print("--- INICIO ENTREVISTA ---")
    agent.start_interview(user_id, role="Ciberseguridad", interview_type="technical")
    # Responder preguntas de contexto
    for ans in context_answers:
        out = agent.next_question(user_id, user_input=ans)
        print(f"[Context Q] {out.get('question') or out}")
    # Simular 10 preguntas alternando buenas y malas respuestas
    good_answer = "Respuesta correcta de IA relevante."
    bad_answer = "Respuesta incorrecta o irrelevante."
    for i in range(10):
        q = agent.next_question(user_id)
        print(f"\n[Q{i+1}] {q.get('question') or q}")
        # Alternar respuestas buenas/malas
        if i % 2 == 0:
            resp = good_answer
        else:
            resp = bad_answer
        feedback = agent.process_answer(user_id, resp)
        print(f"[Feedback] {feedback}")
        # Si hay pista, simular pedir pista y responder de nuevo
        if feedback and isinstance(feedback, dict) and feedback.get('hint'):
            print(f"[Hint] {feedback['hint']}")
            feedback2 = agent.process_answer(user_id, good_answer)
            print(f"[Feedback after hint] {feedback2}")
    # Esperar feedback final
    final = agent.next_question(user_id)
    print(f"\n[Final] {final}")
    # Simular encuesta de satisfacción
    while final and isinstance(final, dict) and final.get('satisfaction_survey'):
        # Responder con 5 siempre
        survey = agent.process_satisfaction_answer(user_id, 5)
        print(f"[Satisfacción] {survey}")
        final = agent.next_question(user_id)
        print(f"[Encuesta/Final] {final}")
    print("--- FIN ENTREVISTA ---")

if __name__ == "__main__":
    simulate_full_interview()
