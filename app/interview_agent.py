# Entrevistador IA robusto para Ingeniería de Sistemas y ramas

from langchain.llms import Ollama
from typing import Dict, Any, List, Optional
import random
import json
from pathlib import Path
import time
# Integración de embeddings
from app.embeddings.embeddings_manager import EmbeddingsManager
from app.services.emotion_analyzer import analyze_emotion
from app.services.ml_feedback_loop import generate_feedback_ml
from app.core.utils_interview import log_llm_error, filter_questions, process_survey_metrics, generate_final_feedback
from app.services.ml_recommendations import suggest_learning_path
from app.services.ml_dynamic_difficulty import adjust_difficulty
from sentence_transformers import util

class InterviewAgent:
    """
    Agente de entrevistas que alterna preguntas técnicas y de habilidades blandas,
    usando datasets personalizados y modelos LLM locales vía Ollama.
    """
    def __init__(self, model_name: str = "llama3"):
        # --- Configuración Principal ---
        self.llm = Ollama(model=model_name)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.emb_mgr = EmbeddingsManager()

        # --- Datasets ---
        self.tech_questions = self._load_dataset('tech_questions.jsonl')
        self.soft_questions = self._load_dataset('soft_skills.jsonl')

        # --- Personalidad del Agente (Emojis y Frases) ---
        self.motivational_phrases = [
            "¡Sigue así, vas por un gran camino! 🚀", "¡Tu esfuerzo se nota! 👏",
            "¡Demuestras dominio y seguridad! 😎", "¡Suma un logro más a tu carrera! 🥇",
            "¡Inspiras confianza! 💪"
        ]
        self.positive_emojis = ["🎉", "✔️", "💡", "🏆", "🙌", "🚀", "👏", "😎", "🥇", "💪", "🌟"]
        self.negative_phrases = [
            "No te desanimes, cada error es una oportunidad para aprender. 💡",
            "¡Ánimo! Repasa el concepto y sigue practicando. 📚",
            "Recuerda: equivocarse es parte del proceso de mejora. 🔁",
            "Sigue intentándolo, la perseverancia es clave. 💪",
            "¡Tú puedes! El siguiente intento será mejor. 🌟"
        ]
        self.negative_emojis = ["🤔", "🧐", "😅", "🔄", "💡", "📚", "🔁", "💪", "🌟"]

        # --- Gamificación ---
        self.badges = [
            {"name": "Primer intento", "condition": lambda s: s.get('points', 0) >= 10},
            {"name": "Racha x5", "condition": lambda s: s.get('streak', 0) >= 5},
            {"name": "Nivel 3", "condition": lambda s: s.get('level', 1) >= 3},
        ]

        # --- Métricas y Estado ---
        self.metrics = {
            'interviews_started': 0, 'interviews_finished': 0,
            'satisfaction_scores': [], 'user_performance': {}
        }
        self.similarity_threshold = 0.75

    def _load_dataset(self, filename: str) -> List[dict]:
        path = Path(__file__).parent / 'datasets' / filename
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def start_interview(self, user_id: str, role: Optional[str] = None, interview_type: Optional[str] = None, mode: str = "practice"):
        self.metrics['interviews_started'] += 1
        context_questions = [
            "¿Para qué rol específico deseas prepararte? (IA, ciberseguridad, devops, infraestructura, etc.)",
            "¿Cuál es tu nivel de experiencia profesional (junior, semi-senior, senior)?",
            "¿Cuántos años de experiencia tienes en el rol?",
            "¿Qué conocimientos o tecnologías consideras tus fortalezas?",
            "¿Qué herramientas dominas mejor?",
            "¿Qué expectativas tienes sobre el trabajo o la empresa?"
        ]
        self.sessions[user_id] = {
            "role": role, "interview_type": interview_type or "technical", "mode": mode,
            "stage": "context", "history": [], "context_asked": 0,
            "question_pool": [], "question_counter": 0,
            "points": 0, "level_num": 1, "streak": 0, "badges": set(),
            "attempts": {}, "error_topics": {}, "topic_success": {}
        }
        question = context_questions[0]
        self.sessions[user_id]["history"].append({"agent": question})
        return {"question": question}

    def _select_interview_questions(self, session: dict, total_questions: int = 10):
        interview_type = session.get("interview_type", "technical").lower()
        user_context_str = f"{session.get('role', '')} {session.get('level', '')} {session.get('knowledge', '')}"

        # Determinar el número de preguntas de cada tipo
        if interview_type == "technical":
            num_tech, num_soft = total_questions, 0
        elif interview_type == "soft":
            num_tech, num_soft = 0, total_questions
        elif interview_type == "mixed":
            num_tech, num_soft = 7, 3
        else: # Fallback
            num_tech, num_soft = total_questions, 0

        # Seleccionar preguntas técnicas
        tech_pool = []
        if num_tech > 0:
            candidates = [q for q in self.tech_questions if session.get("role", "").lower() in q.get("role", "").lower()]
            if len(candidates) < num_tech: candidates = self.tech_questions
            tech_pool = self._select_and_rank_questions(user_context_str, candidates, num_tech)

        # Seleccionar preguntas de habilidades blandas
        soft_pool = []
        if num_soft > 0:
            soft_pool = self._select_and_rank_questions(user_context_str, self.soft_questions, num_soft)

        # Combinar y barajar para entrevistas mixtas
        final_pool = tech_pool + soft_pool
        if interview_type == "mixed":
            random.shuffle(final_pool)

        session["question_pool"] = final_pool
        print(f"[DEBUG] Seleccionadas {len(final_pool)} preguntas ({num_tech}T/{num_soft}S) para entrevista '{interview_type}'.")

    def _select_and_rank_questions(self, user_context: str, question_list: List[dict], num_to_select: int) -> List[dict]:
        """Función auxiliar para seleccionar y clasificar preguntas por relevancia."""
        if not question_list:
            return []

        question_texts = [q.get("question") or q.get("scenario", "") for q in question_list]

        try:
            user_context_emb = self.emb_mgr.model.encode(user_context, convert_to_tensor=True)
            question_embs = self.emb_mgr.model.encode(question_texts, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(user_context_emb, question_embs)[0]

            questions_with_sim = list(zip(question_list, similarities))
            # Filtrar preguntas con similitud muy baja
            questions_with_sim = [item for item in questions_with_sim if item[1] > 0.1]

            # Ordenar por similitud
            sorted_by_sim = sorted(questions_with_sim, key=lambda x: x[1], reverse=True)

            # Extraer las mejores preguntas
            top_questions = [q for q, sim in sorted_by_sim[:num_to_select]]

        except Exception as e:
            print(f"[ERROR] Fallo en el cálculo de similitud, usando selección aleatoria. Error: {e}")
            top_questions = random.sample(question_list, min(num_to_select, len(question_list)))

        # Rellenar si no hay suficientes preguntas
        if len(top_questions) < num_to_select:
            existing_texts = {q.get("question") or q.get("scenario") for q in top_questions}
            remaining = [q for q in question_list if (q.get("question") or q.get("scenario")) not in existing_texts]
            needed = num_to_select - len(top_questions)
            top_questions.extend(random.sample(remaining, min(needed, len(remaining))))

        return top_questions


    def process_context_answer(self, user_id: str, answer: str):
        session = self.sessions.get(user_id)
        if not session: return {"error": "Entrevista no iniciada"}

        context_keys = ["role", "level", "years", "knowledge", "tools", "expectations"]
        context_questions = [
            "¿Cuál es tu nivel de experiencia profesional (junior, semi-senior, senior)?",
            "¿Cuántos años de experiencia tienes en el rol?",
            "¿Qué conocimientos o tecnologías consideras tus fortalezas?",
            "¿Qué herramientas dominas mejor?",
            "¿Qué expectativas tienes sobre el trabajo o la empresa?",
            "¡Excelente! He recopilado tu perfil. Ahora seleccionaré las mejores preguntas para ti. ¡Empecemos! 🚀"
        ]

        idx = session.get("context_asked", 0)
        # La primera respuesta de contexto es el rol, la guardamos.
        if idx == 0:
             session[context_keys[0]] = answer

        session["context_asked"] = idx + 1
        if session["context_asked"] < len(context_keys):
            # Guardar la respuesta anterior en la clave correcta
            session[context_keys[session["context_asked"]-1]] = answer
            question = context_questions[session["context_asked"]]
            session["history"].append({"agent": question})
            return {"question": question}
        else:
            session[context_keys[-1]] = answer # Guardar la última respuesta
            session["stage"] = "interview"

            # Llamar a la selección inteligente de preguntas
            self._select_interview_questions(session)

            return self.next_question(user_id)

    def next_question(self, user_id: str):
        session = self.sessions.get(user_id)
        if not session: return {"error": "Entrevista no iniciada"}

        if session.get('question_counter', 0) >= 10:
            return {"end": True, "message": "¡Has completado la entrevista! 🥳"}

        question_pool = session.get("question_pool", [])
        if not question_pool:
             # Finalizar si no hay más preguntas
            return {"end": True, "message": "¡Felicidades, has respondido todas las preguntas! 🥳"}

        question = question_pool.pop(0)
        question_text = question.get("question") or question.get("scenario")

        session["history"].append({"agent": question_text})
        session["last_question"] = question

        print(f"[DEBUG] Siguiente pregunta: {question_text}")
        return {"question": question_text}

    def process_answer(self, user_id: str, answer: str):
        import re
        session = self.sessions.get(user_id)
        if not session: return {"error": "Entrevista no iniciada"}

        session["history"].append({"user": answer})
        last_question = session.get("last_question")
        if not last_question: return {"error": "No se encontró la última pregunta."}

        # 1. Evaluar la respuesta
        is_correct, scores = self._evaluate_answer(answer, last_question)

        # 2. Generar feedback
        feedback = self._generate_feedback(is_correct, session, last_question, scores)
        session["history"].append({"feedback": feedback})

        # 3. Actualizar estado y gamificación
        if is_correct:
            self._update_session_on_correct(user_id, session, last_question, answer, scores)
            next_q_data = self.next_question(user_id)

            response = {"feedback": feedback, "points": session["points"], "level": session.get("level_num", 1)}
            if next_q_data.get("end"):
                response["end"] = True
                response["message"] = next_q_data.get("message")
            else:
                response["next"] = next_q_data.get("question")
            return response
        else:
            self._update_session_on_incorrect(session, last_question)
            return {"feedback": feedback, "retry": True, "points": session.get("points", 0)}

    def _evaluate_answer(self, user_answer: str, question: dict) -> (bool, dict):
        import re
        question_type = "technical" if "answer" in question else "soft"
        expected_answer = question.get("answer") or question.get("expected", "")

        # Similitud Semántica
        user_emb = self.emb_mgr.model.encode([user_answer], convert_to_tensor=True)
        expected_emb = self.emb_mgr.model.encode([expected_answer], convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(user_emb, expected_emb)[0][0])

        # Cobertura de Palabras Clave
        expected_keywords = set(re.findall(r"\w+", expected_answer.lower()))
        answer_keywords = set(re.findall(r"\w+", user_answer.lower()))
        common_keywords = expected_keywords.intersection(answer_keywords)
        keyword_score = len(common_keywords) / len(expected_keywords) if expected_keywords else 0

        threshold = 0.75 if question_type == "technical" else 0.65
        is_correct = sim_score >= threshold or keyword_score >= 0.6

        scores = {"similarity": sim_score, "keyword_coverage": keyword_score}
        return is_correct, scores

    def _generate_feedback(self, is_correct: bool, session: dict, question: dict, scores: dict) -> str:
        last_agent_msg = question.get("question") or question.get("scenario")
        attempts = session.get('attempts', {}).get(last_agent_msg, 0) + 1

        if is_correct:
            emoji = random.choice(self.positive_emojis)
            phrase = random.choice(self.motivational_phrases)
            points = max(10 - (attempts - 1) * 2, 1)
            feedback = f"<b>¡Respuesta correcta!</b> {emoji} {phrase}<br>"
            feedback += f"<b>Puntaje obtenido:</b> {points} (intentos: {attempts})"
        else:
            emoji = random.choice(self.negative_emojis)
            phrase = random.choice(self.negative_phrases)
            hint = question.get("hint", "Intenta enfocar tu respuesta en los conceptos clave.")
            feedback = f"<b>Respuesta incorrecta</b> {emoji}. {phrase}<br>"
            feedback += f"<b>Pista:</b> {hint}<br>"
            feedback += f"<b>Intentos:</b> {attempts}"

        # Añadir análisis emocional (si existe el servicio)
        try:
            emotion = analyze_emotion(session['history'][-1]['user'])
            if emotion.get('emotion') in ['anger', 'sadness']:
                feedback += "<br>¡Mucho ánimo, vas muy bien!"
        except Exception:
            pass # No fallar si el analizador de emociones no está disponible

        return feedback

    def _update_session_on_correct(self, user_id: str, session: dict, question: dict, answer: str, scores: dict):
        last_agent_msg = question.get("question") or question.get("scenario")
        attempts = session.get('attempts', {}).get(last_agent_msg, 0) + 1
        points = max(10 - (attempts - 1) * 2, 1)

        session["points"] = session.get("points", 0) + points
        session["streak"] = session.get("streak", 0) + 1
        session["question_counter"] += 1
        session['attempts'][last_agent_msg] = 0 # Resetear intentos

        # Actualizar badges
        for badge in self.badges:
            if badge["condition"](session):
                session.setdefault("badges", set()).add(badge["name"])

        # Actualizar temas de éxito
        topic = question.get("topic")
        if topic:
            session.setdefault('topic_success', {})[topic] = session.get('topic_success', {}).get(topic, 0) + 1

        # Aprendizaje activo
        self._save_good_answer(last_agent_msg, answer, session.get("role"))
        try:
            adjust_difficulty(session, performance={'sim_score': scores['similarity'], 'intentos': attempts, 'correcto': True})
        except Exception as e:
            print(f"[ML Error] Ajustando dificultad: {e}")

    def _update_session_on_incorrect(self, session: dict, question: dict):
        last_agent_msg = question.get("question") or question.get("scenario")
        session['attempts'][last_agent_msg] = session.get('attempts', {}).get(last_agent_msg, 0) + 1
        session["streak"] = 0

        # Actualizar temas de error
        topic = question.get("topic")
        if topic:
            session.setdefault('error_topics', {})[topic] = session.get('error_topics', {}).get(topic, 0) + 1

    def process_survey(self, user_id: str, rating: int, comments: str):
        return process_survey_metrics(self.metrics, rating, comments)

    def _save_good_answer(self, question: str, answer: str, role: Optional[str] = None):
        path = Path(__file__).parent / 'datasets' / 'good_answers.jsonl'
        entry = {"role": role, "question": question, "answer": answer}
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # Generar nueva pregunta con LLM
        prompt = (
            f"Basado en la pregunta '{question}' y la respuesta '{answer}', "
            f"genera una nueva pregunta técnica desafiante para un rol de '{role}'."
            f"Devuelve SOLO la nueva pregunta."
        )
        try:
            new_question = self.llm(prompt).strip()
            if new_question and new_question not in [q.get("question","") for q in self.tech_questions]:
                tech_path = Path(__file__).parent / 'datasets' / 'tech_questions.jsonl'
                new_entry = {"type": "technical", "role": role, "question": new_question, "answer": ""}
                with open(tech_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                self.tech_questions.append(new_entry)
                self.emb_mgr.refresh()
        except Exception as e:
            print(f"[LLM Error] No se pudo generar una nueva pregunta: {e}")

    def end_interview(self, user_id: str, satisfaction: Optional[float] = None):
        session = self.sessions.pop(user_id, None)
        if not session: return {"error": "Entrevista no iniciada"}

        self.metrics['interviews_finished'] += 1
        if satisfaction is not None:
            self.metrics['satisfaction_scores'].append(satisfaction)

        # Usar función externa para el feedback final para mantener la clase limpia
        return generate_final_feedback(session, self.tech_questions, self.soft_questions, self.llm)
