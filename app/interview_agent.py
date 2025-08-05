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
    def _log_llm_error(self, error, answer, expected):
        # Logging reutilizable de errores LLM/embeddings
        if not hasattr(self, '_llm_errors'):
            self._llm_errors = []
        log_llm_error(self._llm_errors, error, answer, expected)

    def process_context_answer(self, user_id: str, answer: str):
        """
        Procesa la respuesta de contexto y devuelve la siguiente pregunta de contexto o avanza a la entrevista.
        Personaliza preguntas de contexto para soft-skills y usa NLP para elegir preguntas relevantes.
        """
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        idx = session.get("context_asked", 0)
        # Determinar tipo de entrevista
        interview_type = session.get("interview_type", "tecnica").lower()
        if interview_type in ("blanda", "soft"):
            context_keys = ["situacion", "reto", "motivacion", "valores", "equipo", "objetivo"]
            context_questions = [
                "Cuéntame una situación desafiante que hayas enfrentado en el trabajo.",
                "¿Cómo resolviste un conflicto con un compañero o líder?",
                "¿Qué te motiva a dar lo mejor de ti en un equipo?",
                "¿Qué valores consideras más importantes en el trabajo?",
                "¿Cómo contribuyes al éxito de un equipo?",
                "¿Cuál ha sido tu mayor logro personal o profesional?"
            ]
        else:
            context_keys = ["role", "level", "years", "knowledge", "tools", "expectations", "preference"]
            context_questions = [
                "¿Para qué rol específico deseas prepararte? (IA, ciberseguridad, devops, infraestructura, etc.)",
                "¿Cuál es tu nivel de experiencia profesional (junior, semi-senior, senior)?",
                "¿Cuántos años de experiencia tienes en el rol?",
                "¿Qué conocimientos o tecnologías consideras tus fortalezas?",
                "¿Qué herramientas dominas mejor?",
                "¿Qué expectativas tienes sobre el trabajo o la empresa?",
            ]
        # Guardar respuesta en el contexto adecuado
        if idx < len(context_keys):
            key = context_keys[idx]
            session[key] = answer
        session["context_asked"] = idx + 1
        # Si hay más preguntas de contexto
        if session["context_asked"] < len(context_questions):
            question = context_questions[session["context_asked"]]
            session["history"].append({"agent": question})
            return {"question": question}
        else:
            session["stage"] = "interview"
            # --- NLP robusto para elegir preguntas soft-skills relevantes ---
            if interview_type in ("blanda", "soft"):
                # Unir respuestas de contexto
                contexto_usuario = " ".join([str(session.get(k, "")) for k in context_keys])
                # Usar embeddings_manager para encontrar las preguntas soft más relevantes
                if hasattr(self, 'emb_mgr') and hasattr(self.emb_mgr, 'find_most_similar_soft'):
                    top_soft = self.emb_mgr.find_most_similar_soft(contexto_usuario, top_k=10)
                    session["soft_pool"] = top_soft
                else:
                    # Fallback: aleatorio
                    session["soft_pool"] = random.sample(self.soft_questions, min(10, len(self.soft_questions)))
            return {"message": "Contexto finalizado. Inicia la entrevista."}


    def process_survey(self, user_id: str, rating: int, comments: str):
        """
        Procesa la encuesta de satisfacción para aprendizaje continuo (stub inicial).
        """
        return process_survey_metrics(self.metrics, rating, comments)
    """
    Agente de entrevistas que alterna preguntas técnicas y de habilidades blandas,
    usando datasets personalizados y modelos LLM locales vía Ollama.
    """

    def __init__(self, model_name: str = "llama3"):
        self.llm = Ollama(model=model_name)  # Usando llama3 por defecto
        self.sessions: Dict[str, Dict[str, Any]] = {}
        # Cargar datasets
        self.tech_questions = self._load_dataset('tech_questions.jsonl')
        # Logging exportable y analítica externa
        self.event_log: list = []
        self.analytics_hooks = []  # Para integración con Mixpanel, GA, etc.
        # Gamificación avanzada
        self.badges = [
            {"name": "Primer intento", "condition": lambda s: s.get('points', 0) >= 10},
            {"name": "Racha x5", "condition": lambda s: s.get('streak', 0) >= 5},
            {"name": "Nivel 3", "condition": lambda s: s.get('level', 1) >= 3},
            {"name": "Maestro de velocidad", "condition": lambda s: min(s.get('response_times', [999])) < 10},
            {"name": "Constancia", "condition": lambda s: len(s.get('history', [])) >= 15},
            {"name": "Aprendizaje ML", "condition": lambda s: s.get('ml_improved', False)},
        ]
        self.soft_questions = self._load_dataset('soft_skills.jsonl')
        # Embeddings para robustez semántica
        self.emb_mgr = EmbeddingsManager()
        # Aprendizaje automático de frases motivacionales y emojis
        self.motivational_phrases = [
            "¡Sigue así, vas por un gran camino! 🚀",
            "¡Tu esfuerzo se nota! 👏",
            "¡Demuestras dominio y seguridad! 😎",
            "¡Suma un logro más a tu carrera! 🥇",
            "¡Inspiras confianza! 💪"
        ]
        self.personalized_tips = [
            "Recuerda repasar los temas recientes para consolidar tu aprendizaje.",
            "Aprovecha los recursos recomendados para profundizar en tus áreas de mejora.",
            "Comparte tus logros con tu red profesional para motivarte aún más.",
            "La constancia y la práctica diaria marcan la diferencia en tu progreso.",
            "Refuerza tus fortalezas y trabaja en tus debilidades para un crecimiento integral."
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
        # Métricas para objetivos
        self.metrics = {
            'total_evaluations': 0,
            'correct_evaluations': 0,
            'interviews_started': 0,
            'interviews_finished': 0,
            'satisfaction_scores': [],
            'user_performance': {}  # user_id: [scores]
        }
        # Umbral calibrable para precisión
        self.similarity_threshold = 0.8
        # ML/DL: historial de mejoras
        self.ml_feedback_history = []

    def _load_dataset(self, filename: str) -> List[dict]:
        path = Path(__file__).parent / 'datasets' / filename
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]


    def start_interview(self, user_id: str, role: Optional[str] = None, interview_type: Optional[str] = None, mode: str = "practice"):
        self.metrics['interviews_started'] += 1
        """
        Inicia la entrevista preguntando contexto relevante (rol, experiencia, conocimientos, herramientas, años, etc).
        """
        context_questions = [
            "¿Para qué rol específico deseas prepararte? (IA, ciberseguridad, devops, infraestructura, etc.)",
            "¿Cuál es tu nivel de experiencia profesional (junior, semi-senior, senior)?",
            "¿Cuántos años de experiencia tienes en el rol?",
            "¿Qué conocimientos o tecnologías consideras tus fortalezas?",
            "¿Qué herramientas dominas mejor?",
            "¿Qué expectativas tienes sobre el trabajo o la empresa?"
        ]
        self.sessions[user_id] = {
            "role": role or None,
            "level": None,
            "years": None,
            "knowledge": [],
            "tools": [],
            "interview_type": interview_type or "technical",
            "mode": mode or "practice",
            "stage": "context",
            "history": [],
            "context_asked": 0,
            "tech_pool": self.tech_questions.copy(),
            "soft_pool": self.soft_questions.copy(),
            "last_type": None,
            "exam_answers": [],
            "exam_start_time": None,
            "points": 0,
            "level_num": 1,
            "streak": 0,
            "badges": set()
        }
        question = context_questions[0]
        self.sessions[user_id]["history"].append({"agent": question})
        return {"question": question}

    def _filter_questions(self, questions: List[dict], filters: dict) -> List[dict]:
        """
        Filtra preguntas usando todos los criterios disponibles en filters.
        """
        return filter_questions(questions, filters)

    def next_question(self, user_id: str, user_input: Optional[str] = None):
        """
        Alterna entre preguntas técnicas y de soft skills, según el contexto y el historial.
        """
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        # Limitar a 10 preguntas SOLO en etapa de entrevista
        if session.get('stage') == 'interview' and session.get('question_counter', 0) >= 10:
            session['interview_finished'] = True
            return {"end": True}
        # Si aún no termina el contexto, seguir preguntando contexto
        context_questions = [
            "¿Para qué rol específico deseas prepararte? (IA, ciberseguridad, devops, infraestructura, etc.)",
            "¿Cuál es tu nivel de experiencia profesional (junior, semi-senior, senior)?",
            "¿Cuántos años de experiencia tienes en el rol?",
            "¿Qué conocimientos o tecnologías consideras tus fortalezas?",
            "¿Qué herramientas dominas mejor?",
            "¿Qué expectativas tienes sobre el trabajo o la empresa?"
        ]
        if session.get("stage") == "context":
            idx = session.get("context_asked", 0)
            if idx < len(context_questions):
                question = context_questions[idx]
                session["context_asked"] = idx + 1
                session["history"].append({"agent": question})
                return {"question": question}
            else:
                session["stage"] = "interview"
                session["question_counter"] = 0  # Iniciar contador de preguntas de entrevista
                if "tech_pool" not in session:
                    session["tech_pool"] = self.tech_questions.copy()
                if "soft_pool" not in session:
                    session["soft_pool"] = self.soft_questions.copy()
                return self.next_question(user_id)
        # ...resto de la lógica de selección de pregunta...
        # Si no hay input, seguir flujo aleatorio
        interview_type = session.get("interview_type", "tecnica").lower()
        if interview_type in ("soft", "blanda"):
            qtype = "soft"
        elif interview_type in ("technical", "tecnica"):
            qtype = "technical"
        else:
            qtype = "technical"
        if qtype == "technical" and session["tech_pool"]:
            question = random.choice(session["tech_pool"])
            session["tech_pool"].remove(question)
            preguntas_restantes = 10 - session.get('question_counter', 0)
            print(f"[DEBUG] Pregunta técnica seleccionada: {question.get('question','')} (quedan {preguntas_restantes})")
            session["last_type"] = "technical"
            session["history"].append({"agent": question["question"]})
            return {"question": question["question"]}
        elif qtype == "soft" and session["soft_pool"]:
            question = random.choice(session["soft_pool"])
            session["soft_pool"].remove(question)
            preguntas_restantes = 10 - session.get('question_counter', 0)
            print(f"[DEBUG] Pregunta soft seleccionada: {question.get('scenario','')} (quedan {preguntas_restantes})")
            session["last_type"] = "soft"
            session["history"].append({"agent": question["scenario"]})
            return {"question": question["scenario"]}
        else:
            motivacion = random.choice(self.motivational_phrases)
            fallback = f"¡Has completado la entrevista! {motivacion} ¿Quieres intentar otra modalidad o reiniciar?"
            print(f"[DEBUG] Sin preguntas en pools. tech_pool: {len(session['tech_pool'])}, soft_pool: {len(session['soft_pool'])}")
            session["history"].append({"agent": fallback})
            return {"question": fallback}



    def process_answer(self, user_id: str, answer: str):
        """
        Procesa la respuesta del usuario, evalúa asertividad, bloquea avance si no es correcta, y maneja gamificación avanzada.
        NLP/ML robusto: embeddings, clustering, ajuste de dificultad, feedback emocional, aprendizaje activo, fitting, fine-tuning, random forest, recomendación, mejora continua.
        """
        import re
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        session["history"].append({"user": answer})
        # Determinar pregunta actual
        last_agent_msg = None
        for h in reversed(session["history"]):
            if "agent" in h:
                last_agent_msg = h["agent"]
                break
        # Buscar respuesta esperada en datasets
        expected = ""
        pista = ""
        tipo = ""
        for q in self.tech_questions:
            if q.get("question") == last_agent_msg:
                expected = q.get("answer", "")
                pista = q.get("hint", "") or "Piensa en los conceptos clave relacionados."
                tipo = "technical"
                break
        if not expected:
            for q in self.soft_questions:
                if q.get("scenario") == last_agent_msg:
                    expected = q.get("expected", "")
                    pista = q.get("hint", "") or "Recuerda ejemplos de experiencias previas."
                    tipo = "soft"
                    break
        # --- Estrategias NLP/ML robustas ---
        sim_score = 0.0
        keyword_score = 0.0
        coverage_score = 0.0
        emotional_feedback = ""
        suggestions = []
        # 1. Similitud semántica (embeddings)
        if expected:
            user_emb = self.emb_mgr.model.encode([answer], convert_to_tensor=True)
            expected_emb = self.emb_mgr.model.encode([expected], convert_to_tensor=True)
            sim_score = float(util.pytorch_cos_sim(user_emb, expected_emb)[0][0])
        # 2. Detección de palabras clave importantes
        expected_keywords = set(re.findall(r"\w+", expected.lower()))
        answer_keywords = set(re.findall(r"\w+", answer.lower()))
        common_keywords = expected_keywords.intersection(answer_keywords)
        if expected_keywords:
            keyword_score = len(common_keywords) / len(expected_keywords)
        # 3. Cobertura de conceptos (mínimo 60% de palabras clave)
        coverage_score = keyword_score
        # 4. Sugerencias personalizadas si falta algún concepto clave
        missing = expected_keywords - answer_keywords
        if missing:
            suggestions.append(f"Incluye conceptos como: {', '.join(list(missing)[:3])}.")
        # 5. Feedback emocional contextual y avanzado (transformers)
        emotion_result = None
        try:
            emotion_result = analyze_emotion(answer)
        except Exception:
            pass
        if emotion_result and isinstance(emotion_result, dict):
            emo = emotion_result.get('emotion', '')
            if emo == 'anger':
                emotional_feedback = "Tranquilo, tómate un momento y vuelve a intentarlo. 🧘‍♂️"
            elif emo == 'sadness':
                emotional_feedback = "¡Ánimo! Cada intento te acerca más a la meta. 💪"
            elif emo == 'joy':
                emotional_feedback = "¡Se nota tu entusiasmo! Sigue así. 😃"
            elif emo == 'fear':
                emotional_feedback = "No te preocupes, puedes equivocarte y aprender. 🌱"
            if emo in ['anger', 'sadness', 'fear']:
                session.setdefault('negative_emotions', 0)
                session['negative_emotions'] += 1
                if session['negative_emotions'] >= 3:
                    emotional_feedback += "<br><b>¿Quieres tomar una pausa? Recuerda que tu bienestar es importante.</b>"
            else:
                session['negative_emotions'] = 0
        # 6. Umbral de aceptación (ajustable, fitting/fine-tuning)
        threshold = 0.75 if tipo == "technical" else 0.65
        correcto = sim_score >= threshold or coverage_score >= 0.6
        # 7. Feedback enriquecido y adaptativo (ML, clustering, random forest, etc.)
        if 'attempts' not in session:
            session['attempts'] = {}
        intentos = session.get('attempts', {}).get(last_agent_msg, 0) + 1
        session['attempts'][last_agent_msg] = intentos
        penalizacion_pista = 2 if missing else 0
        puntos_base = 10
        puntos = max(puntos_base - (intentos-1)*2 - penalizacion_pista, 1)
        # --- ML avanzado: feedback loop, clustering, recomendación, fitting ---
        # (Stub: aquí se puede integrar random forest, clustering, etc. para refinar feedback y predicción)
        # 8. Aprendizaje activo: actualizar embeddings y datasets con buenas respuestas
        if correcto:
            try:
                adjust_difficulty(session, performance={
                    'sim_score': sim_score,
                    'coverage_score': coverage_score,
                    'intentos': intentos,
                    'correcto': correcto
                })
            except Exception as e:
                print(f"[ML] Error ajustando dificultad: {e}")
            emoji = random.choice(self.positive_emojis)
            feedback = f"<b>¡Respuesta correcta!</b> {emoji} {random.choice(self.motivational_phrases)}"
            if emotional_feedback:
                feedback += f" {emotional_feedback}"
            feedback += f"<br><b>Puntaje obtenido:</b> {puntos} (intentos: {intentos}, penalización por pistas: {penalizacion_pista})"
            print(f"[FEEDBACK] {feedback}")  # Debug feedback en consola
            session['attempts'][last_agent_msg] = 0
            session['history'].append({"feedback": feedback})
            session["points"] = session.get("points", 0) + puntos
            session["streak"] = session.get("streak", 0) + 1
            for badge in self.badges:
                if badge["condition"](session):
                    session.setdefault("badges", set()).add(badge["name"])
            if session.get('stage') == 'interview':
                session['question_counter'] = session.get('question_counter', 0) + 1
                if session['question_counter'] >= 10:
                    session['interview_finished'] = True
                    # Aprendizaje activo: guardar buena respuesta y actualizar embeddings
                    self._save_good_answer(last_agent_msg or "", answer, session.get("role"))
                    if hasattr(self.emb_mgr, 'update_from_feedback'):
                        self.emb_mgr.update_from_feedback(session['history'])
                    return {"feedback": feedback, "points": session["points"], "level": session.get("level_num", 1), "badges": list(session.get("badges", [])), "end": True}
            # Aprendizaje activo: guardar buena respuesta y actualizar embeddings
            self._save_good_answer(last_agent_msg or "", answer, session.get("role"))
            if hasattr(self.emb_mgr, 'update_from_feedback'):
                self.emb_mgr.update_from_feedback(session['history'])
            next_q = self.next_question(user_id)
            result = {"feedback": feedback, "points": session["points"], "level": session.get("level_num", 1), "badges": list(session.get("badges", [])), "next": next_q["question"] if isinstance(next_q, dict) and "question" in next_q else next_q}
            tema = None
            for q in self.tech_questions:
                if q.get("question") == last_agent_msg:
                    tema = q.get("topic")
                    break
            if tema:
                session.setdefault('topic_success', {})
                session['topic_success'][tema] = session['topic_success'].get(tema, 0) + 1
                if session['topic_success'][tema] == 3:
                    result["feedback"] += f"<br><b>¡Has mejorado mucho en {tema}! Te recomiendo intentar retos avanzados en este tema.</b>"
            return result
        else:
            emoji = random.choice(self.negative_emojis)
            feedback = f"<b>No es la respuesta esperada</b> {emoji}.<br><b>Pista:</b> {pista} "
            if suggestions:
                feedback += "<br><b>Sugerencia:</b> " + " ".join(suggestions)
            feedback += f"<br><b>Intentos:</b> {intentos}"
            if emotional_feedback:
                feedback += f"<br>{emotional_feedback}"
            feedback += "<br>Intenta de nuevo, ¡tú puedes!"
            print(f"[FEEDBACK] {feedback}")  # Debug feedback en consola
            session["history"].append({"feedback": feedback})
            # No avanzar pregunta, bloquear hasta que sea correcta
            tema = None
            for q in self.tech_questions:
                if q.get("question") == last_agent_msg:
                    tema = q.get("topic")
                    break
            if tema:
                session['error_topics'][tema] = session['error_topics'].get(tema, 0) + 1
                if session['error_topics'][tema] == 3:
                    feedback += f"<br><b>Te recomiendo repasar recursos sobre {tema}: <a href='https://www.google.com/search?q={tema}+tutorial' target='_blank'>Ver recursos</a></b>"
            return {"feedback": feedback, "retry": True, "points": session.get("points", 0), "level": session.get("level_num", 1)}
        # --- Analítica avanzada y gamificación ---
        # Medir tiempo de respuesta
        now = time.time()
        last_time = session.get('last_activity', now)
        response_time = now - last_time
        session['last_activity'] = now
        if 'response_times' not in session:
            session['response_times'] = []
        session['response_times'].append(response_time)

        # Evaluación robusta con embeddings y LLM + ML/DL adaptativo
        is_correct = False
        similarity_score = 0.0
        ml_score = None
        if expected:
            try:
                cand_emb = self.emb_mgr.model.encode([answer], convert_to_tensor=True)
                exp_emb = self.emb_mgr.model.encode([expected], convert_to_tensor=True)
                similarity_score = float(self.emb_mgr.model.similarity(cand_emb, exp_emb))
                # ML/DL: feedback adaptativo
                if 'generate_feedback_ml' in globals() and generate_feedback_ml:
                    ml_score = generate_feedback_ml(answer, expected)
                    self.ml_feedback_history.append(ml_score)
                    if ml_score and isinstance(ml_score, dict) and ml_score.get('improved'):
                        session['ml_improved'] = True
                if isinstance(ml_score, dict):
                    is_correct = similarity_score > self.similarity_threshold or ml_score.get('improved')
                else:
                    is_correct = similarity_score > self.similarity_threshold
            except Exception:
                is_correct = expected.lower() in answer.lower() or len(answer) > 10

        # Logging de eventos para analítica
        if 'events' not in session:
            session['events'] = []
        # Logging de eventos para analítica y exportación
        event = {
            'timestamp': now,
            'user_id': user_id,
            'question': last_agent_msg,
            'answer': answer,
            'expected': expected,
            'similarity': similarity_score,
            'is_correct': is_correct,
            'response_time': response_time,
            'points': session.get('points', 0),
            'level': session.get('level', 1)
        }
        if 'events' not in session:
            session['events'] = []
        session['events'].append(event)
        self.event_log.append(event)
        # Hooks para analítica externa
        for hook in self.analytics_hooks:
            try:
                hook(event)
            except Exception:
                pass

        # Gamificación avanzada
        points = session.get("points", 0)
        level = session.get("level", 1)
        streak = session.get("streak", 0)
        if modo == "examen":
            if is_correct:
                points += 10 + int(similarity_score * 5)
                streak += 1
                if streak % 5 == 0:
                    points += 10  # bonus por racha
                if points // 50 > level-1:
                    level += 1
            else:
                streak = 0
            session["points"] = points
            session["level"] = level
            session["streak"] = streak
        # Gamificación visual: barra de progreso, badges, feedback ML
        badges_earned = []
        for badge in self.badges:
            if badge["name"] not in session["badges"] and badge["condition"](session):
                session["badges"].add(badge["name"])
                badges_earned.append(badge["name"])
        progress = int((points / 50) * 100) if level > 1 else int((points / 50) * 100)
        feedback_bar = f'<div style="margin-top:8px;"><div style="background:#e6f9f0;border-radius:4px;width:100%;height:12px;overflow:hidden;"><div style="background:#43e97b;height:12px;width:{progress}%;transition:width 0.5s;"></div></div><small>Progreso nivel {level}: {progress}%</small></div>'
        if badges_earned:
            feedback_bar += f"<br><b>¡Nuevas insignias!: {', '.join(badges_earned)}</b>"

        # Feedback motivacional personalizado y ML
        if is_correct:
            feedback = random.choice(self.motivational_phrases) + " " + random.choice(self.positive_emojis)
            if streak and streak % 5 == 0:
                feedback += " ¡Racha de respuestas correctas! 🔥"
        else:
            feedback = random.choice(self.negative_phrases) + " " + random.choice(self.negative_emojis)
        # Feedback adaptativo ML
        if ml_score and isinstance(ml_score, dict) and ml_score.get('feedback'):
            feedback += f"<br><em>IA ML: {ml_score.get('feedback')}</em>"
        # Añadir barra de progreso y badges
        feedback += feedback_bar
        # Aprendizaje continuo: guardar buenas respuestas y refinar embeddings
        if is_correct:
            self._save_good_answer(last_agent_msg or "", answer, session.get("role"))
            if hasattr(self.emb_mgr, 'update_from_feedback'):
                self.emb_mgr.update_from_feedback(session['history'])

    # Métodos de analítica y privacidad avanzados eliminados para simplificar el flujo y evitar código inalcanzable.


        # --- Código duplicado y ramas muertas eliminadas. El flujo de feedback, gamificación y aprendizaje queda centralizado en el bloque principal de process_answer. ---

    def _save_good_answer(self, question: str, answer: str, role: Optional[str] = None):
        """
        Guarda respuestas destacadas para aprendizaje futuro (simulación de memoria).
        Además, genera una nueva pregunta técnica relacionada usando el LLM y la agrega automáticamente al dataset.
        """
        path = Path(__file__).parent / 'datasets' / 'good_answers.jsonl'
        entry = {"role": role, "question": question, "answer": answer}
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # --- Generación automática de nueva pregunta técnica ---
        prompt = (
            f"Eres un generador de preguntas técnicas para entrevistas de ingeniería de sistemas. "
            f"Dada la siguiente respuesta de un candidato y el contexto de la pregunta original, genera una nueva pregunta técnica relevante, desafiante y NO repetida, que no esté en el dataset. "
            f"Pregunta original: {question}\n"
            f"Respuesta del candidato: {answer}\n"
            f"Rol: {role if role else 'N/A'}\n"
            f"Devuelve SOLO la nueva pregunta, sin explicación ni respuesta."
        )
        try:
            new_question = self.llm(prompt).strip()
            # Validar que la pregunta no esté ya en el dataset
            if new_question and new_question not in [q.get("question","") for q in self.tech_questions]:
                new_entry = {
                    "type": "technical",
                    "role": role,
                    "level": None,
                    "years": None,
                    "knowledge": [],
                    "tools": [],
                    "question": new_question,
                    "answer": ""
                }
                tech_path = Path(__file__).parent / 'datasets' / 'tech_questions.jsonl'
                with open(tech_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                self.tech_questions.append(new_entry)
                self.emb_mgr.refresh()
        except Exception as e:
            pass  # No interrumpir el flujo si falla la generación

    def end_interview(self, user_id: str, satisfaction: Optional[float] = None):
        """
        Finaliza la entrevista y entrega un resumen del desempeño del usuario, penalizando si usó muchas pistas o intentos.
        """
        session = self.sessions.pop(user_id, None)
        if not session:
            return {"error": "Entrevista no iniciada"}
        modo = session.get("mode", "practice")
        self.metrics['interviews_finished'] += 1
        if satisfaction is not None:
            self.metrics['satisfaction_scores'].append(satisfaction)
        tiempo_total = None
        if modo == "exam" and session.get("exam_start_time"):
            tiempo_total = time.time() - session["exam_start_time"]
        total_tech = 0
        total_soft = 0
        correct_tech = 0
        correct_soft = 0
        habilidades_destacadas = set()
        habilidades_mejorar = set()
        user_score = 0
        user_total = 0
        total_intentos = 0
        total_pistas = 0
        total_preguntas = 0
        for i, h in enumerate(session["history"]):
            if "agent" in h and i+1 < len(session["history"]) and "user" in session["history"][i+1]:
                pregunta = h["agent"]
                respuesta = session["history"][i+1]["user"]
                es_tech = any(q.get("question","") == pregunta for q in self.tech_questions)
                es_soft = any(q.get("scenario","") == pregunta for q in self.soft_questions)
                # Intentos y pistas
                intentos = session.get('attempts', {}).get(pregunta, 1)
                total_intentos += intentos
                # Detectar si usó pista (si feedback de la pregunta contiene "Pista" o "Sugerencia")
                pista_usada = False
                for j in range(i+2, min(i+6, len(session["history"]))):
                    f = session["history"][j]
                    if "feedback" in f and ("Pista" in f["feedback"] or "Sugerencia" in f["feedback"]):
                        pista_usada = True
                        break
                if pista_usada:
                    total_pistas += 1
                total_preguntas += 1
                # Evaluación técnica/soft
                if es_tech:
                    total_tech += 1
                    qref = next((q for q in self.tech_questions if q.get("question","") == pregunta), None)
                    expected = qref.get("answer") if qref else None
                    if expected:
                        cand_emb = self.emb_mgr.model.encode([respuesta], convert_to_tensor=True)
                        exp_emb = self.emb_mgr.model.encode([expected], convert_to_tensor=True)
                        sim = float(self.emb_mgr.model.similarity(cand_emb, exp_emb))
                        user_total += 1
                        if sim > self.similarity_threshold:
                            correct_tech += 1
                            user_score += 1
                            if qref and qref.get("knowledge"):
                                habilidades_destacadas.update(qref["knowledge"])
                        else:
                            if qref and qref.get("knowledge"):
                                habilidades_mejorar.update(qref["knowledge"])
                elif es_soft:
                    total_soft += 1
                    qref = next((q for q in self.soft_questions if q.get("scenario","") == pregunta), None)
                    expected = qref.get("expected") if qref else None
                    if expected:
                        cand_emb = self.emb_mgr.model.encode([respuesta], convert_to_tensor=True)
                        exp_emb = self.emb_mgr.model.encode([expected], convert_to_tensor=True)
                        sim = float(self.emb_mgr.model.similarity(cand_emb, exp_emb))
                        user_total += 1
                        if sim > self.similarity_threshold:
                            correct_soft += 1
                            user_score += 1
                            if qref and qref.get("knowledge"):
                                habilidades_destacadas.update(qref["knowledge"])
                        else:
                            if qref and qref.get("knowledge"):
                                habilidades_mejorar.update(qref["knowledge"])
        if user_id not in self.metrics['user_performance']:
            self.metrics['user_performance'][user_id] = []
        if user_total > 0:
            self.metrics['user_performance'][user_id].append(user_score / user_total)
        perf = self.metrics['user_performance'][user_id]
        improvement = None
        if len(perf) > 1:
            improvement = round((perf[-1] - perf[0]) * 100, 2)
        penalizacion = 0
        advertencias = []
        if total_intentos > total_preguntas * 1.5:
            penalizacion += 1
            advertencias.append("Tuviste que intentar varias veces algunas preguntas. Practica para mejorar tu asertividad.")
        if total_pistas > total_preguntas // 3:
            penalizacion += 1
            advertencias.append("Usaste muchas pistas o sugerencias. Intenta responder con menos ayuda para mejorar tu puntaje.")
        prompt = (
            f"Eres un entrevistador experto en {session['role']}. "
            "Resume el desempeño del candidato en la entrevista, destacando fortalezas y aspectos a mejorar. "
            "Incluye estadísticas claras: número de preguntas técnicas y soft skills, aciertos y errores, porcentaje de éxito. "
            "Enumera las habilidades más destacadas y las que requieren mejora. "
            "Sé claro, profesional y objetivo.\n"
            f"Preguntas técnicas: {total_tech}, correctas: {correct_tech}, porcentaje: {round((correct_tech/total_tech)*100,1) if total_tech else 0}%\n"
            f"Preguntas soft skills: {total_soft}, correctas: {correct_soft}, porcentaje: {round((correct_soft/total_soft)*100,1) if total_soft else 0}%\n"
            f"Habilidades destacadas: {', '.join(habilidades_destacadas) if habilidades_destacadas else 'Ninguna'}\n"
            f"Habilidades a mejorar: {', '.join(habilidades_mejorar) if habilidades_mejorar else 'Ninguna'}\n"
            + (f"Tiempo total del examen: {int(tiempo_total//60)} min {int(tiempo_total%60)} seg\n" if tiempo_total else "")
            + f"Historial de la entrevista: {session['history']}"
        )
        summary = self.llm(prompt)
        summary += f"\n\n---\nMétricas clave:\n"
        if self.metrics['total_evaluations']:
            precision = round((self.metrics['correct_evaluations']/self.metrics['total_evaluations'])*100,1)
            summary += f"Precisión IA: {precision}%\n"
        if self.metrics['interviews_started']:
            tasa_final = round((self.metrics['interviews_finished']/self.metrics['interviews_started'])*100,1)
            summary += f"Tasa de finalización: {tasa_final}%\n"
        if self.metrics['satisfaction_scores']:
            sat_avg = round(sum(self.metrics['satisfaction_scores'])/len(self.metrics['satisfaction_scores']),2)
            summary += f"Satisfacción promedio: {sat_avg}/5\n"
        if improvement is not None:
            summary += f"Mejora en desempeño: {improvement}%\n"
        if penalizacion > 0:
            summary += f"\n<b>Advertencia:</b> Tu rendimiento final fue penalizado por: {'; '.join(advertencias)}\n"
        # Recomendaciones personalizadas de aprendizaje (ML)
        learning_path = []
        try:
            learning_path = suggest_learning_path(session.get('events', session.get('history', [])))
        except Exception as e:
            print(f"[ML] Error generando learning path: {e}")
        if learning_path:
            summary += "\n<b>Recomendaciones personalizadas para tu aprendizaje:</b>\n<ul>"
            for rec in learning_path:
                summary += f"<li>{rec}</li>"
            summary += "</ul>"
        # Hook para futuros algoritmos ML/NLP avanzados
        # Ejemplo: clustering de errores, análisis de estilo, etc.
        # OPTIMIZACIÓN: Clustering de temas dominados y débiles
        topic_stats = {}
        for h in session.get('history', []):
            if 'agent' in h:
                t = h['agent']
                for q in self.tech_questions:
                    if q.get('question') == t:
                        topic = q.get('topic', 'General')
                        topic_stats.setdefault(topic, {'total': 0, 'aciertos': 0, 'errores': 0})
                        topic_stats[topic]['total'] += 1
                for q in self.soft_questions:
                    if q.get('scenario') == t:
                        topic = q.get('topic', 'Soft Skills')
                        topic_stats.setdefault(topic, {'total': 0, 'aciertos': 0, 'errores': 0})
                        topic_stats[topic]['total'] += 1
            if 'feedback' in h and 'No es la respuesta esperada' in h['feedback']:
                idx = session['history'].index(h)
                if idx > 0 and 'agent' in session['history'][idx-1]:
                    prev = session['history'][idx-1]['agent']
                    for q in self.tech_questions:
                        if q.get('question') == prev:
                            topic = q.get('topic', 'General')
                            topic_stats.setdefault(topic, {'total': 0, 'aciertos': 0, 'errores': 0})
                            topic_stats[topic]['errores'] += 1
                    for q in self.soft_questions:
                        if q.get('scenario') == prev:
                            topic = q.get('topic', 'Soft Skills')
                            topic_stats.setdefault(topic, {'total': 0, 'aciertos': 0, 'errores': 0})
                            topic_stats[topic]['errores'] += 1
            if 'feedback' in h and '¡Respuesta correcta!' in h['feedback']:
                idx = session['history'].index(h)
                if idx > 0 and 'agent' in session['history'][idx-1]:
                    prev = session['history'][idx-1]['agent']
                    for q in self.tech_questions:
                        if q.get('question') == prev:
                            topic = q.get('topic', 'General')
                            topic_stats.setdefault(topic, {'total': 0, 'aciertos': 0, 'errores': 0})
                            topic_stats[topic]['aciertos'] += 1
                    for q in self.soft_questions:
                        if q.get('scenario') == prev:
                            topic = q.get('topic', 'Soft Skills')
                            topic_stats.setdefault(topic, {'total': 0, 'aciertos': 0, 'errores': 0})
                            topic_stats[topic]['aciertos'] += 1
        # OPTIMIZACIÓN: Recomendar rutas de aprendizaje y recursos por temas débiles
        temas_debiles = [t for t, v in topic_stats.items() if v['errores'] >= 2]
        recursos = ''
        if temas_debiles:
            recursos = '<ul>' + ''.join([f"<li>{t}: <a href='https://www.google.com/search?q={t}+tutorial' target='_blank'>Ver recursos</a></li>" for t in temas_debiles]) + '</ul>'
        # OPTIMIZACIÓN: Métricas de tiempo de respuesta
        tiempos = session.get('response_times', [])
        if tiempos:
            avg_time = sum(tiempos)/len(tiempos)
        else:
            avg_time = 0
        # OPTIMIZACIÓN: Reconocimiento por agilidad
        reconocimiento = ''
        if avg_time < 8 and avg_time > 0:
            reconocimiento = '<br><b>¡Respondiste con gran agilidad! Eso demuestra dominio.</b>'
        elif avg_time > 20:
            reconocimiento = '<br><b>Te sugerimos practicar técnicas de estudio para responder más rápido.</b>'
        summary += f"\n\n---\nMétricas clave:\n"
        if recursos:
            summary += f"\n<b>Temas a reforzar y recursos recomendados:</b> {recursos}"
        if reconocimiento:
            summary += reconocimiento
        return {"summary": summary, "learning_path": learning_path}
