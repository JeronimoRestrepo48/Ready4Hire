
"""
Ready4Hire - Agente de Entrevista IA
====================================

Este módulo implementa el núcleo inteligente del simulador de entrevistas Ready4Hire.

¿Qué es?
---------
Un agente conversacional avanzado que simula entrevistas técnicas y de habilidades blandas, adaptándose al usuario en tiempo real. Utiliza IA generativa, embeddings, clustering, gamificación y análisis emocional para crear una experiencia de aprendizaje personalizada, motivadora y profesional.

¿Para quién es?
---------------
- Técnicos: permite practicar entrevistas reales, recibir feedback detallado, pistas conceptuales, ejemplos de industria y medir progreso con gamificación.
- No técnicos: la interfaz es amigable, las explicaciones son claras y el sistema motiva y guía paso a paso, ayudando a mejorar habilidades blandas y técnicas.

¿Cómo funciona?
----------------
1. Selección inteligente de preguntas: el agente alterna entre preguntas técnicas y blandas, adaptando la dificultad y el tema según el perfil y desempeño del usuario.
2. Feedback motivador y emocional: cada respuesta recibe retroalimentación personalizada, con frases motivacionales, emojis y análisis de emociones.
3. Pistas conceptuales: si la respuesta es incorrecta, el agente genera pistas explicativas, analogías, ejemplos y casos de uso reales, incluso si el modelo LLM está limitado.
4. Gamificación avanzada: el usuario gana puntos, sube de nivel, desbloquea logros y recibe insignias por su desempeño, incentivando la mejora continua.
5. Aprendizaje continuo: todas las interacciones se registran para mejorar el sistema y permitir fine-tuning futuro.

Componentes principales:
-----------------------
- Clase InterviewAgent: gestiona sesiones, preguntas, feedback, gamificación y lógica de interacción.
- Integración con embeddings y RankNet: selecciona preguntas relevantes usando IA profunda y clustering.
- Análisis emocional: detecta emociones en las respuestas para adaptar el feedback y evitar sesgos.

Ejemplo de uso rápido (técnico):
--------------------------------
    agent = InterviewAgent()
    agent.start_interview('usuario1', role='DevOps', interview_type='technical')
    agent.process_answer('usuario1', 'Mi respuesta a la pregunta')

Autor: JeronimoRestrepo48
Licencia: MIT
"""


from langchain.llms import Ollama
import threading
import time
import functools
import signal
from typing import Dict, Any, List, Optional
import random
import json
from pathlib import Path
import time
# Integración de embeddings
from app.embeddings.embeddings_manager import EmbeddingsManager
from app.services.emotion_analyzer import analyze_emotion


class InterviewAgent:
    # Insignias y logros creativos
    EXAM_BADGES = [
        ("Racha de aciertos", "🔥", lambda s: s.get('exam_streak',0) >= 3),
        ("Resiliencia", "💪", lambda s: s.get('exam_incorrect_streak',0) >= 3),
        ("Velocidad", "⏱️", lambda s: s.get('last_answer_time',99) < 10),
        ("Variedad", "🌈", lambda s: len(set(s.get('exam_topics',[]))) >= 3),
        ("Soft Skills Pro", "🧠", lambda s: s.get('soft_correct',0) >= 3),
        ("Precisión", "🎯", lambda s: s.get('exam_correct_count',0) >= 8),
        ("Perseverancia", "🏅", lambda s: s.get('exam_attempts',0) >= 10),
    ]
    def _get_context_memory(self, session):
        """
        Devuelve un resumen de la memoria de contexto relevante para feedback y análisis de emociones/sesgos.
        Incluye últimas respuestas, emociones y feedback.
        """
        history = session.get('history', [])
        last_n = 5
        respuestas = [h['user'] for h in history if 'user' in h][-last_n:]
        feedbacks = [h['agent'] for h in history if 'agent' in h][-last_n:]
        emociones = [h.get('emotion') for h in history if h.get('emotion')]  # Guardar emociones si existen
        return {
            'respuestas': respuestas,
            'feedbacks': feedbacks,
            'emociones': emociones
        }
    # Constantes de clase
    MAX_HINTS = 3
    EXAM_POINTS_PER_CORRECT = 15
    EXAM_LEVEL_THRESHOLDS = [0, 30, 60, 100, 150, 210, 280, 360, 450, 550, 700, 900]
    EXAM_LEVEL_NAMES = [
        "Novato/a", "Aprendiz", "Competente", "Avanzado/a", "Experto/a", "Mentor/a", "Maestro/a", "Leyenda", "Gurú", "Sensei", "Sabio/a", "Ícono"
    ]
    EXAM_ACHIEVEMENTS = [
        (1, "¡Primer paso! 🚶‍♂️ Has respondido tu primera pregunta correctamente."),
        (2, "¡En marcha! 🏃‍♀️ Dos aciertos seguidos, vas tomando ritmo."),
        (3, "¡Racha de 3! 🔥 Tu concentración es admirable."),
        (5, "¡Mitad del reto! ⭐ Ya dominas la mitad del examen."),
        (7, "¡Imparable! 🚀 Siete respuestas correctas, sigue así."),
        (10, "¡Perfecto! 🏆 Todas correctas, eres un/a crack.")
    ]
    SATISFACTION_QUESTIONS = [
        "¿Qué tan útil te resultó la entrevista para prepararte profesionalmente?",
        "¿Las preguntas fueron relevantes para el rol que elegiste?",
        "¿El feedback recibido te ayudó a entender tus fortalezas?",
        "¿El feedback te ayudó a identificar áreas de mejora?",
        "¿Te sentiste motivado/a durante la entrevista?",
        "¿El agente fue empático y humano en sus respuestas?",
        "¿Las explicaciones y consejos fueron claros?",
        "¿Las preguntas de contexto fueron adecuadas?",
        "¿El nivel de dificultad fue el correcto para ti?",
        "¿El tiempo de respuesta fue suficiente?",
        "¿El sistema fue fácil de usar?",
        "¿Te sentiste escuchado/a y comprendido/a?",
        "¿Recomendarías este simulador a otros candidatos?",
        "¿Te gustaría volver a usar el simulador en el futuro?",
        "¿Cómo calificarías tu experiencia general con la entrevista?"
    ]

    def __init__(self, model_name: str = "llama3"):
        # LLM rate limiting and timeout attributes (instance-level)
        self._llm_lock = threading.Lock()
        self._llm_last_call = 0
        self._llm_min_interval = 2.0  # seconds between LLM calls (rate limit)
        self._llm_timeout = 10  # seconds per LLM call
        self.llm = Ollama(model=model_name)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.tech_questions = self._load_dataset('tech_questions.jsonl')
        self.soft_questions = self._load_dataset('soft_skills.jsonl')
        self.emb_mgr = EmbeddingsManager()
        # Inicialización de frases motivacionales y emojis aprendidos
        self.motivational_phrases = [
            "¡Sigue así, vas por un gran camino! 🚀",
            "¡Tu esfuerzo se nota! 👏",
            "¡Demuestras dominio y seguridad! 😎",
            "¡Suma un logro más a tu carrera! 🥇",
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

    def _start_response_timer(self, user_id):
        session = self.sessions.get(user_id)
        if not session:
            return
        if 'response_timer' in session and session['response_timer']:
            session['response_timer'].cancel()
        def timeout_close():
            session['history'].append({"agent": "La entrevista se ha cerrado por inactividad. ¡Gracias por participar!"})
            session['closed'] = True

        def timeout_warning():
            session['history'].append({"agent": "¿Sigues ahí? Han pasado 30 segundos sin respuesta. Si no respondes pronto, la entrevista se cerrará automáticamente."})
            session['waiting_warning'] = True
            t2 = threading.Timer(30, timeout_close)
            session['response_timer'] = t2
            t2.start()

        t1 = threading.Timer(30, timeout_warning)
        session['response_timer'] = t1
        session['waiting_warning'] = False
        session['closed'] = False
        t1.start()

    def _stop_response_timer(self, user_id):
        session = self.sessions.get(user_id)
        if session and 'response_timer' in session and session['response_timer']:
            session['response_timer'].cancel()
            session['response_timer'] = None
            session['waiting_warning'] = False
    def _build_prompt(self, question, context, role, level, last_answers, is_correct=None):
        """
        Construye un prompt adaptativo para el LLM según contexto, rol, nivel y desempeño.
        """
        prompt = f"Eres un entrevistador experto en {role or 'el área correspondiente'}. "
        prompt += f"Nivel: {level or 'N/A'}. "
        prompt += f"Pregunta: {question}\n"
        if context:
            prompt += f"Contexto del candidato: {context}\n"
        if last_answers:
            prompt += f"Respuestas previas del candidato: {' | '.join(last_answers[-3:])}\n"
        if is_correct is not None:
            prompt += "La respuesta anterior fue " + ("correcta." if is_correct else "incorrecta.") + "\n"
        prompt += "Genera feedback motivador, concreto y personalizado. Si la respuesta fue incorrecta, sugiere una pista útil sin dar la solución."
        return prompt

    def _personalized_advice_template(self, cluster_label, performance):
        """
        Devuelve una plantilla de consejo personalizado según el cluster y desempeño.
        """
        templates = {
            0: "Te recomiendo reforzar los conceptos básicos antes de avanzar. Usa recursos introductorios y practica con ejemplos simples.",
            1: "Vas bien, pero puedes profundizar en los detalles técnicos. Intenta explicar los conceptos con tus propias palabras.",
            2: "¡Buen desempeño! Ahora enfócate en casos prácticos y escenarios reales para consolidar tu aprendizaje.",
            3: "Excelente nivel. Busca retos avanzados y comparte tus conocimientos con otros para seguir creciendo.",
            4: "Eres un referente en el área. Considera mentorizar a otros o contribuir a la comunidad."
        }
        base = templates.get(cluster_label, "Sigue practicando y busca feedback específico para tus áreas de mejora.")
        if performance == 'low':
            return base + " Recuerda que la perseverancia es clave."
        elif performance == 'mid':
            return base + " Estás en el camino correcto, sigue así."
        else:
            return base + " ¡Sigue desafiándote!"

    def _score_answer(self, user_answer, expected, similarity_score, soft=False):
        """
        Modelo de scoring mejorado: pondera similitud, completitud y precisión.
        """
        score = 0
        if similarity_score > 0.8:
            score += 2
        elif similarity_score > 0.65:
            score += 1
        if expected and user_answer.strip().lower() == expected.strip().lower():
            score += 1
        if not soft and len(user_answer.split()) > 8:
            score += 1  # Respuestas técnicas más completas
        if soft and any(x in user_answer.lower() for x in ["equipo", "comunicación", "liderazgo", "conflicto", "empatía"]):
            score += 1  # Soft skills clave
        return min(score, 4)

    # (Removed duplicate __init__ and LLM attributes; see top of class for correct implementation)
    def safe_llm_call(self, prompt, fallback=None):
        """
        Call the LLM with global rate limiting and timeout. If busy or slow, return fallback.
        """
        def handler(signum, frame):
            raise TimeoutError("LLM call timed out")
        with self._llm_lock:
            now = time.time()
            wait = self._llm_min_interval - (now - self._llm_last_call)
            if wait > 0:
                time.sleep(wait)
            self._llm_last_call = time.time()
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self._llm_timeout)
        try:
            result = self.llm(prompt)
            signal.alarm(0)
            return result.strip() if isinstance(result, str) else result
        except Exception:
            signal.alarm(0)
            return fallback if fallback is not None else "[Respuesta generada automáticamente: LLM no disponible]"
        finally:
            signal.signal(signal.SIGALRM, old_handler)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.tech_questions = self._load_dataset('tech_questions.jsonl')
        self.soft_questions = self._load_dataset('soft_skills.jsonl')
        self.emb_mgr = EmbeddingsManager()
        # Inicialización de frases motivacionales y emojis aprendidos
        self.motivational_phrases = [
            "¡Sigue así, vas por un gran camino! 🚀",
            "¡Tu esfuerzo se nota! 👏",
            "¡Demuestras dominio y seguridad! 😎",
            "¡Suma un logro más a tu carrera! 🥇",
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

    def _load_dataset(self, filename: str) -> List[dict]:
        path = Path(__file__).parent / 'datasets' / filename
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]


    def start_interview(self, user_id: str, role: Optional[str] = None, interview_type: Optional[str] = None, mode: str = "practice"):
        """
        Inicia la entrevista preguntando contexto relevante según el tipo de entrevista.
        """
        technical_context_questions = [
            "¿Para qué rol específico deseas prepararte? (IA, ciberseguridad, DevOps, infraestructura, Frontend, Backend, QA, Data Engineer, Soporte, Cloud Engineer, Security Analyst, Fullstack, Mobile Developer, Data Scientist.)",
            "¿Cuál es tu nivel de experiencia profesional (junior, semi-senior, senior)?",
            "¿Cuántos años de experiencia tienes en el rol?",
            "¿Qué conocimientos o tecnologías consideras tus fortalezas?",
            "¿Qué herramientas dominas mejor?",
            "¿Qué expectativas tienes sobre el trabajo o la empresa?",
        ]
        soft_context_questions = [
            "¿Qué valoras más en un equipo de trabajo?",
            "¿Cómo manejas el estrés o la presión en el trabajo?",
            "¿Qué habilidades blandas consideras tus fortalezas?",
            "¿Puedes contarme una situación donde resolviste un conflicto?",
            "¿Qué esperas del ambiente laboral ideal?",
        ]
        interview_type = interview_type or "technical"
        if interview_type == "soft":
            context_questions = soft_context_questions
        else:
            context_questions = technical_context_questions
        self.sessions[user_id] = {
            "role": role or None,
            "level": None,
            "years": None,
            "knowledge": [],
            "tools": [],
            "interview_type": interview_type,
            "mode": mode or "practice",
            "stage": "context",
            "history": [],
            "context_asked": 0,
            "tech_pool": [],
            "soft_pool": [],
            "last_type": None,
            "exam_answers": [],
            "exam_start_time": None,
            "context_questions": context_questions,
            # Gamificación
            "score": 0,
            "level": 1,
            "achievements": [],
            "exam_correct_count": 0
        }
        question = context_questions[0]
        self.sessions[user_id]["history"].append({"agent": question})
        # Iniciar temporizador de respuesta
        self._start_response_timer(user_id)
        return {"question": question}

    def _filter_questions(self, questions: List[dict], filters: dict) -> List[dict]:
        """
        Filtra preguntas usando todos los criterios disponibles en filters.
        """
        def match(q):
            for key, val in filters.items():
                if val is None or val == []:
                    continue
                if key in ["knowledge", "tools"]:
                    if not set(val).intersection(set(q.get(key, []))):
                        return False
                elif key == "years":
                    if q.get("years") and int(q["years"]) > int(val)+2 or int(q["years"]) < int(val)-2:
                        return False
                else:
                    if str(q.get(key, "")).lower() != str(val).lower():
                        return False
            return True
        filtered = [q for q in questions if match(q)]
        return filtered if filtered else questions.copy()

    def next_question(self, user_id: str, user_input: Optional[str] = None):
        """
        Alterna entre preguntas técnicas y de soft skills, según el contexto y el historial.
        """
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        # Si aún no termina el contexto, seguir preguntando contexto
        if session["stage"] == "context":
            context_questions = session.get("context_questions") or []
            idx = session["context_asked"] + 1
            if idx < len(context_questions):
                question = context_questions[idx]
                session["context_asked"] = idx
                session["history"].append({"agent": question})
                return {"question": question}
            else:
                # Al terminar contexto, filtrar preguntas técnicas y blandas relevantes al rol usando embeddings
                role = session.get('role','')
                tech_similares = self.emb_mgr.filter_questions_by_role(role, top_k=10, technical=True)
                session["tech_pool"] = tech_similares if tech_similares else self.tech_questions.copy()
                soft_similares = self.emb_mgr.filter_questions_by_role(role, top_k=10, technical=False)
                session["soft_pool"] = soft_similares if soft_similares else self.soft_questions.copy()
                session["stage"] = "interview"

        # Alternar tipo de pregunta según preferencia o tipo de entrevista
        if session.get("interview_type") == "soft":
            qtype = "soft"
        elif session.get("interview_type") == "technical":
            qtype = "technical"
        else:
            qtype = "soft" if session["last_type"] == "technical" else "technical"

        # Selección avanzada de preguntas usando embeddings, clustering y RankNet
        if user_input:
            contexto = f"{session.get('role','')} {session.get('level','')} {session.get('years','')} " \
                       f"{' '.join(session.get('knowledge',[]))} {' '.join(session.get('tools',[]))} " \
                       f"{' '.join([h.get('user','') for h in session['history'] if 'user' in h][-3:])}"
            if qtype == "technical" and session["tech_pool"]:
                # Selección avanzada (UMAP, HDBSCAN, softmax, penalización, RankNet)
                candidates = self.emb_mgr.advanced_question_selector(user_input + " " + contexto, history=session["history"], top_k=3, technical=True)
                # Hook RankNet si está disponible
                if hasattr(self.emb_mgr, 'ranknet'):
                    candidates = self.emb_mgr.ranknet_rank(candidates, user_input + " " + contexto, session["history"])
                question = None
                for q in candidates:
                    if q in session["tech_pool"]:
                        question = q
                        break
                if not question:
                    question = random.choice(session["tech_pool"])
                session["tech_pool"].remove(question)
                session["last_type"] = "technical"
                session["history"].append({"agent": question["question"]})
                return {"question": question["question"]}
            elif qtype == "soft" and session["soft_pool"]:
                candidates = self.emb_mgr.advanced_question_selector(user_input + " " + contexto, history=session["history"], top_k=3, technical=False)
                if hasattr(self.emb_mgr, 'ranknet'):
                    candidates = self.emb_mgr.ranknet_rank(candidates, user_input + " " + contexto, session["history"])
                question = None
                for q in candidates:
                    if q in session["soft_pool"]:
                        question = q
                        break
                if not question:
                    question = random.choice(session["soft_pool"])
                session["soft_pool"].remove(question)
                session["last_type"] = "soft"
                session["history"].append({"agent": question["scenario"]})
                return {"question": question["scenario"]}
        # Si no hay input, usar selección avanzada sin contexto inmediato
        if qtype == "technical" and session["tech_pool"]:
            candidates = self.emb_mgr.advanced_question_selector(
                session.get('role',''), history=session["history"], top_k=3, technical=True)
            if hasattr(self.emb_mgr, 'ranknet'):
                candidates = self.emb_mgr.ranknet_rank(candidates, session.get('role',''), session["history"])
            question = None
            for q in candidates:
                if q in session["tech_pool"]:
                    question = q
                    break
            if not question:
                question = random.choice(session["tech_pool"])
            session["tech_pool"].remove(question)
            session["last_type"] = "technical"
            session["history"].append({"agent": question["question"]})
            return {"question": question["question"]}
        elif qtype == "soft" and session["soft_pool"]:
            candidates = self.emb_mgr.advanced_question_selector(
                session.get('role',''), history=session["history"], top_k=3, technical=False)
            if hasattr(self.emb_mgr, 'ranknet'):
                candidates = self.emb_mgr.ranknet_rank(candidates, session.get('role',''), session["history"])
            question = None
            for q in candidates:
                if q in session["soft_pool"]:
                    question = q
                    break
            if not question:
                question = random.choice(session["soft_pool"])
            session["soft_pool"].remove(question)
            session["last_type"] = "soft"
            session["history"].append({"agent": question["scenario"]})
            return {"question": question["scenario"]}
        else:
            # Al terminar las 10 preguntas, mostrar encuesta de satisfacción antes del feedback final
            if not session.get("satisfaction_pending") and not session.get("satisfaction_done"):
                session["satisfaction_pending"] = True
                session["satisfaction_index"] = 0
                session["satisfaction_answers"] = []
                return {"satisfaction_survey": self.SATISFACTION_QUESTIONS[0], "index": 1, "total": len(self.SATISFACTION_QUESTIONS)}
            elif session.get("satisfaction_pending") and not session.get("satisfaction_done"):
                idx = session.get("satisfaction_index", 0)
                if idx < len(self.SATISFACTION_QUESTIONS):
                    return {"satisfaction_survey": self.SATISFACTION_QUESTIONS[idx], "index": idx+1, "total": len(self.SATISFACTION_QUESTIONS)}
                else:
                    session["satisfaction_pending"] = False
                    session["satisfaction_done"] = True
                    return {"message": "Gracias por responder la encuesta. Ahora recibirás tu feedback final."}
            else:
                return {"message": "Entrevista finalizada. No hay más preguntas."}
    def process_satisfaction_answer(self, user_id: str, rating: int):
        """
        Procesa la respuesta a la encuesta de satisfacción (1-5) y avanza a la siguiente pregunta o finaliza.
        """
        session = self.sessions.get(user_id)
        if not session or not session.get("satisfaction_pending"):
            return {"error": "No hay encuesta de satisfacción activa."}
        idx = session.get("satisfaction_index", 0)
        session["satisfaction_answers"].append(rating)
        idx += 1
        session["satisfaction_index"] = idx
        if idx < len(self.SATISFACTION_QUESTIONS):
            return {"satisfaction_survey": self.SATISFACTION_QUESTIONS[idx], "index": idx+1, "total": len(self.SATISFACTION_QUESTIONS)}
        else:
            session["satisfaction_pending"] = False
            session["satisfaction_done"] = True
            return {"message": "Gracias por responder la encuesta. Ahora recibirás tu feedback final."}


    def process_answer(self, user_id: str, answer: str):
        # Detener temporizador de respuesta al recibir input
        self._stop_response_timer(user_id)
        session = self.sessions.get(user_id)
        # Si la entrevista fue cerrada por inactividad, no procesar más
        if session and session.get('closed'):
            return {"feedback": "La entrevista ya fue cerrada por inactividad."}
        """
        Procesa la respuesta del usuario a una pregunta:
        - Valida la respuesta usando embeddings y NLP.
        - Si es incorrecta, genera feedback motivador y una pista (modo práctica).
        - Si es correcta, motiva y avanza a la siguiente pregunta.
        - En modo práctica: no avanza hasta respuesta correcta o agotar pistas.
        - En modo examen: solo confirma y suma puntos, sin feedback inmediato.
        - Registra la interacción para aprendizaje continuo y fine-tuning.
        Devuelve: dict con feedback, retry (bool) y control de avance.
        """
        """
        Procesa la respuesta del usuario, da feedback humano, motivador y aprende de buenas respuestas.
        Compara embeddings de la respuesta del candidato con la esperada para determinar si es correcta.
        """
    # ...existing code...
        # Analizar emoción del usuario
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        emotion_result = None
        try:
            emotion_result = analyze_emotion(answer)
        except Exception:
            emotion_result = None
        modo = session.get("mode", "practice")
        # Define last_agent early for use in both exam and practice modes
        last_agent = next((h["agent"] for h in reversed(session["history"]) if "agent" in h), "")
        
        # En modo examen, guardar respuestas y sumar puntos si es correcta, sin feedback inmediato
        # Reiniciar temporizador para la siguiente pregunta si la entrevista sigue activa
        if not session.get('closed'):
            self._start_response_timer(user_id)
        # Define last_agent and determine correctness for both modes
        last_agent = next((h["agent"] for h in reversed(session["history"]) if "agent" in h), "")
        similar_tech = self.emb_mgr.find_most_similar_tech(last_agent, top_k=1)
        similar_soft = self.emb_mgr.find_most_similar_soft(last_agent, top_k=1)
        expected = None
        if similar_tech and 'answer' in similar_tech[0]:
            expected = similar_tech[0]['answer']
        elif similar_soft and 'expected' in similar_soft[0]:
            expected = similar_soft[0]['expected']
        
        # NLP: comparar embeddings de la respuesta del candidato y la esperada
        is_correct = False
        similarity_score = 0.0
        if expected:
            try:
                cand_emb = self.emb_mgr.model.encode([answer], convert_to_tensor=True)
                exp_emb = self.emb_mgr.model.encode([expected], convert_to_tensor=True)
                similarity_score = float(self.emb_mgr.model.similarity(cand_emb, exp_emb))
                is_correct = similarity_score > 0.7
            except Exception:
                is_correct = answer.strip().lower() == expected.strip().lower()
        
        if modo == "exam":
            if session["exam_start_time"] is None:
                session["exam_start_time"] = time.time()
            session["exam_answers"].append(answer)
            # Gamificación: sumar puntos, calcular nivel y logros personalizados
            if is_correct:
                session["score"] += self.EXAM_POINTS_PER_CORRECT
                session["exam_correct_count"] += 1
                # Calcular nivel y nombre de nivel
                for i, threshold in enumerate(self.EXAM_LEVEL_THRESHOLDS[::-1]):
                    if session["score"] >= threshold:
                        session["level"] = len(self.EXAM_LEVEL_THRESHOLDS) - i
                        session["level_name"] = self.EXAM_LEVEL_NAMES[len(self.EXAM_LEVEL_THRESHOLDS) - i - 1]
                        break
                # Logros temáticos
                for req, ach in self.EXAM_ACHIEVEMENTS:
                    if session["exam_correct_count"] == req and ach not in session["achievements"]:
                        session["achievements"].append(ach)
            return {"feedback": "Respuesta registrada. Continúa con la siguiente pregunta."}

        # --- Lógica de pistas y reintentos en modo práctica ---
        # (Mover después de definir last_agent, base_feedback, motivacion)
        """
        Procesa la respuesta del usuario, da feedback humano, motivador y aprende de buenas respuestas.
        Compara embeddings de la respuesta del candidato con la esperada para determinar si es correcta.
        """
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        session["history"].append({"user": answer})
        # Inicializar hint_count si no existe
        if "hint_count" not in session or not isinstance(session["hint_count"], int):
            session["hint_count"] = 0
        modo = session.get("mode", "practice")
        # En modo examen, guardar respuestas y no dar feedback inmediato
        if modo == "exam":
            # Inicializar campos de gamificación avanzada
            session.setdefault('exam_streak', 0)
            session.setdefault('exam_incorrect_streak', 0)
            session.setdefault('exam_topics', set())
            session.setdefault('soft_correct', 0)
            session.setdefault('exam_attempts', 0)
            session.setdefault('last_answer_time', 99)
            session.setdefault('last_answer_timestamp', time.time())
            if session["exam_start_time"] is None:
                session["exam_start_time"] = time.time()
            session["exam_answers"].append(answer)
            session['exam_attempts'] += 1
            # Calcular tiempo de respuesta
            now = time.time()
            session['last_answer_time'] = now - session.get('last_answer_timestamp', now)
            session['last_answer_timestamp'] = now
            # Detectar tema (técnico/soft)
            if last_agent:
                if any(q.get('question','') == last_agent for q in self.tech_questions):
                    session['exam_topics'].add('tech')
                elif any(q.get('scenario','') == last_agent for q in self.soft_questions):
                    session['exam_topics'].add('soft')
            # Puntaje base
            points = 0
            if is_correct:
                points += self.EXAM_POINTS_PER_CORRECT
                session['exam_streak'] += 1
                session['exam_incorrect_streak'] = 0
                if any(q.get('scenario','') == last_agent for q in self.soft_questions):
                    session['soft_correct'] += 1
            else:
                session['exam_streak'] = 0
                session['exam_incorrect_streak'] += 1
            # Bonus por racha
            if session['exam_streak'] >= 3:
                points += 5
            # Bonus por rapidez
            if session['last_answer_time'] < 10:
                points += 3
            # Bonus por variedad
            if len(session['exam_topics']) >= 2:
                points += 2
            # Bonus por precisión
            if is_correct and similarity_score > 0.85:
                points += 2
            # Actualizar puntaje y nivel
            session['score'] = session.get('score',0) + points
            # Niveles
            for i, threshold in enumerate(self.EXAM_LEVEL_THRESHOLDS[::-1]):
                if session['score'] >= threshold:
                    session['level'] = len(self.EXAM_LEVEL_THRESHOLDS) - i
                    session['level_name'] = self.EXAM_LEVEL_NAMES[len(self.EXAM_LEVEL_THRESHOLDS) - i - 1]
                    break
            # Logros clásicos
            for req, ach in self.EXAM_ACHIEVEMENTS:
                if session["exam_correct_count"] == req and ach not in session["achievements"]:
                    session["achievements"].append(ach)
            # Insignias creativas
            for badge, emoji, cond in self.EXAM_BADGES:
                if cond(session) and badge not in session.get('achievements', []):
                    session['achievements'].append(f"{badge} {emoji}")
            # Feedback visual/textual inmediato
            feedback = f"<b>+{points} puntos</b> | Nivel: <b>{session.get('level_name','')}</b> | Logros: {' '.join([a for a in session.get('achievements',[]) if any(e in a for e in ['🔥','💪','⏱️','🌈','🧠','🎯','🏅'])])}"
            if is_correct:
                # Guardar interacción para fine-tuning (respuesta correcta)
                self._save_interaction_for_finetune(user_id, last_agent, answer, "", hint=None, correct=True)
        emotion_label = None
        if emotion_result and isinstance(emotion_result, list) and len(emotion_result) > 0:
            # El pipeline puede devolver [[{label,score},...]] o [{label,score},...]
            first_elem = emotion_result[0] # type: ignore
            if isinstance(first_elem, list) and len(first_elem) > 0 and isinstance(first_elem[0], dict) and 'label' in first_elem[0]: # type: ignore
                emotion_label = first_elem[0].get('label') # type: ignore
            elif isinstance(first_elem, dict) and 'label' in first_elem:
                emotion_label = first_elem['label']

        # --- MEMORIA DE CONTEXTO: analizar historial para feedback y sesgos ---
        context_memory = self._get_context_memory(session)
        emociones_hist = context_memory['emociones']
        emociones_neg = [e for e in emociones_hist if e in ["sadness", "fear", "disgust", "anger"]]
        emociones_pos = [e for e in emociones_hist if e in ["joy", "surprise"]]
        # Detectar posible sesgo: si hay 3+ emociones negativas seguidas, sugerir pregunta objetiva
        sesgo_detectado = len(emociones_neg) >= 3 and emociones_neg[-3:] == emociones_neg[-3:]
        # Feedback adaptativo según emoción y memoria
        if is_correct:
                session["hint_count"] = 0  # Reiniciar para la siguiente pregunta
                # Aprendizaje automático: generar nueva frase motivacional y emoji si es posible
                base_feedback = random.choice([
                    "¡Respuesta correcta!",
                    "¡Muy bien!",
                    "¡Excelente!",
                    "¡Perfecto!",
                    "¡Eso es!"
                ])
                # Feedback adaptativo por emoción
                if sesgo_detectado:
                    motivacion = "Detectamos que has tenido varias respuestas con emociones negativas. Recuerda que el objetivo es aprender y mejorar. ¿Te gustaría una pregunta más objetiva para equilibrar el proceso?"
                elif emotion_label in ["sadness", "fear", "disgust", "anger"]:
                    motivacion = "¡Ánimo! Cada paso cuenta, y lo estás haciendo muy bien. Aquí tienes un reto más para seguir creciendo. 💪"
                elif emotion_label in ["joy", "surprise"]:
                    motivacion = "¡Excelente energía! ¿Listo para un reto aún mayor? 🚀"
                elif random.random() < 0.5 and len(self.motivational_phrases) > 0:
                    motivacion = random.choice(self.motivational_phrases)
                else:
                    # Generar nueva frase motivacional con LLM
                    prompt = (
                        "Genera una frase motivacional breve y positiva para un candidato que acaba de responder correctamente en una entrevista técnica. Incluye un emoji nuevo y diferente a los ya usados: "
                        f"{','.join(self.positive_emojis)}. Devuelve solo la frase y emoji."
                    )
                    nueva = self.safe_llm_call(
                        prompt,
                        fallback=random.choice(self.motivational_phrases)
                    )
                    if nueva and nueva not in self.motivational_phrases:
                        self.motivational_phrases.append(nueva)
                        for char in nueva:
                            if char not in self.positive_emojis and ord(char) > 1000:
                                self.positive_emojis.append(char)
                        motivacion = nueva
                    else:
                        motivacion = random.choice(self.motivational_phrases)
                # Añadir emoji aleatorio aprendido
                if self.positive_emojis:
                    base_feedback += " " + random.choice(self.positive_emojis)
        else:
                # Siempre definir base_feedback y motivacion ANTES de usarlas
                base_feedback = random.choice([
                    "La respuesta no es del todo correcta.",
                    "Casi, pero falta un poco más.",
                    "No te preocupes, ¡a todos nos pasa!",
                    "Respuesta incompleta, pero vas bien."
                ])
                if 'emotion_label' not in locals():
                    emotion_label = None
                if sesgo_detectado:
                    motivacion = "Detectamos varias respuestas con emociones negativas. Para evitar sesgos, aquí tienes una pregunta objetiva y neutral. ¡Tú puedes!"
                elif emotion_label in ["sadness", "fear", "disgust", "anger"]:
                    motivacion = "No te preocupes, todos aprendemos de los errores. ¡Sigue adelante, lo lograrás! 💡"
                elif emotion_label in ["joy", "surprise"]:
                    motivacion = "¡Buen ánimo! Repasa el concepto y verás que la próxima será tuya. 📚"
                elif hasattr(self, 'negative_phrases') and random.random() < 0.5 and len(self.negative_phrases) > 0:
                    motivacion = random.choice(self.negative_phrases)
                else:
                    # Generar nueva frase de ánimo con LLM
                    prompt = (
                        "Genera una frase breve de ánimo y refuerzo para un candidato que acaba de fallar una pregunta en una entrevista técnica. Incluye un emoji nuevo y diferente a los ya usados: "
                        f"{','.join(getattr(self, 'negative_emojis', []))}. Devuelve solo la frase y emoji."
                    )
                    nueva = self.safe_llm_call(
                        prompt,
                        fallback=(random.choice(self.negative_phrases) if hasattr(self, 'negative_phrases') and self.negative_phrases else "¡Ánimo!")
                    )
                    if hasattr(self, 'negative_phrases') and nueva and nueva not in self.negative_phrases:
                        self.negative_phrases.append(nueva)
                        for char in nueva:
                            if hasattr(self, 'negative_emojis') and char not in self.negative_emojis and ord(char) > 1000:
                                self.negative_emojis.append(char)
                        motivacion = nueva
                    else:
                        motivacion = random.choice(self.negative_phrases) if hasattr(self, 'negative_phrases') and self.negative_phrases else "¡Ánimo!"
                # Si no ha agotado el máximo de pistas, generar pista y no avanzar
                session["hint_count"] += 1
                hint_prompt = (
                    f"Eres un entrevistador experto en {session['role']}. "
                    f"Pregunta: {last_agent}\nRespuesta del candidato: {answer}\n"
                    f"Respuesta esperada: {expected if expected else 'N/A'}\n"
                    f"Genera una pista interactiva y didáctica para ayudar al candidato a acercarse a la respuesta correcta. "
                    f"No des la respuesta directa, pero explica el concepto clave de la pregunta, proporciona una analogía sencilla, un caso de uso real en la industria y un ejemplo práctico. "
                    f"Hazlo en tono motivador y dinámico. Si puedes, invita al candidato a reflexionar o a imaginar una situación real. "
                    f"Esta es la pista número {min(session['hint_count'], self.MAX_HINTS)}/{self.MAX_HINTS}."
                )
                pista = self.safe_llm_call(
                    hint_prompt,
                    fallback=(
                        f"Concepto clave: {expected if expected else 'Revisa la definición principal del tema.'}\n"
                        f"Ejemplo práctico: Imagina que aplicas este concepto en un proyecto real, ¿cómo lo usarías?\n"
                        f"Caso de uso en la industria: Este conocimiento es fundamental en roles como {session.get('role','el área correspondiente')}."
                    )
                )
                if session["hint_count"] <= self.MAX_HINTS:
                    feedback = f"{base_feedback}\n{motivacion}\nPista {session['hint_count']}/{self.MAX_HINTS}: {pista}"
                    retry = True
                else:
                    feedback = f"{base_feedback}\n{motivacion}\nHas alcanzado el máximo de {self.MAX_HINTS} pistas. La respuesta esperada era: {expected}.\nÚltima pista extra: {pista}"
                    session["hint_count"] = 0
                    retry = False
                # Guardar emoción en el historial para memoria de contexto
                session["history"].append({"agent": feedback, "emotion": emotion_label})
                self._save_interaction_for_finetune(user_id, last_agent, answer, base_feedback + "\n" + motivacion, hint=pista, correct=False)
                return {"feedback": feedback, "retry": retry}

    def _save_interaction_for_finetune(self, user_id: str, question: str, answer: str, feedback: str, hint: Optional[str] = None, correct: bool = False):
        """
        Guarda la interacción relevante para procesos de fine-tuning y mejora del modelo.
        - user_id: identificador del usuario
        - question: pregunta realizada
        - answer: respuesta del usuario
        - feedback: feedback generado
        - hint: pista generada (si aplica)
        - correct: si la respuesta fue correcta
        Almacena en datasets/finetune_interactions.jsonl
        """
        """
        Guarda la interacción para posibles procesos de fine-tuning o análisis posterior.
        """
        try:
            new_entry = {
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "feedback": feedback,
                "hint": hint,
                "correct": correct
            }
            # Guardar en un archivo de interacciones para fine-tuning
            finetune_path = Path(__file__).parent / 'datasets' / 'finetune_interactions.jsonl'
            with open(finetune_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass  # No interrumpir el flujo si falla la escritura

    def end_interview(self, user_id: str):
        """
        Finaliza la entrevista y entrega un resumen robusto:
        - Analiza el historial de respuestas y feedback.
        - Calcula estadísticas, fortalezas, puntos de mejora y habilidades destacadas.
        - Integra resultados de gamificación y encuesta de satisfacción.
        - Genera un resumen profesional usando LLM.
        """
        """
        Finaliza la entrevista y entrega un resumen del desempeño del usuario, con estadísticas, fortalezas y puntos de mejora.
        Si hay encuesta de satisfacción pendiente, la solicita antes del feedback final.
        """
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        # Si la entrevista está cerrada por inactividad, no permitir más preguntas
        if session and session.get('closed'):
            mensaje = (
                "La entrevista ya fue cerrada por inactividad. "
                "Recuerda que puedes volver a intentarlo cuando quieras. "
                "Te recomendamos prepararte con calma, repasar los temas clave y regresar cuando estés listo/a. "
                "¡Gracias por usar Ready4Hire!"
            )
            session['history'].append({"agent": mensaje})
            return {"error": mensaje}
        # Si la encuesta no se ha completado, forzarla antes del feedback final
        if not session.get("satisfaction_done"):
            if not session.get("satisfaction_pending"):
                session["satisfaction_pending"] = True
                session["satisfaction_index"] = 0
                session["satisfaction_answers"] = []
            return {"satisfaction_survey": self.SATISFACTION_QUESTIONS[session["satisfaction_index"]], "index": session["satisfaction_index"]+1, "total": len(self.SATISFACTION_QUESTIONS)}
        # Una vez completada la encuesta, entregar el feedback final y limpiar la sesión
        # (No hacer pop hasta después de usar la sesión)
        modo = session.get("mode", "practice")
        tiempo_total = None
        if modo == "exam" and session.get("exam_start_time"):
            tiempo_total = time.time() - session["exam_start_time"]
        # Analizar historial para estadísticas
        total_tech = 0
        total_soft = 0
        correct_tech = 0
        correct_soft = 0
        habilidades_destacadas = set()
        habilidades_mejorar = set()
        for i, h in enumerate(session["history"]):
            if "agent" in h and i+1 < len(session["history"]) and "user" in session["history"][i+1]:
                pregunta = h["agent"]
                respuesta = session["history"][i+1]["user"]
                # Determinar si es técnica o soft
                es_tech = any(q.get("question","") == pregunta for q in self.tech_questions)
                es_soft = any(q.get("scenario","") == pregunta for q in self.soft_questions)
                # Buscar la pregunta original
                if es_tech:
                    total_tech += 1
                    qref = next((q for q in self.tech_questions if q.get("question","") == pregunta), None)
                    expected = qref.get("answer") if qref else None
                    if expected:
                        cand_emb = self.emb_mgr.model.encode([respuesta], convert_to_tensor=True)
                        exp_emb = self.emb_mgr.model.encode([expected], convert_to_tensor=True)
                        sim = float(self.emb_mgr.model.similarity(cand_emb, exp_emb))
                        if sim > 0.7:
                            correct_tech += 1
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
                        if sim > 0.7:
                            correct_soft += 1
                            if qref and qref.get("knowledge"):
                                habilidades_destacadas.update(qref["knowledge"])
                        else:
                            if qref and qref.get("knowledge"):
                                habilidades_mejorar.update(qref["knowledge"])
        # Preparar prompt para LLM
        gamification_text = ""
        if session.get("mode") == "exam":
            level_name = session.get("level_name", self.EXAM_LEVEL_NAMES[0])
            next_level_idx = min(session.get("level",1), len(self.EXAM_LEVEL_NAMES)-1)
            next_level_name = self.EXAM_LEVEL_NAMES[next_level_idx]
            next_level_points = self.EXAM_LEVEL_THRESHOLDS[next_level_idx] if next_level_idx < len(self.EXAM_LEVEL_THRESHOLDS) else 'MAX'
            points_to_next = max(0, next_level_points - session.get("score",0)) if isinstance(next_level_points, int) else 0
            gamification_text = (
                f"\n\n---\n🎮 Sistema de Gamificación Avanzada 🎮\n"
                f"Puntaje total: {session.get('score',0)} puntos\n"
                f"Nivel alcanzado: {level_name} (Nivel {session.get('level',1)})\n"
                f"Próximo nivel: {next_level_name} (a {points_to_next} puntos)\n"
                f"Logros desbloqueados: {', '.join(session.get('achievements', [])) if session.get('achievements') else 'Ninguno'}\n"
                f"Respuestas correctas: {session.get('exam_correct_count',0)} de 10\n"
                "\n¡Sigue practicando para desbloquear más logros y alcanzar niveles legendarios! "
                "Cada respuesta correcta suma puntos y te acerca a ser un/a Sensei del conocimiento. "
                "¿Podrás llegar a Ícono? ¡Vuelve a intentarlo y supera tu récord!\n"
            )
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
            + f"Historial de la entrevista: {session['history']}\n"
            + f"Encuesta de satisfacción (1-5): {session.get('satisfaction_answers',[])}"
            + gamification_text
        )
        summary = self.safe_llm_call(
            prompt,
            fallback="[Resumen generado automáticamente: LLM no disponible]"
        )
        self.sessions.pop(user_id, None)
        return {"summary": summary}
