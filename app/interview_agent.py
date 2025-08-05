

# ===============================
# Ready4Hire - Agente de Entrevista IA
# ===============================
# Este módulo implementa la lógica central del agente de entrevistas:
# - Selección inteligente de preguntas técnicas y blandas
# - Feedback motivador, emocional y adaptativo
# - Pistas generadas por LLM
# - Gamificación avanzada (puntos, niveles, logros)
# - Análisis emocional y personalización
# - Aprendizaje continuo y registro para fine-tuning
#
# Autor: JeronimoRestrepo48
# Licencia: MIT


from langchain.llms import Ollama
from typing import Dict, Any, List, Optional
import random
import json
from pathlib import Path
import time
# Integración de embeddings
from app.embeddings.embeddings_manager import EmbeddingsManager
from app.emotion_analyzer import analyze_emotion

class InterviewAgent:
    MAX_HINTS = 3
    # Configuración de gamificación
    EXAM_POINTS_PER_CORRECT = 15
    EXAM_LEVEL_THRESHOLDS = [0, 30, 60, 100, 150, 210, 280, 360, 450, 550, 700, 900]  # Puntos requeridos para cada nivel
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

    """
    Agente de entrevistas IA para simulación de procesos técnicos y de soft skills.
    - Alterna preguntas técnicas y blandas según contexto y tipo de entrevista.
    - Usa embeddings y NLP para seleccionar y validar preguntas/respuestas.
    - Proporciona feedback emocional, motivador y adaptativo.
    - Integra gamificación, análisis emocional y aprendizaje continuo.
    - Registra todas las interacciones relevantes para mejorar el modelo (fine-tuning).
    """

    def __init__(self, model_name: str = "llama3"):
        """
        Inicializa el agente de entrevista.
        - model_name: nombre del modelo LLM a usar (por defecto llama3 vía Ollama).
        - Carga datasets de preguntas técnicas y blandas.
        - Inicializa gestor de embeddings y frases motivacionales/emojis.
        """
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

    def _load_dataset(self, filename: str) -> List[dict]:
        path = Path(__file__).parent / 'datasets' / filename
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]


    def start_interview(self, user_id: str, role: Optional[str] = None, interview_type: Optional[str] = None, mode: str = "practice"):
        """
        Inicia la entrevista preguntando contexto relevante según el tipo de entrevista.
        """
        technical_context_questions = [
            "¿Para qué rol específico deseas prepararte? (IA, ciberseguridad, devops, infraestructura, etc.)",
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
                # Al terminar contexto, usar embeddings para seleccionar las 10 preguntas técnicas más relevantes
                contexto = f"{session.get('role','')} {session.get('level','')} {session.get('years','')} " \
                           f"{' '.join(session.get('knowledge',[]))} {' '.join(session.get('tools',[]))}"
                tech_similares = self.emb_mgr.find_most_similar_tech(contexto, top_k=10)
                session["tech_pool"] = tech_similares if tech_similares else self.tech_questions.copy()
                # Para soft skills, usar embeddings para seleccionar las 10 más relevantes
                soft_similares = self.emb_mgr.find_most_similar_soft(contexto, top_k=10)
                session["soft_pool"] = soft_similares if soft_similares else self.soft_questions.copy()
                session["stage"] = "interview"

        # Alternar tipo de pregunta según preferencia o tipo de entrevista
        if session.get("interview_type") == "soft":
            qtype = "soft"
        elif session.get("interview_type") == "technical":
            qtype = "technical"
        else:
            qtype = "soft" if session["last_type"] == "technical" else "technical"

        # Si el usuario da un input, buscar pregunta relevante por embeddings y contexto
        if user_input:
            contexto = f"{session.get('role','')} {session.get('level','')} {session.get('years','')} " \
                       f"{' '.join(session.get('knowledge',[]))} {' '.join(session.get('tools',[]))} " \
                       f"{' '.join([h.get('user','') for h in session['history'] if 'user' in h][-3:])}"
            if qtype == "technical" and session["tech_pool"]:
                # Buscar pregunta relevante considerando contexto y últimas respuestas
                similar = self.emb_mgr.find_most_similar_tech(user_input + " " + contexto, top_k=2)
                # Si la última fue correcta, priorizar dificultad mayor si existe
                question = None
                for q in similar:
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
                similar = self.emb_mgr.find_most_similar_soft(user_input + " " + contexto, top_k=2)
                question = None
                for q in similar:
                    if q in session["soft_pool"]:
                        question = q
                        break
                if not question:
                    question = random.choice(session["soft_pool"])
                session["soft_pool"].remove(question)
                session["last_type"] = "soft"
                session["history"].append({"agent": question["scenario"]})
                return {"question": question["scenario"]}
        # Si no hay input, seguir flujo aleatorio
        if qtype == "technical" and session["tech_pool"]:
            question = random.choice(session["tech_pool"])
            session["tech_pool"].remove(question)
            session["last_type"] = "technical"
            session["history"].append({"agent": question["question"]})
            return {"question": question["question"]}
        elif qtype == "soft" and session["soft_pool"]:
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
        session = self.sessions.get(user_id)
        if not session:
            return {"error": "Entrevista no iniciada"}
        # Inicializar hint_count si no existe
        if "hint_count" not in session:
            session["hint_count"] = 0
        session["history"].append({"user": answer})
        # Analizar emoción del usuario
        emotion_result = None
        try:
            emotion_result = analyze_emotion(answer)
        except Exception:
            emotion_result = None
        modo = session.get("mode", "practice")
        # En modo examen, guardar respuestas y sumar puntos si es correcta, sin feedback inmediato
        if modo == "exam":
            if session["exam_start_time"] is None:
                session["exam_start_time"] = time.time()
            session["exam_answers"].append(answer)
            # Determinar si es correcta (igual que en feedback normal)
            last_agent = next((h["agent"] for h in reversed(session["history"]) if "agent" in h), "")
            similar_tech = self.emb_mgr.find_most_similar_tech(last_agent, top_k=1)
            similar_soft = self.emb_mgr.find_most_similar_soft(last_agent, top_k=1)
            expected = None
            if similar_tech and 'answer' in similar_tech[0]:
                expected = similar_tech[0]['answer']
            elif similar_soft and 'expected' in similar_soft[0]:
                expected = similar_soft[0]['expected']
            is_correct = False
            if expected:
                try:
                    cand_emb = self.emb_mgr.model.encode([answer], convert_to_tensor=True)
                    exp_emb = self.emb_mgr.model.encode([expected], convert_to_tensor=True)
                    similarity_score = float(self.emb_mgr.model.similarity(cand_emb, exp_emb))
                    is_correct = similarity_score > 0.7
                except Exception:
                    is_correct = answer.strip().lower() == expected.strip().lower()
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
        modo = session.get("mode", "practice")
        # En modo examen, guardar respuestas y no dar feedback inmediato
        if modo == "exam":
            if session["exam_start_time"] is None:
                session["exam_start_time"] = time.time()
            session["exam_answers"].append(answer)
            # Solo devolver confirmación, no feedback
            return {"feedback": "Respuesta registrada. Continúa con la siguiente pregunta."}
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

        # Mensaje base
        # Determinar emoción dominante (si hay)
                # Guardar interacción para fine-tuning (respuesta correcta)
                self._save_interaction_for_finetune(user_id, last_agent, answer, "", hint=None, correct=True)
        emotion_label = None
        if emotion_result and isinstance(emotion_result, list) and len(emotion_result) > 0:
            # El pipeline puede devolver [[{label,score},...]] o [{label,score},...]
            first_elem = emotion_result[0]
            if isinstance(first_elem, list) and len(first_elem) > 0 and isinstance(first_elem[0], dict) and 'label' in first_elem[0]: # type: ignore
                emotion_label = first_elem[0].get('label') # type: ignore
            elif isinstance(first_elem, dict) and 'label' in first_elem:
                emotion_label = first_elem['label']

        # Feedback adaptativo según emoción
        # Si detecta tristeza, frustración, miedo, etc., usar frases más empáticas y de apoyo
        # Si detecta alegría/confianza, usar frases más retadoras o de celebración
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
            if emotion_label in ["sadness", "fear", "disgust", "anger"]:
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
                try:
                    nueva = self.llm(prompt).strip()
                    if nueva and nueva not in self.motivational_phrases:
                        self.motivational_phrases.append(nueva)
                        for char in nueva:
                            if char not in self.positive_emojis and ord(char) > 1000:
                                self.positive_emojis.append(char)
                        motivacion = nueva
                    else:
                        motivacion = random.choice(self.motivational_phrases)
                except Exception:
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
            if emotion_label in ["sadness", "fear", "disgust", "anger"]:
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
                try:
                    nueva = self.llm(prompt).strip()
                    if hasattr(self, 'negative_phrases') and nueva and nueva not in self.negative_phrases:
                        self.negative_phrases.append(nueva)
                        for char in nueva:
                            if hasattr(self, 'negative_emojis') and char not in self.negative_emojis and ord(char) > 1000:
                                self.negative_emojis.append(char)
                        motivacion = nueva
                    else:
                        motivacion = random.choice(self.negative_phrases) if hasattr(self, 'negative_phrases') and self.negative_phrases else "¡Ánimo!"
                except Exception:
                    motivacion = random.choice(self.negative_phrases) if hasattr(self, 'negative_phrases') and self.negative_phrases else "¡Ánimo!"
            # Si no ha agotado el máximo de pistas, generar pista y no avanzar
            if session["hint_count"] < self.MAX_HINTS:
                session["hint_count"] += 1
                hint_prompt = (
                    f"Eres un entrevistador experto en {session['role']}. "
                    f"Pregunta: {last_agent}\nRespuesta del candidato: {answer}\n"
                    f"Respuesta esperada: {expected if expected else 'N/A'}\n"
                    f"Genera una pista breve, útil y concreta para ayudar al candidato a acercarse a la respuesta correcta. "
                    f"No des la respuesta, pero orienta con un concepto, ejemplo, analogía o palabra clave relevante. "
                    f"Esta es la pista número {session['hint_count']} de {self.MAX_HINTS}."
                )
                pista = self.llm(hint_prompt).strip()
                feedback = f"{base_feedback}\n{motivacion}\nPista {session['hint_count']}/{self.MAX_HINTS}: {pista}"
                session["history"].append({"agent": feedback})
                self._save_interaction_for_finetune(user_id, last_agent, answer, base_feedback + "\n" + motivacion, hint=pista, correct=False)
                return {"feedback": feedback, "retry": True}
            else:
                feedback = f"{base_feedback}\n{motivacion}\nHas alcanzado el máximo de {self.MAX_HINTS} pistas. La respuesta esperada era: {expected}."
                session["history"].append({"agent": feedback})
                self._save_interaction_for_finetune(user_id, last_agent, answer, base_feedback + "\n" + motivacion, hint=None, correct=False)
                session["hint_count"] = 0
                return {"feedback": feedback, "retry": False}

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
        summary = self.llm(prompt)
        self.sessions.pop(user_id, None)
        return {"summary": summary}
