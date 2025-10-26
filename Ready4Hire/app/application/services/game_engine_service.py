"""
Game Engine Service
Motor de juegos con IA para gamificación
"""

import logging
import random
from typing import Dict, List, Optional
from datetime import datetime, timezone
from app.domain.entities.gamification import Game, GameSession, GameType

logger = logging.getLogger(__name__)


class GameEngineService:
    """Motor de juegos con IA"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.active_sessions: Dict[str, GameSession] = {}

    # ========================================================================
    # CODE CHALLENGE - Desafíos de programación
    # ========================================================================

    def create_code_challenge(self, profession: str, difficulty: str, skill: str) -> Dict:
        """Crea un desafío de código con IA"""
        prompt = f"""Genera un desafío de programación para un {profession} de nivel {difficulty}.

Enfoque en: {skill}

Incluye:
1. Descripción del problema
2. Ejemplos de entrada/salida
3. Restricciones
4. Función inicial (esqueleto de código)

Formato:
PROBLEMA: [descripción clara del problema]
EJEMPLOS:
- Input: ... Output: ...
RESTRICCIONES: [lista de restricciones]
CODIGO_INICIAL: [función/clase base]
"""

        try:
            response = self.llm_service.generate(prompt, max_tokens=500)

            # Parse response
            challenge = {
                "type": "code_challenge",
                "problem": "",
                "examples": [],
                "constraints": [],
                "initial_code": "",
                "test_cases": [],
                "difficulty": difficulty,
                "skill": skill,
            }

            # Simple parsing (mejorar con regex)
            sections = response.split("\n\n")
            for section in sections:
                if section.startswith("PROBLEMA:"):
                    challenge["problem"] = section.replace("PROBLEMA:", "").strip()
                elif section.startswith("EJEMPLOS:"):
                    challenge["examples"] = [line.strip() for line in section.split("\n")[1:] if line.strip()]
                elif section.startswith("RESTRICCIONES:"):
                    challenge["constraints"] = [line.strip() for line in section.split("\n")[1:] if line.strip()]
                elif section.startswith("CODIGO_INICIAL:"):
                    challenge["initial_code"] = section.replace("CODIGO_INICIAL:", "").strip()

            return challenge

        except Exception as e:
            logger.error(f"Error creating code challenge: {e}")
            return self._fallback_code_challenge(skill, difficulty)

    def _fallback_code_challenge(self, skill: str, difficulty: str) -> Dict:
        """Challenge de respaldo si falla la IA"""
        challenges = {
            "python": {
                "junior": {
                    "problem": "Escribe una función que retorne el factorial de un número",
                    "initial_code": "def factorial(n):\n    # Tu código aquí\n    pass",
                    "examples": ["Input: 5, Output: 120", "Input: 3, Output: 6"],
                }
            }
        }
        return challenges.get(skill, {}).get(difficulty, challenges["python"]["junior"])

    # ========================================================================
    # QUICK QUIZ - Quiz rápido
    # ========================================================================

    def create_quick_quiz(self, profession: str, difficulty: str, topic: str, num_questions: int = 5) -> Dict:
        """Crea un quiz rápido con IA"""
        prompt = f"""Genera {num_questions} preguntas de opción múltiple para un {profession} de nivel {difficulty}.

Tema: {topic}

Para cada pregunta incluye:
- Pregunta clara y concisa
- 4 opciones de respuesta (A, B, C, D)
- Indicar la respuesta correcta
- Breve explicación de por qué es correcta

Formato:
Q1: [pregunta]
A) [opción A]
B) [opción B]
C) [opción C]
D) [opción D]
Correcta: [letra]
Explicación: [razón]
"""

        try:
            response = self.llm_service.generate(prompt, max_tokens=800)

            questions = []
            # Parse questions (simplificado)
            current_q = {}
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("Q"):
                    if current_q:
                        questions.append(current_q)
                    current_q = {
                        "question": line.split(":", 1)[1].strip(),
                        "options": [],
                        "correct": "",
                        "explanation": "",
                    }
                elif line.startswith(("A)", "B)", "C)", "D)")):
                    current_q["options"].append(line)
                elif line.startswith("Correcta:"):
                    current_q["correct"] = line.split(":", 1)[1].strip()
                elif line.startswith("Explicación:"):
                    current_q["explanation"] = line.split(":", 1)[1].strip()

            if current_q:
                questions.append(current_q)

            return {
                "type": "quick_quiz",
                "questions": questions[:num_questions],
                "topic": topic,
                "difficulty": difficulty,
            }

        except Exception as e:
            logger.error(f"Error creating quiz: {e}")
            return self._fallback_quiz(topic, num_questions)

    def _fallback_quiz(self, topic: str, num_questions: int) -> Dict:
        """Quiz de respaldo"""
        return {
            "type": "quick_quiz",
            "questions": [
                {
                    "question": f"Pregunta sobre {topic}",
                    "options": ["A) Opción 1", "B) Opción 2", "C) Opción 3", "D) Opción 4"],
                    "correct": "A",
                    "explanation": "Explicación de la respuesta correcta",
                }
            ]
            * num_questions,
        }

    # ========================================================================
    # SCENARIO SIMULATOR - Simulador de escenarios
    # ========================================================================

    def create_scenario_simulator(self, profession: str, scenario_type: str) -> Dict:
        """Crea simulación de escenario laboral real"""
        prompt = f"""Crea un escenario laboral realista para un {profession}.

Tipo de escenario: {scenario_type}

Incluye:
1. Contexto y situación inicial
2. 3-4 opciones de acción
3. Consecuencias de cada acción
4. Mejor opción y por qué

El escenario debe ser realista y desafiante.

Formato:
CONTEXTO: [descripción de la situación]
OPCIONES:
A) [acción 1]
B) [acción 2]
C) [acción 3]
D) [acción 4]
CONSECUENCIAS:
A: [consecuencias de elegir A]
B: [consecuencias de elegir B]
...
MEJOR_OPCION: [letra]
RAZONAMIENTO: [explicación]
"""

        try:
            response = self.llm_service.generate(prompt, max_tokens=600)

            # Parse scenario
            scenario = {
                "type": "scenario_simulator",
                "context": "",
                "options": [],
                "consequences": {},
                "best_option": "",
                "reasoning": "",
                "profession": profession,
                "scenario_type": scenario_type,
            }

            # Simple parsing
            sections = response.split("\n\n")
            for section in sections:
                if section.startswith("CONTEXTO:"):
                    scenario["context"] = section.replace("CONTEXTO:", "").strip()
                elif section.startswith("OPCIONES:"):
                    scenario["options"] = [line.strip() for line in section.split("\n")[1:] if line.strip()]
                elif section.startswith("MEJOR_OPCION:"):
                    scenario["best_option"] = section.replace("MEJOR_OPCION:", "").strip()
                elif section.startswith("RAZONAMIENTO:"):
                    scenario["reasoning"] = section.replace("RAZONAMIENTO:", "").strip()

            return scenario

        except Exception as e:
            logger.error(f"Error creating scenario: {e}")
            return self._fallback_scenario(profession)

    def _fallback_scenario(self, profession: str) -> Dict:
        """Escenario de respaldo"""
        return {
            "type": "scenario_simulator",
            "context": f"Eres un {profession} enfrentando un desafío típico en tu trabajo.",
            "options": [
                "A) Consultar con tu equipo",
                "B) Tomar una decisión inmediata",
                "C) Investigar más antes de actuar",
                "D) Escalar al supervisor",
            ],
            "best_option": "C",
            "reasoning": "Es importante investigar antes de actuar",
        }

    # ========================================================================
    # SPEED ROUND - Ronda rápida
    # ========================================================================

    def create_speed_round(self, profession: str, num_questions: int = 10) -> Dict:
        """Crea ronda rápida de preguntas simples"""
        questions = []

        for i in range(num_questions):
            # Generate quick true/false or simple questions
            question_type = random.choice(["true_false", "definition", "acronym"])

            if question_type == "true_false":
                q = self._generate_true_false(profession)
            elif question_type == "definition":
                q = self._generate_definition(profession)
            else:
                q = self._generate_acronym(profession)

            questions.append(q)

        return {"type": "speed_round", "questions": questions, "time_limit_per_question": 15, "profession": profession}

    def _generate_true_false(self, profession: str) -> Dict:
        """Genera pregunta verdadero/falso"""
        # Usar IA o base de datos
        return {"type": "true_false", "question": f"Pregunta sobre {profession}", "answer": True, "points": 10}

    def _generate_definition(self, profession: str) -> Dict:
        """Genera pregunta de definición"""
        return {
            "type": "definition",
            "term": "Término técnico",
            "options": ["Def 1", "Def 2", "Def 3"],
            "correct": 0,
            "points": 15,
        }

    def _generate_acronym(self, profession: str) -> Dict:
        """Genera pregunta de acrónimo"""
        return {
            "type": "acronym",
            "acronym": "API",
            "question": "¿Qué significa API?",
            "answer": "Application Programming Interface",
            "points": 10,
        }

    # ========================================================================
    # SKILL BUILDER - Constructor de habilidades
    # ========================================================================

    def create_skill_builder(self, skill: str, current_level: str) -> Dict:
        """Crea ejercicio de construcción de habilidad"""
        exercises = []

        # Generate progressive exercises
        for i in range(5):
            exercise = {
                "exercise_number": i + 1,
                "description": f"Ejercicio {i+1} para mejorar {skill}",
                "difficulty": current_level,
                "points": 20 * (i + 1),
                "hints": [f"Pista {j+1}" for j in range(3)],
            }
            exercises.append(exercise)

        return {"type": "skill_builder", "skill": skill, "exercises": exercises, "adaptive": True}

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    def start_game_session(self, user_id: str, game: Game) -> GameSession:
        """Inicia una sesión de juego"""
        session = GameSession(
            id=f"{game.id}_{user_id}_{datetime.now(timezone.utc).timestamp()}",
            user_id=user_id,
            game_id=game.id,
            game_type=game.type,
            started_at=datetime.now(timezone.utc),
        )

        self.active_sessions[session.id] = session
        logger.info(f"Game session started: {session.id}")
        return session

    def submit_answer(self, session_id: str, answer: Dict) -> Dict:
        """Procesa respuesta en un juego"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]
        session.answers.append(answer)

        # Evaluate answer
        is_correct = self._evaluate_answer(answer, session.game_type)
        points = answer.get("points", 0) if is_correct else 0

        session.score += points

        return {"correct": is_correct, "points": points, "total_score": session.score, "feedback": ""}

    def _evaluate_answer(self, answer: Dict, game_type: GameType) -> bool:
        """Evalúa si una respuesta es correcta"""
        # Implementación específica por tipo de juego
        return True  # Placeholder

    def complete_session(self, session_id: str) -> GameSession:
        """Completa una sesión de juego"""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        final_score = session.score

        session.complete(final_score)
        logger.info(f"Game session completed: {session_id} - Score: {final_score}")

        return session
