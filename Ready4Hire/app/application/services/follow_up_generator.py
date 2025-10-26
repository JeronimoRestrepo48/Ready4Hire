"""
Generador de Preguntas Follow-Up Dinámicas (v2.2 - Mejora #3)

Genera preguntas de seguimiento personalizadas basadas en:
- Respuestas anteriores del candidato
- Conceptos débiles identificados
- Fortalezas que vale la pena explorar más
- Inconsistencias detectadas
- Nivel de profundidad técnica demostrado

Mejora la efectividad de las entrevistas adaptándose al candidato en tiempo real.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from app.domain.entities.interview import Interview
from app.domain.entities.answer import Answer
from app.application.services.contextual_evaluator import InterviewHistoryAnalyzer
from app.infrastructure.llm.llm_service import OllamaLLMService

logger = logging.getLogger(__name__)


@dataclass
class FollowUpQuestion:
    """
    Pregunta de seguimiento generada dinámicamente.
    """

    question_text: str
    reason: str  # Por qué se genera esta pregunta
    focus_area: str  # "weakness" | "strength" | "inconsistency" | "exploration"
    concepts_to_probe: List[str]  # Conceptos a explorar
    difficulty: str  # "junior" | "mid" | "senior"
    category: str  # "technical" | "soft_skills"
    expected_depth: str  # "basic" | "intermediate" | "advanced"


class FollowUpQuestionGenerator:
    """
    Genera preguntas de seguimiento adaptativas basadas en el contexto de la entrevista.

    Estrategias:
    1. Profundizar en conceptos débiles (weakness probe)
    2. Explorar fortalezas técnicas (strength exploration)
    3. Aclarar inconsistencias (consistency check)
    4. Evaluar aplicación práctica (practical scenario)
    """

    def __init__(
        self,
        llm_service: Optional[OllamaLLMService] = None,
        history_analyzer: Optional[InterviewHistoryAnalyzer] = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.7,  # Más alta para creatividad en preguntas
    ):
        """
        Inicializa el generador de follow-ups.

        Args:
            llm_service: Servicio LLM para generar preguntas (opcional)
            history_analyzer: Analizador de historial (se crea si no se provee)
            model: Modelo LLM a usar
            temperature: Temperatura para generación (más alta = más creativo)
        """
        self.llm_service = llm_service or OllamaLLMService(model=model, temperature=temperature, max_tokens=512)
        self.history_analyzer = history_analyzer or InterviewHistoryAnalyzer()
        self.model = model
        self.temperature = temperature

        logger.info(f"FollowUpQuestionGenerator inicializado (model={model}, temp={temperature})")

    def generate_follow_up(
        self,
        interview: Interview,
        last_answer: Answer,
        last_question_text: str,
        category: str,
        difficulty: str,
        role: str,
    ) -> Optional[FollowUpQuestion]:
        """
        Genera pregunta de seguimiento basada en la última respuesta.

        Args:
            interview: Entrevista con historial
            last_answer: Última respuesta del candidato
            last_question_text: Texto de la última pregunta
            category: Categoría (technical, soft_skills)
            difficulty: Dificultad actual
            role: Rol del candidato

        Returns:
            FollowUpQuestion o None si no se puede/debe generar
        """
        # Analizar contexto
        context = self.history_analyzer.analyze(interview)

        if not context["has_sufficient_context"]:
            logger.debug("Contexto insuficiente para generar follow-up")
            return None

        # Decidir estrategia basada en el contexto
        strategy = self._select_strategy(context=context, last_answer=last_answer, category=category)

        logger.info(f"Estrategia seleccionada para follow-up: {strategy}")

        # Generar pregunta según estrategia
        if strategy == "weakness_probe":
            return self._generate_weakness_probe(
                context=context, last_answer=last_answer, category=category, difficulty=difficulty, role=role
            )
        elif strategy == "strength_exploration":
            return self._generate_strength_exploration(
                context=context, last_answer=last_answer, category=category, difficulty=difficulty, role=role
            )
        elif strategy == "consistency_check":
            return self._generate_consistency_check(
                context=context,
                last_answer=last_answer,
                last_question_text=last_question_text,
                category=category,
                difficulty=difficulty,
                role=role,
            )
        elif strategy == "practical_scenario":
            return self._generate_practical_scenario(
                context=context, last_answer=last_answer, category=category, difficulty=difficulty, role=role
            )
        else:
            return None

    def _select_strategy(self, context: Dict[str, Any], last_answer: Answer, category: str) -> str:
        """
        Selecciona estrategia de follow-up basada en el contexto.

        Prioridades:
        1. Si hay conceptos débiles recurrentes → weakness_probe
        2. Si score alto y conceptos fuertes → strength_exploration
        3. Si inconsistencia detectada → consistency_check
        4. Por defecto → practical_scenario
        """
        last_score = last_answer.score if isinstance(last_answer.score, float) else last_answer.score

        # Si hay conceptos débiles y score bajo/medio
        if context["weak_concepts"] and last_score < 7.0:
            return "weakness_probe"

        # Si score alto y hay conceptos fuertes
        if last_score >= 8.0 and context["strong_concepts"]:
            return "strength_exploration"

        # Si consistencia baja
        if context["consistency_score"] < 0.6:
            return "consistency_check"

        # Por defecto: escenario práctico
        return "practical_scenario"

    def _generate_weakness_probe(
        self, context: Dict[str, Any], last_answer: Answer, category: str, difficulty: str, role: str
    ) -> FollowUpQuestion:
        """
        Genera pregunta para profundizar en conceptos débiles.
        """
        weak_concepts = context["weak_concepts"][:2]  # Top 2 conceptos débiles

        if not weak_concepts:
            # Fallback: usar missing_concepts de la última respuesta
            weak_concepts = last_answer.evaluation_details.get("missing_concepts", ["concepto básico"])[:2]

        # Generar pregunta con LLM
        prompt = self._build_weakness_probe_prompt(
            weak_concepts=weak_concepts, role=role, category=category, difficulty=difficulty
        )

        question_text = self._generate_with_llm(prompt)

        return FollowUpQuestion(
            question_text=question_text,
            reason=f"Profundizar en conceptos débiles: {', '.join(weak_concepts)}",
            focus_area="weakness",
            concepts_to_probe=weak_concepts,
            difficulty=difficulty,
            category=category,
            expected_depth="intermediate",
        )

    def _generate_strength_exploration(
        self, context: Dict[str, Any], last_answer: Answer, category: str, difficulty: str, role: str
    ) -> FollowUpQuestion:
        """
        Genera pregunta para explorar fortalezas en mayor profundidad.
        """
        strong_concepts = context["strong_concepts"][:2]  # Top 2 conceptos fuertes

        if not strong_concepts:
            # Fallback: usar concepts_covered de la última respuesta
            strong_concepts = last_answer.evaluation_details.get("concepts_covered", ["concepto avanzado"])[:2]

        # Generar pregunta con LLM (nivel más avanzado)
        prompt = self._build_strength_exploration_prompt(
            strong_concepts=strong_concepts, role=role, category=category, difficulty=difficulty
        )

        question_text = self._generate_with_llm(prompt)

        # Aumentar dificultad para explorar fortaleza
        elevated_difficulty = self._elevate_difficulty(difficulty)

        return FollowUpQuestion(
            question_text=question_text,
            reason=f"Explorar dominio avanzado de: {', '.join(strong_concepts)}",
            focus_area="strength",
            concepts_to_probe=strong_concepts,
            difficulty=elevated_difficulty,
            category=category,
            expected_depth="advanced",
        )

    def _generate_consistency_check(
        self,
        context: Dict[str, Any],
        last_answer: Answer,
        last_question_text: str,
        category: str,
        difficulty: str,
        role: str,
    ) -> FollowUpQuestion:
        """
        Genera pregunta para verificar consistencia en respuestas.
        """
        # Identificar concepto a verificar
        concepts_to_check = last_answer.evaluation_details.get("concepts_covered", [])[:1]

        if not concepts_to_check:
            concepts_to_check = ["el concepto previamente mencionado"]

        # Generar pregunta desde ángulo diferente
        prompt = self._build_consistency_check_prompt(
            concepts=concepts_to_check,
            previous_question=last_question_text,
            role=role,
            category=category,
            difficulty=difficulty,
        )

        question_text = self._generate_with_llm(prompt)

        return FollowUpQuestion(
            question_text=question_text,
            reason=f"Verificar consistencia en: {', '.join(concepts_to_check)}",
            focus_area="inconsistency",
            concepts_to_probe=concepts_to_check,
            difficulty=difficulty,
            category=category,
            expected_depth="intermediate",
        )

    def _generate_practical_scenario(
        self, context: Dict[str, Any], last_answer: Answer, category: str, difficulty: str, role: str
    ) -> FollowUpQuestion:
        """
        Genera pregunta de escenario práctico/aplicación real.
        """
        # Usar conceptos cubiertos de la última respuesta
        concepts = last_answer.evaluation_details.get("concepts_covered", ["programación"])[:2]

        # Generar escenario práctico
        prompt = self._build_practical_scenario_prompt(
            concepts=concepts, role=role, category=category, difficulty=difficulty
        )

        question_text = self._generate_with_llm(prompt)

        return FollowUpQuestion(
            question_text=question_text,
            reason=f"Aplicación práctica de: {', '.join(concepts)}",
            focus_area="exploration",
            concepts_to_probe=concepts,
            difficulty=difficulty,
            category=category,
            expected_depth="intermediate",
        )

    # =========================================================================
    # PROMPT BUILDERS
    # =========================================================================

    def _build_weakness_probe_prompt(self, weak_concepts: List[str], role: str, category: str, difficulty: str) -> str:
        """Construye prompt para generar pregunta sobre conceptos débiles."""
        return f"""Genera UNA pregunta de entrevista técnica para {role} que profundice en estos conceptos débiles: {', '.join(weak_concepts)}.

Contexto:
- Categoría: {category}
- Nivel: {difficulty}
- El candidato mostró debilidad en estos conceptos en respuestas anteriores
- La pregunta debe ayudar a evaluar si realmente comprende estos conceptos

Requisitos:
- Pregunta clara y específica
- Enfocada en uno o dos de los conceptos débiles
- Apropiada para nivel {difficulty}
- No mencionar que es una "pregunta de seguimiento"

Responde SOLO con el texto de la pregunta, sin explicaciones adicionales."""

    def _build_strength_exploration_prompt(
        self, strong_concepts: List[str], role: str, category: str, difficulty: str
    ) -> str:
        """Construye prompt para explorar fortalezas en profundidad."""
        return f"""Genera UNA pregunta avanzada de entrevista técnica para {role} que explore en mayor profundidad: {', '.join(strong_concepts)}.

Contexto:
- Categoría: {category}
- Nivel: {difficulty} (pero haz la pregunta más desafiante)
- El candidato demostró buen dominio de estos conceptos
- Queremos evaluar qué tan profundo es su conocimiento

Requisitos:
- Pregunta desafiante pero justa
- Puede incluir casos edge, optimizaciones, o trade-offs
- Apropiada para evaluar expertise real
- No mencionar que es una "pregunta de seguimiento"

Responde SOLO con el texto de la pregunta, sin explicaciones adicionales."""

    def _build_consistency_check_prompt(
        self, concepts: List[str], previous_question: str, role: str, category: str, difficulty: str
    ) -> str:
        """Construye prompt para verificar consistencia."""
        return f"""Genera UNA pregunta de entrevista técnica para {role} que verifique la comprensión de: {', '.join(concepts)}.

Contexto:
- Categoría: {category}
- Nivel: {difficulty}
- Pregunta anterior: "{previous_question}"
- Queremos verificar consistencia abordando el mismo concepto desde ángulo diferente

Requisitos:
- Pregunta relacionada pero NO idéntica a la anterior
- Aborda el mismo concepto desde perspectiva diferente
- Apropiada para nivel {difficulty}
- No mencionar la pregunta anterior

Responde SOLO con el texto de la pregunta, sin explicaciones adicionales."""

    def _build_practical_scenario_prompt(self, concepts: List[str], role: str, category: str, difficulty: str) -> str:
        """Construye prompt para escenario práctico."""
        return f"""Genera UNA pregunta de escenario práctico para {role} que aplique: {', '.join(concepts)}.

Contexto:
- Categoría: {category}
- Nivel: {difficulty}
- Queremos evaluar cómo el candidato aplica estos conceptos en situaciones reales

Requisitos:
- Describe un escenario laboral realista
- Pregunta cómo resolverían el problema usando los conceptos
- Apropiada para nivel {difficulty}
- No muy larga (2-3 oraciones máximo)

Responde SOLO con el texto de la pregunta/escenario, sin explicaciones adicionales."""

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _generate_with_llm(self, prompt: str) -> str:
        """
        Genera pregunta usando LLM.

        Args:
            prompt: Prompt construido

        Returns:
            Texto de la pregunta generada
        """
        try:
            response = self.llm_service.generate(prompt=prompt, temperature=self.temperature, max_tokens=256)

            # Limpiar respuesta
            question = response.strip()

            # Remover posibles prefijos como "Pregunta: "
            if question.lower().startswith("pregunta:"):
                question = question[9:].strip()

            return question

        except Exception as e:
            logger.error(f"Error generando follow-up con LLM: {e}")
            # Fallback a pregunta genérica
            return "¿Puedes explicar este concepto con más detalle y dar un ejemplo práctico?"

    def _elevate_difficulty(self, current_difficulty: str) -> str:
        """Aumenta nivel de dificultad en un grado."""
        difficulty_map = {"junior": "mid", "mid": "senior", "senior": "senior"}  # Ya es máximo
        return difficulty_map.get(current_difficulty, current_difficulty)

    def generate_multiple_follow_ups(
        self,
        interview: Interview,
        last_answer: Answer,
        last_question_text: str,
        category: str,
        difficulty: str,
        role: str,
        count: int = 3,
    ) -> List[FollowUpQuestion]:
        """
        Genera múltiples opciones de preguntas follow-up.

        Útil para dar opciones al entrevistador o para sistema adaptativo.

        Args:
            interview: Entrevista con historial
            last_answer: Última respuesta
            last_question_text: Texto de última pregunta
            category: Categoría
            difficulty: Dificultad
            role: Rol
            count: Número de opciones a generar (default: 3)

        Returns:
            Lista de FollowUpQuestion (puede ser menor a count si no hay suficiente contexto)
        """
        follow_ups = []

        # Intentar generar usando diferentes estrategias
        strategies = ["weakness_probe", "strength_exploration", "practical_scenario"]

        for strategy in strategies[:count]:
            # Temporalmente forzar estrategia
            original_select = self._select_strategy
            self._select_strategy = lambda *args, **kwargs: strategy

            try:
                follow_up = self.generate_follow_up(
                    interview=interview,
                    last_answer=last_answer,
                    last_question_text=last_question_text,
                    category=category,
                    difficulty=difficulty,
                    role=role,
                )

                if follow_up:
                    follow_ups.append(follow_up)
            except Exception as e:
                logger.warning(f"Error generando follow-up con estrategia {strategy}: {e}")
            finally:
                # Restaurar método original
                self._select_strategy = original_select

        return follow_ups
