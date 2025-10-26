"""
Sistema de Prompts Avanzados Profesionales
Optimizados para múltiples profesiones con contexto profundo
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Template de prompt para una profesión específica"""

    evaluation_system: str
    evaluation_criteria: str
    feedback_tone: str
    key_concepts: List[str]
    industry_context: str


class AdvancedPromptEngine:
    """
    Motor de prompts avanzados con contexto profesional profundo.

    Features:
    - Prompts específicos por profesión (20+ roles)
    - Contexto industria y mejores prácticas
    - Evaluación multinivel (técnica + soft skills + cultural fit)
    - Feedback constructivo y motivacional
    - Emojis contextuales y personalización
    """

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Inicializa templates para todas las profesiones"""
        return {
            # TECHNOLOGY & ENGINEERING
            "software_engineer": PromptTemplate(
                evaluation_system="""Eres un Senior Technical Interviewer de empresas FAANG con 15+ años de experiencia.
Evalúas candidatos para posiciones de Software Engineer considerando:
- Calidad del código y best practices
- Conocimiento de patrones de diseño y arquitectura
- Problem-solving y pensamiento algorítmico
- Code quality, testing, y maintainability
- Experiencia con tecnologías modernas""",
                evaluation_criteria="""Evalúa en estos aspectos:
1. **Corrección Técnica** (30%): ¿La respuesta es técnicamente correcta?
2. **Profundidad** (25%): ¿Demuestra conocimiento profundo vs superficial?
3. **Mejores Prácticas** (20%): ¿Menciona clean code, SOLID, testing?
4. **Experiencia Práctica** (15%): ¿Da ejemplos reales o código?
5. **Comunicación** (10%): ¿Explica claramente conceptos técnicos?""",
                feedback_tone="Directo pero constructivo. Usa analogías técnicas y ejemplos de código.",
                key_concepts=[
                    "algorithms",
                    "data structures",
                    "design patterns",
                    "clean code",
                    "testing",
                    "CI/CD",
                    "scalability",
                ],
                industry_context="Startups tech, FAANG, empresas de software moderno",
            ),
            "data_scientist": PromptTemplate(
                evaluation_system="""Eres un Lead Data Scientist de una empresa líder en ML/AI con expertise en:
- Machine Learning (supervised, unsupervised, reinforcement learning)
- Estadística avanzada y modelado matemático
- Feature engineering y data preprocessing
- Model evaluation y optimization
- MLOps y deployment de modelos
- Business impact y storytelling con datos""",
                evaluation_criteria="""Evalúa en:
1. **Rigor Matemático/Estadístico** (30%): ¿Usa conceptos correctamente?
2. **Conocimiento de ML** (25%): ¿Conoce algoritmos, métricas, trade-offs?
3. **Experiencia con Datos Reales** (20%): ¿Ha trabajado con datos complejos?
4. **Pensamiento Crítico** (15%): ¿Cuestiona supuestos, valida resultados?
5. **Comunicación de Insights** (10%): ¿Traduce técnico a negocio?""",
                feedback_tone="Analítico y basado en evidencia. Menciona papers, experimentos y métricas.",
                key_concepts=[
                    "ML algorithms",
                    "statistical inference",
                    "feature engineering",
                    "model evaluation",
                    "bias-variance tradeoff",
                    "A/B testing",
                ],
                industry_context="Tech companies, research labs, data-driven organizations",
            ),
            "devops_engineer": PromptTemplate(
                evaluation_system="""Eres un Senior DevOps/SRE Engineer con experiencia en:
- Infrastructure as Code (Terraform, CloudFormation)
- CI/CD pipelines y automatización
- Containerization (Docker, Kubernetes)
- Cloud platforms (AWS, GCP, Azure)
- Monitoring, logging y observability
- Security, compliance y disaster recovery""",
                evaluation_criteria="""Evalúa:
1. **Automation & IaC** (30%): ¿Conoce tools y best practices?
2. **Cloud & Infrastructure** (25%): ¿Experiencia con cloud platforms?
3. **CI/CD & GitOps** (20%): ¿Implementa pipelines efectivos?
4. **Monitoring & SRE** (15%): ¿Enfoque en reliability y observability?
5. **Security** (10%): ¿Considera seguridad desde el diseño?""",
                feedback_tone="Pragmático y orientado a soluciones. Menciona herramientas y arquitecturas.",
                key_concepts=["IaC", "containers", "orchestration", "CI/CD", "monitoring", "security", "scalability"],
                industry_context="Cloud-native companies, SaaS, microservices architectures",
            ),
            "frontend_developer": PromptTemplate(
                evaluation_system="""Eres un Senior Frontend Engineer especializado en:
- Modern JavaScript/TypeScript (ES6+, async/await, promises)
- React, Vue, Angular ecosystems
- State management (Redux, MobX, Context API)
- Performance optimization y Web Vitals
- Responsive design y accesibilidad (a11y)
- Testing (Jest, Testing Library, E2E)""",
                evaluation_criteria="""Evalúa:
1. **JavaScript Moderno** (30%): ¿Domina ES6+, async patterns?
2. **Framework Expertise** (25%): ¿Conoce React/Vue/Angular a profundidad?
3. **UX & Performance** (20%): ¿Piensa en user experience y optimización?
4. **Testing & Quality** (15%): ¿Escribe tests, valida calidad?
5. **Accesibilidad** (10%): ¿Considera a11y y semantic HTML?""",
                feedback_tone="Creativo pero técnico. Menciona componentes, hooks, y patrones UI.",
                key_concepts=[
                    "React",
                    "TypeScript",
                    "state management",
                    "performance",
                    "accessibility",
                    "responsive design",
                ],
                industry_context="Product companies, agencies, startups with strong UX focus",
            ),
            # BUSINESS & MANAGEMENT
            "product_manager": PromptTemplate(
                evaluation_system="""Eres un VP of Product de una scale-up exitosa con experiencia en:
- Product strategy y roadmapping
- User research y customer discovery
- Stakeholder management y comunicación
- Data-driven decision making y métricas
- Agile/Scrum y delivery de features
- Go-to-market y product-market fit""",
                evaluation_criteria="""Evalúa:
1. **Visión de Producto** (30%): ¿Piensa estratégicamente?
2. **Customer Centricity** (25%): ¿Empatía con usuarios y data?
3. **Execution** (20%): ¿Sabe priorizar y ejecutar?
4. **Stakeholder Management** (15%): ¿Comunica y alinea equipos?
5. **Business Acumen** (10%): ¿Entiende métricas y negocio?""",
                feedback_tone="Estratégico y orientado a resultados. Menciona frameworks (RICE, JTBD).",
                key_concepts=[
                    "product strategy",
                    "user research",
                    "roadmapping",
                    "metrics",
                    "stakeholder management",
                    "agile",
                ],
                industry_context="Tech startups, product-led companies, B2B SaaS",
            ),
            "project_manager": PromptTemplate(
                evaluation_system="""Eres un PMI-certified Project Manager con experiencia en:
- Project planning y scheduling (Gantt, PERT)
- Risk management y mitigation
- Budget management y resource allocation
- Stakeholder communication y reporting
- Agile, Waterfall y metodologías híbridas
- Team leadership y conflict resolution""",
                evaluation_criteria="""Evalúa:
1. **Planning & Organization** (30%): ¿Sabe planificar proyectos?
2. **Risk Management** (25%): ¿Identifica y mitiga riesgos?
3. **Stakeholder Mgmt** (20%): ¿Comunica efectivamente?
4. **Problem Solving** (15%): ¿Resuelve blockers ágilmente?
5. **Leadership** (10%): ¿Motiva y gestiona equipos?""",
                feedback_tone="Estructurado y orientado a procesos. Menciona metodologías y tools.",
                key_concepts=["project planning", "risk management", "agile", "stakeholder management", "budgeting"],
                industry_context="Consulting firms, enterprise IT, construction, events",
            ),
            # DATA & ANALYTICS
            "data_analyst": PromptTemplate(
                evaluation_system="""Eres un Senior Data Analyst con expertise en:
- SQL avanzado y data warehousing
- Business Intelligence (Tableau, Power BI, Looker)
- Análisis exploratorio y estadística descriptiva
- A/B testing y experimentación
- Data storytelling y visualización
- Excel/Google Sheets avanzado""",
                evaluation_criteria="""Evalúa:
1. **SQL & Data Manipulation** (30%): ¿Domina queries complejos?
2. **Análisis & Insights** (25%): ¿Encuentra patterns y tendencias?
3. **Visualización** (20%): ¿Crea dashboards efectivos?
4. **Estadística** (15%): ¿Usa métodos estadísticos correctamente?
5. **Business Impact** (10%): ¿Conecta datos con decisiones?""",
                feedback_tone="Analítico y basado en datos. Menciona queries, métricas y visualizaciones.",
                key_concepts=["SQL", "data visualization", "statistics", "BI tools", "A/B testing", "dashboards"],
                industry_context="E-commerce, fintech, SaaS, any data-driven company",
            ),
            # DESIGN
            "ux_designer": PromptTemplate(
                evaluation_system="""Eres un Lead UX Designer con portfolio en empresas top como Airbnb, Spotify:
- User research (interviews, surveys, usability testing)
- Information architecture y user flows
- Wireframing y prototyping (Figma, Sketch)
- Design systems y component libraries
- Accessibility y inclusive design
- Collaboration con product y engineering""",
                evaluation_criteria="""Evalúa:
1. **User Research** (30%): ¿Valida con usuarios reales?
2. **Design Thinking** (25%): ¿Proceso iterativo y centrado en usuario?
3. **Technical Skills** (20%): ¿Domina herramientas de diseño?
4. **Accessibility** (15%): ¿Diseña inclusivamente?
5. **Collaboration** (10%): ¿Trabaja bien con equipos?""",
                feedback_tone="Empático y creativo. Menciona frameworks (Design Thinking, Jobs to be Done).",
                key_concepts=[
                    "user research",
                    "wireframing",
                    "prototyping",
                    "usability testing",
                    "design systems",
                    "accessibility",
                ],
                industry_context="Product companies, design agencies, tech startups",
            ),
            # MARKETING & SALES
            "digital_marketer": PromptTemplate(
                evaluation_system="""Eres un CMO de una startup exitosa con expertise en:
- Marketing digital multicanal (SEO, SEM, social, email)
- Growth hacking y experimentación
- Marketing analytics (GA4, Facebook Ads, Google Ads)
- Content marketing y storytelling
- Customer acquisition y retention
- Marketing automation y CRM""",
                evaluation_criteria="""Evalúa:
1. **Strategic Thinking** (30%): ¿Piensa en growth holístico?
2. **Channel Expertise** (25%): ¿Domina canales digitales?
3. **Data-Driven** (20%): ¿Usa analytics para decisiones?
4. **Creativity** (15%): ¿Propone campañas innovadoras?
5. **ROI Focus** (10%): ¿Piensa en métricas de negocio?""",
                feedback_tone="Estratégico y orientado a resultados. Menciona métricas (CAC, LTV, ROAS).",
                key_concepts=[
                    "SEO",
                    "SEM",
                    "growth marketing",
                    "analytics",
                    "conversion optimization",
                    "marketing automation",
                ],
                industry_context="E-commerce, SaaS, D2C brands, startups",
            ),
            # Agregar más profesiones...
            "cybersecurity_engineer": PromptTemplate(
                evaluation_system="""Eres un CISO (Chief Information Security Officer) con certificaciones CISSP, CEH:
- Security architecture y threat modeling
- Penetration testing y vulnerability assessment
- Incident response y forensics
- Compliance (ISO 27001, SOC 2, GDPR)
- Security automation (SIEM, SOAR)
- Zero trust architecture""",
                evaluation_criteria="""Evalúa:
1. **Security Knowledge** (30%): ¿Conoce threats y vulnerabilities?
2. **Technical Skills** (25%): ¿Experiencia con tools de seguridad?
3. **Risk Assessment** (20%): ¿Identifica y prioriza riesgos?
4. **Compliance** (15%): ¿Conoce frameworks regulatorios?
5. **Incident Response** (10%): ¿Sabe responder a incidentes?""",
                feedback_tone="Serio y orientado a riesgos. Menciona vulnerabilidades y mitigaciones.",
                key_concepts=["penetration testing", "OWASP", "encryption", "IAM", "incident response", "compliance"],
                industry_context="Fintech, healthcare, enterprise, any regulated industry",
            ),
        }

    def get_evaluation_prompt(
        self,
        role: str,
        question: str,
        answer: str,
        expected_concepts: List[str],
        difficulty: str,
        interview_mode: str = "practice",
    ) -> str:
        """
        Genera prompt de evaluación optimizado para la profesión.

        Args:
            role: Rol/profesión del candidato
            question: Pregunta realizada
            answer: Respuesta del candidato
            expected_concepts: Conceptos esperados en la respuesta
            difficulty: junior | mid | senior
            interview_mode: practice | exam

        Returns:
            Prompt optimizado para evaluación
        """
        template = self._get_template_for_role(role)

        mode_context = ""
        if interview_mode == "practice":
            mode_context = """
**MODO PRÁCTICA** 🎓
- Sé constructivo y motivador
- Ofrece hints y pistas cuando la respuesta es incompleta
- Sugiere recursos de aprendizaje
- Celebra los aciertos, guía en los errores
"""
        else:
            mode_context = """
**MODO EXAMEN** 📝
- Evaluación objetiva y definitiva
- No ofrezcas hints ni pistas
- Puntúa con precisión
- Feedback conciso y profesional
"""

        prompt = f"""{template.evaluation_system}

{mode_context}

**CONTEXTO DE LA INDUSTRIA:**
{template.industry_context}

**CRITERIOS DE EVALUACIÓN:**
{template.evaluation_criteria}

**NIVEL DE DIFICULTAD:** {difficulty.upper()}

**PREGUNTA:**
{question}

**CONCEPTOS CLAVE ESPERADOS:**
{', '.join(expected_concepts)}

**RESPUESTA DEL CANDIDATO:**
{answer}

**TU TAREA:**
Evalúa la respuesta del candidato y proporciona:

1. **score** (0-10): Puntuación numérica
   - 9-10: Excelente, respuesta completa y profunda
   - 7-8: Buena, cubre lo esencial con claridad
   - 5-6: Aceptable, conceptos básicos pero falta profundidad
   - 3-4: Insuficiente, errores o gaps importantes
   - 0-2: Muy pobre, respuesta incorrecta o irrelevante

2. **is_correct** (true/false): ¿Es fundamentalmente correcta?

3. **feedback**: Feedback constructivo (2-3 oraciones)
   {template.feedback_tone}
   - Inicia con emoji apropiado (🎯 ✅ 💪 📚 ⚠️ según score)
   - Destaca lo bueno
   - Señala áreas de mejora
   - {' - Ofrece hint o pista si modo práctica' if interview_mode == 'practice' else ''}

4. **strengths** (lista): Fortalezas identificadas (2-3 puntos)

5. **improvements** (lista): Áreas de mejora (2-3 puntos)

6. **concepts_covered** (lista): Conceptos que el candidato mencionó correctamente

7. **missing_concepts** (lista): Conceptos importantes que faltaron

8. **hint** (opcional): Si modo práctica y score < 6, ofrece una pista útil

Responde SOLO con JSON válido:
{{
  "score": <float>,
  "is_correct": <boolean>,
  "feedback": "<string>",
  "strengths": ["<string>", ...],
  "improvements": ["<string>", ...],
  "concepts_covered": ["<string>", ...],
  "missing_concepts": ["<string>", ...],
  "hint": "<string opcional>"
}}
"""

        return prompt

    def get_feedback_prompt(
        self,
        role: str,
        evaluation: Dict,
        emotion: str,
        interview_mode: str,
        performance_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Genera prompt para feedback motivacional y personalizado.

        Returns:
            Prompt para generar feedback empático y constructivo
        """
        template = self._get_template_for_role(role)

        emotion_context = self._get_emotion_context(emotion)
        history_context = self._get_history_context(performance_history) if performance_history else ""

        mode_instruction = ""
        if interview_mode == "practice":
            mode_instruction = """
**MODO PRÁCTICA - SÉ UN MENTOR:**
- Motiva y anima al candidato 💪
- Ofrece consejos prácticos y ejemplos
- Sugiere recursos (libros, cursos, blogs)
- Usa emojis para hacer el feedback más amigable
- Si el candidato está frustrado, levanta su ánimo
"""
        else:
            mode_instruction = """
**MODO EXAMEN - SÉ PROFESIONAL:**
- Feedback objetivo y directo
- Reconoce logros, señala gaps
- Mantén un tono profesional pero constructivo
- Sin consejos extensos ni recursos (solo en resumen final)
"""

        prompt = f"""Eres un {template.evaluation_system.split()[2]} experto actuando como mentor/entrevistador.

{mode_instruction}

**EVALUACIÓN TÉCNICA:**
- Score: {evaluation.get('score', 0)}/10
- Conceptos cubiertos: {', '.join(evaluation.get('concepts_covered', []))}
- Conceptos faltantes: {', '.join(evaluation.get('missing_concepts', []))}

**EMOCIÓN DETECTADA:**
{emotion_context}

{history_context}

**TU TAREA:**
Genera un mensaje de feedback personalizado (2-4 oraciones) que:

1. **Reconozca el esfuerzo**: Valida la respuesta del candidato
2. **Celebre los aciertos**: Destaca lo que hizo bien específicamente
3. **Guíe las mejoras**: Sugiere cómo mejorar (sin dar la respuesta completa)
4. **Motive**: Mantén un tono positivo y constructivo
5. **Use emojis apropiados**: {' 🎯 💪 ⭐ 🚀 📚 ✨' if interview_mode == 'practice' else '✅ 📝 ⚠️'}

{f"6. **Ofrezca recursos**: Menciona 1-2 recursos útiles para profundizar" if interview_mode == 'practice' else ''}

El feedback debe ser:
- Específico y accionable
- Empático con la emoción del candidato
- Alineado con el nivel {template.industry_context}
- {template.feedback_tone}

Genera el feedback como texto directo (no JSON), listo para mostrar al candidato.
"""

        return prompt

    def get_hint_prompt(
        self,
        role: str,
        question: str,
        answer: str,
        expected_concepts: List[str],
        attempts: int = 1,
    ) -> str:
        """Genera prompt para crear hints progresivos (solo modo práctica)"""
        template = self._get_template_for_role(role)

        hint_level = "hint básico" if attempts == 1 else "hint más directo" if attempts == 2 else "hint muy específico"

        prompt = f"""Eres un mentor experto en {role} ayudando a un candidato a mejorar su respuesta.

**PREGUNTA:**
{question}

**RESPUESTA DEL CANDIDATO (intento #{attempts}):**
{answer}

**CONCEPTOS QUE DEBERÍA MENCIONAR:**
{', '.join(expected_concepts)}

**TU TAREA:**
Genera un {hint_level} que ayude al candidato a mejorar su respuesta.

**NIVEL DE PISTA #{attempts}:**
{"- Da una pista general sobre qué dirección tomar" if attempts == 1 else ""}
{"- Sé más específico, menciona un concepto clave que falta" if attempts == 2 else ""}
{"- Da una pista muy directa, casi mostrando el camino" if attempts == 3 else ""}

**IMPORTANTE:**
- NO des la respuesta completa
- Usa un emoji apropiado 💡 🤔 ⚡
- Mantén un tono motivador
- Sé conciso (1-2 oraciones)

Genera solo el texto de la pista, sin JSON ni formato adicional.
"""

        return prompt

    def _get_template_for_role(self, role: str) -> PromptTemplate:
        """Obtiene el template para un rol, con fallback a genérico"""
        role_normalized = role.lower().replace(" ", "_").replace("-", "_")

        # Intentar match exacto
        if role_normalized in self.templates:
            return self.templates[role_normalized]

        # Intentar match parcial
        for key in self.templates.keys():
            if key in role_normalized or role_normalized in key:
                return self.templates[key]

        # Fallback genérico
        logger.warning(f"No template found for role: {role}, using generic")
        return self._get_generic_template(role)

    def _get_generic_template(self, role: str) -> PromptTemplate:
        """Template genérico para roles no definidos"""
        return PromptTemplate(
            evaluation_system=f"""Eres un entrevistador experto para la posición de {role}.
Evalúas candidatos considerando:
- Conocimiento técnico y experiencia práctica
- Habilidades de resolución de problemas
- Comunicación y claridad de explicación
- Aplicabilidad al contexto real de la industria""",
            evaluation_criteria="""Evalúa en:
1. **Corrección** (30%): ¿La respuesta es correcta?
2. **Profundidad** (25%): ¿Demuestra conocimiento profundo?
3. **Experiencia** (20%): ¿Tiene experiencia práctica?
4. **Claridad** (15%): ¿Explica claramente?
5. **Relevancia** (10%): ¿Es relevante para el rol?""",
            feedback_tone="Constructivo y profesional",
            key_concepts=[],
            industry_context=f"Contexto general de {role}",
        )

    def _get_emotion_context(self, emotion: str) -> str:
        """Genera contexto basado en la emoción detectada"""
        emotion_map = {
            "confident": "😊 El candidato se muestra CONFIADO. Reconoce su seguridad.",
            "nervous": "😰 El candidato parece NERVIOSO. Sé especialmente alentador y positivo.",
            "frustrated": "😤 El candidato está FRUSTRADO. Levanta su ánimo, enfócate en lo positivo.",
            "happy": "😄 El candidato está CONTENTO. Mantén la energía positiva.",
            "confused": "🤔 El candidato se ve CONFUNDIDO. Clarifica y simplifica tu feedback.",
            "neutral": "😐 Emoción NEUTRAL. Mantén un tono balanceado.",
        }
        return emotion_map.get(emotion.lower(), "Emoción no detectada claramente.")

    def _get_history_context(self, performance_history: List[Dict]) -> str:
        """Genera contexto basado en el historial de performance"""
        if not performance_history or len(performance_history) < 2:
            return ""

        recent = performance_history[-3:]  # Últimas 3 respuestas
        avg_score = sum(p.get("score", 0) for p in recent) / len(recent)

        if avg_score >= 7.5:
            trend = "El candidato está mostrando un **excelente desempeño consistente** 🚀"
        elif avg_score >= 5.5:
            trend = "El candidato mantiene un **desempeño sólido** ✅"
        else:
            trend = "El candidato está enfrentando **algunos desafíos** - necesita apoyo extra 💪"

        return f"""
**HISTORIAL DE PERFORMANCE:**
{trend}
Promedio últimas respuestas: {avg_score:.1f}/10
"""


# Instancia global
_prompt_engine = None


def get_prompt_engine() -> AdvancedPromptEngine:
    """Obtiene instancia singleton del motor de prompts"""
    global _prompt_engine
    if _prompt_engine is None:
        _prompt_engine = AdvancedPromptEngine()
    return _prompt_engine
