"""
Context Questions Configuration
Preguntas iniciales para conocer el contexto del candidato antes de la entrevista formal.

IMPORTANTE: Estas preguntas están diseñadas para generar embeddings que alimentan
el sistema de clustering y selección inteligente de preguntas.

Los factores clave que capturan son:
1. Experiencia y nivel técnico
2. Tecnologías y herramientas conocidas
3. Áreas de especialización y expertise
4. Estilo de trabajo y preferencias
5. Objetivos de aprendizaje y desarrollo
"""

from typing import Dict, List


# Preguntas de contexto UNIVERSALES para soft skills (aplicables a todas las profesiones)
SOFT_SKILLS_CONTEXT_QUESTIONS: List[str] = [
    "¿Cómo describirías tu estilo de trabajo en equipo? ¿Prefieres liderar, colaborar o trabajar de forma independiente? Da ejemplos concretos.",
    "Describe una situación donde tuviste que comunicar ideas complejas o resolver un conflicto en el equipo. ¿Qué estrategias utilizaste?",
    "Cuéntame sobre una ocasión donde tuviste que adaptarte rápidamente a un cambio significativo (proyecto, tecnología, metodología). ¿Cómo lo manejaste?",
    "¿Cómo tomas decisiones importantes cuando trabajas bajo presión y con plazos ajustados? Proporciona un ejemplo específico.",
    "¿Qué te motiva profesionalmente y qué valores son importantes para ti en un ambiente de trabajo? ¿Cómo impactan en tu desempeño diario?",
]

# Preguntas de contexto específicas por profesión
PROFESSION_CONTEXT_QUESTIONS: Dict[str, List[str]] = {
    "software_engineer": [
        "¿Cuántos años de experiencia tienes desarrollando software y en qué tipos de proyectos has trabajado (web, móvil, sistemas, etc.)?",
        "¿Qué lenguajes de programación y frameworks dominas? Describe tu nivel en cada uno y proyectos relevantes donde los usaste.",
        "¿Cómo abordas el diseño de arquitectura de software? Describe patrones y principios que aplicas regularmente.",
        "Cuéntame sobre un bug crítico que resolviste: ¿Cómo lo detectaste, debuggeaste y solucionaste?",
        "¿Qué prácticas de desarrollo sigues (TDD, code reviews, CI/CD)? ¿Cómo impactan en la calidad del código?",
    ],
    "data_scientist": [
        "¿Cuál es tu experiencia en análisis de datos y machine learning? Menciona proyectos y resultados obtenidos.",
        "¿Qué técnicas y algoritmos de ML dominas? Describe casos de uso donde los aplicaste exitosamente.",
        "¿Cómo manejas el preprocesamiento y limpieza de datos? Describe desafíos que enfrentaste con datos reales.",
        "Explica un modelo de ML que desarrollaste: problema, datos, enfoque, métricas y resultados.",
        "¿Qué herramientas y frameworks usas (Pandas, TensorFlow, PyTorch, etc.) y cómo las integras en tu workflow?",
    ],
    "devops_engineer": [
        "¿Cuántos años de experiencia tienes en DevOps/SRE y qué infraestructuras has gestionado?",
        "¿Qué plataformas cloud dominas (AWS, Azure, GCP)? Describe arquitecturas que has implementado.",
        "¿Cómo diseñas e implementas pipelines de CI/CD? Describe herramientas y mejores prácticas que usas.",
        "Cuéntame sobre un incidente de producción crítico que resolviste: causa, solución y prevención.",
        "¿Qué estrategias usas para monitoreo, logging y observabilidad? Describe herramientas y métricas clave.",
    ],
    "frontend_developer": [
        "¿Cuántos años de experiencia tienes en desarrollo frontend y qué tipos de aplicaciones has construido?",
        "¿Qué frameworks y librerías dominas (React, Angular, Vue)? Describe proyectos complejos donde los usaste.",
        "¿Cómo garantizas rendimiento y accesibilidad en tus aplicaciones? Menciona técnicas y herramientas.",
        "Describe tu enfoque para responsive design y cross-browser compatibility. ¿Qué desafíos has enfrentado?",
        "¿Cómo manejas el estado en aplicaciones complejas? Describe patrones y librerías que prefieres.",
    ],
    "backend_developer": [
        "¿Cuántos años de experiencia tienes desarrollando backend y qué arquitecturas has implementado?",
        "¿Qué lenguajes y frameworks backend dominas? Describe APIs y servicios que has construido.",
        "¿Cómo diseñas bases de datos y optimizas queries? Menciona SQL vs NoSQL y cuándo usas cada uno.",
        "Describe tu experiencia con microservicios: comunicación, manejo de errores, escalabilidad.",
        "¿Cómo garantizas seguridad en APIs y servicios? Menciona authentication, authorization y best practices.",
    ],
    "fullstack_developer": [
        "¿Cuántos años de experiencia tienes como fullstack y qué stacks has utilizado (MERN, MEAN, Django+React, etc.)?",
        "¿Cómo balanceas desarrollo frontend y backend? Describe un proyecto donde trabajaste en ambas capas.",
        "¿Qué bases de datos has utilizado y cómo las integras con el frontend? Menciona ORMs y estrategias.",
        "Describe tu experiencia con deployment: ¿cómo llevas aplicaciones completas a producción?",
        "¿Qué desafíos específicos encuentras siendo fullstack y cómo los superas?",
    ],
    "mobile_developer_android": [
        "¿Cuántos años de experiencia tienes desarrollando para Android y qué tipos de apps has creado?",
        "¿Dominas Kotlin o Java? Describe proyectos donde utilizaste las características modernas de Kotlin.",
        "¿Cómo manejas arquitectura en Android (MVVM, MVI, Clean Architecture)? Describe tu enfoque preferido.",
        "¿Qué experiencia tienes con Jetpack Compose? Compáralo con XML layouts.",
        "¿Cómo optimizas rendimiento y batería en apps Android? Menciona técnicas y herramientas.",
    ],
    "mobile_developer_ios": [
        "¿Cuántos años de experiencia tienes desarrollando para iOS y qué tipos de apps has publicado?",
        "¿Dominas Swift? Describe características del lenguaje que usas frecuentemente.",
        "¿Cómo manejas arquitectura en iOS (MVC, MVVM, VIPER)? Describe tu enfoque preferido.",
        "¿Qué experiencia tienes con SwiftUI vs UIKit? ¿Cuándo usas cada uno?",
        "¿Cómo manejas persistencia de datos en iOS (Core Data, Realm, SQLite)?",
    ],
    "qa_manual": [
        "¿Cuántos años de experiencia tienes en QA manual y qué tipos de aplicaciones has testeado?",
        "¿Cómo diseñas casos de prueba efectivos? Describe tu metodología y cobertura.",
        "¿Qué herramientas de gestión de pruebas has utilizado (Jira, TestRail, etc.)?",
        "Describe un bug crítico que encontraste: ¿cómo lo detectaste, documentaste y comunicaste?",
        "¿Cómo priorizas pruebas cuando el tiempo es limitado? Describe tu estrategia de testing.",
    ],
    "qa_automation": [
        "¿Cuántos años de experiencia tienes en QA automation y qué frameworks has utilizado?",
        "¿Qué herramientas de automatización dominas (Selenium, Cypress, Playwright)? Describe proyectos.",
        "¿Cómo diseñas frameworks de testing escalables y mantenibles?",
        "¿Qué estrategias usas para tests flaky? ¿Cómo garantizas estabilidad en CI/CD?",
        "Describe tu experiencia con diferentes tipos de testing: E2E, integration, API, performance.",
    ],
    "data_engineer": [
        "¿Cuántos años de experiencia tienes construyendo pipelines de datos y qué volúmenes has manejado?",
        "¿Qué tecnologías de big data dominas (Spark, Hadoop, Kafka)? Describe casos de uso.",
        "¿Cómo diseñas ETL pipelines escalables y eficientes?",
        "¿Qué experiencia tienes con data warehouses (Redshift, BigQuery, Snowflake)?",
        "¿Cómo garantizas calidad y consistencia de datos en pipelines distribuidos?",
    ],
    "data_analyst": [
        "¿Cuántos años de experiencia tienes en análisis de datos y qué herramientas dominas?",
        "¿Cómo abordas análisis exploratorio de datos? Describe técnicas y visualizaciones que usas.",
        "¿Qué experiencia tienes con SQL y optimización de queries complejas?",
        "Describe un análisis que realizaste que impactó decisiones de negocio significativas.",
        "¿Cómo comunicas insights técnicos a stakeholders no técnicos? Menciona herramientas de visualización.",
    ],
    "product_manager": [
        "¿Cuántos años de experiencia tienes como PM y qué productos has gestionado?",
        "¿Cómo defines y priorizas el roadmap de producto? Describe frameworks que usas.",
        "¿Cómo trabajas con equipos de ingeniería, diseño y negocio? Describe tu enfoque colaborativo.",
        "Describe un producto que lanzaste: desde ideación hasta métricas de éxito post-launch.",
        "¿Cómo manejas trade-offs entre features, tiempo y recursos? Da ejemplos específicos.",
    ],
    "business_analyst": [
        "¿Cuántos años de experiencia tienes en análisis de negocio y qué industrias has trabajado?",
        "¿Cómo recolectas y documentas requisitos de negocio? Describe técnicas que usas.",
        "¿Qué herramientas analíticas y de modelado dominas (SQL, Excel, Power BI)?",
        "Describe un proceso de negocio que optimizaste: análisis, propuesta e implementación.",
        "¿Cómo traduces necesidades de negocio en requisitos técnicos para equipos de desarrollo?",
    ],
    "security_analyst": [
        "¿Cuántos años de experiencia tienes en seguridad y qué tipos de sistemas has protegido?",
        "¿Qué herramientas de seguridad y pentesting dominas? Describe evaluaciones que has realizado.",
        "¿Cómo identificas y mitigas vulnerabilidades? Describe tu metodología.",
        "Cuéntame sobre un incidente de seguridad que manejaste: detección, respuesta y remediación.",
        "¿Qué frameworks de seguridad sigues (OWASP, NIST)? ¿Cómo los aplicas?",
    ],
    "cloud_architect": [
        "¿Cuántos años de experiencia tienes diseñando arquitecturas cloud y en qué plataformas?",
        "¿Cómo diseñas arquitecturas altamente disponibles y escalables? Describe patrones que usas.",
        "¿Qué experiencia tienes con Infrastructure as Code (Terraform, CloudFormation)?",
        "Describe una migración cloud que lideraste: planificación, ejecución y resultados.",
        "¿Cómo optimizas costos cloud sin sacrificar rendimiento y disponibilidad?",
    ],
    "ux_designer": [
        "¿Cuántos años de experiencia tienes en UX/UI y qué productos has diseñado?",
        "¿Cómo conduces research de usuarios? Describe metodologías y herramientas que usas.",
        "¿Cómo validas diseños antes de desarrollo? Menciona prototyping y user testing.",
        "Describe un problema de UX complejo que resolviste: proceso, iteraciones y solución final.",
        "¿Cómo colaboras con PMs y desarrolladores para llevar diseños a producción?",
    ],
    "digital_marketer": [
        "¿Cuántos años de experiencia tienes en marketing digital y qué canales dominas?",
        "¿Cómo diseñas y ejecutas campañas digitales? Describe métricas y ROI que has logrado.",
        "¿Qué herramientas de analytics y automation usas? Describe tu stack de marketing.",
        "Cuéntame sobre una campaña exitosa: estrategia, ejecución y resultados medibles.",
        "¿Cómo optimizas conversión y customer journey? Menciona técnicas y A/B testing.",
    ],
    "hr_specialist": [
        "¿Cuántos años de experiencia tienes en HR y qué áreas has gestionado (recruitment, L&D, etc.)?",
        "¿Cómo manejas procesos de reclutamiento? Describe tu estrategia de sourcing y selección.",
        "¿Qué experiencia tienes con sistemas HRIS y gestión de talento?",
        "Describe una iniciativa de L&D que implementaste y su impacto en la organización.",
        "¿Cómo manejas conflictos laborales y employee relations? Da ejemplos.",
    ],
    "sales_representative": [
        "¿Cuántos años de experiencia tienes en ventas y qué productos/servicios has vendido?",
        "¿Cómo prospectas y calificas leads? Describe tu proceso de ventas.",
        "¿Qué técnicas de cierre usas y cuál es tu tasa de conversión promedio?",
        "Cuéntame sobre tu venta más compleja: desafíos, objeciones y cómo cerraste el deal.",
        "¿Cómo construyes y mantienes relaciones a largo plazo con clientes?",
    ],
    "graphic_designer": [
        "¿Cuántos años de experiencia tienes en diseño gráfico y qué industrias has trabajado?",
        "¿Qué herramientas de diseño dominas (Adobe Suite, Figma)? Describe proyectos destacados.",
        "¿Cómo desarrollas identidades de marca y sistemas de diseño cohesivos?",
        "Describe un proyecto creativo desafiante: brief, proceso y entregables finales.",
        "¿Cómo incorporas feedback de clientes mientras mantienes integridad creativa?",
    ],
    "financial_analyst": [
        "¿Cuántos años de experiencia tienes en análisis financiero y qué sectores has cubierto?",
        "¿Qué modelos financieros desarrollas regularmente (DCF, LBO, etc.)?",
        "¿Cómo realizas forecast y budgeting? Describe herramientas y metodologías.",
        "Cuéntame sobre un análisis que influyó en una decisión estratégica importante.",
        "¿Qué KPIs y métricas financieras monitored regularmente según industria?",
    ],
    "project_manager_tech": [
        "¿Cuántos años de experiencia tienes gestionando proyectos técnicos y de qué magnitud?",
        "¿Qué metodologías sigues (Agile, Scrum, Waterfall)? Describe cuándo usas cada una.",
        "¿Cómo manejas stakeholders, expectativas y comunicación en proyectos?",
        "Describe un proyecto que rescataste: problemas, acciones y resultados.",
        "¿Cómo gestionas riesgos y cambios de scope en proyectos técnicos?",
    ],
    "customer_support": [
        "¿Cuántos años de experiencia tienes en soporte y qué productos/servicios has soportado?",
        "¿Cómo priorizas y resuelves tickets? Describe tu enfoque para customer satisfaction.",
        "¿Qué herramientas de helpdesk y CRM has utilizado?",
        "Cuéntame sobre un cliente difícil que manejaste exitosamente: situación y resolución.",
        "¿Cómo contribuyes a mejorar el producto basándote en feedback de clientes?",
    ],
    "content_writer": [
        "¿Cuántos años de experiencia tienes escribiendo contenido y qué formatos dominas?",
        "¿Cómo investigas temas y aseguras precisión y calidad en tu writing?",
        "¿Qué conocimientos de SEO aplicas en tu contenido?",
        "Describe tu proceso creativo: desde brief hasta entregable pulido.",
        "¿Cómo adaptas tu tono y estilo según audiencia y plataforma?",
    ],
    "scrum_master": [
        "¿Cuántos años de experiencia tienes como Scrum Master y qué equipos has facilitado?",
        "¿Cómo facilitas ceremonias de Scrum (daily, planning, retro, review)? Describe tu enfoque.",
        "¿Cómo identificas y remueves impedimentos del equipo? Da ejemplos específicos.",
        "Describe cómo has ayudado a un equipo a mejorar su madurez ágil y productividad.",
        "¿Cómo manejas conflictos dentro del equipo y con stakeholders externos?",
    ],
    "accountant": [
        "¿Cuántos años de experiencia tienes en contabilidad y qué sectores has trabajado?",
        "¿Qué sistemas contables y ERPs dominas? Describe tu experiencia con software financiero.",
        "¿Cómo manejas cierres mensuales y anuales? Describe tu proceso y timelines.",
        "¿Qué experiencia tienes con auditorías externas? Describe tu rol y preparación.",
        "¿Cómo garantizas cumplimiento con estándares contables (IFRS, GAAP)? Da ejemplos.",
    ],
}


# Mapeo de nombres de profesión del frontend a claves internas
PROFESSION_NAME_MAPPING = {
    "Software Engineer": "software_engineer",
    "Frontend Developer": "frontend_developer",
    "Backend Developer": "backend_developer",
    "Full Stack Developer": "fullstack_developer",
    "Android Developer": "mobile_developer_android",
    "iOS Developer": "mobile_developer_ios",
    "DevOps Engineer": "devops_engineer",
    "Cloud Architect": "cloud_architect",
    "QA Engineer (Manual)": "qa_manual",
    "QA Engineer (Automation)": "qa_automation",
    "Security Analyst": "security_analyst",
    "Technical Project Manager": "project_manager_tech",
    "Scrum Master": "scrum_master",
    "Data Scientist": "data_scientist",
    "Data Engineer": "data_engineer",
    "Data Analyst": "data_analyst",
    "UX/UI Designer": "ux_designer",
    "Graphic Designer": "graphic_designer",
    "Product Manager": "product_manager",
    "Business Analyst": "business_analyst",
    "Financial Analyst": "financial_analyst",
    "Accountant": "accountant",
    "Digital Marketing Specialist": "digital_marketer",
    "Sales Representative": "sales_representative",
    "Content Writer": "content_writer",
    "HR Specialist": "hr_specialist",
    "Customer Support Specialist": "customer_support",
}


def get_context_questions(interview_type: str, profession: str = None) -> List[str]:
    """
    Obtiene las preguntas de contexto según el tipo de entrevista y profesión.

    Args:
        interview_type: Tipo de entrevista ("technical" o "soft_skills")
        profession: Profesión específica del candidato (opcional)

    Returns:
        Lista de preguntas de contexto
    """
    # Si es entrevista de soft skills, retornar preguntas universales
    if interview_type == "soft_skills":
        return SOFT_SKILLS_CONTEXT_QUESTIONS
    
    # Si es técnica y hay profesión, mapear nombre y buscar preguntas específicas
    if interview_type == "technical" and profession:
        # Intentar mapeo directo del nombre
        profession_key = PROFESSION_NAME_MAPPING.get(profession)
        
        # Si no hay mapeo, intentar normalización (fallback)
        if not profession_key:
            profession_key = profession.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        
        # Buscar preguntas específicas de la profesión
        if profession_key and profession_key in PROFESSION_CONTEXT_QUESTIONS:
            return PROFESSION_CONTEXT_QUESTIONS[profession_key]
    
    # Fallback: retornar soft skills si no hay match
    return SOFT_SKILLS_CONTEXT_QUESTIONS


def build_user_profile_context(
    profession: str = None,
    technical_skills: List[str] = None,
    soft_skills: List[str] = None,
    interests: List[str] = None,
    experience_level: str = None,
) -> str:
    """
    Construye un contexto textual del perfil del usuario para mejorar clustering.
    Este texto se concatenará con las respuestas de contexto para embeddings más precisos.

    Args:
        profession: Profesión del usuario
        technical_skills: Lista de habilidades técnicas
        soft_skills: Lista de habilidades blandas
        interests: Lista de intereses profesionales
        experience_level: Nivel de experiencia (Junior, Mid, Senior)

    Returns:
        String con el perfil del usuario en formato natural
    """
    profile_parts = []
    
    if profession:
        profile_parts.append(f"Profesión: {profession}")
    
    if experience_level:
        profile_parts.append(f"Nivel de experiencia: {experience_level}")
    
    if technical_skills and len(technical_skills) > 0:
        skills_str = ", ".join(technical_skills)
        profile_parts.append(f"Habilidades técnicas: {skills_str}")
    
    if soft_skills and len(soft_skills) > 0:
        skills_str = ", ".join(soft_skills)
        profile_parts.append(f"Habilidades blandas: {skills_str}")
    
    if interests and len(interests) > 0:
        interests_str = ", ".join(interests)
        profile_parts.append(f"Áreas de interés: {interests_str}")
    
    return ". ".join(profile_parts) + "." if profile_parts else ""


# Número de preguntas de contexto antes de comenzar la entrevista formal
CONTEXT_QUESTIONS_COUNT = 5

# Número de preguntas principales de la entrevista
MAIN_QUESTIONS_COUNT = 10

# Número máximo de intentos por pregunta
MAX_ATTEMPTS_PER_QUESTION = 3

# Umbral de score para considerar una respuesta correcta
PASS_THRESHOLD = 6.0

# Umbral de score para considerar una respuesta excelente
EXCELLENT_THRESHOLD = 9.0
