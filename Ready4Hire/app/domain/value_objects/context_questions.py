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
import logging

logger = logging.getLogger(__name__)


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
    "mobile_developer": [
        "¿Cuántos años de experiencia tienes desarrollando aplicaciones móviles y qué plataformas has trabajado (Android, iOS, multiplataforma)?",
        "¿Qué frameworks y tecnologías móviles dominas (React Native, Flutter, nativo)? Describe proyectos donde los usaste.",
        "¿Cómo manejas arquitectura en aplicaciones móviles? Describe patrones y principios que aplicas.",
        "¿Qué experiencia tienes con integración de APIs y manejo de estado en apps móviles?",
        "¿Cómo optimizas rendimiento, batería y experiencia de usuario en aplicaciones móviles?",
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
    # SALUD Y MEDICINA
    "doctor": [
        "¿Cuántos años de experiencia tienes como médico y en qué especialidad?",
        "¿Qué tipos de pacientes y casos has tratado? Describe tu experiencia clínica.",
        "¿Cómo manejas diagnósticos complejos y casos que requieren múltiples especialistas?",
        "Cuéntame sobre un caso desafiante que trataste: diagnóstico, tratamiento y resultado.",
        "¿Cómo mantienes actualizado tu conocimiento médico y qué recursos utilizas?",
    ],
    "nurse": [
        "¿Cuántos años de experiencia tienes en enfermería y en qué áreas has trabajado?",
        "¿Qué tipos de pacientes y cuidados has manejado? Describe tu experiencia clínica.",
        "¿Cómo manejas situaciones de emergencia y pacientes críticos?",
        "Cuéntame sobre una situación desafiante donde tu cuidado hizo la diferencia.",
        "¿Cómo colaboras con médicos y otros profesionales de la salud?",
    ],
    "dentist": [
        "¿Cuántos años de experiencia tienes en odontología y qué especialidades dominas?",
        "¿Qué tipos de procedimientos realizas regularmente? Describe tu práctica clínica.",
        "¿Cómo manejas pacientes con ansiedad dental? Describe técnicas y enfoques.",
        "Cuéntame sobre un caso complejo que trataste: diagnóstico, plan de tratamiento y resultado.",
        "¿Qué tecnologías y equipos dentales utilizas en tu práctica?",
    ],
    "pharmacist": [
        "¿Cuántos años de experiencia tienes en farmacia y en qué entornos has trabajado?",
        "¿Cómo manejas la dispensación de medicamentos y la verificación de recetas?",
        "¿Qué experiencia tienes con consultoría farmacéutica y consejo a pacientes?",
        "Describe cómo identificas y previenes interacciones medicamentosas peligrosas.",
        "¿Cómo colaboras con médicos para optimizar tratamientos farmacológicos?",
    ],
    "psychologist": [
        "¿Cuántos años de experiencia tienes en psicología y qué enfoques terapéuticos utilizas?",
        "¿Qué tipos de pacientes y condiciones has tratado? Describe tu experiencia clínica.",
        "¿Cómo estableces rapport y confianza con pacientes? Describe tu enfoque terapéutico.",
        "Cuéntame sobre un caso complejo que manejaste: evaluación, tratamiento y progreso.",
        "¿Cómo manejas situaciones de crisis y pacientes en riesgo?",
    ],
    "physical_therapist": [
        "¿Cuántos años de experiencia tienes en fisioterapia y qué áreas especializas?",
        "¿Qué tipos de lesiones y condiciones has tratado? Describe tu experiencia clínica.",
        "¿Cómo desarrollas planes de tratamiento personalizados para pacientes?",
        "Cuéntame sobre un caso donde ayudaste a un paciente a recuperar funcionalidad significativa.",
        "¿Qué técnicas y modalidades terapéuticas utilizas regularmente?",
    ],
    "nutritionist": [
        "¿Cuántos años de experiencia tienes en nutrición y qué áreas especializas?",
        "¿Qué tipos de pacientes y condiciones nutricionales has manejado?",
        "¿Cómo desarrollas planes nutricionales personalizados? Describe tu metodología.",
        "Cuéntame sobre un caso donde tu intervención nutricional tuvo impacto significativo.",
        "¿Cómo mantienes actualizado tu conocimiento sobre nutrición y dietética?",
    ],
    "veterinarian": [
        "¿Cuántos años de experiencia tienes en veterinaria y qué tipos de animales tratas?",
        "¿Qué especialidades veterinarias dominas? Describe tu experiencia clínica.",
        "¿Cómo manejas diagnósticos complejos en animales? Describe tu proceso.",
        "Cuéntame sobre un caso desafiante que trataste: diagnóstico, tratamiento y resultado.",
        "¿Cómo comunicas diagnósticos y tratamientos a dueños de mascotas?",
    ],
    # EDUCACIÓN
    "teacher": [
        "¿Cuántos años de experiencia tienes enseñando y qué niveles y materias has cubierto?",
        "¿Qué metodologías de enseñanza utilizas? Describe tu enfoque pedagógico.",
        "¿Cómo manejas diferentes estilos de aprendizaje y necesidades especiales?",
        "Cuéntame sobre una estrategia de enseñanza que implementaste exitosamente.",
        "¿Cómo evalúas el progreso de los estudiantes y adaptas tu enseñanza?",
    ],
    "principal": [
        "¿Cuántos años de experiencia tienes como director y qué tipos de escuelas has gestionado?",
        "¿Cómo lideras el equipo docente y administrativo? Describe tu estilo de liderazgo.",
        "¿Qué iniciativas educativas has implementado? Describe impacto y resultados.",
        "Cuéntame sobre un desafío significativo que enfrentaste como director y cómo lo resolviste.",
        "¿Cómo manejas relaciones con padres, comunidad y autoridades educativas?",
    ],
    "tutor": [
        "¿Cuántos años de experiencia tienes como tutor y qué materias y niveles cubres?",
        "¿Cómo identificas las necesidades de aprendizaje de cada estudiante?",
        "¿Qué estrategias utilizas para ayudar a estudiantes con dificultades?",
        "Cuéntame sobre un estudiante que mejoró significativamente con tu tutoría.",
        "¿Cómo adaptas tu enfoque según el estilo de aprendizaje del estudiante?",
    ],
    "educational_counselor": [
        "¿Cuántos años de experiencia tienes en orientación educativa y qué niveles has trabajado?",
        "¿Cómo ayudas a estudiantes a identificar sus intereses y objetivos académicos?",
        "¿Qué herramientas y evaluaciones utilizas para orientación vocacional?",
        "Cuéntame sobre un caso donde ayudaste a un estudiante a tomar decisiones académicas importantes.",
        "¿Cómo colaboras con docentes y padres para apoyar el desarrollo estudiantil?",
    ],
    "librarian": [
        "¿Cuántos años de experiencia tienes en bibliotecología y qué tipos de bibliotecas has gestionado?",
        "¿Cómo organizas y catalogas colecciones? Describe sistemas y metodologías que usas.",
        "¿Qué servicios de referencia y apoyo a usuarios proporcionas?",
        "Cuéntame sobre una iniciativa que implementaste para mejorar el acceso a recursos.",
        "¿Cómo integras tecnología y recursos digitales en servicios bibliotecarios?",
    ],
    "training_specialist": [
        "¿Cuántos años de experiencia tienes en capacitación y qué áreas has cubierto?",
        "¿Cómo diseñas programas de capacitación efectivos? Describe tu metodología.",
        "¿Qué técnicas de enseñanza y herramientas utilizas para adultos?",
        "Cuéntame sobre un programa de capacitación exitoso que desarrollaste e implementaste.",
        "¿Cómo evalúas la efectividad de la capacitación y mides ROI?",
    ],
    # NEGOCIOS Y FINANZAS
    "investment_advisor": [
        "¿Cuántos años de experiencia tienes como asesor de inversiones y qué tipos de clientes has asesorado?",
        "¿Cómo evalúas el perfil de riesgo de clientes y desarrollas estrategias de inversión?",
        "¿Qué productos y mercados financieros conoces? Describe tu expertise.",
        "Cuéntame sobre una recomendación de inversión exitosa que hiciste y su resultado.",
        "¿Cómo manejas volatilidad de mercados y comunicas cambios a clientes?",
    ],
    "operations_manager": [
        "¿Cuántos años de experiencia tienes gestionando operaciones y en qué industrias?",
        "¿Cómo optimizas procesos operativos? Describe metodologías y herramientas que usas.",
        "¿Qué experiencia tienes con gestión de inventario, supply chain y logística?",
        "Cuéntame sobre una mejora operativa significativa que implementaste y su impacto.",
        "¿Cómo manejas equipos operativos y garantizas eficiencia y calidad?",
    ],
    "consultant": [
        "¿Cuántos años de experiencia tienes como consultor y en qué áreas especializas?",
        "¿Qué tipos de proyectos y clientes has asesorado? Describe tu experiencia.",
        "¿Cómo identificas problemas y desarrollas soluciones para clientes?",
        "Cuéntame sobre un proyecto consultivo exitoso: desafío, solución e impacto.",
        "¿Cómo manejas expectativas de clientes y entregas en plazos ajustados?",
    ],
    "entrepreneur": [
        "¿Cuántos negocios has fundado o cofundado y en qué industrias?",
        "¿Cómo identificas oportunidades de negocio y validas ideas? Describe tu proceso.",
        "¿Qué experiencia tienes con fundraising, pitch a inversores y gestión de startups?",
        "Cuéntame sobre un desafío significativo que enfrentaste como emprendedor y cómo lo superaste.",
        "¿Cómo construyes y lideras equipos en etapas tempranas de startups?",
    ],
    # MARKETING Y VENTAS
    "marketing_manager": [
        "¿Cuántos años de experiencia tienes en marketing y qué industrias has trabajado?",
        "¿Cómo desarrollas estrategias de marketing integrales? Describe tu enfoque.",
        "¿Qué canales y herramientas de marketing dominas? Describe tu stack.",
        "Cuéntame sobre una campaña exitosa que lideraste: estrategia, ejecución y resultados.",
        "¿Cómo mides ROI y efectividad de campañas de marketing?",
    ],
    "brand_manager": [
        "¿Cuántos años de experiencia tienes gestionando marcas y qué marcas has manejado?",
        "¿Cómo desarrollas y mantienes identidad de marca? Describe tu proceso.",
        "¿Qué experiencia tienes con posicionamiento de marca y diferenciación competitiva?",
        "Cuéntame sobre una iniciativa de marca exitosa que implementaste y su impacto.",
        "¿Cómo manejas crisis de marca y reputación corporativa?",
    ],
    "pr_specialist": [
        "¿Cuántos años de experiencia tienes en relaciones públicas y qué sectores has cubierto?",
        "¿Cómo desarrollas estrategias de comunicación y relaciones con medios?",
        "¿Qué experiencia tienes con gestión de crisis y comunicación corporativa?",
        "Cuéntame sobre una campaña de PR exitosa que manejaste: objetivos, ejecución y resultados.",
        "¿Cómo construyes y mantienes relaciones con periodistas y medios?",
    ],
    "social_media_manager": [
        "¿Cuántos años de experiencia tienes gestionando redes sociales y qué plataformas dominas?",
        "¿Cómo desarrollas estrategias de contenido para redes sociales? Describe tu enfoque.",
        "¿Qué herramientas y métricas utilizas para gestión y análisis de social media?",
        "Cuéntame sobre una campaña de redes sociales exitosa: estrategia, ejecución y engagement.",
        "¿Cómo manejas community management y respuesta a comentarios y mensajes?",
    ],
    # LEGAL Y JURÍDICO
    "lawyer": [
        "¿Cuántos años de experiencia tienes como abogado y en qué áreas del derecho especializas?",
        "¿Qué tipos de casos has manejado? Describe tu experiencia legal.",
        "¿Cómo preparas casos y desarrollas estrategias legales?",
        "Cuéntame sobre un caso complejo que manejaste: desafíos, estrategia y resultado.",
        "¿Cómo manejas negociaciones y resolución de disputas?",
    ],
    "paralegal": [
        "¿Cuántos años de experiencia tienes como paralegal y en qué áreas legales has trabajado?",
        "¿Qué tareas realizas regularmente? Describe tu rol y responsabilidades.",
        "¿Cómo organizas y gestionas documentación legal y casos?",
        "Cuéntame sobre cómo apoyaste en un caso complejo: tu contribución y resultado.",
        "¿Qué sistemas y herramientas legales utilizas para gestión de casos?",
    ],
    "judge": [
        "¿Cuántos años de experiencia tienes como juez y en qué jurisdicciones has servido?",
        "¿Qué tipos de casos has presidido? Describe tu experiencia judicial.",
        "¿Cómo preparas decisiones judiciales y aplicas la ley? Describe tu proceso.",
        "Cuéntame sobre un caso complejo que presidiste: desafíos legales y tu decisión.",
        "¿Cómo manejas imparcialidad y justicia en casos sensibles?",
    ],
    "legal_advisor": [
        "¿Cuántos años de experiencia tienes como asesor legal y en qué áreas especializas?",
        "¿Qué tipos de clientes y situaciones legales has asesorado?",
        "¿Cómo proporcionas consejo legal estratégico? Describe tu enfoque.",
        "Cuéntame sobre un asesoramiento legal exitoso que proporcionaste y su impacto.",
        "¿Cómo manejas compliance y gestión de riesgos legales para clientes?",
    ],
    "notary": [
        "¿Cuántos años de experiencia tienes como notario y qué tipos de documentos has notarizado?",
        "¿Qué servicios notariales proporcionas regularmente?",
        "¿Cómo verificas identidad y capacidad legal de las partes?",
        "Cuéntame sobre un documento complejo que notarizaste: proceso y consideraciones.",
        "¿Cómo mantienes registros y garantizas cumplimiento legal en notarización?",
    ],
    # INGENIERÍA Y CONSTRUCCIÓN
    "civil_engineer": [
        "¿Cuántos años de experiencia tienes en ingeniería civil y qué tipos de proyectos has diseñado?",
        "¿Qué áreas especializas (estructuras, transporte, hidráulica)? Describe tu expertise.",
        "¿Cómo manejas diseño, cálculos y especificaciones técnicas?",
        "Cuéntame sobre un proyecto complejo que diseñaste: desafíos técnicos y solución.",
        "¿Qué software y herramientas de ingeniería utilizas regularmente?",
    ],
    "mechanical_engineer": [
        "¿Cuántos años de experiencia tienes en ingeniería mecánica y en qué industrias has trabajado?",
        "¿Qué tipos de sistemas y máquinas has diseñado? Describe tu experiencia.",
        "¿Cómo manejas diseño mecánico, análisis de estrés y selección de materiales?",
        "Cuéntame sobre un proyecto complejo que diseñaste: desafíos técnicos y solución.",
        "¿Qué software CAD/CAE y herramientas utilizas para diseño mecánico?",
    ],
    "electrical_engineer": [
        "¿Cuántos años de experiencia tienes en ingeniería eléctrica y qué sistemas has diseñado?",
        "¿Qué áreas especializas (potencia, control, electrónica)? Describe tu expertise.",
        "¿Cómo manejas diseño de circuitos, sistemas eléctricos y cumplimiento de códigos?",
        "Cuéntame sobre un proyecto complejo que diseñaste: desafíos técnicos y solución.",
        "¿Qué software y herramientas utilizas para diseño y simulación eléctrica?",
    ],
    "architect": [
        "¿Cuántos años de experiencia tienes en arquitectura y qué tipos de proyectos has diseñado?",
        "¿Cómo integras funcionalidad, estética y sostenibilidad en tus diseños?",
        "¿Qué experiencia tienes con códigos de construcción, permisos y coordinación con ingenieros?",
        "Cuéntame sobre un proyecto arquitectónico complejo: concepto, diseño y construcción.",
        "¿Qué software de diseño arquitectónico y BIM utilizas?",
    ],
    "construction_manager": [
        "¿Cuántos años de experiencia tienes gestionando construcción y qué tipos de proyectos?",
        "¿Cómo planificas y coordinas proyectos de construcción? Describe tu metodología.",
        "¿Qué experiencia tienes con gestión de contratistas, subcontratistas y equipos?",
        "Cuéntame sobre un proyecto complejo que gestionaste: desafíos, solución y resultado.",
        "¿Cómo manejas seguridad, calidad y cumplimiento de plazos en construcción?",
    ],
    "urban_planner": [
        "¿Cuántos años de experiencia tienes en planificación urbana y qué tipos de proyectos has planificado?",
        "¿Cómo desarrollas planes urbanos? Describe tu proceso y metodología.",
        "¿Qué experiencia tienes con zonificación, desarrollo sostenible y participación ciudadana?",
        "Cuéntame sobre un plan urbano que desarrollaste: objetivos, proceso e impacto.",
        "¿Cómo integras consideraciones ambientales, sociales y económicas en planificación?",
    ],
    # RECURSOS HUMANOS
    "recruiter": [
        "¿Cuántos años de experiencia tienes en reclutamiento y qué tipos de posiciones has cubierto?",
        "¿Cómo identificas y atraes candidatos calificados? Describe tu estrategia de sourcing.",
        "¿Qué experiencia tienes con screening, entrevistas y evaluación de candidatos?",
        "Cuéntame sobre un proceso de reclutamiento exitoso: desde requisición hasta contratación.",
        "¿Cómo construyes relaciones con candidatos y mantienes pipelines de talento?",
    ],
    "hr_analyst": [
        "¿Cuántos años de experiencia tienes en análisis de RRHH y qué áreas has analizado?",
        "¿Qué métricas y KPIs de RRHH monitoreas regularmente?",
        "¿Cómo analizas datos de empleados para insights estratégicos?",
        "Cuéntame sobre un análisis de RRHH que influyó en decisiones organizacionales.",
        "¿Qué herramientas y sistemas HRIS utilizas para análisis de datos?",
    ],
    "training_coordinator": [
        "¿Cuántos años de experiencia tienes coordinando capacitación y qué programas has gestionado?",
        "¿Cómo identificas necesidades de capacitación organizacional?",
        "¿Qué experiencia tienes con diseño, desarrollo e implementación de programas de L&D?",
        "Cuéntame sobre un programa de capacitación exitoso que coordinaste: diseño, ejecución e impacto.",
        "¿Cómo evalúas efectividad de capacitación y mides ROI de programas L&D?",
    ],
    "compensation_specialist": [
        "¿Cuántos años de experiencia tienes en compensaciones y qué tipos de estructuras has diseñado?",
        "¿Cómo desarrollas estrategias de compensación competitivas? Describe tu metodología.",
        "¿Qué experiencia tienes con análisis de mercado salarial y benchmarking?",
        "Cuéntame sobre una estructura de compensación que diseñaste: objetivos, diseño e implementación.",
        "¿Cómo manejas equity, beneficios y paquetes de compensación total?",
    ],
    # COMUNICACIÓN Y MEDIOS
    "journalist": [
        "¿Cuántos años de experiencia tienes en periodismo y qué tipos de historias has cubierto?",
        "¿Cómo investigas y verificas información para tus reportajes?",
        "¿Qué experiencia tienes con diferentes formatos (noticias, reportajes, entrevistas)?",
        "Cuéntame sobre una historia importante que reportaste: investigación, desarrollo e impacto.",
        "¿Cómo manejas ética periodística y objetividad en tu trabajo?",
    ],
    "editor": [
        "¿Cuántos años de experiencia tienes editando y qué tipos de contenido has editado?",
        "¿Cómo editas contenido para claridad, precisión y estilo? Describe tu proceso.",
        "¿Qué experiencia tienes con diferentes estilos editoriales y guías de estilo?",
        "Cuéntame sobre un proyecto editorial complejo que manejaste: desafíos y resultado.",
        "¿Cómo colaboras con escritores y otros editores en proyectos editoriales?",
    ],
    "photographer": [
        "¿Cuántos años de experiencia tienes en fotografía y qué géneros fotografías?",
        "¿Qué tipos de proyectos y clientes has trabajado? Describe tu portfolio.",
        "¿Cómo planificas sesiones fotográficas? Describe tu proceso creativo y técnico.",
        "Cuéntame sobre un proyecto fotográfico destacado: concepto, ejecución y resultado.",
        "¿Qué equipo y técnicas fotográficas utilizas regularmente?",
    ],
    "video_producer": [
        "¿Cuántos años de experiencia tienes produciendo video y qué tipos de proyectos?",
        "¿Cómo desarrollas conceptos y guiones para videos? Describe tu proceso creativo.",
        "¿Qué experiencia tienes con preproducción, producción y postproducción?",
        "Cuéntame sobre un proyecto de video exitoso: concepto, producción e impacto.",
        "¿Qué software y herramientas de producción de video utilizas?",
    ],
    "radio_host": [
        "¿Cuántos años de experiencia tienes como locutor y qué tipos de programas has conducido?",
        "¿Cómo preparas y estructura programas de radio? Describe tu proceso.",
        "¿Qué experiencia tienes con diferentes formatos (noticias, entretenimiento, talk shows)?",
        "Cuéntame sobre un programa o segmento destacado que produjiste y su recepción.",
        "¿Cómo manejas interacción con audiencia y entrevistas en vivo?",
    ],
    "translator": [
        "¿Cuántos años de experiencia tienes traduciendo y qué pares de idiomas dominas?",
        "¿Qué tipos de documentos y contenidos traduces? Describe tu especialización.",
        "¿Cómo manejas traducción de contenido técnico o especializado?",
        "Cuéntame sobre un proyecto de traducción complejo que manejaste: desafíos y solución.",
        "¿Qué herramientas y tecnologías de traducción utilizas?",
    ],
    # SERVICIOS Y ATENCIÓN AL CLIENTE
    "hotel_manager": [
        "¿Cuántos años de experiencia tienes gestionando hoteles y qué tipos de propiedades?",
        "¿Cómo gestionas operaciones hoteleras diarias? Describe tus responsabilidades clave.",
        "¿Qué experiencia tienes con gestión de personal, housekeeping y servicios de huéspedes?",
        "Cuéntame sobre una iniciativa que implementaste para mejorar experiencia de huéspedes.",
        "¿Cómo manejas gestión de ingresos, ocupación y satisfacción de huéspedes?",
    ],
    "travel_agent": [
        "¿Cuántos años de experiencia tienes como agente de viajes y qué destinos conoces?",
        "¿Cómo ayudas a clientes a planificar viajes? Describe tu proceso de consultoría.",
        "¿Qué experiencia tienes con reservas, itinerarios y gestión de viajes complejos?",
        "Cuéntame sobre un viaje complejo que planificaste: desafíos, solución y satisfacción del cliente.",
        "¿Qué sistemas de reservas y herramientas de viaje utilizas?",
    ],
    "event_coordinator": [
        "¿Cuántos años de experiencia tienes coordinando eventos y qué tipos de eventos has organizado?",
        "¿Cómo planificas y ejecutas eventos? Describe tu proceso de coordinación.",
        "¿Qué experiencia tienes con gestión de proveedores, logística y presupuestos de eventos?",
        "Cuéntame sobre un evento complejo que coordinaste: planificación, ejecución y resultado.",
        "¿Cómo manejas imprevistos y garantizas éxito de eventos?",
    ],
    "restaurant_manager": [
        "¿Cuántos años de experiencia tienes gestionando restaurantes y qué tipos de establecimientos?",
        "¿Cómo gestionas operaciones diarias de restaurante? Describe tus responsabilidades.",
        "¿Qué experiencia tienes con gestión de personal, cocina, servicio y costos?",
        "Cuéntame sobre una iniciativa que implementaste para mejorar operaciones o experiencia de clientes.",
        "¿Cómo manejas control de calidad, inventario y rentabilidad de restaurante?",
    ],
    # CIENCIAS E INVESTIGACIÓN
    "research_scientist": [
        "¿Cuántos años de experiencia tienes en investigación científica y en qué áreas?",
        "¿Qué tipos de proyectos de investigación has liderado o participado?",
        "¿Cómo diseñas experimentos y metodologías de investigación?",
        "Cuéntame sobre un proyecto de investigación significativo: hipótesis, metodología y resultados.",
        "¿Qué técnicas y herramientas de investigación utilizas regularmente?",
    ],
    "laboratory_technician": [
        "¿Cuántos años de experiencia tienes como técnico de laboratorio y en qué tipos de laboratorios?",
        "¿Qué técnicas y procedimientos de laboratorio realizas regularmente?",
        "¿Cómo manejas muestras, análisis y documentación de resultados?",
        "Cuéntame sobre un análisis complejo que realizaste: procedimiento, desafíos y resultados.",
        "¿Qué equipos y sistemas de laboratorio operas y mantienes?",
    ],
    "environmental_scientist": [
        "¿Cuántos años de experiencia tienes en ciencias ambientales y qué áreas especializas?",
        "¿Qué tipos de estudios y proyectos ambientales has realizado?",
        "¿Cómo evalúas impacto ambiental y desarrollas soluciones sostenibles?",
        "Cuéntame sobre un proyecto ambiental significativo: objetivos, metodología e impacto.",
        "¿Qué herramientas y tecnologías utilizas para monitoreo y análisis ambiental?",
    ],
    "statistician": [
        "¿Cuántos años de experiencia tienes en estadística y en qué áreas aplicas?",
        "¿Qué tipos de análisis estadísticos realizas regularmente?",
        "¿Cómo diseñas estudios y experimentos estadísticos?",
        "Cuéntame sobre un análisis estadístico complejo que realizaste: metodología y insights.",
        "¿Qué software y herramientas estadísticas utilizas (R, Python, SPSS, etc.)?",
    ],
    "quality_control_analyst": [
        "¿Cuántos años de experiencia tienes en control de calidad y en qué industrias?",
        "¿Qué procesos y productos has analizado para garantizar calidad?",
        "¿Cómo desarrollas y ejecutas protocolos de control de calidad?",
        "Cuéntame sobre un problema de calidad que identificaste y resolviste: análisis y solución.",
        "¿Qué herramientas y metodologías de QC utilizas (Six Sigma, ISO, etc.)?",
    ],
    # ARTE Y CREATIVIDAD
    "interior_designer": [
        "¿Cuántos años de experiencia tienes en diseño de interiores y qué tipos de espacios has diseñado?",
        "¿Cómo desarrollas conceptos de diseño? Describe tu proceso creativo.",
        "¿Qué experiencia tienes con selección de materiales, mobiliario y decoración?",
        "Cuéntame sobre un proyecto de diseño de interiores destacado: concepto, ejecución y resultado.",
        "¿Qué software de diseño y herramientas utilizas para visualización?",
    ],
    "musician": [
        "¿Cuántos años de experiencia tienes como músico y qué instrumentos y géneros dominas?",
        "¿Qué tipos de proyectos musicales has realizado? Describe tu experiencia.",
        "¿Cómo desarrollas composiciones y arreglos musicales?",
        "Cuéntame sobre un proyecto musical destacado: concepto, producción y recepción.",
        "¿Qué herramientas y tecnologías musicales utilizas (DAWs, instrumentos, etc.)?",
    ],
    "artist": [
        "¿Cuántos años de experiencia tienes como artista y en qué medios y estilos trabajas?",
        "¿Qué tipos de proyectos artísticos has realizado? Describe tu portfolio.",
        "¿Cómo desarrollas conceptos y ejecutas obras artísticas? Describe tu proceso creativo.",
        "Cuéntame sobre una obra o proyecto artístico destacado: concepto, proceso y resultado.",
        "¿Qué técnicas, materiales y herramientas artísticas utilizas?",
    ],
    "fashion_designer": [
        "¿Cuántos años de experiencia tienes en diseño de moda y qué tipos de colecciones has creado?",
        "¿Cómo desarrollas conceptos y diseños de moda? Describe tu proceso creativo.",
        "¿Qué experiencia tienes con patronaje, confección y producción de prendas?",
        "Cuéntame sobre una colección o diseño destacado: concepto, desarrollo y resultado.",
        "¿Qué técnicas, materiales y procesos de diseño de moda utilizas?",
    ],
    # LOGÍSTICA Y TRANSPORTE
    "logistics_coordinator": [
        "¿Cuántos años de experiencia tienes coordinando logística y en qué industrias?",
        "¿Cómo planificas y coordinas operaciones logísticas? Describe tu proceso.",
        "¿Qué experiencia tienes con gestión de inventario, almacenamiento y distribución?",
        "Cuéntame sobre una operación logística compleja que coordinaste: desafíos y solución.",
        "¿Qué sistemas y herramientas de gestión logística utilizas?",
    ],
    "supply_chain_manager": [
        "¿Cuántos años de experiencia tienes gestionando supply chain y en qué industrias?",
        "¿Cómo optimizas cadenas de suministro? Describe tu estrategia y metodología.",
        "¿Qué experiencia tienes con gestión de proveedores, inventario y distribución?",
        "Cuéntame sobre una mejora significativa en supply chain que implementaste: impacto y resultados.",
        "¿Cómo manejas riesgos y resiliencia en cadenas de suministro?",
    ],
    "truck_driver": [
        "¿Cuántos años de experiencia tienes conduciendo camiones y qué tipos de rutas y cargas?",
        "¿Qué tipos de vehículos y licencias tienes? Describe tu experiencia de conducción.",
        "¿Cómo manejas planificación de rutas, cumplimiento de regulaciones y seguridad?",
        "Cuéntame sobre una ruta o entrega compleja que manejaste: desafíos y solución.",
        "¿Cómo mantienes registros y cumplimiento con regulaciones de transporte?",
    ],
    "warehouse_manager": [
        "¿Cuántos años de experiencia tienes gestionando almacenes y qué tipos de operaciones?",
        "¿Cómo organizas y optimizas operaciones de almacén? Describe tu metodología.",
        "¿Qué experiencia tienes con gestión de inventario, picking, packing y envíos?",
        "Cuéntame sobre una mejora significativa en operaciones de almacén que implementaste.",
        "¿Qué sistemas WMS y tecnologías de almacén utilizas?",
    ],
    "pilot": [
        "¿Cuántos años de experiencia tienes como piloto y qué tipos de aeronaves y rutas?",
        "¿Qué licencias y certificaciones tienes? Describe tu experiencia de vuelo.",
        "¿Cómo manejas planificación de vuelo, navegación y cumplimiento de regulaciones?",
        "Cuéntame sobre un vuelo desafiante que manejaste: condiciones y solución.",
        "¿Cómo mantienes competencia y cumplimiento con regulaciones de aviación?",
    ],
}


# Mapeo de nombres de profesión del frontend a claves internas
PROFESSION_NAME_MAPPING = {
    "Software Engineer": "software_engineer",
    "Software Developer": "software_engineer",
    "Frontend Developer": "frontend_developer",
    "Backend Developer": "backend_developer",
    "Full Stack Developer": "fullstack_developer",
    "Mobile Developer": "mobile_developer_android",  # Mapeo genérico a Android
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
    "HR Manager": "hr_specialist",
    "Customer Support Specialist": "customer_support",
    "Customer Service Representative": "customer_support",
    "QA Engineer": "qa_manual",  # Mapeo genérico a QA Manual
    "Cybersecurity Analyst": "security_analyst",
    "Security Analyst": "security_analyst",
    "AI/ML Engineer": "data_scientist",  # Mapeo a Data Scientist
    "Content Creator": "content_writer",
    "Content Writer": "content_writer",
    "Project Manager": "project_manager_tech",
    "Technical Project Manager": "project_manager_tech",
    # SALUD Y MEDICINA
    "Doctor": "doctor",
    "Nurse": "nurse",
    "Dentist": "dentist",
    "Pharmacist": "pharmacist",
    "Psychologist": "psychologist",
    "Physical Therapist": "physical_therapist",
    "Nutritionist": "nutritionist",
    "Veterinarian": "veterinarian",
    # EDUCACIÓN
    "Teacher": "teacher",
    "Principal": "principal",
    "Tutor": "tutor",
    "Educational Counselor": "educational_counselor",
    "Librarian": "librarian",
    "Training Specialist": "training_specialist",
    # NEGOCIOS Y FINANZAS
    "Investment Advisor": "investment_advisor",
    "Operations Manager": "operations_manager",
    "Consultant": "consultant",
    "Entrepreneur": "entrepreneur",
    # MARKETING Y VENTAS
    "Marketing Manager": "marketing_manager",
    "Brand Manager": "brand_manager",
    "PR Specialist": "pr_specialist",
    "Social Media Manager": "social_media_manager",
    # LEGAL Y JURÍDICO
    "Lawyer": "lawyer",
    "Paralegal": "paralegal",
    "Judge": "judge",
    "Legal Advisor": "legal_advisor",
    "Notary": "notary",
    # INGENIERÍA Y CONSTRUCCIÓN
    "Civil Engineer": "civil_engineer",
    "Mechanical Engineer": "mechanical_engineer",
    "Electrical Engineer": "electrical_engineer",
    "Architect": "architect",
    "Construction Manager": "construction_manager",
    "Urban Planner": "urban_planner",
    # RECURSOS HUMANOS
    "Recruiter": "recruiter",
    "HR Analyst": "hr_analyst",
    "Training Coordinator": "training_coordinator",
    "Compensation Specialist": "compensation_specialist",
    # COMUNICACIÓN Y MEDIOS
    "Journalist": "journalist",
    "Editor": "editor",
    "Photographer": "photographer",
    "Video Producer": "video_producer",
    "Radio Host": "radio_host",
    "Translator": "translator",
    # SERVICIOS Y ATENCIÓN AL CLIENTE
    "Hotel Manager": "hotel_manager",
    "Travel Agent": "travel_agent",
    "Event Coordinator": "event_coordinator",
    "Restaurant Manager": "restaurant_manager",
    # CIENCIAS E INVESTIGACIÓN
    "Research Scientist": "research_scientist",
    "Laboratory Technician": "laboratory_technician",
    "Environmental Scientist": "environmental_scientist",
    "Statistician": "statistician",
    "Quality Control Analyst": "quality_control_analyst",
    # ARTE Y CREATIVIDAD
    "Interior Designer": "interior_designer",
    "Musician": "musician",
    "Artist": "artist",
    "Fashion Designer": "fashion_designer",
    # LOGÍSTICA Y TRANSPORTE
    "Logistics Coordinator": "logistics_coordinator",
    "Supply Chain Manager": "supply_chain_manager",
    "Truck Driver": "truck_driver",
    "Pilot": "pilot",
    "Warehouse Manager": "warehouse_manager",
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
            profession_key = profession.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        # Buscar preguntas específicas de la profesión
        if profession_key and profession_key in PROFESSION_CONTEXT_QUESTIONS:
            return PROFESSION_CONTEXT_QUESTIONS[profession_key]
        
        # Si no hay preguntas específicas, usar preguntas genéricas de software engineer como fallback
        # Esto evita el warning "No template found"
        logger.warning(f"⚠️ No se encontraron preguntas de contexto específicas para '{profession}' (key: '{profession_key}'), usando preguntas genéricas de software engineer")
        return PROFESSION_CONTEXT_QUESTIONS.get("software_engineer", SOFT_SKILLS_CONTEXT_QUESTIONS)
    
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
