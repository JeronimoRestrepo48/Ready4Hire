#!/usr/bin/env python3
"""
Script para expandir el dataset de preguntas técnicas con todas las profesiones.
Genera preguntas para 84 profesiones x 3 niveles x 15-20 preguntas = 3000+ preguntas.
"""

import json
import random
from pathlib import Path

# Todas las profesiones del frontend
PROFESSIONS = [
    # TECNOLOGÍA E INFORMÁTICA
    'Software Developer', 'Frontend Developer', 'Backend Developer', 'Full Stack Developer', 
    'DevOps Engineer', 'Data Scientist', 'Product Manager', 'UX/UI Designer', 'QA Engineer', 
    'Mobile Developer', 'Cloud Architect', 'Cybersecurity Analyst', 'AI/ML Engineer',
    
    # SALUD Y MEDICINA
    'Doctor', 'Nurse', 'Dentist', 'Pharmacist', 'Psychologist', 'Physical Therapist', 
    'Nutritionist', 'Veterinarian',
    
    # EDUCACIÓN
    'Teacher', 'Principal', 'Tutor', 'Educational Counselor', 'Librarian', 'Training Specialist',
    
    # NEGOCIOS Y FINANZAS
    'Financial Analyst', 'Accountant', 'Investment Advisor', 'Business Analyst', 
    'Operations Manager', 'Project Manager', 'Consultant', 'Entrepreneur',
    
    # MARKETING Y VENTAS
    'Marketing Manager', 'Sales Representative', 'Digital Marketing Specialist', 
    'Content Creator', 'Social Media Manager', 'Brand Manager', 'PR Specialist',
    
    # LEGAL Y JURÍDICO
    'Lawyer', 'Paralegal', 'Judge', 'Legal Advisor', 'Notary',
    
    # INGENIERÍA Y CONSTRUCCIÓN
    'Civil Engineer', 'Mechanical Engineer', 'Electrical Engineer', 'Architect', 
    'Construction Manager', 'Urban Planner',
    
    # RECURSOS HUMANOS
    'HR Manager', 'Recruiter', 'HR Analyst', 'Training Coordinator', 'Compensation Specialist',
    
    # COMUNICACIÓN Y MEDIOS
    'Journalist', 'Editor', 'Photographer', 'Video Producer', 'Radio Host', 'Translator',
    
    # SERVICIOS Y ATENCIÓN
    'Customer Service Representative', 'Hotel Manager', 'Travel Agent', 'Event Coordinator', 
    'Restaurant Manager',
    
    # CIENCIAS E INVESTIGACIÓN
    'Research Scientist', 'Laboratory Technician', 'Environmental Scientist', 'Statistician', 
    'Quality Control Analyst',
    
    # ARTE Y CREATIVIDAD
    'Graphic Designer', 'Interior Designer', 'Musician', 'Artist', 'Fashion Designer',
    
    # LOGÍSTICA Y TRANSPORTE
    'Logistics Coordinator', 'Supply Chain Manager', 'Truck Driver', 'Pilot', 'Warehouse Manager'
]

# Plantillas de preguntas por categoría profesional
QUESTION_TEMPLATES = {
    "Software Developer": [
        "¿Qué es {concept} y cómo se implementa?",
        "¿Cuál es la diferencia entre {concept1} y {concept2}?",
        "¿Cómo optimizarías {scenario}?",
        "¿Qué patrones de diseño usarías para {scenario}?",
        "¿Cómo manejarías {error_scenario}?",
        "¿Qué metodología usarías para {project_type}?",
        "¿Cómo asegurarías la calidad en {development_stage}?",
        "¿Qué herramientas utilizarías para {task}?",
        "¿Cómo documentarías {code_component}?",
        "¿Qué consideraciones de seguridad aplicarías en {scenario}?"
    ],
    
    "Doctor": [
        "¿Cómo diagnosticarías {symptoms}?",
        "¿Qué tratamiento recomendarías para {condition}?",
        "¿Cuáles son los signos de {emergency_condition}?",
        "¿Cómo manejarías {patient_scenario}?",
        "¿Qué protocolos seguirías para {medical_procedure}?",
        "¿Cómo comunicarías {diagnosis} al paciente?",
        "¿Qué medidas preventivas recomendarías para {disease}?",
        "¿Cómo manejarías la interacción entre {medication1} y {medication2}?",
        "¿Qué consideraciones éticas aplicarías en {scenario}?",
        "¿Cómo actualizas tus conocimientos sobre {medical_field}?"
    ],
    
    "Teacher": [
        "¿Cómo adaptarías tu enseñanza para {student_type}?",
        "¿Qué estrategias usarías para enseñar {subject}?",
        "¿Cómo manejarías {classroom_challenge}?",
        "¿Qué métodos de evaluación utilizarías para {skill}?",
        "¿Cómo integrarías {technology} en tu clase?",
        "¿Cómo motivarías a estudiantes con {learning_difficulty}?",
        "¿Qué haría si un estudiante {behavioral_issue}?",
        "¿Cómo comunicarías {concern} a los padres?",
        "¿Qué recursos utilizarías para {educational_goal}?",
        "¿Cómo fomentarías {skill} en tus estudiantes?"
    ],
    
    "Lawyer": [
        "¿Cómo abordarías un caso de {legal_area}?",
        "¿Qué precedentes aplicarías en {legal_scenario}?",
        "¿Cómo prepararías {legal_document}?",
        "¿Qué estrategia seguirías para {court_case}?",
        "¿Cómo manejarías {ethical_dilemma}?",
        "¿Qué investigación realizarías para {case_type}?",
        "¿Cómo negociarías {agreement_type}?",
        "¿Qué argumentos presentarías para {legal_position}?",
        "¿Cómo asegurarías {legal_compliance}?",
        "¿Qué consejo darías a un cliente sobre {legal_matter}?"
    ],
    
    "Marketing Manager": [
        "¿Cómo desarrollarías una estrategia para {product_type}?",
        "¿Qué canales utilizarías para llegar a {target_audience}?",
        "¿Cómo medirías el éxito de {campaign_type}?",
        "¿Qué haría si {marketing_challenge}?",
        "¿Cómo segmentarías {market}?",
        "¿Qué presupuesto asignarías a {marketing_channel}?",
        "¿Cómo posicionarías {brand} frente a {competitor}?",
        "¿Qué métricas utilizarías para {marketing_objective}?",
        "¿Cómo adaptarías tu estrategia para {demographic}?",
        "¿Qué tendencias seguirías en {industry}?"
    ]
}

# Conceptos específicos por nivel y profesión
CONCEPTS_BY_PROFESSION = {
    "Software Developer": {
        "junior": ["variables", "funciones", "clases", "arrays", "loops", "condicionales", "debugging", "git", "testing básico", "APIs"],
        "mid": ["arquitectura MVC", "bases de datos", "APIs REST", "testing avanzado", "performance", "seguridad básica", "microservicios", "docker", "CI/CD", "refactoring"],
        "senior": ["arquitectura de sistemas", "escalabilidad", "seguridad avanzada", "cloud computing", "leadership técnico", "code review", "arquitectura distribuida", "monitoring", "disaster recovery", "mentoring"]
    },
    "Doctor": {
        "junior": ["anatomía básica", "signos vitales", "historia clínica", "examen físico", "diagnóstico diferencial", "medicamentos básicos", "emergencias comunes", "comunicación paciente", "ética médica", "procedimientos básicos"],
        "mid": ["patología compleja", "interpretación estudios", "tratamientos especializados", "complicaciones", "manejo dolor", "farmacología avanzada", "procedimientos intermedios", "trabajo en equipo", "calidad atención", "investigación clínica"],
        "senior": ["casos complejos", "liderazgo médico", "protocolos hospitalarios", "supervisión residentes", "innovación médica", "gestión riesgos", "políticas salud", "docencia médica", "investigación avanzada", "administración sanitaria"]
    },
    "Teacher": {
        "junior": ["planificación clases", "manejo aula", "evaluación básica", "metodologías enseñanza", "recursos educativos", "comunicación estudiantes", "motivación estudiantil", "tecnología educativa", "desarrollo curricular", "atención diversidad"],
        "mid": ["pedagogía avanzada", "evaluación integral", "proyectos educativos", "inclusión educativa", "innovación pedagógica", "liderazgo educativo", "investigación educativa", "gestión conflictos", "colaboración colegas", "desarrollo profesional"],
        "senior": ["dirección académica", "políticas educativas", "mentoría docentes", "investigación pedagógica avanzada", "transformación educativa", "calidad educativa", "administración educativa", "innovación institucional", "evaluación institucional", "liderazgo sistémico"]
    }
}

def generate_question(profession, level, template, concepts):
    """Generate a specific question for a profession and level."""
    concept = random.choice(concepts)
    
    # Replace placeholders in template
    question = template.replace("{concept}", concept)
    question = question.replace("{concept1}", random.choice(concepts))
    question = question.replace("{concept2}", random.choice(concepts))
    
    # Add more specific replacements based on profession
    if "Developer" in profession or "Engineer" in profession:
        question = question.replace("{scenario}", random.choice(["una aplicación web", "un sistema distribuido", "una API REST", "un microservicio"]))
        question = question.replace("{error_scenario}", random.choice(["errores de memoria", "fallos de red", "timeout de API", "concurrencia"]))
        
    elif profession == "Doctor":
        question = question.replace("{symptoms}", random.choice(["dolor abdominal agudo", "disnea", "cefalea persistente", "fiebre alta"]))
        question = question.replace("{condition}", random.choice(["diabetes tipo 2", "hipertensión", "depresión", "artritis"]))
        
    elif profession == "Teacher":
        question = question.replace("{student_type}", random.choice(["estudiantes con dificultades de aprendizaje", "estudiantes avanzados", "estudiantes desmotivados"]))
        question = question.replace("{subject}", random.choice(["matemáticas", "ciencias", "historia", "literatura"]))
    
    return question

def expand_dataset():
    """Expand the dataset with questions for all professions."""
    
    # Load existing dataset
    dataset_path = Path("app/datasets/technical_questions_by_profession_v3.jsonl")
    existing_questions = []
    
    if dataset_path.exists():
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        existing_questions.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                        continue
    
    print(f"Existing questions: {len(existing_questions)}")
    
    # Track existing professions
    existing_roles = set(q.get('role', '') for q in existing_questions)
    print(f"Existing roles: {len(existing_roles)}")
    
    new_questions = []
    
    # Generate questions for each profession
    for profession in PROFESSIONS:
        if profession in existing_roles:
            print(f"Skipping {profession} - already exists")
            continue
            
        print(f"Generating questions for {profession}...")
        
        # Get templates for this profession or use generic software dev templates
        templates = QUESTION_TEMPLATES.get(profession, QUESTION_TEMPLATES["Software Developer"])
        
        # Get concepts for this profession or create generic ones
        if profession in CONCEPTS_BY_PROFESSION:
            profession_concepts = CONCEPTS_BY_PROFESSION[profession]
        else:
            # Create generic concepts based on profession type
            profession_concepts = {
                "junior": [f"{profession.lower()} básico", "fundamentos", "herramientas básicas", "procesos estándar", "comunicación"],
                "mid": [f"{profession.lower()} avanzado", "liderazgo", "proyectos complejos", "optimización", "gestión"],
                "senior": [f"{profession.lower()} experto", "estrategia", "mentoría", "innovación", "transformación"]
            }
        
        # Generate questions for each level
        for level in ["junior", "mid", "senior"]:
            concepts = profession_concepts[level]
            questions_per_level = 15 if level != "mid" else 20  # More mid-level questions
            
            for i in range(questions_per_level):
                template = random.choice(templates)
                question = generate_question(profession, level, template, concepts)
                
                new_question = {
                    "type": "technical",
                    "role": profession,
                    "level": level,
                    "question": question,
                    "expected_concepts": random.sample(concepts, min(2, len(concepts))),
                    "difficulty": level
                }
                
                new_questions.append(new_question)
    
    print(f"Generated {len(new_questions)} new questions")
    
    # Combine with existing questions
    all_questions = existing_questions + new_questions
    
    # Write expanded dataset
    backup_path = dataset_path.with_suffix('.jsonl.backup')
    if dataset_path.exists():
        dataset_path.rename(backup_path)
        print(f"Backed up original dataset to {backup_path}")
    
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for question in all_questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"✅ Dataset expanded! Total questions: {len(all_questions)}")
    print(f"📁 Saved to: {dataset_path}")
    
    # Statistics
    roles_count = {}
    for q in all_questions:
        role = q.get('role', 'Unknown')
        roles_count[role] = roles_count.get(role, 0) + 1
    
    print(f"\n📊 Questions per profession:")
    for role, count in sorted(roles_count.items()):
        print(f"  {role}: {count}")

if __name__ == "__main__":
    expand_dataset()
