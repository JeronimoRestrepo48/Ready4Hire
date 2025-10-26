"""
Profession Domain Entity
Sistema expandido de profesiones y habilidades
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from enum import Enum


class ProfessionCategory(Enum):
    """Categorías principales de profesiones"""

    TECHNOLOGY = "technology"
    BUSINESS = "business"
    CREATIVE = "creative"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENGINEERING = "engineering"
    SCIENCE = "science"
    LEGAL = "legal"
    FINANCE = "finance"
    MARKETING = "marketing"
    SALES = "sales"
    HUMAN_RESOURCES = "human_resources"
    OPERATIONS = "operations"
    CUSTOMER_SERVICE = "customer_service"
    DESIGN = "design"
    MEDIA = "media"
    CONSTRUCTION = "construction"
    MANUFACTURING = "manufacturing"
    LOGISTICS = "logistics"
    HOSPITALITY = "hospitality"
    OTHER = "other"


@dataclass
class Skill:
    """Habilidad técnica o blanda"""

    id: str
    name: str
    category: str  # technical, soft, domain
    description: str
    professions: List[str] = field(default_factory=list)
    level: str = "beginner"  # beginner, intermediate, advanced, expert
    related_skills: List[str] = field(default_factory=list)


@dataclass
class Profession:
    """Profesión completa con habilidades asociadas"""

    id: str
    name: str
    category: ProfessionCategory
    description: str
    technical_skills: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)
    common_tools: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    career_paths: List[str] = field(default_factory=list)
    related_professions: List[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"
    average_salary_range: str = ""
    demand_level: str = "medium"  # low, medium, high
    remote_friendly: bool = False


# ============================================================================
# COMPREHENSIVE PROFESSION DATABASE
# ============================================================================

TECH_SKILLS = {
    # Programming Languages
    "python": "Python Programming",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "csharp": "C#",
    "cpp": "C++",
    "go": "Go",
    "rust": "Rust",
    "php": "PHP",
    "ruby": "Ruby",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "scala": "Scala",
    "r": "R Programming",
    "sql": "SQL",
    # Frameworks & Libraries
    "react": "React",
    "angular": "Angular",
    "vue": "Vue.js",
    "nodejs": "Node.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring Framework",
    "dotnet": ".NET",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit": "Scikit-learn",
    # Databases
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "elasticsearch": "Elasticsearch",
    "cassandra": "Cassandra",
    # Cloud & DevOps
    "aws": "Amazon Web Services",
    "azure": "Microsoft Azure",
    "gcp": "Google Cloud Platform",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "jenkins": "Jenkins",
    "gitlab_ci": "GitLab CI/CD",
    "github_actions": "GitHub Actions",
    # Data & Analytics
    "data_analysis": "Data Analysis",
    "data_visualization": "Data Visualization",
    "machine_learning": "Machine Learning",
    "deep_learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "computer_vision": "Computer Vision",
    "big_data": "Big Data",
    "hadoop": "Hadoop",
    "spark": "Apache Spark",
    # Security
    "cybersecurity": "Cybersecurity",
    "penetration_testing": "Penetration Testing",
    "security_audit": "Security Auditing",
    "encryption": "Encryption",
    # Mobile
    "ios_dev": "iOS Development",
    "android_dev": "Android Development",
    "react_native": "React Native",
    "flutter": "Flutter",
    # Other
    "git": "Git Version Control",
    "agile": "Agile Methodologies",
    "scrum": "Scrum",
    "api_design": "API Design",
    "microservices": "Microservices",
    "rest": "REST API",
    "graphql": "GraphQL",
    "testing": "Software Testing",
    "ci_cd": "CI/CD",
    "system_design": "System Design",
    # Design & UX
    "design": "Design",
    "prototyping": "Prototyping",
    "user_research": "User Research",
    "illustration": "Illustration",
    "typography": "Typography",
    "html": "HTML",
    "css": "CSS",
    # CMS & Content
    "cms": "Content Management Systems",
    "wordpress": "WordPress",
    # Support & Tools
    "ticketing_systems": "Ticketing Systems",
    "zendesk": "Zendesk",
    "selenium": "Selenium Testing",
    "cypress": "Cypress",
}

SOFT_SKILLS = {
    "communication": "Communication",
    "leadership": "Leadership",
    "teamwork": "Teamwork",
    "problem_solving": "Problem Solving",
    "critical_thinking": "Critical Thinking",
    "creativity": "Creativity",
    "adaptability": "Adaptability",
    "time_management": "Time Management",
    "organization": "Organization",
    "attention_detail": "Attention to Detail",
    "decision_making": "Decision Making",
    "conflict_resolution": "Conflict Resolution",
    "emotional_intelligence": "Emotional Intelligence",
    "negotiation": "Negotiation",
    "presentation": "Presentation Skills",
    "active_listening": "Active Listening",
    "empathy": "Empathy",
    "patience": "Patience",
    "stress_management": "Stress Management",
    "work_ethic": "Work Ethic",
    "initiative": "Initiative",
    "accountability": "Accountability",
    "collaboration": "Collaboration",
    "mentoring": "Mentoring",
    "customer_service": "Customer Service",
    "sales": "Sales Skills",
    "networking": "Networking",
    "public_speaking": "Public Speaking",
    "writing": "Written Communication",
    "analytical_thinking": "Analytical Thinking",
    "research": "Research Skills",
    "coaching": "Coaching",
    "persuasion": "Persuasion",
    "resilience": "Resilience",
    "integrity": "Integrity",
    "strategic_planning": "Strategic Planning",
}

BUSINESS_SKILLS = {
    "project_management": "Project Management",
    "product_management": "Product Management",
    "business_analysis": "Business Analysis",
    "strategic_planning": "Strategic Planning",
    "financial_analysis": "Financial Analysis",
    "budgeting": "Budgeting",
    "accounting": "Accounting",
    "marketing": "Marketing",
    "digital_marketing": "Digital Marketing",
    "seo": "SEO",
    "content_marketing": "Content Marketing",
    "social_media": "Social Media Marketing",
    "brand_management": "Brand Management",
    "market_research": "Market Research",
    "sales_strategy": "Sales Strategy",
    "crm": "CRM Management",
    "supply_chain": "Supply Chain Management",
    "operations": "Operations Management",
    "quality_assurance": "Quality Assurance",
    "risk_management": "Risk Management",
    "compliance": "Compliance",
    "hr_management": "HR Management",
    "recruitment": "Recruitment",
    "training": "Training & Development",
}

# Profesiones principales por categoría
PROFESSIONS_DATABASE = {
    # Technology
    "software_engineer": Profession(
        id="software_engineer",
        name="Software Engineer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Develops and maintains software applications",
        technical_skills=["python", "javascript", "sql", "git", "api_design", "testing"],
        soft_skills=["problem_solving", "teamwork", "communication", "time_management"],
        common_tools=["VS Code", "Git", "Docker", "AWS"],
        remote_friendly=True,
    ),
    "data_scientist": Profession(
        id="data_scientist",
        name="Data Scientist",
        category=ProfessionCategory.TECHNOLOGY,
        description="Analyzes and interprets complex data",
        technical_skills=["python", "r", "machine_learning", "data_analysis", "sql", "tensorflow"],
        soft_skills=["analytical_thinking", "problem_solving", "communication", "creativity"],
        common_tools=["Jupyter", "Pandas", "TensorFlow", "Tableau"],
        remote_friendly=True,
    ),
    "devops_engineer": Profession(
        id="devops_engineer",
        name="DevOps Engineer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Manages infrastructure and deployment pipelines",
        technical_skills=["docker", "kubernetes", "aws", "terraform", "ci_cd", "python"],
        soft_skills=["problem_solving", "communication", "teamwork", "adaptability"],
        common_tools=["Docker", "Kubernetes", "Jenkins", "AWS"],
        remote_friendly=True,
    ),
    # Business
    "product_manager": Profession(
        id="product_manager",
        name="Product Manager",
        category=ProfessionCategory.BUSINESS,
        description="Manages product lifecycle and strategy",
        technical_skills=["product_management", "agile", "data_analysis"],
        soft_skills=["leadership", "communication", "strategic_planning", "decision_making"],
        common_tools=["Jira", "Confluence", "Figma", "Analytics"],
        remote_friendly=True,
    ),
    "business_analyst": Profession(
        id="business_analyst",
        name="Business Analyst",
        category=ProfessionCategory.BUSINESS,
        description="Analyzes business processes and requirements",
        technical_skills=["business_analysis", "data_analysis", "sql", "project_management"],
        soft_skills=["analytical_thinking", "communication", "problem_solving"],
        common_tools=["Excel", "Power BI", "SQL", "Jira"],
        remote_friendly=True,
    ),
    # Frontend Development
    "frontend_developer": Profession(
        id="frontend_developer",
        name="Frontend Developer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Builds user interfaces and experiences",
        technical_skills=["javascript", "react", "html", "css", "typescript", "testing"],
        soft_skills=["creativity", "attention_detail", "problem_solving", "communication"],
        common_tools=["VS Code", "Chrome DevTools", "Figma", "Git"],
        remote_friendly=True,
    ),
    "backend_developer": Profession(
        id="backend_developer",
        name="Backend Developer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Develops server-side logic and APIs",
        technical_skills=["python", "nodejs", "sql", "api_design", "microservices", "docker"],
        soft_skills=["problem_solving", "analytical_thinking", "teamwork", "communication"],
        common_tools=["VS Code", "Postman", "Docker", "Git"],
        remote_friendly=True,
    ),
    "fullstack_developer": Profession(
        id="fullstack_developer",
        name="Full Stack Developer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Works on both frontend and backend systems",
        technical_skills=["javascript", "python", "react", "nodejs", "sql", "docker"],
        soft_skills=["problem_solving", "adaptability", "time_management", "communication"],
        common_tools=["VS Code", "Docker", "Git", "Postman"],
        remote_friendly=True,
    ),
    # Mobile Development
    "mobile_developer_android": Profession(
        id="mobile_developer_android",
        name="Android Developer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Develops native Android applications",
        technical_skills=["kotlin", "java", "android_dev", "sql", "git"],
        soft_skills=["problem_solving", "creativity", "attention_detail", "teamwork"],
        common_tools=["Android Studio", "Git", "Firebase"],
        remote_friendly=True,
    ),
    "mobile_developer_ios": Profession(
        id="mobile_developer_ios",
        name="iOS Developer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Develops native iOS applications",
        technical_skills=["swift", "ios_dev", "git", "api_design"],
        soft_skills=["problem_solving", "creativity", "attention_detail", "patience"],
        common_tools=["Xcode", "Git", "TestFlight"],
        remote_friendly=True,
    ),
    # QA & Testing
    "qa_manual": Profession(
        id="qa_manual",
        name="QA Engineer (Manual)",
        category=ProfessionCategory.TECHNOLOGY,
        description="Tests software manually to find bugs",
        technical_skills=["testing", "sql", "api_design"],
        soft_skills=["attention_detail", "analytical_thinking", "communication", "patience"],
        common_tools=["Jira", "TestRail", "Postman"],
        remote_friendly=True,
    ),
    "qa_automation": Profession(
        id="qa_automation",
        name="QA Engineer (Automation)",
        category=ProfessionCategory.TECHNOLOGY,
        description="Automates testing processes",
        technical_skills=["python", "javascript", "testing", "ci_cd", "selenium"],
        soft_skills=["problem_solving", "attention_detail", "analytical_thinking"],
        common_tools=["Selenium", "Cypress", "Jenkins", "Git"],
        remote_friendly=True,
    ),
    # Security
    "security_analyst": Profession(
        id="security_analyst",
        name="Security Analyst",
        category=ProfessionCategory.TECHNOLOGY,
        description="Protects systems from security threats",
        technical_skills=["cybersecurity", "penetration_testing", "security_audit", "encryption"],
        soft_skills=["analytical_thinking", "problem_solving", "attention_detail", "communication"],
        common_tools=["Wireshark", "Nmap", "Burp Suite", "Metasploit"],
        remote_friendly=True,
    ),
    # Cloud & Infrastructure
    "cloud_architect": Profession(
        id="cloud_architect",
        name="Cloud Architect",
        category=ProfessionCategory.TECHNOLOGY,
        description="Designs cloud infrastructure solutions",
        technical_skills=["aws", "azure", "gcp", "terraform", "kubernetes", "docker"],
        soft_skills=["strategic_planning", "problem_solving", "communication", "leadership"],
        common_tools=["AWS Console", "Terraform", "Kubernetes", "Docker"],
        remote_friendly=True,
    ),
    # Data Roles
    "data_engineer": Profession(
        id="data_engineer",
        name="Data Engineer",
        category=ProfessionCategory.TECHNOLOGY,
        description="Builds and maintains data pipelines",
        technical_skills=["python", "sql", "big_data", "spark", "hadoop"],
        soft_skills=["problem_solving", "analytical_thinking", "attention_detail"],
        common_tools=["Apache Spark", "Airflow", "SQL", "AWS"],
        remote_friendly=True,
    ),
    "data_analyst": Profession(
        id="data_analyst",
        name="Data Analyst",
        category=ProfessionCategory.BUSINESS,
        description="Analyzes data to drive business decisions",
        technical_skills=["sql", "data_analysis", "data_visualization", "python", "r"],
        soft_skills=["analytical_thinking", "communication", "attention_detail", "problem_solving"],
        common_tools=["SQL", "Excel", "Tableau", "Power BI"],
        remote_friendly=True,
    ),
    # Marketing & Sales
    "digital_marketer": Profession(
        id="digital_marketer",
        name="Digital Marketing Specialist",
        category=ProfessionCategory.MARKETING,
        description="Manages digital marketing campaigns",
        technical_skills=["digital_marketing", "seo", "social_media", "content_marketing"],
        soft_skills=["creativity", "communication", "analytical_thinking", "adaptability"],
        common_tools=["Google Analytics", "SEMrush", "HubSpot", "Mailchimp"],
        remote_friendly=True,
    ),
    "sales_representative": Profession(
        id="sales_representative",
        name="Sales Representative",
        category=ProfessionCategory.SALES,
        description="Sells products or services to customers",
        technical_skills=["crm", "sales_strategy"],
        soft_skills=["communication", "negotiation", "persuasion", "active_listening", "resilience"],
        common_tools=["Salesforce", "HubSpot", "LinkedIn"],
        remote_friendly=True,
    ),
    # HR & Recruitment
    "hr_specialist": Profession(
        id="hr_specialist",
        name="HR Specialist",
        category=ProfessionCategory.HUMAN_RESOURCES,
        description="Manages human resources functions",
        technical_skills=["hr_management", "recruitment", "compliance"],
        soft_skills=["communication", "empathy", "organization", "conflict_resolution"],
        common_tools=["BambooHR", "Workday", "LinkedIn Recruiter"],
        remote_friendly=True,
    ),
    # Design
    "ux_designer": Profession(
        id="ux_designer",
        name="UX/UI Designer",
        category=ProfessionCategory.DESIGN,
        description="Designs user experiences and interfaces",
        technical_skills=["design", "prototyping", "user_research"],
        soft_skills=["creativity", "empathy", "communication", "problem_solving"],
        common_tools=["Figma", "Adobe XD", "Sketch", "InVision"],
        remote_friendly=True,
    ),
    "graphic_designer": Profession(
        id="graphic_designer",
        name="Graphic Designer",
        category=ProfessionCategory.DESIGN,
        description="Creates visual content for brands",
        technical_skills=["design", "illustration", "typography"],
        soft_skills=["creativity", "attention_detail", "communication", "time_management"],
        common_tools=["Adobe Photoshop", "Illustrator", "InDesign", "Figma"],
        remote_friendly=True,
    ),
    # Finance
    "financial_analyst": Profession(
        id="financial_analyst",
        name="Financial Analyst",
        category=ProfessionCategory.FINANCE,
        description="Analyzes financial data and trends",
        technical_skills=["financial_analysis", "accounting", "budgeting", "data_analysis"],
        soft_skills=["analytical_thinking", "attention_detail", "communication", "decision_making"],
        common_tools=["Excel", "SAP", "QuickBooks", "Bloomberg Terminal"],
        remote_friendly=True,
    ),
    "accountant": Profession(
        id="accountant",
        name="Accountant",
        category=ProfessionCategory.FINANCE,
        description="Manages financial records and compliance",
        technical_skills=["accounting", "financial_analysis", "compliance", "budgeting"],
        soft_skills=["attention_detail", "organization", "analytical_thinking", "integrity"],
        common_tools=["QuickBooks", "Excel", "SAP", "Xero"],
        remote_friendly=True,
    ),
    # Project Management
    "project_manager_tech": Profession(
        id="project_manager_tech",
        name="Technical Project Manager",
        category=ProfessionCategory.TECHNOLOGY,
        description="Manages technical projects and teams",
        technical_skills=["project_management", "agile", "scrum", "ci_cd"],
        soft_skills=["leadership", "communication", "organization", "problem_solving"],
        common_tools=["Jira", "Confluence", "MS Project", "Slack"],
        remote_friendly=True,
    ),
    "scrum_master": Profession(
        id="scrum_master",
        name="Scrum Master",
        category=ProfessionCategory.TECHNOLOGY,
        description="Facilitates Agile/Scrum processes",
        technical_skills=["agile", "scrum", "project_management"],
        soft_skills=["leadership", "communication", "coaching", "conflict_resolution"],
        common_tools=["Jira", "Confluence", "Miro", "Trello"],
        remote_friendly=True,
    ),
    # Customer Service
    "customer_support": Profession(
        id="customer_support",
        name="Customer Support Specialist",
        category=ProfessionCategory.CUSTOMER_SERVICE,
        description="Provides support to customers",
        technical_skills=["crm", "ticketing_systems"],
        soft_skills=["communication", "empathy", "patience", "problem_solving", "active_listening"],
        common_tools=["Zendesk", "Intercom", "Freshdesk", "Slack"],
        remote_friendly=True,
    ),
    # Content Creation
    "content_writer": Profession(
        id="content_writer",
        name="Content Writer",
        category=ProfessionCategory.CREATIVE,
        description="Creates written content for various platforms",
        technical_skills=["seo", "content_marketing", "cms"],
        soft_skills=["writing", "creativity", "research", "attention_detail"],
        common_tools=["WordPress", "Google Docs", "Grammarly", "Ahrefs"],
        remote_friendly=True,
    ),
}


def get_all_professions() -> List[Profession]:
    """Retorna todas las profesiones disponibles"""
    return list(PROFESSIONS_DATABASE.values())


def get_professions_by_category(category: ProfessionCategory) -> List[Profession]:
    """Retorna profesiones por categoría"""
    return [p for p in PROFESSIONS_DATABASE.values() if p.category == category]


def get_skills_for_profession(profession_id: str) -> Dict[str, List[str]]:
    """Retorna todas las habilidades requeridas para una profesión"""
    if profession_id not in PROFESSIONS_DATABASE:
        return {"technical": [], "soft": [], "business": []}

    prof = PROFESSIONS_DATABASE[profession_id]
    return {"technical": prof.technical_skills, "soft": prof.soft_skills, "business": []}
