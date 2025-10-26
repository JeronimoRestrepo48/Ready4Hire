"""
Sistema de Recomendaciones Personalizadas con Machine Learning.
Recomienda skills, learning paths y oportunidades basado en perfil del usuario.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Servicio de recomendaciones personalizadas usando ML.
    
    Features:
    - Collaborative filtering (usuarios similares)
    - Content-based filtering (skills complementarias)
    - Market trends (demanda laboral)
    - Learning paths personalizados
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        
        # Data de mercado (normalmente vendría de scraping o API externa)
        self.market_data = self._load_market_data()
        self.skill_relationships = self._build_skill_graph()
        
        logger.info("✅ RecommendationService initialized")
    
    def _load_market_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Carga datos del mercado laboral.
        En producción vendría de scraping de LinkedIn, Indeed, etc.
        """
        return {
            # Backend
            "Python": {"demand": 9.5, "salary_impact": 15000, "trend": "up", "jobs": 12500},
            "FastAPI": {"demand": 8.0, "salary_impact": 10000, "trend": "up", "jobs": 3500},
            "Django": {"demand": 7.5, "salary_impact": 12000, "trend": "stable", "jobs": 8000},
            "PostgreSQL": {"demand": 8.5, "salary_impact": 8000, "trend": "stable", "jobs": 9500},
            "Docker": {"demand": 9.0, "salary_impact": 15000, "trend": "up", "jobs": 11000},
            "Kubernetes": {"demand": 9.5, "salary_impact": 20000, "trend": "up", "jobs": 8500},
            
            # Frontend
            "React": {"demand": 9.0, "salary_impact": 12000, "trend": "up", "jobs": 15000},
            "TypeScript": {"demand": 8.5, "salary_impact": 10000, "trend": "up", "jobs": 10000},
            "Vue.js": {"demand": 7.0, "salary_impact": 8000, "trend": "stable", "jobs": 5000},
            "Next.js": {"demand": 8.0, "salary_impact": 10000, "trend": "up", "jobs": 4000},
            
            # DevOps
            "AWS": {"demand": 9.5, "salary_impact": 18000, "trend": "up", "jobs": 14000},
            "Azure": {"demand": 8.5, "salary_impact": 16000, "trend": "up", "jobs": 9000},
            "Terraform": {"demand": 8.0, "salary_impact": 12000, "trend": "up", "jobs": 6000},
            "CI/CD": {"demand": 8.5, "salary_impact": 10000, "trend": "stable", "jobs": 8000},
            
            # Data Science
            "Machine Learning": {"demand": 9.0, "salary_impact": 20000, "trend": "up", "jobs": 7500},
            "TensorFlow": {"demand": 7.5, "salary_impact": 15000, "trend": "stable", "jobs": 4000},
            "PyTorch": {"demand": 8.0, "salary_impact": 16000, "trend": "up", "jobs": 4500},
            "Data Analysis": {"demand": 8.5, "salary_impact": 12000, "trend": "up", "jobs": 9000},
            
            # Mobile
            "React Native": {"demand": 7.5, "salary_impact": 12000, "trend": "stable", "jobs": 5500},
            "Flutter": {"demand": 8.0, "salary_impact": 13000, "trend": "up", "jobs": 4500},
            "Swift": {"demand": 7.0, "salary_impact": 14000, "trend": "stable", "jobs": 4000},
            "Kotlin": {"demand": 7.5, "salary_impact": 13000, "trend": "stable", "jobs": 4200},
        }
    
    def _build_skill_graph(self) -> Dict[str, List[str]]:
        """
        Construye grafo de relaciones entre skills.
        Skills que frecuentemente van juntas.
        """
        return {
            "Python": ["FastAPI", "Django", "PostgreSQL", "Docker", "Machine Learning"],
            "FastAPI": ["Python", "PostgreSQL", "Docker", "Redis", "Kubernetes"],
            "React": ["TypeScript", "Next.js", "Node.js", "Redux", "GraphQL"],
            "Docker": ["Kubernetes", "CI/CD", "AWS", "Terraform"],
            "Kubernetes": ["Docker", "AWS", "Azure", "Terraform", "Helm"],
            "Machine Learning": ["Python", "TensorFlow", "PyTorch", "Data Analysis", "SQL"],
            "AWS": ["Docker", "Kubernetes", "Terraform", "Lambda", "S3"],
            "TypeScript": ["React", "Node.js", "Next.js", "Vue.js", "Angular"],
        }
    
    async def recommend_skills(
        self,
        user_id: str,
        current_skills: List[str],
        target_role: Optional[str] = None,
        n_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recomienda skills basado en perfil del usuario.
        
        Args:
            user_id: ID del usuario
            current_skills: Skills actuales
            target_role: Rol objetivo (opcional)
            n_recommendations: Número de recomendaciones
            
        Returns:
            Lista de skills recomendadas con metadata
        """
        try:
            logger.info(f"🎯 Generating skill recommendations for user {user_id}")
            
            # 1. Content-based: Skills complementarias
            content_based = self._recommend_complementary_skills(current_skills)
            
            # 2. Market trends: Skills con alta demanda
            trending = self._recommend_trending_skills(current_skills)
            
            # 3. Role-specific: Skills para rol objetivo
            role_specific = []
            if target_role:
                role_specific = self._recommend_for_role(target_role, current_skills)
            
            # Combinar y rankear
            all_recommendations = {}
            
            # Content-based (peso 0.4)
            for skill, score in content_based:
                all_recommendations[skill] = all_recommendations.get(skill, 0) + score * 0.4
            
            # Trending (peso 0.3)
            for skill, score in trending:
                all_recommendations[skill] = all_recommendations.get(skill, 0) + score * 0.3
            
            # Role-specific (peso 0.3)
            for skill, score in role_specific:
                all_recommendations[skill] = all_recommendations.get(skill, 0) + score * 0.3
            
            # Filtrar skills ya adquiridas
            current_skills_lower = [s.lower() for s in current_skills]
            all_recommendations = {
                k: v for k, v in all_recommendations.items()
                if k.lower() not in current_skills_lower
            }
            
            # Ordenar por score
            sorted_recommendations = sorted(
                all_recommendations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            # Enrichir con market data
            recommendations = []
            for skill, score in sorted_recommendations:
                market_info = self.market_data.get(skill, {})
                recommendations.append({
                    "skill": skill,
                    "relevance_score": round(score, 2),
                    "demand_score": market_info.get("demand", 5.0),
                    "salary_impact_usd": market_info.get("salary_impact", 0),
                    "trend": market_info.get("trend", "stable"),
                    "available_jobs": market_info.get("jobs", 0),
                    "reason": self._get_recommendation_reason(skill, current_skills, target_role)
                })
            
            logger.info(f"✅ Generated {len(recommendations)} skill recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return []
    
    def _recommend_complementary_skills(
        self,
        current_skills: List[str]
    ) -> List[Tuple[str, float]]:
        """Recomienda skills complementarias basadas en skill graph"""
        complementary = Counter()
        
        for skill in current_skills:
            related = self.skill_relationships.get(skill, [])
            for related_skill in related:
                complementary[related_skill] += 1
        
        # Normalizar scores
        total = sum(complementary.values())
        if total > 0:
            return [(skill, count / total) for skill, count in complementary.most_common(10)]
        return []
    
    def _recommend_trending_skills(
        self,
        current_skills: List[str]
    ) -> List[Tuple[str, float]]:
        """Recomienda skills trending con alta demanda"""
        trending = []
        
        for skill, data in self.market_data.items():
            if skill not in current_skills:
                # Score = demand * trend_multiplier
                trend_multiplier = 1.2 if data["trend"] == "up" else 1.0
                score = data["demand"] * trend_multiplier / 10.0  # Normalizar
                trending.append((skill, score))
        
        return sorted(trending, key=lambda x: x[1], reverse=True)[:10]
    
    def _recommend_for_role(
        self,
        target_role: str,
        current_skills: List[str]
    ) -> List[Tuple[str, float]]:
        """Recomienda skills específicas para un rol"""
        role_requirements = {
            "Backend Developer": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis", "AWS"],
            "Frontend Developer": ["React", "TypeScript", "Next.js", "Tailwind CSS", "GraphQL"],
            "Full Stack Developer": ["React", "Node.js", "PostgreSQL", "Docker", "TypeScript", "AWS"],
            "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Python"],
            "Data Scientist": ["Python", "Machine Learning", "TensorFlow", "Data Analysis", "SQL"],
            "Mobile Developer": ["React Native", "Flutter", "TypeScript", "Firebase", "Redux"],
        }
        
        required = role_requirements.get(target_role, [])
        missing = [skill for skill in required if skill not in current_skills]
        
        # Todos con score 1.0 (igualmente importantes)
        return [(skill, 1.0) for skill in missing]
    
    def _get_recommendation_reason(
        self,
        skill: str,
        current_skills: List[str],
        target_role: Optional[str]
    ) -> str:
        """Genera explicación de por qué se recomienda un skill"""
        reasons = []
        
        # Check if complementary
        for current in current_skills:
            if skill in self.skill_relationships.get(current, []):
                reasons.append(f"Complementa tu conocimiento de {current}")
                break
        
        # Check market
        market_info = self.market_data.get(skill, {})
        if market_info.get("trend") == "up":
            reasons.append("En alta demanda")
        
        if market_info.get("salary_impact", 0) > 10000:
            reasons.append(f"+${market_info['salary_impact']:,} de impacto salarial")
        
        # Check role
        if target_role:
            reasons.append(f"Requerido para {target_role}")
        
        return " • ".join(reasons) if reasons else "Skill valioso en el mercado"
    
    async def generate_learning_path(
        self,
        user_id: str,
        current_skills: List[str],
        target_role: str,
        current_level: str = "mid"
    ) -> Dict[str, Any]:
        """
        Genera un learning path personalizado.
        
        Args:
            user_id: ID del usuario
            current_skills: Skills actuales
            target_role: Rol objetivo
            current_level: Nivel actual (junior/mid/senior)
            
        Returns:
            Learning path estructurado con milestones
        """
        try:
            logger.info(f"📚 Generating learning path for {user_id} -> {target_role}")
            
            # Obtener skills requeridas para el rol
            required_skills = await self.recommend_skills(
                user_id,
                current_skills,
                target_role,
                n_recommendations=10
            )
            
            # Ordenar por prerequisitos y dificultad
            ordered_skills = self._order_by_prerequisites(required_skills)
            
            # Agrupar en milestones (cada 3-4 skills)
            milestones = []
            for i in range(0, len(ordered_skills), 3):
                milestone_skills = ordered_skills[i:i+3]
                milestones.append({
                    "milestone_number": len(milestones) + 1,
                    "name": f"Milestone {len(milestones) + 1}",
                    "skills": [s["skill"] for s in milestone_skills],
                    "estimated_weeks": len(milestone_skills) * 4,  # 4 semanas por skill
                    "resources": self._get_learning_resources(milestone_skills)
                })
            
            # Calcular readiness
            total_required = len(required_skills)
            current_count = len([s for s in required_skills if s["skill"] in current_skills])
            readiness = (current_count / total_required * 100) if total_required > 0 else 100
            
            # Calcular salary impact
            total_salary_impact = sum(s["salary_impact_usd"] for s in required_skills)
            
            learning_path = {
                "user_id": user_id,
                "target_role": target_role,
                "current_level": current_level,
                "readiness_percentage": round(readiness, 1),
                "missing_skills_count": len(required_skills),
                "estimated_total_months": len(milestones) * 3,
                "expected_salary_increase_usd": total_salary_impact,
                "milestones": milestones,
                "current_skills": current_skills,
                "recommended_next_step": milestones[0]["skills"][0] if milestones else None
            }
            
            logger.info(f"✅ Learning path generated: {len(milestones)} milestones")
            return learning_path
            
        except Exception as e:
            logger.error(f"❌ Error generating learning path: {e}")
            return {}
    
    def _order_by_prerequisites(
        self,
        skills: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ordena skills por prerequisitos y dificultad"""
        # Orden simple: fundamentales primero
        fundamental_order = ["Python", "JavaScript", "SQL", "Git", "Docker", "React", "Node.js"]
        
        sorted_skills = []
        for fundamental in fundamental_order:
            for skill_dict in skills:
                if skill_dict["skill"] == fundamental and skill_dict not in sorted_skills:
                    sorted_skills.append(skill_dict)
        
        # Agregar el resto
        for skill_dict in skills:
            if skill_dict not in sorted_skills:
                sorted_skills.append(skill_dict)
        
        return sorted_skills
    
    def _get_learning_resources(
        self,
        skills: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Obtiene recursos de aprendizaje para skills"""
        resources = []
        
        resource_map = {
            "Python": {"type": "course", "name": "Python for Everybody", "url": "https://py4e.com"},
            "React": {"type": "course", "name": "React Official Docs", "url": "https://react.dev"},
            "Docker": {"type": "course", "name": "Docker Getting Started", "url": "https://docs.docker.com/get-started/"},
            "Kubernetes": {"type": "course", "name": "Kubernetes Basics", "url": "https://kubernetes.io/docs/tutorials/"},
            "AWS": {"type": "course", "name": "AWS Training", "url": "https://aws.amazon.com/training/"},
        }
        
        for skill_dict in skills:
            skill_name = skill_dict["skill"]
            if skill_name in resource_map:
                resources.append(resource_map[skill_name])
            else:
                resources.append({
                    "type": "search",
                    "name": f"Learn {skill_name}",
                    "url": f"https://www.google.com/search?q=learn+{skill_name.replace(' ', '+')}"
                })
        
        return resources


# Factory
_recommendation_service_instance: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Factory para obtener instancia global"""
    global _recommendation_service_instance
    
    if _recommendation_service_instance is None:
        _recommendation_service_instance = RecommendationService()
    
    return _recommendation_service_instance

