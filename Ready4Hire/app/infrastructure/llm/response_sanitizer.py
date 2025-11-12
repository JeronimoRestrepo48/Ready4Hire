"""
Servicio de sanitización de respuestas LLM.
Transforma respuestas genéricas de LLM en feedback de agente especializado.
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ResponseSanitizer:
    """
    Sanitiza y mejora respuestas de LLM para que parezcan de un agente especializado,
    no de un modelo de lenguaje genérico.
    
    Elimina:
    - Frases típicas de LLM ("Como modelo de IA...", "No puedo...", etc.)
    - Disculpas excesivas
    - Lenguaje robótico
    - Formato excesivamente estructurado
    
    Mejora:
    - Tono más natural y personalizado
    - Elimina redundancias
    - Hace el feedback más directo y específico
    """

    # Patrones de frases típicas de LLM a eliminar/reemplazar
    LLM_PHRASES_TO_REMOVE = [
        r"como (un )?modelo (de )?IA",
        r"como (un )?asistente (de )?IA",
        r"como (un )?sistema (de )?IA",
        r"no puedo (proporcionar|generar|crear|hacer)",
        r"no tengo (acceso|información|conocimiento)",
        r"no estoy (programado|diseñado|capaz)",
        r"lamento (mucho|profundamente)",
        r"disculpa (por|si)",
        r"perdón (por|si)",
        r"debo (advertir|mencionar|señalar)",
        r"es importante (notar|mencionar|señalar|recordar)",
        r"vale la pena (mencionar|señalar|recordar)",
        r"cabe (mencionar|destacar|señalar)",
        r"en (resumen|conclusión),",
        r"para (resumir|concluir)",
        r"\*\*[^*]+\*\*:",  # Headers en markdown innecesarios
        r"^#+\s+",  # Headers markdown
    ]

    # Patrones de redundancia a eliminar
    REDUNDANCY_PATTERNS = [
        (r"\b(muy|extremadamente|sumamente)\s+(muy|extremadamente|sumamente)\b", r"\1"),
        (r"\b(es|está|son|están)\s+es\b", r"\1"),
        (r"\b(la|el)\s+\1\b", r"\1"),
        (r"\b(para|por|con|de)\s+\1\b", r"\1"),
        (r"\s+", r" "),  # Múltiples espacios
    ]

    # Reemplazos para hacer el tono más natural
    TONE_REPLACEMENTS = [
        (r"\b(debemos|debéis|deben)\b", r"deberías"),
        (r"\b(es crucial|es esencial|es fundamental)\b", r"es importante"),
        (r"\b(es muy importante|es sumamente importante)\b", r"es importante"),
        (r"\b(te recomiendo (fuertemente|encarecidamente))\b", r"te recomiendo"),
        (r"\b(por favor,?\s*)?(intenta|intente|intenten)\b", r"intenta"),
        (r"\b(deberías considerar|sería bueno considerar)\b", r"considera"),
    ]

    def sanitize_feedback(self, feedback: str, role: str = "", category: str = "") -> str:
        """
        Sanitiza feedback de LLM para que parezca de un agente especializado.
        
        Args:
            feedback: Feedback generado por LLM
            role: Rol/profesión (para personalización)
            category: Categoría (technical/soft_skills)
            
        Returns:
            Feedback sanitizado y natural
        """
        if not feedback or not isinstance(feedback, str):
            return feedback

        # Paso 1: Eliminar frases típicas de LLM
        sanitized = self._remove_llm_phrases(feedback)
        
        # Paso 2: Eliminar redundancias
        sanitized = self._remove_redundancies(sanitized)
        
        # Paso 3: Mejorar tono
        sanitized = self._improve_tone(sanitized)
        
        # Paso 4: Personalizar según rol
        sanitized = self._personalize_for_role(sanitized, role, category)
        
        # Paso 5: Limpiar formato
        sanitized = self._clean_formatting(sanitized)
        
        # Paso 6: Asegurar que no quede vacío
        if not sanitized or len(sanitized.strip()) < 10:
            return self._generate_fallback_feedback(role, category)
        
        return sanitized.strip()

    def sanitize_evaluation_response(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza toda la respuesta de evaluación.
        
        Args:
            evaluation: Diccionario con evaluación completa
            
        Returns:
            Evaluación sanitizada
        """
        sanitized = evaluation.copy()
        
        # Sanitizar justification/feedback
        if "justification" in sanitized:
            sanitized["justification"] = self.sanitize_feedback(
                sanitized["justification"],
                role=sanitized.get("role", ""),
                category=sanitized.get("category", "")
            )
        
        if "feedback" in sanitized:
            sanitized["feedback"] = self.sanitize_feedback(
                sanitized["feedback"],
                role=sanitized.get("role", ""),
                category=sanitized.get("category", "")
            )
        
        # Sanitizar strengths y improvements
        if "strengths" in sanitized:
            sanitized["strengths"] = [
                self.sanitize_feedback(s, sanitized.get("role", ""), sanitized.get("category", ""))
                for s in sanitized["strengths"]
            ]
        
        if "improvements" in sanitized:
            sanitized["improvements"] = [
                self.sanitize_feedback(i, sanitized.get("role", ""), sanitized.get("category", ""))
                for i in sanitized["improvements"]
            ]
        
        # Sanitizar hint si existe
        if "hint" in sanitized and sanitized["hint"]:
            sanitized["hint"] = self.sanitize_feedback(
                sanitized["hint"],
                role=sanitized.get("role", ""),
                category=sanitized.get("category", "")
            )
        
        return sanitized

    def _remove_llm_phrases(self, text: str) -> str:
        """Elimina frases típicas de LLM"""
        for pattern in self.LLM_PHRASES_TO_REMOVE:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text

    def _remove_redundancies(self, text: str) -> str:
        """Elimina redundancias y repeticiones"""
        for pattern, replacement in self.REDUNDANCY_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _improve_tone(self, text: str) -> str:
        """Mejora el tono para que sea más natural"""
        for pattern, replacement in self.TONE_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _personalize_for_role(self, text: str, role: str, category: str) -> str:
        """Personaliza el feedback según el rol y categoría"""
        if not role:
            return text
        
        role_lower = role.lower()
        
        # Ajustes específicos por tipo de rol
        if any(tech in role_lower for tech in ["developer", "engineer", "programmer"]):
            # Para roles técnicos, usar terminología más directa
            text = re.sub(r"\bpuedes (mejorar|considerar)\b", r"mejora o considera", text, flags=re.IGNORECASE)
        elif any(mgmt in role_lower for mgmt in ["manager", "leader", "director"]):
            # Para roles de gestión, tono más ejecutivo
            text = re.sub(r"\b(te recomiendo|sugiero)\b", r"considera", text, flags=re.IGNORECASE)
        
        # Ajustes por categoría
        if category == "technical":
            # Eliminar explicaciones innecesarias en feedback técnico
            text = re.sub(r"\b(es decir|en otras palabras|dicho de otra manera)\b", r"", text, flags=re.IGNORECASE)
        
        return text

    def _clean_formatting(self, text: str) -> str:
        """Limpia formato excesivo"""
        # Eliminar múltiples saltos de línea
        text = re.sub(r"\n{3,}", r"\n\n", text)
        
        # Eliminar espacios al inicio/final de líneas
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
        
        # Eliminar listas de viñetas excesivas (más de 5)
        bullet_pattern = r"^[\-\*\+]\s+"
        lines = text.split("\n")
        bullet_count = sum(1 for line in lines if re.match(bullet_pattern, line))
        if bullet_count > 5:
            # Mantener solo las primeras 5
            kept_bullets = 0
            result_lines = []
            for line in lines:
                if re.match(bullet_pattern, line):
                    if kept_bullets < 5:
                        result_lines.append(line)
                        kept_bullets += 1
                else:
                    result_lines.append(line)
            text = "\n".join(result_lines)
        
        return text

    def _generate_fallback_feedback(self, role: str, category: str) -> str:
        """Genera feedback de fallback si la sanitización deja el texto vacío"""
        if category == "technical":
            return f"Tu respuesta muestra un buen inicio. Considera profundizar en los aspectos técnicos específicos de {role}."
        else:
            return f"Has proporcionado una respuesta. Considera agregar ejemplos concretos relevantes para {role}."

