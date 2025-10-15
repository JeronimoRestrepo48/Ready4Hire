"""
Prompt Guard Service
Servicio de detección de ataques de prompt injection
"""
import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class PromptGuard:
    """
    Servicio de protección contra prompt injection y jailbreaking.
    
    Features:
    - Detección de patrones maliciosos
    - Detección de intentos de jailbreak
    - Sistema de scoring de riesgo
    - Logging de intentos de ataque
    """
    
    # Patrones de prompt injection
    INJECTION_PATTERNS = [
        # Intentos de ignorar instrucciones previas
        r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?|commands?)',
        r'disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?|commands?)',
        r'forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?|commands?)',
        
        # Intentos de cambiar el rol del sistema
        r'you\s+are\s+(now|not)\s+(a|an)\s+',
        r'act\s+as\s+(a|an)\s+',
        r'pretend\s+(you|to)\s+',
        r'from\s+now\s+on',
        
        # Intentos de jailbreak comunes
        r'DAN\s+mode',
        r'developer\s+mode',
        r'sudo\s+mode',
        r'admin\s+mode',
        r'unrestricted\s+mode',
        
        # Intentos de extraer información del sistema
        r'show\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)',
        r'what\s+(is|are)\s+(your|the)\s+(system\s+)?(prompt|instructions)',
        r'reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)',
        
        # Intentos de manipulación
        r'execute\s+code',
        r'run\s+code',
        r'\beval\(',
        r'\bexec\(',
    ]
    
    def __init__(self, threshold: float = 0.5):
        """
        Inicializa el guard.
        
        Args:
            threshold: Umbral de riesgo (0.0 a 1.0) para considerar input malicioso
        """
        self.threshold = threshold
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INJECTION_PATTERNS
        ]
    
    def detect(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Detecta intentos de prompt injection.
        
        Args:
            text: Texto a analizar
        
        Returns:
            Tupla (is_malicious, risk_score, matched_patterns)
            - is_malicious: True si se detectó ataque
            - risk_score: Puntuación de riesgo (0.0 a 1.0)
            - matched_patterns: Lista de patrones que coincidieron
        """
        if not text:
            return False, 0.0, []
        
        matched = []
        
        # Buscar patrones maliciosos
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                matched.append(self.INJECTION_PATTERNS[i])
        
        # Calcular score de riesgo
        risk_score = min(len(matched) * 0.25, 1.0)
        is_malicious = risk_score >= self.threshold
        
        if is_malicious:
            logger.warning(
                f"🚨 PROMPT INJECTION DETECTED!\n"
                f"Risk Score: {risk_score:.2f}\n"
                f"Matched Patterns: {len(matched)}\n"
                f"Text Preview: {text[:100]}..."
            )
        
        return is_malicious, risk_score, matched
    
    def check_and_raise(self, text: str):
        """
        Verifica el texto y lanza excepción si es malicioso.
        
        Args:
            text: Texto a verificar
        
        Raises:
            ValueError: Si se detecta prompt injection
        """
        is_malicious, risk_score, patterns = self.detect(text)
        
        if is_malicious:
            raise ValueError(
                f"Prompt injection detectado (riesgo: {risk_score:.2f}). "
                f"Patrones: {', '.join(patterns[:3])}"
            )
    
    def is_safe(self, text: str) -> bool:
        """
        Verifica si el texto es seguro.
        
        Args:
            text: Texto a verificar
        
        Returns:
            True si es seguro, False si es malicioso
        """
        is_malicious, _, _ = self.detect(text)
        return not is_malicious


# Instancia global (singleton)
_guard = None


def get_prompt_guard(threshold: float = 0.5) -> PromptGuard:
    """Obtiene la instancia global del guard"""
    global _guard
    if _guard is None:
        _guard = PromptGuard(threshold)
    return _guard
