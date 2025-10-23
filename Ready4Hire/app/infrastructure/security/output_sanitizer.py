"""
Output Sanitizer para respuestas del LLM.
Previene que outputs maliciosos o inapropiados lleguen al frontend.
"""
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class OutputSanitizer:
    """
    Sanitiza outputs del LLM para prevenir:
    - Código malicioso (scripts, HTML peligroso)
    - Información sensible leaked (API keys, passwords)
    - Instrucciones system prompt leaked
    - Contenido inapropiado
    """
    
    # Patrones peligrosos
    DANGEROUS_PATTERNS = [
        # Scripts
        (r'<script[^>]*>.*?</script>', 'Inline script detectado', ''),
        (r'<iframe[^>]*>.*?</iframe>', 'Iframe detectado', ''),
        (r'javascript:', 'JavaScript URL detectado', ''),
        (r'on\w+\s*=', 'Event handler HTML detectado', ''),
        
        # System prompt leakage
        (r'You are (a|an|the) (AI|assistant|system|bot).*?(?=\n|$)', 'System prompt leakage', '[REDACTED]'),
        (r'(ignore|disregard) (previous|all) instructions', 'Prompt injection attempt', '[BLOCKED]'),
        
        # Información sensible
        (r'\b[A-Za-z0-9]{32,}\b', 'Posible API key/token', '[REDACTED-TOKEN]'),  # Tokens largos
        (r'password[\s:=]+[\w!@#$%^&*]+', 'Password detectado', '[REDACTED-PASSWORD]'),
        (r'sk-[A-Za-z0-9]{32,}', 'OpenAI API key detectado', '[REDACTED-API-KEY]'),
        
        # SQL/Command injection
        (r';\s*(DROP|DELETE|UPDATE|INSERT)\s+', 'SQL command detectado', ''),
        (r'\|\s*(rm|cat|ls|grep|curl)\s+', 'Shell command detectado', ''),
    ]
    
    # Palabras prohibidas (contenido inapropiado)
    BLOCKED_WORDS = [
        # Agregar según necesidad
        # 'palabra_prohibida',
    ]
    
    # Límites
    MAX_OUTPUT_LENGTH = 10000  # Caracteres máximos
    MAX_LINE_LENGTH = 500  # Evitar líneas extremadamente largas
    
    def __init__(
        self,
        strict_mode: bool = False,
        log_violations: bool = True,
        block_urls: bool = False
    ):
        """
        Inicializa el sanitizador de outputs.
        
        Args:
            strict_mode: Si True, aplica reglas más estrictas
            log_violations: Si True, loggea violaciones detectadas
            block_urls: Si True, elimina todas las URLs
        """
        self.strict_mode = strict_mode
        self.log_violations = log_violations
        self.block_urls = block_urls
        
        self.violations = []  # Historial de violaciones
    
    def sanitize(self, text: str) -> str:
        """
        Sanitiza output del LLM.
        
        Args:
            text: Texto a sanitizar
        
        Returns:
            Texto sanitizado y seguro
        """
        if not text:
            return ""
        
        original_length = len(text)
        
        # 1. Truncar si es muy largo
        if len(text) > self.MAX_OUTPUT_LENGTH:
            self._log_violation(
                f"Output demasiado largo: {len(text)} > {self.MAX_OUTPUT_LENGTH}",
                "length_exceeded"
            )
            text = text[:self.MAX_OUTPUT_LENGTH] + "... [truncado]"
        
        # 2. Aplicar patrones de limpieza
        for pattern, reason, replacement in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                self._log_violation(reason, "dangerous_pattern")
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)
        
        # 3. Bloquear palabras prohibidas
        for word in self.BLOCKED_WORDS:
            if word.lower() in text.lower():
                self._log_violation(f"Palabra bloqueada: {word}", "blocked_word")
                text = re.sub(
                    rf'\b{re.escape(word)}\b',
                    '[BLOQUEADO]',
                    text,
                    flags=re.IGNORECASE
                )
        
        # 4. Bloquear URLs si está configurado
        if self.block_urls:
            url_pattern = r'https?://\S+'
            if re.search(url_pattern, text):
                self._log_violation("URL detectada", "url_blocked")
                text = re.sub(url_pattern, '[URL REMOVIDA]', text)
        
        # 5. Normalizar líneas muy largas
        lines = text.split('\n')
        sanitized_lines = []
        for line in lines:
            if len(line) > self.MAX_LINE_LENGTH:
                # Truncar líneas muy largas (posible attack)
                line = line[:self.MAX_LINE_LENGTH] + "..."
            sanitized_lines.append(line)
        text = '\n'.join(sanitized_lines)
        
        # 6. Normalizar espacios excesivos
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Máx 3 newlines
        text = re.sub(r' {3,}', '  ', text)  # Máx 2 espacios
        
        # 7. Remover caracteres de control peligrosos
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        final_length = len(text)
        
        if self.log_violations and original_length != final_length:
            logger.info(
                f"✂️ Output sanitizado: {original_length} → {final_length} chars "
                f"({original_length - final_length} removidos)"
            )
        
        return text.strip()
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza todos los valores string de un diccionario.
        
        Args:
            data: Diccionario con outputs del LLM
        
        Returns:
            Diccionario sanitizado
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _log_violation(self, reason: str, violation_type: str):
        """Registra una violación detectada."""
        violation = {
            "reason": reason,
            "type": violation_type,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        self.violations.append(violation)
        
        if self.log_violations:
            logger.warning(f"⚠️ Output violation: {reason} (type={violation_type})")
    
    def get_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene las últimas violaciones detectadas.
        
        Args:
            limit: Número máximo de violaciones a retornar
        
        Returns:
            Lista de violaciones
        """
        return self.violations[-limit:]
    
    def clear_violations(self):
        """Limpia el historial de violaciones."""
        self.violations = []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sanitizador.
        
        Returns:
            Dict con estadísticas
        """
        if not self.violations:
            return {
                "total_violations": 0,
                "by_type": {}
            }
        
        by_type = {}
        for v in self.violations:
            vtype = v["type"]
            by_type[vtype] = by_type.get(vtype, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "by_type": by_type,
            "most_common": max(by_type, key=by_type.get) if by_type else None
        }


# ============================================================================
# Instancia global (singleton)
# ============================================================================

_output_sanitizer: Optional[OutputSanitizer] = None


def get_output_sanitizer(
    strict_mode: bool = False,
    log_violations: bool = True,
    block_urls: bool = False
) -> OutputSanitizer:
    """Obtiene la instancia global del sanitizador de outputs."""
    global _output_sanitizer
    if _output_sanitizer is None:
        _output_sanitizer = OutputSanitizer(
            strict_mode=strict_mode,
            log_violations=log_violations,
            block_urls=block_urls
        )
        logger.info("✅ OutputSanitizer inicializado")
    return _output_sanitizer

