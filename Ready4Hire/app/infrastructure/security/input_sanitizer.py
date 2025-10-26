"""
Input Sanitizer Service
Servicio de sanitización de entradas de usuario
"""

import re
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """
    Servicio de sanitización de inputs del usuario.

    Features:
    - Limpieza de caracteres especiales
    - Normalización de espacios
    - Truncado de texto muy largo
    - Eliminación de código malicioso
    """

    def __init__(self, max_length: int = 5000, strip_html: bool = True, strip_scripts: bool = True):
        """
        Inicializa el sanitizador.

        Args:
            max_length: Longitud máxima de texto
            strip_html: Si eliminar tags HTML
            strip_scripts: Si eliminar scripts
        """
        self.max_length = max_length
        self.strip_html = strip_html
        self.strip_scripts = strip_scripts

        # Patrones de limpieza
        self.html_pattern = re.compile(r"<[^>]+>")
        self.script_pattern = re.compile(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", re.IGNORECASE)
        self.excessive_spaces = re.compile(r"\s+")

    def sanitize(self, text: str) -> str:
        """
        Sanitiza un texto de entrada.

        Args:
            text: Texto a sanitizar

        Returns:
            Texto sanitizado
        """
        if not text:
            return ""

        # Truncar si es muy largo
        if len(text) > self.max_length:
            logger.warning(f"⚠️ Texto truncado de {len(text)} a {self.max_length} caracteres")
            text = text[: self.max_length]

        # Eliminar scripts
        if self.strip_scripts:
            text = self.script_pattern.sub("", text)

        # Eliminar HTML tags
        if self.strip_html:
            text = self.html_pattern.sub("", text)

        # Normalizar espacios
        text = self.excessive_spaces.sub(" ", text)

        # Eliminar espacios al inicio/final
        text = text.strip()

        return text

    def sanitize_dict(self, data: dict) -> dict:
        """
        Sanitiza todos los valores string de un diccionario.

        Args:
            data: Diccionario a sanitizar

        Returns:
            Diccionario con valores sanitizados
        """
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self.sanitize(v) if isinstance(v, str) else v for v in value]
            else:
                sanitized[key] = value

        return sanitized


# Instancia global (singleton)
_sanitizer = None


def get_sanitizer(max_length: int = 5000, strip_html: bool = True, strip_scripts: bool = True) -> InputSanitizer:
    """Obtiene la instancia global del sanitizador"""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = InputSanitizer(max_length, strip_html, strip_scripts)
    return _sanitizer
