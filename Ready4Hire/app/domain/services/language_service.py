"""
Language Detection Service
Servicio de detección de idioma usando langid
"""

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Verificar disponibilidad de langid
_LANGID_AVAILABLE = False
try:
    import langid

    _LANGID_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ langid not available. Install with: pip install langid")


# Términos globales multilenguaje
GLOBAL_TERMS = {
    "es": {
        "greeting": "hola",
        "farewell": "adiós",
        "thanks": "gracias",
        "yes": "sí",
        "no": "no",
        "help": "ayuda",
        "start": "comenzar",
        "end": "terminar",
    },
    "en": {
        "greeting": "hello",
        "farewell": "goodbye",
        "thanks": "thanks",
        "yes": "yes",
        "no": "no",
        "help": "help",
        "start": "start",
        "end": "end",
    },
    "fr": {
        "greeting": "bonjour",
        "farewell": "au revoir",
        "thanks": "merci",
        "yes": "oui",
        "no": "non",
        "help": "aide",
        "start": "commencer",
        "end": "terminer",
    },
}


class LanguageService:
    """
    Servicio de detección y manejo de idiomas.

    Features:
    - Detección automática de idioma (langid)
    - Diccionario de términos globales
    - Soporte multilenguaje (ES/EN/FR)
    - Fallback a español si no detecta
    """

    def __init__(self, default_language: str = "es"):
        """
        Inicializa el servicio de idiomas.

        Args:
            default_language: Idioma por defecto ('es', 'en', 'fr')
        """
        self.default_language = default_language
        self._available = _LANGID_AVAILABLE

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detecta el idioma de un texto.

        Args:
            text: Texto a analizar

        Returns:
            Tupla (language_code, confidence)
            - language_code: Código ISO 639-1 (ej: 'es', 'en')
            - confidence: Nivel de confianza (0.0 a 1.0)
        """
        if not text or not text.strip():
            return self.default_language, 0.0

        if not self._available:
            logger.warning("⚠️ langid no disponible, usando idioma por defecto")
            return self.default_language, 0.5

        try:
            import langid

            lang_code, confidence = langid.classify(text)

            logger.info(f"🌐 Idioma detectado: {lang_code} " f"(confianza: {confidence:.2f})")

            return lang_code, confidence

        except Exception as e:
            logger.error(f"❌ Error detectando idioma: {str(e)}")
            return self.default_language, 0.0

    def detect_simple(self, text: str) -> str:
        """
        Detecta el idioma y retorna solo el código.

        Args:
            text: Texto a analizar

        Returns:
            Código de idioma ('es', 'en', etc.)
        """
        lang_code, _ = self.detect(text)
        return lang_code

    def get_term(self, term_key: str, language: Optional[str] = None) -> str:
        """
        Obtiene un término en el idioma especificado.

        Args:
            term_key: Clave del término (ej: 'greeting', 'thanks')
            language: Código de idioma (None = default)

        Returns:
            Término traducido
        """
        lang = language or self.default_language

        if lang not in GLOBAL_TERMS:
            logger.warning(f"⚠️ Idioma '{lang}' no soportado, usando '{self.default_language}'")
            lang = self.default_language

        terms = GLOBAL_TERMS[lang]

        if term_key not in terms:
            logger.warning(f"⚠️ Término '{term_key}' no encontrado para idioma '{lang}'")
            return term_key

        return terms[term_key]

    def get_all_terms(self, language: Optional[str] = None) -> dict:
        """
        Obtiene todos los términos de un idioma.

        Args:
            language: Código de idioma (None = default)

        Returns:
            Diccionario de términos
        """
        lang = language or self.default_language

        if lang not in GLOBAL_TERMS:
            lang = self.default_language

        return GLOBAL_TERMS[lang].copy()

    def is_available(self) -> bool:
        """Verifica si langid está disponible"""
        return self._available

    def get_supported_languages(self) -> list:
        """Retorna lista de idiomas soportados"""
        return list(GLOBAL_TERMS.keys())


# Instancia global (singleton)
_language_service = None


def get_language_service(default_language: str = "es") -> LanguageService:
    """Obtiene la instancia global del servicio de idiomas"""
    global _language_service
    if _language_service is None:
        _language_service = LanguageService(default_language)
    return _language_service
