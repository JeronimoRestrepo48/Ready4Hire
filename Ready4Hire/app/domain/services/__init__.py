"""
Domain Services Module
Servicios de dominio para Ready4Hire
"""

from .language_service import LanguageService, get_language_service, GLOBAL_TERMS
from .text_service import TextService, get_text_service

__all__ = [
    "LanguageService",
    "get_language_service",
    "GLOBAL_TERMS",
    "TextService",
    "get_text_service",
]
