"""
Text Processing Service
Servicio de procesamiento y normalización de texto
"""
import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextService:
    """
    Servicio de procesamiento de texto.
    
    Features:
    - Normalización de texto (lowercase, acentos, etc.)
    - Limpieza de caracteres especiales
    - Truncado inteligente de texto
    - Extracción de palabras clave
    """
    
    def __init__(self):
        """Inicializa el servicio de texto"""
        # Patrones de limpieza
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.multiple_spaces = re.compile(r'\s{2,}')
    
    def normalize(
        self,
        text: str,
        lowercase: bool = True,
        remove_accents: bool = True,
        remove_punctuation: bool = False
    ) -> str:
        """
        Normaliza un texto.
        
        Args:
            text: Texto a normalizar
            lowercase: Si convertir a minúsculas
            remove_accents: Si eliminar acentos
            remove_punctuation: Si eliminar puntuación
        
        Returns:
            Texto normalizado
        """
        if not text:
            return ""
        
        # Convertir a minúsculas
        if lowercase:
            text = text.lower()
        
        # Eliminar acentos
        if remove_accents:
            text = self._remove_accents(text)
        
        # Eliminar puntuación
        if remove_punctuation:
            text = self.punctuation_pattern.sub('', text)
        
        # Normalizar espacios
        text = self.multiple_spaces.sub(' ', text)
        text = text.strip()
        
        return text
    
    def _remove_accents(self, text: str) -> str:
        """
        Elimina acentos de un texto.
        
        Args:
            text: Texto con acentos
        
        Returns:
            Texto sin acentos
        """
        # Normalizar a NFD (descomponer acentos)
        nfd = unicodedata.normalize('NFD', text)
        
        # Eliminar caracteres diacríticos
        without_accents = ''.join(
            char for char in nfd
            if unicodedata.category(char) != 'Mn'
        )
        
        return without_accents
    
    def truncate(
        self,
        text: str,
        max_length: int,
        suffix: str = '...'
    ) -> str:
        """
        Trunca un texto de forma inteligente.
        
        Args:
            text: Texto a truncar
            max_length: Longitud máxima
            suffix: Sufijo a añadir si se trunca
        
        Returns:
            Texto truncado
        """
        if not text or len(text) <= max_length:
            return text
        
        # Truncar en límite de palabra si es posible
        truncated = text[:max_length - len(suffix)]
        
        # Buscar último espacio
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Si está cerca del límite
            truncated = truncated[:last_space]
        
        return truncated + suffix
    
    def extract_keywords(
        self,
        text: str,
        min_length: int = 3,
        max_keywords: int = 10
    ) -> list:
        """
        Extrae palabras clave de un texto.
        
        Args:
            text: Texto a analizar
            min_length: Longitud mínima de palabra
            max_keywords: Número máximo de keywords
        
        Returns:
            Lista de palabras clave (ordenadas por frecuencia)
        """
        if not text:
            return []
        
        # Normalizar texto
        normalized = self.normalize(
            text,
            lowercase=True,
            remove_accents=True,
            remove_punctuation=True
        )
        
        # Extraer palabras
        words = normalized.split()
        
        # Filtrar por longitud
        words = [w for w in words if len(w) >= min_length]
        
        # Eliminar stopwords comunes (ES/EN)
        stopwords = {
            # Español
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se',
            'no', 'haber', 'por', 'con', 'su', 'para', 'como', 'estar',
            'tener', 'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o',
            'poder', 'decir', 'este', 'ir', 'otro', 'ese', 'la', 'si',
            'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
            'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno',
            # Inglés
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
        }
        
        words = [w for w in words if w not in stopwords]
        
        # Contar frecuencia
        from collections import Counter
        word_counts = Counter(words)
        
        # Retornar top keywords
        top_keywords = [
            word for word, count in word_counts.most_common(max_keywords)
        ]
        
        return top_keywords
    
    def clean(self, text: str) -> str:
        """
        Limpieza básica de texto.
        
        Args:
            text: Texto a limpiar
        
        Returns:
            Texto limpio
        """
        if not text:
            return ""
        
        # Eliminar espacios múltiples
        text = self.multiple_spaces.sub(' ', text)
        
        # Eliminar espacios al inicio/final
        text = text.strip()
        
        return text
    
    def is_empty(self, text: str) -> bool:
        """
        Verifica si un texto está vacío (o solo espacios).
        
        Args:
            text: Texto a verificar
        
        Returns:
            True si está vacío
        """
        return not text or not text.strip()


# Instancia global (singleton)
_text_service = None


def get_text_service() -> TextService:
    """Obtiene la instancia global del servicio de texto"""
    global _text_service
    if _text_service is None:
        _text_service = TextService()
    return _text_service
