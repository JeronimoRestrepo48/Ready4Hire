# Utilidades varias para Ready4Hire


import re
import unicodedata
from typing import List


# Manejo seguro de NLTK y fallback si no hay recursos
word_tokenize = None
stopwords = None
try:
    from nltk.corpus import stopwords as _stopwords
    from nltk.tokenize import word_tokenize as _word_tokenize
    import nltk
    word_tokenize = _word_tokenize
    stopwords = _stopwords
    _nltk_ready = True
    try:
        _spanish_stopwords = set(stopwords.words('spanish'))
    except LookupError:
        nltk.download('stopwords')
        _spanish_stopwords = set(stopwords.words('spanish'))
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    _nltk_ready = False
    _spanish_stopwords = set()
    stopwords = None

def clean_text(text: str) -> str:
    """
    Limpia y normaliza texto para procesamiento NLP: quita tildes, minúsculas, espacios, signos y stopwords.
    Si NLTK no está disponible, solo hace limpieza básica.
    """
    text = text.strip().replace("\n", " ")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    if _nltk_ready and _spanish_stopwords and word_tokenize:
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in _spanish_stopwords]
            text = ' '.join(tokens)
        except Exception:
            pass
    return text.strip()

def remove_stopwords(tokens: List[str], lang: str = 'spanish') -> List[str]:
    """Elimina stopwords del idioma especificado. Si NLTK no está disponible, retorna tokens sin filtrar."""
    if _nltk_ready and stopwords is not None:
        try:
            stops = set(stopwords.words(lang))
            return [t for t in tokens if t not in stops]
        except Exception:
            return tokens
    return tokens

def normalize_unicode(text: str) -> str:
    """Normaliza texto unicode a ASCII simple."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
