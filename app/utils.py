import langid

# Diccionario de términos globales traducibles (puede expandirse)
GLOBAL_TERMS = {
    'en': {
        'docker': 'Docker',
        'pipeline': 'Pipeline',
        'cloud': 'Cloud',
        'database': 'Database',
        'api': 'API',
        'machine learning': 'Machine Learning',
        'devops': 'DevOps',
        'frontend': 'Frontend',
        'backend': 'Backend',
        'security': 'Security',
        'test': 'Test',
        'infrastructure': 'Infrastructure',
        'etl': 'ETL',
        'ci/cd': 'CI/CD',
        'microservices': 'Microservices',
    },
    'es': {
        'docker': 'Docker',
        'pipeline': 'Pipeline',
        'nube': 'Cloud',
        'base de datos': 'Database',
        'api': 'API',
        'aprendizaje automático': 'Machine Learning',
        'devops': 'DevOps',
        'frontend': 'Frontend',
        'backend': 'Backend',
        'seguridad': 'Security',
        'prueba': 'Test',
        'infraestructura': 'Infrastructure',
        'etl': 'ETL',
        'ci/cd': 'CI/CD',
        'microservicios': 'Microservices',
    },
    # Agregar más idiomas y términos según necesidad
}

def detect_language(text: str) -> str:
    """
    Detecta el idioma de un texto usando langid. Retorna el código ISO (ej: 'es', 'en', 'pt', 'fr').
    """
    lang, _ = langid.classify(text)
    return lang

from typing import Optional

def standardize_global_terms(text: str, lang: Optional[str] = None) -> str:
    """
    Reemplaza términos globales en el texto por su versión estándar en inglés para mejorar la compatibilidad NLP.
    Si lang no se especifica, se detecta automáticamente.
    """
    if not lang:
        lang = detect_language(text)
    terms = GLOBAL_TERMS.get(lang, {})
    for k, v in terms.items():
        # Reemplazo insensible a mayúsculas/minúsculas
        text = re.sub(rf'\b{k}\b', v, text, flags=re.IGNORECASE)
    return text

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

def clean_text(text: str, lang: Optional[str] = None) -> str:
    """
    Limpia y normaliza texto para procesamiento NLP multilingüe:
    - Detecta idioma automáticamente si no se especifica
    - Quita tildes, minúsculas, espacios, signos y stopwords del idioma
    - Estandariza términos globales a inglés
    Si NLTK no está disponible, solo hace limpieza básica.
    """
    text = text.strip().replace("\n", " ")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    if not lang:
        lang = detect_language(text)
    text = standardize_global_terms(text, lang)
    if _nltk_ready and word_tokenize:
        try:
            tokens = word_tokenize(text)
            # Usa stopwords del idioma detectado si existen
            stops = set()
            try:
                if stopwords is not None:
                    stops = set(stopwords.words(lang))
            except Exception:
                pass
            tokens = [t for t in tokens if t not in stops]
            text = ' '.join(tokens)
        except Exception:
            pass
    return text.strip()

def remove_stopwords(tokens: List[str], lang: Optional[str] = None) -> List[str]:
    """
    Elimina stopwords del idioma especificado o detectado automáticamente.
    Si NLTK no está disponible, retorna tokens sin filtrar.
    """
    if not lang and tokens:
        # Detecta idioma a partir de los tokens unidos
        lang_detected = detect_language(' '.join(tokens))
        lang = lang_detected if isinstance(lang_detected, str) else 'spanish'
    elif not lang:
        lang = 'spanish'
    if _nltk_ready and stopwords is not None and isinstance(lang, str):
        try:
            stops = set(stopwords.words(lang))
            return [t for t in tokens if t not in stops]
        except Exception:
            return tokens
    return tokens

def normalize_unicode(text: str) -> str:
    """Normaliza texto unicode a ASCII simple."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
