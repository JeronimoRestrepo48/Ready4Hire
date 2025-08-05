# audio_utils.py
"""
Utilidades para Speech-to-Text (STT) y Text-to-Speech (TTS) en el backend.
"""
import tempfile
from fastapi import UploadFile
import os


# Carga global del modelo Whisper para evitar recarga en cada llamada
WHISPER_AVAILABLE = False
WHISPER_MODEL = None
try:
    import whisper
    WHISPER_MODEL = whisper.load_model('base', device='cpu') # type: ignore
    WHISPER_AVAILABLE = True
except ImportError:
    pass
except Exception:
    pass

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

def transcribe_audio(file: UploadFile, lang: str = 'es') -> str:
    if not WHISPER_AVAILABLE or WHISPER_MODEL is None:
        raise RuntimeError('Whisper no está instalado o no se pudo cargar el modelo')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        result = WHISPER_MODEL.transcribe(tmp_path, language=lang)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return result['text']

from typing import Optional

def synthesize_text(text: str, lang: str = 'es', out_path: Optional[str] = None) -> str:
    if not PYTTSX3_AVAILABLE:
        raise RuntimeError('pyttsx3 no está instalado')
    import pyttsx3
    engine = pyttsx3.init()
    try:
        voices = engine.getProperty('voices')
        selected_voice = None
        from collections.abc import Iterable
        if isinstance(voices, Iterable):
            for v in voices:
                langs = []
                try:
                    langs = [l.decode('utf-8') if isinstance(l, bytes) else l for l in getattr(v, 'languages', [])]
                except Exception:
                    pass
                if any(lang in l for l in langs) or lang in v.id:
                    selected_voice = v.id
                    break
        if selected_voice:
            engine.setProperty('voice', selected_voice)
        if not out_path:
            out_path = tempfile.mktemp(suffix='.wav')
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path
    finally:
        try:
            engine.stop()
        except Exception:
            pass
