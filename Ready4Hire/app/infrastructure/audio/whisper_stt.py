"""
Whisper Speech-to-Text Service
Servicio de transcripción de audio usando OpenAI Whisper
"""

import tempfile
import os
import logging
from typing import Optional
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

# Lazy loading del modelo Whisper
_WHISPER_MODEL = None
_WHISPER_AVAILABLE = False


def _load_whisper_model():
    """Carga el modelo Whisper (lazy loading)"""
    global _WHISPER_MODEL, _WHISPER_AVAILABLE

    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    try:
        import whisper

        # Verificar que el módulo tenga el método load_model
        if not hasattr(whisper, "load_model"):
            logger.error(
                "❌ El módulo 'whisper' no tiene 'load_model'. "
                "Asegúrate de tener instalado 'openai-whisper' (no 'whisper'). "
                "Instala con: pip uninstall whisper && pip install openai-whisper"
            )
            _WHISPER_AVAILABLE = False
            return None

        _WHISPER_MODEL = whisper.load_model("base", device="cpu")  # type: ignore
        _WHISPER_AVAILABLE = True
        logger.info("✅ Whisper model loaded successfully")
        return _WHISPER_MODEL
    except ImportError:
        logger.error("❌ Whisper not installed. Install with: pip install openai-whisper")
        _WHISPER_AVAILABLE = False
        return None
    except Exception as e:
        logger.error(f"❌ Error loading Whisper model: {str(e)}")
        _WHISPER_AVAILABLE = False
        return None


class WhisperSTT:
    """
    Speech-to-Text service usando OpenAI Whisper.

    Features:
    - Transcripción de audio a texto
    - Soporte multilenguaje (ES/EN)
    - Lazy loading del modelo
    - Manejo de errores robusto
    """

    def __init__(self):
        """Inicializa el servicio STT"""
        self.model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Asegura que el modelo esté cargado"""
        if not self._initialized:
            self.model = _load_whisper_model()
            self._initialized = True

            if self.model is None:
                raise RuntimeError("Whisper no está disponible. " "Instala con: pip install openai-whisper")

    def transcribe(self, audio_file: UploadFile, language: str = "es") -> str:
        """
        Transcribe un archivo de audio a texto.

        Args:
            audio_file: Archivo de audio subido
            language: Idioma del audio ('es', 'en', etc.)

        Returns:
            Texto transcrito

        Raises:
            HTTPException: Si hay error en la transcripción
        """
        self._ensure_initialized()

        # Validar tipo de archivo (más permisivo para webm desde navegador)
        import mimetypes

        mime, _ = mimetypes.guess_type(audio_file.filename or "")
        
        # Aceptar audio/webm desde navegadores además de otros formatos
        content_type = getattr(audio_file, "content_type", None) or ""
        is_audio = (
            (mime and mime.startswith("audio")) or
            content_type.startswith("audio/") or
            audio_file.filename and audio_file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"))
        )

        if not is_audio:
            raise HTTPException(
                status_code=400, 
                detail=f"El archivo debe ser de tipo audio. Recibido: {content_type or mime or 'desconocido'}"
            )

        # Determinar extensión del archivo
        filename = audio_file.filename or "audio"
        file_ext = None
        if "." in filename:
            file_ext = filename.rsplit(".", 1)[1].lower()
        
        # Usar extensión apropiada (webm necesita conversión, pero Whisper puede manejarlo)
        # Whisper soporta múltiples formatos, usar .wav como fallback
        suffix = f".{file_ext}" if file_ext in ["wav", "mp3", "m4a", "ogg", "flac"] else ".wav"
        
        # Guardar temporalmente el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.file.read())
            tmp_path = tmp.name

        try:
            logger.info(f"🎤 Transcribiendo audio en idioma: {language}, formato: {suffix}")

            # Transcribir con Whisper
            if self.model is None:
                raise RuntimeError("Whisper model is not loaded")

            # Whisper puede manejar múltiples formatos, pero si es webm puede necesitar conversión
            # Por ahora intentamos directamente (Whisper moderno puede manejar webm)
            result = self.model.transcribe(tmp_path, language=language if language != "auto" else None)
            text = str(result["text"]).strip()

            logger.info(f"✅ Transcripción exitosa: {len(text)} caracteres")
            return text

        except Exception as e:
            logger.error(f"❌ Error en transcripción: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al transcribir audio: {str(e)}")

        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def is_available(self) -> bool:
        """Verifica si Whisper está disponible"""
        try:
            self._ensure_initialized()
            return self.model is not None
        except Exception:
            return False

    def get_supported_languages(self) -> list:
        """Retorna lista de idiomas soportados"""
        return [
            "es",
            "en",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
            # Whisper soporta ~100 idiomas, estos son los más comunes
        ]


# Instancia global (singleton)
_stt_service = None


def get_stt_service() -> WhisperSTT:
    """Obtiene la instancia global del servicio STT"""
    global _stt_service
    if _stt_service is None:
        _stt_service = WhisperSTT()
    return _stt_service
