"""
pyttsx3 Text-to-Speech Service
Servicio de síntesis de voz usando pyttsx3
"""

import logging
from typing import Optional
from fastapi import HTTPException
import tempfile
import os

logger = logging.getLogger(__name__)

# Verificar disponibilidad de pyttsx3
_PYTTSX3_AVAILABLE = False
try:
    import pyttsx3

    _PYTTSX3_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ pyttsx3 not available. Install with: pip install pyttsx3")


class Pyttsx3TTS:
    """
    Text-to-Speech service usando pyttsx3.

    Features:
    - Síntesis de texto a audio
    - Soporte multilenguaje (ES/EN)
    - Configuración de velocidad y volumen
    - Generación de archivos MP3/WAV
    """

    def __init__(self):
        """Inicializa el servicio TTS"""
        self.engine = None
        self._initialized = False
        self._available = _PYTTSX3_AVAILABLE

    def _ensure_initialized(self):
        """Asegura que el engine esté inicializado"""
        if not self._available:
            raise RuntimeError("pyttsx3 no está disponible. " "Instala con: pip install pyttsx3")

        if not self._initialized:
            try:
                import pyttsx3

                self.engine = pyttsx3.init()
                self._initialized = True
                logger.info("✅ pyttsx3 engine initialized")
            except Exception as e:
                logger.error(f"❌ Error initializing pyttsx3: {str(e)}")
                raise RuntimeError(f"Error inicializando TTS: {str(e)}")

    def synthesize(
        self, text: str, language: str = "es", rate: int = 150, volume: float = 1.0, output_format: str = "mp3"
    ) -> str:
        """
        Sintetiza texto a audio y retorna la ruta del archivo.

        Args:
            text: Texto a sintetizar
            language: Idioma ('es' o 'en')
            rate: Velocidad de habla (palabras por minuto, default 150)
            volume: Volumen (0.0 a 1.0, default 1.0)
            output_format: Formato de salida ('mp3' o 'wav')

        Returns:
            Ruta del archivo de audio generado

        Raises:
            HTTPException: Si hay error en la síntesis
        """
        self._ensure_initialized()

        if self.engine is None:
            raise HTTPException(status_code=500, detail="TTS engine no inicializado correctamente")

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

        try:
            logger.info(f"🔊 Sintetizando texto: {len(text)} caracteres")

            # Configurar propiedades
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)

            # Configurar voz según idioma
            voices = self.engine.getProperty("voices")
            if voices and isinstance(voices, (list, tuple)):
                if language == "es" and len(voices) > 1:
                    self.engine.setProperty("voice", voices[1].id)  # Voz en español
                elif language == "en" and len(voices) > 0:
                    self.engine.setProperty("voice", voices[0].id)  # Voz en inglés

            # Crear archivo temporal
            suffix = f".{output_format}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name

            # Generar audio
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()

            logger.info(f"✅ Audio generado: {tmp_path}")
            return tmp_path

        except Exception as e:
            logger.error(f"❌ Error en síntesis TTS: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al sintetizar audio: {str(e)}")

    def synthesize_to_bytes(self, text: str, language: str = "es", rate: int = 150, volume: float = 1.0) -> bytes:
        """
        Sintetiza texto y retorna bytes del audio.

        Args:
            text: Texto a sintetizar
            language: Idioma
            rate: Velocidad
            volume: Volumen

        Returns:
            Bytes del audio generado
        """
        audio_path = self.synthesize(text, language, rate, volume)

        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(audio_path)
            except Exception:
                pass

    def is_available(self) -> bool:
        """Verifica si pyttsx3 está disponible"""
        return self._available

    def get_voices(self) -> list:
        """Retorna lista de voces disponibles"""
        if not self.is_available():
            return []

        try:
            self._ensure_initialized()
            if self.engine is None:
                return []
            voices = self.engine.getProperty("voices")
            if not voices or not isinstance(voices, (list, tuple)):
                return []
            return [{"id": v.id, "name": v.name, "languages": v.languages} for v in voices]
        except Exception:
            return []


# Instancia global (singleton)
_tts_service = None


def get_tts_service() -> Pyttsx3TTS:
    """Obtiene la instancia global del servicio TTS"""
    global _tts_service
    if _tts_service is None:
        _tts_service = Pyttsx3TTS()
    return _tts_service
