"""
Audio Infrastructure Module
Servicios de audio (STT/TTS) para Ready4Hire
"""

from .whisper_stt import WhisperSTT, get_stt_service
from .pyttsx3_tts import Pyttsx3TTS, get_tts_service

__all__ = [
    "WhisperSTT",
    "get_stt_service",
    "Pyttsx3TTS",
    "get_tts_service",
]
