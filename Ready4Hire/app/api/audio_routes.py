"""
Audio API Routes
Endpoints para Speech-to-Text (STT) y Text-to-Speech (TTS)
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from typing import Optional
import logging
import tempfile
import os

from app.infrastructure.audio import get_stt_service, get_tts_service
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/audio", tags=["Audio"])


# ============================================================================
# DTOs
# ============================================================================

class STTResponse(BaseModel):
    """Respuesta del servicio STT"""
    text: str
    language: str
    confidence: Optional[float] = None


class TTSRequest(BaseModel):
    """Request para síntesis de voz"""
    text: str
    language: str = "es"
    rate: int = 150
    volume: float = 1.0
    output_format: str = "mp3"


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/speech-to-text", response_model=STTResponse)
async def speech_to_text(
    audio_file: UploadFile = File(...),
    language: str = Form("es")
):
    """
    Convierte audio a texto usando Whisper STT.
    
    Args:
        audio_file: Archivo de audio (wav, mp3, m4a, etc.)
        language: Idioma del audio (es, en, etc.)
    
    Returns:
        Texto transcrito
    """
    try:
        logger.info(f"🎤 STT request - Language: {language}, File: {audio_file.filename}")
        
        # Obtener servicio STT
        stt_service = get_stt_service()
        
        if not stt_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servicio STT no disponible. Instala openai-whisper."
            )
        
        # Transcribir audio
        transcribed_text = stt_service.transcribe(audio_file, language)
        
        logger.info(f"✅ STT successful - Length: {len(transcribed_text)} chars")
        
        return STTResponse(
            text=transcribed_text,
            language=language,
            confidence=0.95  # Whisper no proporciona confidence, usamos valor fijo
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ STT error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en transcripción: {str(e)}"
        )


@router.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """
    Convierte texto a audio usando pyttsx3 TTS.
    
    Args:
        request: Parámetros de síntesis
    
    Returns:
        Archivo de audio generado
    """
    try:
        logger.info(f"🔊 TTS request - Language: {request.language}, Length: {len(request.text)} chars")
        
        # Obtener servicio TTS
        tts_service = get_tts_service()
        
        if not tts_service._available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servicio TTS no disponible. Instala pyttsx3."
            )
        
        # Sintetizar audio
        audio_path = tts_service.synthesize(
            text=request.text,
            language=request.language,
            rate=request.rate,
            volume=request.volume,
            output_format=request.output_format
        )
        
        logger.info(f"✅ TTS successful - File: {audio_path}")
        
        # Retornar archivo de audio
        return FileResponse(
            path=audio_path,
            media_type=f"audio/{request.output_format}",
            filename=f"tts_audio.{request.output_format}",
            background=lambda: os.unlink(audio_path) if os.path.exists(audio_path) else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ TTS error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en síntesis: {str(e)}"
        )


@router.post("/text-to-speech-bytes")
async def text_to_speech_bytes(request: TTSRequest):
    """
    Convierte texto a audio y retorna los bytes directamente.
    Útil para integración con JavaScript que espera un array de bytes.
    
    Args:
        request: Parámetros de síntesis
    
    Returns:
        Bytes del archivo de audio
    """
    try:
        logger.info(f"🔊 TTS bytes request - Language: {request.language}, Length: {len(request.text)} chars")
        
        # Obtener servicio TTS
        tts_service = get_tts_service()
        
        if not tts_service._available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servicio TTS no disponible. Instala pyttsx3."
            )
        
        # Sintetizar audio a bytes
        audio_bytes = tts_service.synthesize_to_bytes(
            text=request.text,
            language=request.language,
            rate=request.rate,
            volume=request.volume,
            output_format=request.output_format
        )
        
        logger.info(f"✅ TTS bytes successful - Size: {len(audio_bytes)} bytes")
        
        # Retornar bytes directamente
        from fastapi.responses import Response
        return Response(
            content=audio_bytes,
            media_type=f"audio/{request.output_format}",
            headers={
                "Content-Disposition": f"attachment; filename=tts_audio.{request.output_format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ TTS bytes error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en síntesis: {str(e)}"
        )


@router.get("/status")
async def audio_status():
    """
    Verifica el estado de los servicios de audio.
    
    Returns:
        Estado de STT y TTS
    """
    try:
        stt_service = get_stt_service()
        tts_service = get_tts_service()
        
        return {
            "stt": {
                "available": stt_service.is_available(),
                "supported_languages": stt_service.get_supported_languages() if stt_service.is_available() else []
            },
            "tts": {
                "available": tts_service._available,
                "supported_languages": ["es", "en"] if tts_service._available else []
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Audio status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error verificando estado de audio: {str(e)}"
        )
