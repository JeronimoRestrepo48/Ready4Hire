"""
Certificate API Routes
Endpoints para generación y validación de certificados
"""

from fastapi import APIRouter, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging
from io import BytesIO
from datetime import datetime

from app.infrastructure.llm.certificate_generator import get_certificate_generator, CertificateData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/certificates", tags=["Certificates"])


# ============================================================================
# DTOs
# ============================================================================


class GenerateCertificateRequest(BaseModel):
    """Request para generar certificado"""

    candidate_name: str
    role: str
    score: float
    percentile: int
    interview_id: str


class CertificateResponse(BaseModel):
    """Respuesta de generación de certificado"""

    certificate_id: str
    validation_url: str
    download_url: str


class ValidationResponse(BaseModel):
    """Respuesta de validación"""

    is_valid: bool
    certificate_id: str
    issued_date: Optional[datetime]
    candidate_name: Optional[str]
    role: Optional[str]
    score: Optional[float]


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/generate", response_model=CertificateResponse, status_code=status.HTTP_201_CREATED)
async def generate_certificate(request: GenerateCertificateRequest):
    """
    Genera un certificado para una entrevista completada exitosamente.

    Requisitos:
    - Score >= 7.5
    - Modo examen (verificado externamente)
    """
    try:
        if request.score < 7.5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Score insuficiente para certificación (mínimo 7.5)"
            )

        generator = get_certificate_generator()

        # Crear datos del certificado
        cert_data = CertificateData(
            certificate_id="",  # Se generará automáticamente
            candidate_name=request.candidate_name,
            role=request.role,
            completion_date=datetime.now(),
            score=request.score,
            percentile=request.percentile,
            interview_id=request.interview_id,
            validation_url="",  # Se generará automáticamente
        )

        # Generar ID y URL
        cert_data.certificate_id = generator._generate_certificate_id(request.score)
        cert_data.validation_url = f"{generator.base_url}/verify/{cert_data.certificate_id}"

        # TODO: Guardar en base de datos para validación

        logger.info(f"✅ Certificate generated: {cert_data.certificate_id}")

        return CertificateResponse(
            certificate_id=cert_data.certificate_id,
            validation_url=cert_data.validation_url,
            download_url=f"/api/v2/certificates/{cert_data.certificate_id}/download",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating certificate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate certificate: {str(e)}"
        )


@router.get("/{certificate_id}/download")
async def download_certificate(certificate_id: str, format: str = "svg"):
    """
    Descarga un certificado en formato SVG o PDF.

    Args:
        certificate_id: ID del certificado
        format: Formato de salida (svg o pdf)
    """
    try:
        generator = get_certificate_generator()

        # TODO: Recuperar datos del certificado desde DB
        # Por ahora, datos de ejemplo
        cert_data = CertificateData(
            certificate_id=certificate_id,
            candidate_name="Candidate Name",
            role="Software Engineer",
            completion_date=datetime.now(),
            score=8.5,
            percentile=85,
            interview_id="interview_123",
            validation_url=f"{generator.base_url}/verify/{certificate_id}",
        )

        if format.lower() == "svg":
            # Generar SVG
            svg_content = generator.generate_preview_svg(cert_data)

            return Response(
                content=svg_content,
                media_type="image/svg+xml",
                headers={"Content-Disposition": f'attachment; filename="certificate_{certificate_id}.svg"'},
            )

        elif format.lower() == "pdf":
            # Generar PDF (placeholder - requiere weasyprint)
            pdf_bytes = generator.generate_certificate(cert_data)

            return StreamingResponse(
                BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="certificate_{certificate_id}.pdf"'},
            )

        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid format. Use 'svg' or 'pdf'")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading certificate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to download certificate: {str(e)}"
        )


@router.get("/{certificate_id}/validate", response_model=ValidationResponse)
async def validate_certificate(certificate_id: str):
    """
    Valida la autenticidad de un certificado.

    Args:
        certificate_id: ID del certificado a validar
    """
    try:
        generator = get_certificate_generator()

        # Validar formato
        is_valid_format = generator.validate_certificate(certificate_id)

        if not is_valid_format:
            return ValidationResponse(
                is_valid=False,
                certificate_id=certificate_id,
                issued_date=None,
                candidate_name=None,
                role=None,
                score=None,
            )

        # TODO: Buscar en base de datos
        # Por ahora, asumir válido si el formato es correcto
        return ValidationResponse(
            is_valid=True,
            certificate_id=certificate_id,
            issued_date=datetime.now(),
            candidate_name="Candidate Name",
            role="Software Engineer",
            score=8.5,
        )

    except Exception as e:
        logger.error(f"Error validating certificate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to validate certificate: {str(e)}"
        )


@router.get("/{certificate_id}/preview")
async def preview_certificate(certificate_id: str):
    """
    Vista previa del certificado en SVG.
    """
    try:
        return await download_certificate(certificate_id, format="svg")
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate preview: {str(e)}"
        )
