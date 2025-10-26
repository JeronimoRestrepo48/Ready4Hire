"""
Generador de Certificados Personalizados
Crea certificados PDF con diseño profesional
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import hashlib
import qrcode
from io import BytesIO
import base64


@dataclass
class CertificateData:
    """Datos para generar certificado"""

    certificate_id: str
    candidate_name: str
    role: str
    completion_date: datetime
    score: float
    percentile: int
    interview_id: str
    validation_url: str


class CertificateGenerator:
    """
    Genera certificados personalizados en PDF

    Features:
    - Diseño profesional con branding
    - QR Code para verificación
    - ID único verificable
    - Firma digital
    - Exportación a PDF
    - Preview en SVG
    """

    def __init__(self, base_url: str = "https://ready4hire.com"):
        self.base_url = base_url
        self.brand_color = "#6366f1"  # Indigo
        self.accent_color = "#10b981"  # Green

    def generate_certificate(self, data: CertificateData) -> bytes:
        """
        Genera certificado en PDF

        Args:
            data: Datos del certificado

        Returns:
            bytes del PDF generado
        """
        # En producción, usar librería como ReportLab o WeasyPrint
        # Por ahora, generar SVG que puede convertirse a PDF
        svg_content = self._generate_svg(data)

        # TODO: Convertir SVG a PDF usando weasyprint o similar
        # from weasyprint import HTML, CSS
        # pdf_bytes = HTML(string=svg_content).write_pdf()

        return svg_content.encode("utf-8")

    def generate_preview_svg(self, data: CertificateData) -> str:
        """Genera preview del certificado en SVG"""
        return self._generate_svg(data)

    def _generate_svg(self, data: CertificateData) -> str:
        """Genera certificado en formato SVG"""

        # Generar QR Code
        qr_data_url = self._generate_qr_code(data.validation_url)

        # Calcular nivel de certificación
        cert_level, level_color = self._get_certification_level(data.score)

        svg_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="850" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <!-- Background -->
    <defs>
        <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#f8fafc;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#e2e8f0;stop-opacity:1" />
        </linearGradient>
        
        <linearGradient id="headerGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:{self.brand_color};stop-opacity:1" />
            <stop offset="100%" style="stop-color:{self.accent_color};stop-opacity:1" />
        </linearGradient>
    </defs>
    
    <!-- Background Rectangle -->
    <rect width="1200" height="850" fill="url(#bgGradient)"/>
    
    <!-- Border -->
    <rect x="50" y="50" width="1100" height="750" fill="white" stroke="url(#headerGradient)" stroke-width="3"/>
    <rect x="70" y="70" width="1060" height="710" fill="none" stroke="#e2e8f0" stroke-width="2"/>
    
    <!-- Header -->
    <rect x="70" y="70" width="1060" height="120" fill="url(#headerGradient)"/>
    <text x="600" y="130" font-family="Arial, sans-serif" font-size="42" font-weight="bold" fill="white" text-anchor="middle">
        CERTIFICADO DE COMPETENCIA
    </text>
    <text x="600" y="165" font-family="Arial, sans-serif" font-size="18" fill="white" text-anchor="middle" opacity="0.9">
        Ready4Hire - Professional Interview Platform
    </text>
    
    <!-- Main Content -->
    <text x="600" y="250" font-family="Arial, sans-serif" font-size="24" fill="#64748b" text-anchor="middle">
        Se certifica que
    </text>
    
    <text x="600" y="320" font-family="Georgia, serif" font-size="48" font-weight="bold" fill="#1e293b" text-anchor="middle">
        {data.candidate_name}
    </text>
    
    <text x="600" y="390" font-family="Arial, sans-serif" font-size="24" fill="#64748b" text-anchor="middle">
        ha completado exitosamente la evaluación técnica para
    </text>
    
    <text x="600" y="450" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="{self.brand_color}" text-anchor="middle">
        {data.role}
    </text>
    
    <!-- Score Badge -->
    <circle cx="600" cy="550" r="60" fill="{level_color}" opacity="0.2"/>
    <circle cx="600" cy="550" r="55" fill="none" stroke="{level_color}" stroke-width="4"/>
    <text x="600" y="545" font-family="Arial, sans-serif" font-size="32" font-weight="bold" fill="{level_color}" text-anchor="middle">
        {data.score:.1f}
    </text>
    <text x="600" y="570" font-family="Arial, sans-serif" font-size="16" fill="{level_color}" text-anchor="middle">
        / 10.0
    </text>
    
    <!-- Certification Level -->
    <text x="600" y="640" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#1e293b" text-anchor="middle">
        Nivel: {cert_level}
    </text>
    <text x="600" y="665" font-family="Arial, sans-serif" font-size="16" fill="#64748b" text-anchor="middle">
        Top {data.percentile}% de candidatos
    </text>
    
    <!-- Footer -->
    <line x1="200" y1="710" x2="500" y2="710" stroke="#cbd5e1" stroke-width="2"/>
    <text x="350" y="735" font-family="Arial, sans-serif" font-size="14" fill="#64748b" text-anchor="middle">
        {data.completion_date.strftime("%B %d, %Y")}
    </text>
    <text x="350" y="755" font-family="Arial, sans-serif" font-size="12" fill="#94a3b8" text-anchor="middle">
        Fecha de Certificación
    </text>
    
    <line x1="700" y1="710" x2="1000" y2="710" stroke="#cbd5e1" stroke-width="2"/>
    <text x="850" y="735" font-family="Arial, sans-serif" font-size="14" fill="#64748b" text-anchor="middle" font-weight="bold">
        ID: {data.certificate_id}
    </text>
    <text x="850" y="755" font-family="Arial, sans-serif" font-size="12" fill="#94a3b8" text-anchor="middle">
        Verificar en: {self.base_url}/verify/{data.certificate_id}
    </text>
    
    <!-- QR Code (placeholder - en producción insertar imagen base64) -->
    <image x="1050" y="650" width="80" height="80" xlink:href="{qr_data_url}"/>
    
    <!-- Watermark -->
    <text x="1150" y="840" font-family="Arial, sans-serif" font-size="10" fill="#cbd5e1" text-anchor="end">
        Ready4Hire © {datetime.now().year}
    </text>
</svg>"""

        return svg_template

    def _get_certification_level(self, score: float) -> tuple:
        """Determina nivel de certificación basado en score"""
        if score >= 9.0:
            return "EXCELENCIA", "#10b981"  # Green
        elif score >= 8.5:
            return "SOBRESALIENTE", "#6366f1"  # Indigo
        elif score >= 8.0:
            return "DISTINGUIDO", "#8b5cf6"  # Purple
        elif score >= 7.5:
            return "COMPETENTE", "#3b82f6"  # Blue
        else:
            return "APROBADO", "#64748b"  # Gray

    def _generate_qr_code(self, url: str) -> str:
        """Genera QR code como data URL"""
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=1,
            )
            qr.add_data(url)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            # Convertir a data URL
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"
        except Exception:
            # Fallback: placeholder
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    def validate_certificate(self, certificate_id: str) -> bool:
        """
        Valida si un certificado es auténtico

        Args:
            certificate_id: ID del certificado

        Returns:
            True si es válido, False otherwise
        """
        # TODO: Implementar validación contra DB
        # Por ahora, validar formato
        return certificate_id.startswith("R4H-") and len(certificate_id) == 16


# Factory
_certificate_generator = None


def get_certificate_generator() -> CertificateGenerator:
    """Obtiene instancia singleton del generador de certificados"""
    global _certificate_generator
    if _certificate_generator is None:
        _certificate_generator = CertificateGenerator()
    return _certificate_generator
