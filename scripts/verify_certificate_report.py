#!/usr/bin/env python3
"""
Script para verificar que se generaron correctamente el certificado y reporte
de una entrevista completada.
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8001"
USER_EMAIL = "jeronimorestrepo48@gmail.com"
USER_PASSWORD = "Jerorepo0228"

async def verify_certificate_and_report():
    """Verifica certificado y reporte de entrevistas completadas"""
    print("=" * 80)
    print("🔍 VERIFICACIÓN DE CERTIFICADO Y REPORTE")
    print("=" * 80)
    
    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    user_id = f"user_{USER_EMAIL.replace('@', '_at_').replace('.', '_')}"
    
    try:
        # 1. Autenticar
        print("\n🔐 Autenticando...")
        auth_resp = await client.post(
            f"{BASE_URL}/api/v2/auth/login",
            json={"email": USER_EMAIL, "password": USER_PASSWORD}
        )
        
        if auth_resp.status_code != 200:
            print(f"❌ Error de autenticación: {auth_resp.status_code}")
            return
        
        token = auth_resp.json().get("token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        print("✅ Autenticado")
        
        # 2. Obtener entrevistas completadas
        print(f"\n📋 Obteniendo entrevistas completadas para {user_id}...")
        completed_resp = await client.get(
            f"{BASE_URL}/api/v2/interviews/user/{user_id}/completed",
            headers=headers
        )
        
        if completed_resp.status_code != 200:
            print(f"❌ Error obteniendo entrevistas: {completed_resp.status_code}")
            print(completed_resp.text[:500])
            return
        
        completed_data = completed_resp.json()
        interviews = completed_data.get("interviews", [])
        
        if not interviews:
            print("⚠️ No se encontraron entrevistas completadas")
            print("   Ejecuta primero el script de testeo completo para generar una entrevista")
            return
        
        print(f"✅ Encontradas {len(interviews)} entrevista(s) completada(s)\n")
        
        # 3. Verificar cada entrevista
        for idx, interview in enumerate(interviews, 1):
            interview_id = interview.get("interview_id")
            role = interview.get("role")
            mode = interview.get("mode")
            has_report = interview.get("has_report", False)
            has_certificate = interview.get("has_certificate", False)
            certificate_id = interview.get("certificate_id")
            avg_score = interview.get("average_score", 0)
            
            print(f"{'='*80}")
            print(f"📊 Entrevista {idx}: {interview_id[:30]}...")
            print(f"   Rol: {role}")
            print(f"   Modo: {mode}")
            print(f"   Score promedio: {avg_score}/10")
            print(f"   Preguntas: {interview.get('total_questions', 0)}/10")
            print(f"   Completada: {interview.get('completed_at', 'N/A')}")
            
            # Verificar reporte
            print(f"\n📄 REPORTE:")
            if has_report:
                print("   ✅ Reporte generado")
                try:
                    report_resp = await client.get(
                        f"{BASE_URL}/api/v2/interviews/{interview_id}/report",
                        headers=headers
                    )
                    if report_resp.status_code == 200:
                        report_data = report_resp.json()
                        report_content = report_data.get("report", {})
                        metrics = report_content.get("metrics", {})
                        
                        print(f"   📊 Métricas:")
                        print(f"      - Score promedio: {metrics.get('average_score', 'N/A')}")
                        print(f"      - Tasa de éxito: {metrics.get('success_rate', 'N/A'):.1f}%")
                        print(f"      - Percentil: {metrics.get('percentile', 'N/A')}")
                        print(f"      - Pistas usadas: {metrics.get('hints_used', 'N/A')}")
                        
                        strengths = report_content.get("strengths", [])
                        improvements = report_content.get("improvements", [])
                        
                        if strengths:
                            print(f"   💪 Fortalezas ({len(strengths)}):")
                            for s in strengths[:3]:
                                print(f"      • {s}")
                        
                        if improvements:
                            print(f"   📈 Áreas de mejora ({len(improvements)}):")
                            for i in improvements[:3]:
                                print(f"      • {i}")
                        
                        resources = report_content.get("recommended_resources", [])
                        if resources:
                            print(f"   📚 Recursos recomendados ({len(resources)}):")
                            for r in resources[:2]:
                                print(f"      • {r.get('title', 'N/A')} ({r.get('type', 'N/A')})")
                        
                        print(f"   🔗 URL compartible: {report_data.get('shareable_url', 'N/A')}")
                    else:
                        print(f"   ⚠️ Error obteniendo reporte: {report_resp.status_code}")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            else:
                print("   ❌ Reporte NO generado")
            
            # Verificar certificado
            print(f"\n🏆 CERTIFICADO:")
            if has_certificate and certificate_id:
                print(f"   ✅ Certificado generado: {certificate_id}")
                try:
                    cert_resp = await client.get(
                        f"{BASE_URL}/api/v2/interviews/{interview_id}/certificate?format=json",
                        headers=headers
                    )
                    if cert_resp.status_code == 200:
                        cert_data = cert_resp.json()
                        print(f"   📜 Detalles del certificado:")
                        print(f"      - ID: {cert_data.get('certificate_id', 'N/A')}")
                        print(f"      - Candidato: {cert_data.get('candidate_name', 'N/A')}")
                        print(f"      - Rol: {cert_data.get('role', 'N/A')}")
                        print(f"      - Score: {cert_data.get('score', 'N/A')}/10")
                        print(f"      - Percentil: Top {cert_data.get('percentile', 'N/A')}%")
                        print(f"      - Fecha: {cert_data.get('completion_date', 'N/A')}")
                        print(f"   🔗 URLs:")
                        print(f"      - Preview SVG: {BASE_URL}{cert_data.get('preview_url', '')}")
                        print(f"      - Descarga PDF: {BASE_URL}{cert_data.get('download_url', '')}")
                        print(f"      - Validación: {cert_data.get('validation_url', 'N/A')}")
                        
                        # Intentar obtener preview SVG
                        svg_resp = await client.get(
                            f"{BASE_URL}/api/v2/interviews/{interview_id}/certificate?format=svg",
                            headers=headers
                        )
                        if svg_resp.status_code == 200:
                            svg_size = len(svg_resp.content)
                            print(f"   ✅ Preview SVG disponible ({svg_size} bytes)")
                        else:
                            print(f"   ⚠️ Error obteniendo preview SVG: {svg_resp.status_code}")
                    else:
                        print(f"   ⚠️ Error obteniendo certificado: {cert_resp.status_code}")
                        print(f"   Response: {cert_resp.text[:200]}")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            elif interview.get("certificate_eligible", False):
                print("   ⚠️ Elegible para certificado pero no generado")
            else:
                print("   ℹ️ No elegible para certificado (modo práctica o score < 7.5)")
            
            print()
        
        print("=" * 80)
        print("✅ Verificación completada")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(verify_certificate_and_report())

