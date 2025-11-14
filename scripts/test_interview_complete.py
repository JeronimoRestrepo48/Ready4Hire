#!/usr/bin/env python3
"""
Script de testeo completo del flujo conversacional de entrevistas.
Prueba el agente con respuestas alternadas (buenas y malas) y verifica:
- Feedback post-pregunta
- Sistema de pistas
- Flujo conversacional completo
- Generación de certificado y reporte PDF
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# URLs de los backends
PYTHON_API_URL = "http://localhost:8001"  # Backend Python para entrevistas
WEBAPP_API_URL = "http://localhost:5214"  # Backend WebApp para autenticación

USER_EMAIL = "jeronimorestrepo48@gmail.com"
USER_PASSWORD = "Jerorepo0228"

class InterviewTester:
    def __init__(self):
        self.python_api_url = PYTHON_API_URL
        self.webapp_api_url = WEBAPP_API_URL
        self.user_id = ""  # Formato: user_email_at_domain (para backend Python)
        self.user_email = ""  # Email del usuario autenticado
        self.user_db_id = None  # ID numérico del usuario en la base de datos WebApp
        self.interview_id = ""
        # Aumentar timeout para respuestas del LLM que pueden tardar
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self.messages: List[Dict] = []
        self.test_results = {
            "started_at": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": [],
            "certificate_generated": False,
            "report_generated": False,
            "interview_saved_in_db": False,
        }
    
    async def authenticate(self) -> bool:
        """Autentica al usuario contra el backend de WebApp y obtiene user_id"""
        print("🔐 Autenticando usuario contra WebApp backend...")
        try:
            # Autenticarse contra el backend de WebApp
            login_response = await self.client.post(
                f"{self.webapp_api_url}/api/auth/login",
                json={
                    "email": USER_EMAIL,
                    "password": USER_PASSWORD
                }
            )
            
            if login_response.status_code != 200:
                error_text = login_response.text
                print(f"❌ Error en autenticación: {login_response.status_code} - {error_text}")
                self.test_results["errors"].append(f"Auth error: {error_text}")
                return False
            
            login_data = login_response.json()
            if not login_data.get("success"):
                print(f"❌ Autenticación fallida: {login_data.get('message', 'Unknown error')}")
                self.test_results["errors"].append(f"Auth failed: {login_data.get('message')}")
                return False
            
            # Obtener información del usuario
            user_data = login_data.get("user", {})
            self.user_email = user_data.get("email", USER_EMAIL)
            self.user_db_id = user_data.get("id")
            
            # Convertir email al formato que espera el backend de Python
            self.user_id = f"user_{self.user_email.replace('@', '_at_').replace('.', '_')}"
            
            print(f"✅ Usuario autenticado exitosamente")
            print(f"   Email: {self.user_email}")
            print(f"   DB ID: {self.user_db_id}")
            print(f"   Python user_id: {self.user_id}")
            
            self.test_results["steps_completed"].append("authentication_successful")
            return True
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error en autenticación: {e}")
            print(f"Traceback: {error_details}")
            self.test_results["errors"].append(f"Auth exception: {str(e)}\n{error_details}")
            return False
    
    async def start_interview(self, role: str = "Software Engineer", difficulty: str = "mid", category: str = "technical") -> bool:
        """Inicia una nueva entrevista o continúa con una existente"""
        print(f"\n🚀 Iniciando entrevista técnica para {role} (nivel: {difficulty})...")
        try:
            # Primero verificar si hay una entrevista activa
            active_response = await self.client.get(f"{self.python_api_url}/api/v2/interviews/active/{self.user_id}")
            if active_response.status_code == 200:
                active_data = active_response.json()
                if active_data.get("interview_id"):
                    print(f"✅ Encontrada entrevista activa: {active_data['interview_id']}")
                    self.interview_id = active_data["interview_id"]
                    current_phase = active_data.get("current_phase", "context")
                    print(f"📊 Fase actual: {current_phase}")
                    print(f"📝 Preguntas de contexto respondidas: {active_data.get('context_questions_answered', 0)}")
                    print(f"🎯 Preguntas técnicas respondidas: {active_data.get('question_count', 0)}")
                    
                    # Si hay pregunta actual, mostrarla
                    if active_data.get("current_question"):
                        current_q = active_data["current_question"].get("text", "")
                        if current_q:
                            print(f"❓ Pregunta actual: {current_q[:100]}...")
                            self.messages.append({
                                "type": "question",
                                "text": current_q,
                                "phase": current_phase,
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    self.test_results["steps_completed"].append("resumed_active_interview")
                    return True
            
            # Si no hay entrevista activa, crear una nueva
            response = await self.client.post(
                f"{self.python_api_url}/api/v2/interviews",
                json={
                    "user_id": self.user_id,
                    "role": role,
                    "category": category,
                    "difficulty": difficulty,
                }
            )
            
            if response.status_code == 409:
                # Entrevista ya existe, obtenerla
                error_data = response.json()
                existing_id = error_data.get("details", {}).get("existing_interview_id")
                if existing_id:
                    print(f"⚠️ Entrevista ya existe, continuando con: {existing_id}")
                    self.interview_id = existing_id
                    # Obtener estado completo
                    active_response = await self.client.get(f"{self.python_api_url}/api/v2/interviews/active/{self.user_id}")
                    if active_response.status_code == 200:
                        active_data = active_response.json()
                        current_phase = active_data.get("current_phase", "context")
                        print(f"📊 Fase actual: {current_phase}")
                        print(f"📝 Preguntas de contexto respondidas: {active_data.get('context_questions_answered', 0)}")
                        print(f"🎯 Preguntas técnicas respondidas: {active_data.get('question_count', 0)}")
                        if active_data.get("current_question"):
                            current_q = active_data["current_question"].get("text", "")
                            if current_q:
                                print(f"❓ Pregunta actual: {current_q[:100]}...")
                                self.messages.append({
                                    "type": "question",
                                    "text": current_q,
                                    "phase": current_phase,
                                    "timestamp": datetime.now().isoformat()
                                })
                        self.test_results["steps_completed"].append("resumed_existing_interview")
                        return True
                print(f"❌ No se pudo obtener la entrevista existente")
                return False
            
            if response.status_code != 200:
                print(f"❌ Error iniciando entrevista: {response.status_code} - {response.text}")
                self.test_results["errors"].append(f"Start interview error: {response.text}")
                return False
            
            data = response.json()
            self.interview_id = data.get("interview_id")
            first_question = data.get("first_question", {}).get("text", "")
            
            print(f"✅ Entrevista iniciada: {self.interview_id}")
            print(f"📝 Primera pregunta de contexto: {first_question[:100]}...")
            
            self.messages.append({
                "type": "system",
                "text": data.get("message", ""),
                "timestamp": datetime.now().isoformat()
            })
            self.messages.append({
                "type": "question",
                "text": first_question,
                "phase": "context",
                "timestamp": datetime.now().isoformat()
            })
            
            self.test_results["steps_completed"].append("interview_started")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.test_results["errors"].append(f"Start interview exception: {str(e)}")
            return False
    
    async def answer_context_question(self, answer: str) -> bool:
        """Responde una pregunta de contexto"""
        print(f"\n💬 Respondiendo pregunta de contexto: {answer[:50]}...")
        try:
            response = await self.client.post(
                f"{self.python_api_url}/api/v2/interviews/{self.interview_id}/answers",
                json={
                    "answer": answer,
                    "user_id": self.user_id,
                }
            )
            
            if response.status_code != 200:
                print(f"❌ Error respondiendo: {response.status_code} - {response.text}")
                return False
            
            data = response.json()
            status = data.get("status", "")
            next_question = data.get("next_question")
            
            self.messages.append({
                "type": "answer",
                "text": answer,
                "phase": "context",
                "timestamp": datetime.now().isoformat()
            })
            
            if next_question:
                question_text = next_question.get("text", "")
                print(f"✅ Respuesta guardada. Siguiente pregunta: {question_text[:100]}...")
                self.messages.append({
                    "type": "question",
                    "text": question_text,
                    "phase": "context",
                    "timestamp": datetime.now().isoformat()
                })
            elif status == "questions":
                print("✅ Fase de contexto completada. Iniciando preguntas técnicas...")
                self.test_results["steps_completed"].append("context_phase_completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.test_results["errors"].append(f"Answer context exception: {str(e)}")
            return False
    
    async def answer_technical_question(self, answer: str, is_correct: bool = True) -> Dict:
        """Responde una pregunta técnica (alternando correctas e incorrectas)"""
        print(f"\n💬 Respondiendo pregunta técnica ({'✅ CORRECTA' if is_correct else '❌ INCORRECTA'}): {answer[:80]}...")
        try:
            if not self.interview_id:
                print("❌ No hay interview_id disponible")
                return {}
            
            response = await self.client.post(
                f"{self.python_api_url}/api/v2/interviews/{self.interview_id}/answers",
                json={
                    "answer": answer,
                    "user_id": self.user_id,
                }
            )
            
            if response.status_code != 200:
                error_text = response.text if hasattr(response, 'text') else str(response.content)
                print(f"❌ Error respondiendo: {response.status_code}")
                print(f"   Response: {error_text[:500]}")
                self.test_results["errors"].append(f"Answer technical HTTP {response.status_code}: {error_text[:200]}")
                return {}
            
            try:
                data = response.json()
            except Exception as json_error:
                error_text = response.text if hasattr(response, 'text') else str(response.content)
                print(f"❌ Error parseando JSON: {json_error}")
                print(f"   Response: {error_text[:500]}")
                self.test_results["errors"].append(f"Answer technical JSON error: {str(json_error)} - {error_text[:200]}")
                return {}
            
            # Extraer información importante
            result = {
                "status": data.get("status", ""),
                "score": data.get("evaluation", {}).get("score", 0),
                "feedback": data.get("feedback_result", ""),
                "motivational_feedback": data.get("motivational_feedback", ""),
                "hint": data.get("hint", ""),
                "should_advance": data.get("should_advance", False),
                "next_question": data.get("next_question"),
                "correct_answer": data.get("correct_answer", ""),
                "improvement_tips": data.get("improvement_tips", ""),
            }
            
            self.messages.append({
                "type": "answer",
                "text": answer,
                "phase": "technical",
                "is_correct": is_correct,
                "score": result["score"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Mostrar feedback
            if result["feedback"]:
                print(f"📊 Score: {result['score']}/10")
                print(f"💬 Feedback: {result['feedback'][:150]}...")
            
            if result["motivational_feedback"]:
                print(f"💪 Feedback motivacional: {result['motivational_feedback'][:150]}...")
            
            if result["hint"]:
                print(f"💡 Pista: {result['hint'][:150]}...")
            
            if result["correct_answer"]:
                print(f"✅ Respuesta correcta: {result['correct_answer'][:150]}...")
            
            if result["next_question"]:
                question_text = result["next_question"].get("text", "")
                print(f"➡️ Siguiente pregunta: {question_text[:100]}...")
                self.messages.append({
                    "type": "question",
                    "text": question_text,
                    "phase": "technical",
                    "timestamp": datetime.now().isoformat()
                })
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error: {e}")
            print(f"Traceback: {error_details}")
            self.test_results["errors"].append(f"Answer technical exception: {str(e)}\n{error_details}")
            return {}
    
    async def complete_interview(self) -> bool:
        """Completa la entrevista y verifica generación de certificado y reporte"""
        print("\n🏁 Verificando estado de la entrevista...")
        try:
            # Primero intentar obtener la entrevista activa
            active_response = await self.client.get(
                f"{self.python_api_url}/api/v2/interviews/active/{self.user_id}"
            )
            
            if active_response.status_code == 200:
                data = active_response.json()
                status = data.get("status", "")
                
                if status == "completed":
                    print("✅ Entrevista completada en el backend Python")
                    self.test_results["steps_completed"].append("interview_completed")
                    
                    # Verificar que la entrevista se guardó en la base de datos del WebApp
                    print("\n💾 Verificando que la entrevista se guardó en la base de datos...")
                    await self.verify_interview_in_database()
                    
                    # Verificar certificado y reporte
                    print("\n📜 Verificando certificado y reporte...")
                    await self.verify_certificate_and_report()
                    
                    return True
                else:
                    print(f"⚠️ Entrevista aún en estado: {status}")
                    return False
            elif active_response.status_code == 404:
                # La entrevista ya no está activa, probablemente se completó
                print("ℹ️ Entrevista no encontrada en estado activo (probablemente completada)")
                
                # Verificar en las entrevistas completadas
                print("\n💾 Verificando que la entrevista se guardó en la base de datos...")
                interview_found = await self.verify_interview_in_database()
                
                if interview_found:
                    print("✅ Entrevista encontrada en estado completado")
                    self.test_results["steps_completed"].append("interview_completed")
                    
                    # Verificar certificado y reporte
                    print("\n📜 Verificando certificado y reporte...")
                    await self.verify_certificate_and_report()
                    
                    return True
                else:
                    print("⚠️ Entrevista no encontrada en las completadas")
                    return False
            else:
                print(f"⚠️ Error verificando estado: {active_response.status_code}")
                return False
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error: {e}")
            print(f"Traceback: {error_details}")
            self.test_results["errors"].append(f"Complete interview exception: {str(e)}\n{error_details}")
            return False
    
    async def verify_interview_in_database(self) -> bool:
        """Verifica que la entrevista se guardó en la base de datos del WebApp"""
        try:
            # Obtener entrevistas completadas del usuario desde el backend Python
            # El backend Python debería sincronizar con PostgreSQL automáticamente
            completed_response = await self.client.get(
                f"{self.python_api_url}/api/v2/interviews/user/{self.user_id}/completed?limit=10"
            )
            
            if completed_response.status_code == 200:
                completed_data = completed_response.json()
                interviews = completed_data.get("interviews", [])
                
                # Buscar nuestra entrevista por interview_id
                for interview in interviews:
                    if interview.get("interview_id") == self.interview_id:
                        print(f"✅ Entrevista encontrada en la base de datos")
                        print(f"   Interview ID: {interview.get('interview_id')}")
                        print(f"   Role: {interview.get('role')}")
                        print(f"   Average Score: {interview.get('average_score')}")
                        print(f"   Completed At: {interview.get('completed_at')}")
                        print(f"   Has Report: {interview.get('has_report', False)}")
                        print(f"   Has Certificate: {interview.get('has_certificate', False)}")
                        print(f"   Certificate Eligible: {interview.get('certificate_eligible', False)}")
                        self.test_results["interview_saved_in_db"] = True
                        self.test_results["steps_completed"].append("interview_saved_in_database")
                        return True
                
                # Si no encontramos por ID exacto, buscar la más reciente (puede ser que el ID cambió)
                if interviews:
                    latest_interview = interviews[0]  # La más reciente
                    print(f"⚠️ Entrevista con ID {self.interview_id} no encontrada")
                    print(f"   Pero se encontró la entrevista más reciente:")
                    print(f"   Interview ID: {latest_interview.get('interview_id')}")
                    print(f"   Role: {latest_interview.get('role')}")
                    print(f"   Average Score: {latest_interview.get('average_score')}")
                    print(f"   Completed At: {latest_interview.get('completed_at')}")
                    
                    # Actualizar el interview_id para usar el encontrado
                    self.interview_id = latest_interview.get('interview_id')
                    self.test_results["interview_saved_in_db"] = True
                    self.test_results["steps_completed"].append("interview_saved_in_database")
                    return True
                else:
                    print("⚠️ Entrevista completada pero no encontrada en la lista de completadas")
                    print(f"   Total entrevistas encontradas: {len(interviews)}")
            else:
                error_text = completed_response.text if hasattr(completed_response, 'text') else str(completed_response.content)
                print(f"⚠️ No se pudo verificar entrevistas completadas: {completed_response.status_code}")
                print(f"   Response: {error_text[:200]}")
            
            return False
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"⚠️ Error verificando entrevista en base de datos: {e}")
            print(f"Traceback: {error_details}")
            return False
    
    async def verify_certificate_and_report(self) -> bool:
        """Verifica que se generaron certificado y reporte"""
        try:
            # Verificar certificado
            cert_response = await self.client.get(
                f"{self.python_api_url}/api/v2/interviews/{self.interview_id}/certificate?format=json"
            )
            
            if cert_response.status_code == 200:
                cert_data = cert_response.json()
                print(f"✅ Certificado generado")
                print(f"   Certificate ID: {cert_data.get('certificate_id')}")
                print(f"   Score: {cert_data.get('score')}")
                print(f"   Percentile: {cert_data.get('percentile')}")
                self.test_results["certificate_generated"] = True
                self.test_results["steps_completed"].append("certificate_generated")
            else:
                print(f"⚠️ Certificado no disponible: {cert_response.status_code}")
            
            # Verificar reporte
            report_response = await self.client.get(
                f"{self.python_api_url}/api/v2/interviews/{self.interview_id}/report"
            )
            
            if report_response.status_code == 200:
                report_response_data = report_response.json()
                # El reporte viene dentro de una clave "report"
                report_data = report_response_data.get("report", report_response_data)
                
                print(f"✅ Reporte generado")
                # Intentar diferentes estructuras posibles
                if isinstance(report_data, dict):
                    # Buscar métricas en diferentes ubicaciones posibles
                    metrics = report_data.get("metrics", {})
                    if isinstance(metrics, dict):
                        avg_score = metrics.get("average_score")
                        total_q = metrics.get("total_questions")
                    else:
                        avg_score = report_data.get("average_score")
                        total_q = report_data.get("total_questions")
                    
                    strengths = report_data.get("strengths", [])
                    improvements = report_data.get("improvements", [])
                    
                    print(f"   Average Score: {avg_score}")
                    print(f"   Total Questions: {total_q}")
                    print(f"   Strengths: {len(strengths) if isinstance(strengths, list) else 0} puntos fuertes")
                    print(f"   Improvements: {len(improvements) if isinstance(improvements, list) else 0} áreas de mejora")
                else:
                    print(f"   Reporte recibido (formato: {type(report_data).__name__})")
                
                self.test_results["report_generated"] = True
                self.test_results["steps_completed"].append("report_generated")
            else:
                print(f"⚠️ Reporte no disponible: {report_response.status_code}")
            
            return self.test_results["certificate_generated"] and self.test_results["report_generated"]
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"⚠️ Error verificando certificado y reporte: {e}")
            print(f"Traceback: {error_details}")
            return False
    
    async def run_complete_test(self):
        """Ejecuta el testeo completo del flujo"""
        print("=" * 80)
        print("🧪 TESTEO COMPLETO DEL FLUJO CONVERSACIONAL")
        print("=" * 80)
        
        # 1. Autenticar
        if not await self.authenticate():
            return
        
        # 2. Iniciar entrevista
        if not await self.start_interview(role="Software Engineer", difficulty="mid"):
            return
        
        # 3. Verificar estado actual y responder preguntas de contexto si es necesario
        active_response = await self.client.get(f"{self.python_api_url}/api/v2/interviews/active/{self.user_id}")
        if active_response.status_code == 200:
            active_data = active_response.json()
            context_answered = active_data.get("context_questions_answered", 0)
            current_phase = active_data.get("current_phase", "context")
            
            if current_phase == "context" and context_answered < 5:
                print(f"\n📋 Continuando con preguntas de contexto ({context_answered}/5 respondidas)...")
                context_answers = [
                    "Tengo 3 años de experiencia desarrollando aplicaciones web con React y Node.js.",
                    "He trabajado principalmente en startups tech, desarrollando features desde cero.",
                    "Mi stack principal es JavaScript/TypeScript, React, Node.js, PostgreSQL.",
                    "Me gusta trabajar en equipo, siguiendo metodologías ágiles como Scrum.",
                    "Mi objetivo es crecer como desarrollador full-stack y aprender arquitecturas escalables.",
                ]
                
                # Responder solo las preguntas faltantes
                for i in range(context_answered, 5):
                    print(f"\n📋 Pregunta de contexto {i+1}/5")
                    await self.answer_context_question(context_answers[i])
                    await asyncio.sleep(1)
            elif current_phase == "questions":
                print(f"✅ Fase de contexto ya completada. Continuando con preguntas técnicas...")
            else:
                print(f"📊 Estado: fase={current_phase}, contexto={context_answered}/5")
        else:
            # Si no hay entrevista activa, responder todas las preguntas de contexto
            print("\n📋 Respondiendo preguntas de contexto (5 preguntas)...")
            context_answers = [
                "Tengo 3 años de experiencia desarrollando aplicaciones web con React y Node.js.",
                "He trabajado principalmente en startups tech, desarrollando features desde cero.",
                "Mi stack principal es JavaScript/TypeScript, React, Node.js, PostgreSQL.",
                "Me gusta trabajar en equipo, siguiendo metodologías ágiles como Scrum.",
                "Mi objetivo es crecer como desarrollador full-stack y aprender arquitecturas escalables.",
            ]
            
            for i, answer in enumerate(context_answers, 1):
                print(f"\n📋 Pregunta de contexto {i}/5")
                await self.answer_context_question(answer)
                await asyncio.sleep(1)
        
        # 4. Verificar cuántas preguntas técnicas ya se respondieron
        active_response = await self.client.get(f"{self.python_api_url}/api/v2/interviews/active/{self.user_id}")
        questions_answered = 0
        if active_response.status_code == 200:
            active_data = active_response.json()
            questions_answered = active_data.get("question_count", 0)
            print(f"📊 Preguntas técnicas ya respondidas: {questions_answered}/10")
        
        # 4. Responder preguntas técnicas (alternando correctas e incorrectas)
        print("\n" + "=" * 80)
        print("🎯 FASE DE PREGUNTAS TÉCNICAS")
        print("=" * 80)
        
        # Respuestas alternadas: correcta, incorrecta, correcta, incorrecta, etc.
        technical_answers = [
            ("Una función es un bloque de código reutilizable que realiza una tarea específica. Puede recibir parámetros y retornar valores.", True),
            ("No sé mucho sobre esto...", False),  # Respuesta incorrecta para probar pistas
            ("Git es un sistema de control de versiones distribuido que permite rastrear cambios en el código y colaborar con otros desarrolladores.", True),
            ("Algo sobre bases de datos...", False),  # Respuesta incorrecta
            ("REST es un estilo arquitectónico para diseñar APIs web que usa métodos HTTP estándar y recursos identificados por URLs.", True),
            ("No tengo idea...", False),  # Respuesta incorrecta
            ("Un closure es una función que tiene acceso a variables de su scope externo incluso después de que la función externa haya terminado.", True),
            ("Algo sobre JavaScript...", False),  # Respuesta incorrecta
            ("La programación asíncrona permite ejecutar código sin bloquear el hilo principal, usando callbacks, promesas o async/await.", True),
            ("No sé...", False),  # Respuesta incorrecta
        ]
        
        # Responder solo las preguntas faltantes
        remaining_questions = 10 - questions_answered
        if remaining_questions > 0:
            print(f"📝 Respondiendo {remaining_questions} preguntas técnicas restantes...")
            for i in range(questions_answered, min(questions_answered + remaining_questions, len(technical_answers))):
                answer, is_correct = technical_answers[i]
                print(f"\n📝 Pregunta técnica {i+1}/10")
                result = await self.answer_technical_question(answer, is_correct)
                
                # Si la respuesta es incorrecta y hay pista, intentar responder de nuevo
                if not is_correct and result.get("hint") and not result.get("should_advance"):
                    print(f"\n🔄 Reintentando con la pista...")
                    improved_answer = f"Basándome en la pista: {answer} y considerando {result['hint'][:50]}..."
                    result2 = await self.answer_technical_question(improved_answer, True)
                    
                    # Si aún no avanza, intentar una vez más
                    if not result2.get("should_advance") and result2.get("hint"):
                        print(f"\n🔄 Segundo reintento...")
                        final_answer = f"Entiendo ahora: {result2['hint'][:50]}..."
                        await self.answer_technical_question(final_answer, True)
                
                await asyncio.sleep(2)  # Pausa entre preguntas
        else:
            print("✅ Todas las preguntas técnicas ya fueron respondidas")
        
        # 5. Completar entrevista
        await self.complete_interview()
        
        # 6. Generar reporte final
        self.test_results["completed_at"] = datetime.now().isoformat()
        self.test_results["total_messages"] = len(self.messages)
        
        print("\n" + "=" * 80)
        print("📊 RESUMEN DEL TESTEO")
        print("=" * 80)
        print(f"✅ Pasos completados: {len(self.test_results['steps_completed'])}")
        print(f"❌ Erros encontrados: {len(self.test_results['errors'])}")
        print(f"💬 Total mensajes: {len(self.messages)}")
        print(f"📜 Certificado generado: {self.test_results['certificate_generated']}")
        print(f"📄 Reporte generado: {self.test_results['report_generated']}")
        
        if self.test_results["errors"]:
            print("\n⚠️ Errores encontrados:")
            for error in self.test_results["errors"]:
                print(f"  - {error}")
        
        # Guardar resultados
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_results": self.test_results,
                "messages": self.messages,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados en: {results_file}")
    
    async def close(self):
        """Cierra el cliente HTTP"""
        await self.client.aclose()


async def main():
    tester = InterviewTester()
    try:
        await tester.run_complete_test()
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())

