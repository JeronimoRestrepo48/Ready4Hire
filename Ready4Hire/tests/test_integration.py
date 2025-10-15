#!/usr/bin/env python3
"""
Test de Integración - Ready4Hire v2.0
Verifica que todos los componentes funcionen correctamente con Ollama local
"""
import sys
import os
import time
from colorama import init, Fore, Style

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Inicializar colorama
init(autoreset=True)

def print_header(text: str):
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")

def print_test(test_name: str):
    print(f"\n{Fore.YELLOW}🔧 Test {test_name}")
    print(f"{Fore.YELLOW}{'-'*50}{Style.RESET_ALL}")

def print_success(message: str):
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

def print_error(message: str):
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

def print_info(message: str):
    print(f"{Fore.WHITE}   {message}{Style.RESET_ALL}")


# ============================================================================
# TEST 1: OllamaClient
# ============================================================================
def test_ollama_client():
    """Test del cliente Ollama básico"""
    print_test("1: OllamaClient")
    
    try:
        from app.infrastructure.llm.ollama_client import OllamaClient
        
        # Inicializar cliente
        client = OllamaClient(
            base_url="http://localhost:11434",
            default_model="llama3:latest"
        )
        print_success("Cliente inicializado")
        
        # Listar modelos
        models = client.list_models()
        print_info(f"Modelos disponibles: {models[:3]}")
        
        # Test de generación
        start = time.time()
        response = client.generate(
            prompt="Di solo 'Hola'",
            max_tokens=10
        )
        latency = int((time.time() - start) * 1000)
        
        print_success("Generación exitosa")
        if isinstance(response, dict):
            print_info(f"Respuesta: {response['response'][:50]}")
            print_info(f"Latencia: {response['latency_ms']:.0f}ms")
            print_info(f"Modelo: {response['model']}")
        else:
            print_info(f"Respuesta: {str(response)[:50]}")
        
        # Métricas
        metrics = client.get_metrics()
        print_success("Métricas:")
        print_info(f"Total requests: {metrics['total_requests']}")
        print_info(f"Successful: {metrics['successful_requests']}")
        print_info(f"Avg latency: {metrics['avg_latency']:.0f}ms")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False


# ============================================================================
# TEST 2: OllamaLLMService
# ============================================================================
def test_llm_service():
    """Test del servicio LLM abstracto"""
    print_test("2: OllamaLLMService")
    
    try:
        from app.infrastructure.llm.llm_service import OllamaLLMService
        
        # Inicializar servicio
        service = OllamaLLMService(
            base_url="http://localhost:11434",
            model="llama3:latest"
        )
        print_success("Servicio inicializado")
        
        # Test de generación
        response = service.generate(
            prompt="¿Qué es Python? Responde en una oración.",
            max_tokens=50
        )
        
        print_success("Generación exitosa")
        print_info(f"Respuesta: {response[:100]}")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False


# ============================================================================
# TEST 3: EvaluationService
# ============================================================================
def test_evaluation_service():
    """Test del servicio de evaluación"""
    print_test("3: EvaluationService")
    
    try:
        from app.application.services.evaluation_service import EvaluationService
        from app.infrastructure.llm.llm_service import OllamaLLMService
        
        # Inicializar servicio
        llm_service = OllamaLLMService(
            base_url="http://localhost:11434",
            model="llama3:latest"
        )
        
        eval_service = EvaluationService(
            llm_service=llm_service,
            model="llama3:latest",
            temperature=0.3
        )
        print_success("Servicio inicializado")
        
        # Evaluar respuesta
        result = eval_service.evaluate_answer(
            question="¿Qué es Python?",
            answer="Python es un lenguaje de programación interpretado de alto nivel",
            expected_concepts=["lenguaje", "programación", "interpretado"],
            keywords=["Python", "lenguaje"],
            category="technical",
            difficulty="junior",
            role="Backend Developer"
        )
        
        print_success("Evaluación exitosa")
        print_info(f"Score: {result['score']}/10")
        print_info("Breakdown:")
        for key, value in result['breakdown'].items():
            print_info(f"  - {key}: {value}")
        print_info(f"Justificación: {result['justification'][:80]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: FeedbackService
# ============================================================================
def test_feedback_service():
    """Test del servicio de feedback"""
    print_test("4: FeedbackService")
    
    try:
        from app.application.services.feedback_service import FeedbackService
        from app.infrastructure.llm.llm_service import OllamaLLMService
        from app.domain.value_objects.emotion import Emotion
        
        # Inicializar servicio
        llm_service = OllamaLLMService(
            base_url="http://localhost:11434",
            model="llama3:latest"
        )
        
        feedback_service = FeedbackService(
            llm_service=llm_service,
            model="llama3:latest"
        )
        print_success("Servicio inicializado")
        
        # Generar feedback
        feedback = feedback_service.generate_feedback(
            question="¿Qué es Docker?",
            answer="Docker es una plataforma de contenedores",
            evaluation={
                "score": 7.5,
                "breakdown": {
                    "completeness": 2.0,
                    "technical_depth": 2.0,
                    "clarity": 1.5,
                    "key_concepts": 2.0
                },
                "justification": "Buena respuesta básica"
            },
            emotion=Emotion.JOY,
            role="DevOps Engineer",
            category="technical"
        )
        
        print_success("Feedback generado")
        print_info(f"Feedback: {feedback[:100]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: Container (DI)
# ============================================================================
def test_container():
    """Test del container de DI"""
    print_test("5: Container (Dependency Injection)")
    
    try:
        from app.container import Container
        
        # Obtener container (sin parámetros usa defaults)
        container = Container(
            ollama_url="http://localhost:11434",
            ollama_model="llama3:latest"
        )
        print_success("Container inicializado")
        
        # Health check
        health = container.health_check()
        print_success("Health check:")
        for component, status in health.items():
            color = Fore.GREEN if "healthy" in status else Fore.RED
            print(f"   {color}{component}: {status}{Style.RESET_ALL}")
        
        # Verificar servicios
        eval_service = container.get_evaluation_service()
        feedback_service = container.get_feedback_service()
        
        print_success("Servicios obtenidos del container")
        print_info(f"EvaluationService: {type(eval_service).__name__}")
        print_info(f"FeedbackService: {type(feedback_service).__name__}")
        
        return True
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Ejecuta todos los tests"""
    print_header("🎯 Ready4Hire v2.0 - Test de Integración")
    print(f"{Fore.WHITE}Verificando infraestructura Ollama local...{Style.RESET_ALL}\n")
    
    # Verificar que Ollama está corriendo
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print_success("Ollama está corriendo ✓")
        else:
            print_error("Ollama no responde correctamente")
            print_info("Ejecuta: ollama serve &")
            sys.exit(1)
    except Exception:
        print_error("No se puede conectar a Ollama")
        print_info("Ejecuta: ollama serve &")
        sys.exit(1)
    
    # Ejecutar tests
    tests = [
        ("OllamaClient", test_ollama_client),
        ("OllamaLLMService", test_llm_service),
        ("EvaluationService", test_evaluation_service),
        ("FeedbackService", test_feedback_service),
        ("Container", test_container),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Error inesperado en {name}: {str(e)}")
            results.append((name, False))
    
    # Resumen
    print_header("📊 Resumen de Tests")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Fore.GREEN}✅ PASS" if result else f"{Fore.RED}❌ FAIL"
        print(f"{status}{Style.RESET_ALL} - {name}")
    
    print(f"\n{Fore.CYAN}{'='*50}")
    percentage = (passed / total) * 100
    color = Fore.GREEN if percentage == 100 else Fore.YELLOW if percentage >= 60 else Fore.RED
    print(f"{color}🎯 Resultado: {passed}/{total} tests pasados ({percentage:.0f}%){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")
    
    if percentage == 100:
        print(f"{Fore.GREEN}✨ ¡Todos los tests pasaron! Sistema listo para usar.{Style.RESET_ALL}\n")
        return 0
    elif percentage >= 60:
        print(f"{Fore.YELLOW}⚠️  La mayoría de tests pasaron, pero hay algunos fallos.{Style.RESET_ALL}\n")
        return 1
    else:
        print(f"{Fore.RED}❌ Muchos tests fallaron. Revisa la configuración.{Style.RESET_ALL}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
