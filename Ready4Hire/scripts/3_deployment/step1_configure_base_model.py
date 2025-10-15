#!/usr/bin/env python3
"""
PASO 3.1 (ALT): Configurar Modelo Base
Configura un modelo base de Ollama con prompt especializado para Ready4Hire.
"""
import sys
import subprocess
from pathlib import Path

# Agregar paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.common import *


def create_modelfile_from_base(base_model: str, output_path: Path):
    """Crea Modelfile desde modelo base."""
    
    modelfile_content = f"""FROM {base_model}

SYSTEM You are an expert technical interviewer and soft skills evaluator for Ready4Hire. Your task is to evaluate candidate responses in technical and soft skills interviews. When evaluating, consider: Completeness (Does the answer cover all aspects?), Technical depth (Understanding of concepts?), Clarity (Well-structured explanation?), and Key concepts (Important concepts mentioned?). Provide scores from 0-10 and detailed feedback with strengths and areas for improvement.

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1

TEMPLATE \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{{{ .System }}}}

### Input:
{{{{ .Prompt }}}}

### Response:
\"\"\"
"""
    
    with open(output_path, 'w') as f:
        f.write(modelfile_content)
    
    return output_path


def verify_base_model(base_model: str) -> bool:
    """Verifica que el modelo base exista en Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Extraer nombre sin tag si incluye :
        model_base = base_model.split(':')[0]
        return model_base in result.stdout
        
    except Exception:
        return False


def pull_model_if_needed(base_model: str) -> bool:
    """Descarga modelo base si no existe."""
    try:
        print_info(f"Verificando modelo base '{base_model}'...")
        
        if verify_base_model(base_model):
            print_success(f"Modelo '{base_model}' ya existe")
            return True
        
        print_warning(f"Modelo '{base_model}' no encontrado")
        print_info(f"Descargando '{base_model}' (esto puede tomar varios minutos)...")
        
        result = subprocess.run(
            ["ollama", "pull", base_model],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos máximo
        )
        
        if result.returncode == 0:
            print_success(f"Modelo '{base_model}' descargado")
            return True
        else:
            print_error(f"Error descargando: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Timeout descargando modelo (>10 minutos)")
        return False
    except FileNotFoundError:
        print_error("Ollama no encontrado")
        print_info("Instala desde: https://ollama.com")
        return False


def create_custom_model(modelfile_path: Path, model_name: str) -> bool:
    """Crea modelo personalizado en Ollama."""
    try:
        print_info(f"Creando modelo personalizado '{model_name}'...")
        
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return True
        else:
            print_error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Timeout creando modelo")
        return False


def test_model(model_name: str) -> bool:
    """Hace un test rápido del modelo."""
    try:
        print_info("Ejecutando test de evaluación...")
        
        test_prompt = """Evalúa esta respuesta:

Pregunta: ¿Qué es un closure en JavaScript?
Respuesta: Un closure es cuando una función recuerda las variables de su scope externo.

Proporciona un score del 0-10 y justificación."""
        
        result = subprocess.run(
            ["ollama", "run", model_name, test_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print_success("Test completado")
            print_info("Respuesta del modelo:")
            print("─" * 70)
            response = result.stdout.strip()
            print(response[:300] + "..." if len(response) > 300 else response)
            print("─" * 70)
            return True
        else:
            print_warning("Test falló (no crítico)")
            return False
            
    except subprocess.TimeoutExpired:
        print_warning("Test timeout (no crítico)")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Configura modelo base para Ready4Hire")
    parser.add_argument("--base-model", default="llama3.2:3b", 
                       help="Modelo base de Ollama (default: llama3.2:3b)")
    parser.add_argument("--model-name", default="ready4hire:latest", 
                       help="Nombre del modelo personalizado")
    parser.add_argument("--skip-test", action="store_true", 
                       help="Saltar test")
    
    args = parser.parse_args()
    
    print_header("PASO 3.1: CONFIGURAR MODELO BASE")
    
    paths = get_paths()
    config_dir = paths['root'] / "models" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Verificar/descargar modelo base
        print_section("1. Verificando modelo base")
        if not pull_model_if_needed(args.base_model):
            print_error("No se pudo obtener el modelo base")
            return 1
        
        # 2. Crear Modelfile personalizado
        print_section("2. Creando configuración personalizada")
        modelfile_path = config_dir / f"Modelfile.{args.model_name.replace(':', '_')}"
        create_modelfile_from_base(args.base_model, modelfile_path)
        print_success(f"Modelfile creado")
        print_info(f"Ubicación: {modelfile_path}")
        
        # 3. Crear modelo personalizado
        print_section("3. Creando modelo Ready4Hire")
        if not create_custom_model(modelfile_path, args.model_name):
            print_error("No se pudo crear el modelo personalizado")
            return 1
        
        print_success(f"Modelo '{args.model_name}' creado")
        
        # 4. Verificar
        print_section("4. Verificando instalación")
        if verify_base_model(args.model_name):
            print_success(f"Modelo '{args.model_name}' verificado")
        else:
            print_warning("No se pudo verificar el modelo")
        
        # 5. Test (opcional)
        if not args.skip_test:
            print_section("5. Test de evaluación")
            test_model(args.model_name)
        
        # Resumen final
        print_header("✅ PASO 3.1 COMPLETADO")
        
        print_section("Configuración")
        print(f"  Modelo base: {args.base_model}")
        print(f"  Modelo Ready4Hire: {args.model_name}")
        print(f"  Modelfile: {modelfile_path}")
        
        print_section("Próximos pasos")
        print(f"1. Configurar .env:")
        print(f"   MODEL_NAME={args.model_name}")
        print()
        print(f"2. Ejecutar tests:")
        print(f"   python3 scripts/4_testing/step1_test_model.py --model {args.model_name}")
        print()
        print(f"3. Iniciar aplicación:")
        print(f"   python3 -m uvicorn app.main:app --reload")
        
        print_section("Comandos útiles")
        print(f"  ollama run {args.model_name}        # Probar modelo")
        print(f"  ollama list                         # Listar modelos")
        print(f"  ollama rm {args.model_name}         # Eliminar modelo")
        
        return 0
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
