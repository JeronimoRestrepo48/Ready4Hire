#!/usr/bin/env python3
"""
PASO 3.1: Importar Modelo a Ollama
Crea Modelfile e importa el modelo fine-tuned a Ollama.
"""
import sys
import subprocess
from pathlib import Path

# Agregar paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.common import *


def create_modelfile(gguf_path: Path, output_path: Path):
    """Crea Modelfile para Ollama."""
    
    modelfile_content = f"""FROM {gguf_path}

# System prompt para Ready4Hire
SYSTEM You are an expert technical interviewer and soft skills evaluator. Your task is to evaluate candidate responses considering completeness, technical depth, clarity, and key concepts coverage.

# Parámetros de generación
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# Template
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


def import_to_ollama(modelfile_path: Path, model_name: str) -> bool:
    """Importa modelo a Ollama."""
    try:
        print_info(f"Importando como '{model_name}'...")
        print_warning("Esto puede tomar 1-2 minutos")
        
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return True
        else:
            print_error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Timeout (>5 minutos)")
        return False
    except FileNotFoundError:
        print_error("Ollama no encontrado")
        print_info("Instala desde: https://ollama.com")
        return False


def verify_model(model_name: str) -> bool:
    """Verifica que el modelo esté en Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        return model_name in result.stdout
        
    except Exception:
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Importa modelo a Ollama")
    parser.add_argument("--model-dir", default="models/ready4hire-finetuned", help="Directorio del modelo")
    parser.add_argument("--model-name", default="ready4hire:latest", help="Nombre en Ollama")
    parser.add_argument("--skip-test", action="store_true", help="Saltar test")
    
    args = parser.parse_args()
    
    print_header("PASO 3.1: IMPORTAR A OLLAMA")
    
    paths = get_paths()
    model_dir = paths['root'] / args.model_dir
    
    # Buscar GGUF
    gguf_files = list(model_dir.glob("*.gguf"))
    
    if not gguf_files:
        print_error(f"No se encontró archivo GGUF en {model_dir}")
        print_warning("Ejecuta primero: python3 scripts/2_training/step1_finetune_model.py")
        return 1
    
    gguf_path = gguf_files[0]
    
    print_section("Modelo encontrado")
    print_file_info(gguf_path, "GGUF")
    
    try:
        # Crear Modelfile
        print_section("Creando Modelfile")
        modelfile_path = model_dir / "Modelfile"
        create_modelfile(gguf_path, modelfile_path)
        print_success(f"Modelfile: {modelfile_path}")
        
        # Importar
        print_section("Importando a Ollama")
        if not import_to_ollama(modelfile_path, args.model_name):
            return 1
        
        print_success(f"Modelo importado como '{args.model_name}'")
        
        # Verificar
        print_section("Verificando importación")
        if verify_model(args.model_name):
            print_success(f"Modelo '{args.model_name}' verificado")
        else:
            print_warning(f"No se pudo verificar el modelo")
        
        # Test (opcional)
        if not args.skip_test:
            print_section("Test rápido")
            print_info("Ejecutando test...")
            
            result = subprocess.run(
                ["ollama", "run", args.model_name, "Hello, can you help me?"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print_success("Test completado")
                print_info("Respuesta:")
                print("─" * 60)
                print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
                print("─" * 60)
            else:
                print_warning("Test falló (no crítico)")
        
        print_header("✅ PASO 3.1 COMPLETADO")
        print_info(f"Modelo '{args.model_name}' listo para usar")
        print_info("Próximo paso: python3 scripts/4_testing/step1_test_model.py")
        
        print_section("Comandos útiles")
        print(f"  ollama run {args.model_name}")
        print(f"  ollama list")
        print(f"  ollama rm {args.model_name}")
        
        return 0
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
