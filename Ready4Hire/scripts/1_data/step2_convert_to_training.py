#!/usr/bin/env python3
"""
PASO 1.2: Conversión a Training Data
Convierte audit_log.jsonl al formato TrainingExample.
"""
import sys
from pathlib import Path

# Agregar paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.common import *
from app.infrastructure.ml.training_data_collector import TrainingExample
import json
from datetime import datetime


def convert_audit_log(input_path: Path, output_path: Path) -> int:
    """Convierte audit_log a training data."""
    
    if not input_path.exists():
        print_error(f"No se encuentra: {input_path}")
        return 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    entry = json.loads(line)
                    data = entry.get('data', {})
                    
                    example = TrainingExample(
                        # Input
                        question=data.get('question', ''),
                        answer=data.get('answer', ''),
                        expected_concepts=[],
                        keywords=[],
                        category=data.get('category', 'technical'),
                        difficulty=data.get('metadata', {}).get('difficulty', 'medium'),
                        role="Software Engineer",
                        
                        # Output
                        score=data.get('score', 0.0),
                        breakdown={
                            "completeness": data.get('score', 0.0) * 0.3,
                            "depth": data.get('score', 0.0) * 0.3,
                            "clarity": data.get('score', 0.0) * 0.2,
                            "concepts": data.get('score', 0.0) * 0.2
                        },
                        justification=f"Evaluación demo con score {data.get('score', 0.0)}/10",
                        strengths=["Respuesta coherente"] if data.get('score', 0) >= 7 else [],
                        improvements=["Mejorar profundidad"] if data.get('score', 0) < 7 else [],
                        concepts_covered=[],
                        missing_concepts=[],
                        
                        # Metadata
                        example_id=f"demo_{count}_{int(datetime.now().timestamp())}",
                        created_at=data.get('timestamp', datetime.now().isoformat()),
                        model_used="demo",
                        evaluation_source="demo"
                    )
                    
                    f_out.write(json.dumps(example.to_dict(), ensure_ascii=False) + '\n')
                    count += 1
                    
                    if count % 50 == 0:
                        print_info(f"Convertidos {count} ejemplos")
                    
                except Exception as e:
                    print_warning(f"Error en línea: {e}")
                    continue
    
    return count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convierte audit log a training data")
    parser.add_argument("--input", default="logs/audit_log.jsonl", help="Audit log de entrada")
    parser.add_argument("--output", default="data/training/evaluations.jsonl", help="Training data de salida")
    
    args = parser.parse_args()
    
    print_header("PASO 1.2: CONVERSIÓN A TRAINING DATA")
    
    paths = get_paths()
    input_path = paths['root'] / args.input
    output_path = paths['root'] / args.output
    
    try:
        print_section("Convirtiendo formato")
        print_info(f"Entrada: {input_path}")
        print_info(f"Salida: {output_path}")
        
        count = convert_audit_log(input_path, output_path)
        
        if count > 0:
            print_success(f"Convertidos {count} ejemplos")
            print_file_info(output_path, "Training data")
            
            print_header("✅ PASO 1.2 COMPLETADO")
            print_info("Próximo paso: python3 scripts/1_data/step3_create_dataset.py")
            return 0
        else:
            print_error("No se convirtieron ejemplos")
            print_warning("Ejecuta primero: python3 scripts/1_data/step1_generate_demo_data.py")
            return 1
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
