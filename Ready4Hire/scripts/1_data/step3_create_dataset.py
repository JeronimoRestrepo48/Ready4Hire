#!/usr/bin/env python3
"""
PASO 1.3: Crear Dataset Alpaca
Genera dataset en formato Alpaca para fine-tuning.
"""
import sys
from pathlib import Path

# Agregar paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.common import *
from app.infrastructure.ml.training_data_collector import TrainingDataCollector
from app.infrastructure.ml.dataset_generator import DatasetGenerator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Crea dataset Alpaca")
    parser.add_argument("--min-score", type=float, default=7.0, help="Score mínimo")
    parser.add_argument("--train-split", type=float, default=0.8, help="Proporción training")
    
    args = parser.parse_args()
    
    print_header("PASO 1.3: CREAR DATASET ALPACA")
    
    try:
        # Cargar ejemplos
        print_section("Cargando ejemplos de entrenamiento")
        collector = TrainingDataCollector()
        all_examples = collector.load_all_examples()
        
        print_info(f"Cargados {len(all_examples)} ejemplos totales")
        
        # Filtrar por score
        examples = [ex for ex in all_examples if ex.score >= args.min_score]
        print_info(f"Filtrados {len(examples)} ejemplos (score >= {args.min_score})")
        
        if len(examples) == 0:
            print_error("No hay ejemplos con el score mínimo")
            print_warning("Ajusta --min-score o genera más datos")
            return 1
        
        # Generar dataset
        print_section("Generando dataset Alpaca")
        generator = DatasetGenerator(collector)
        
        stats = generator.generate_dataset(
            output_path="app/datasets/ready4hire_dataset.jsonl",
            train_split=args.train_split,
            filter_low_quality=True,
            min_score_quality=args.min_score,
            balance_categories=True
        )
        
        # Mostrar resultados
        print_success("Dataset generado exitosamente")
        
        print_section("Estadísticas del Dataset")
        print_info(f"Total ejemplos: {stats['total_examples']}")
        print_info(f"Training: {stats['train_size']} ejemplos")
        print_info(f"Validation: {stats['val_size']} ejemplos")
        print_info(f"Filtrados: {stats.get('filtered_out', 0)} ejemplos")
        
        print_section("Archivos generados")
        paths = get_paths()
        train_path = paths['root'] / stats['train_path']
        val_path = paths['root'] / stats['validation_path']
        
        print_file_info(train_path, "Training")
        print_file_info(val_path, "Validation")
        
        # Distribución por categoría
        if 'category_distribution' in stats:
            print_section("Distribución por categoría")
            for cat, count in stats['category_distribution'].items():
                pct = (count / stats['total_examples']) * 100
                print_info(f"{cat}: {count} ({pct:.1f}%)")
        
        print_header("✅ PASO 1.3 COMPLETADO")
        print_info("Dataset listo para fine-tuning")
        print_info("Próximo paso: python3 scripts/2_training/step1_finetune_model.py")
        
        return 0
        
    except FileNotFoundError as e:
        print_error(f"Archivo no encontrado: {e}")
        print_warning("Ejecuta primero: python3 scripts/1_data/step2_convert_to_training.py")
        return 1
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
