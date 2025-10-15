#!/usr/bin/env python3
"""
PASO 2.1: Fine-Tuning del Modelo con Unsloth
Entrena el modelo usando los datasets generados.
"""
import sys
from pathlib import Path

# Agregar paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.common import *

# Verificar dependencias de fine-tuning
try:
    from unsloth import FastLanguageModel
    import torch
    from trl.trainer.sft_trainer import SFTTrainer
    from transformers import TrainingArguments
    HAS_UNSLOTH = True
except ImportError:
    import torch  # Import torch separately to avoid unbound error
    HAS_UNSLOTH = False
    FastLanguageModel = None
    SFTTrainer = None
    TrainingArguments = None

def check_requirements():
    """Verifica requisitos para fine-tuning."""
    print_section("Verificando requisitos")
    
    issues = []
    
    # GPU
    if not torch.cuda.is_available():
        issues.append("GPU con CUDA no disponible")
        print_warning("GPU no detectada")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print_success(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        
        if vram_gb < 8:
            issues.append(f"VRAM insuficiente: {vram_gb:.1f} GB (mínimo: 8 GB)")
    
    # Unsloth
    if not HAS_UNSLOTH:
        issues.append("Unsloth no instalado")
        print_warning("Unsloth no encontrado")
    else:
        print_success("Unsloth instalado")
    
    return len(issues) == 0, issues


def load_dataset(train_path: Path, val_path: Path):
    """Carga datasets."""
    import json
    
    train_data = []
    val_data = []
    
    print_info(f"Cargando training: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    print_info(f"Cargando validation: {val_path}")
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    print_success(f"Training: {len(train_data)} ejemplos")
    print_success(f"Validation: {len(val_data)} ejemplos")
    
    return train_data, val_data


def format_prompt(example: dict) -> str:
    """Formatea ejemplo para el modelo."""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune modelo con Unsloth")
    parser.add_argument("--model", default="unsloth/llama-3-8b-bnb-4bit", help="Modelo base")
    parser.add_argument("--epochs", type=int, default=3, help="Número de epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    
    args = parser.parse_args()
    
    print_header("PASO 2.1: FINE-TUNING CON UNSLOTH")
    
    paths = get_paths()
    
    # Verificar requisitos
    ok, issues = check_requirements()
    if not ok:
        print_error("Requisitos no cumplidos:")
        for issue in issues:
            print_warning(f"  - {issue}")
        
        if not HAS_UNSLOTH:
            print_info("\n💡 Para instalar Unsloth:")
            print("   pip install unsloth")
        
        return 1
    
    # Double-check that imports succeeded
    if not HAS_UNSLOTH or FastLanguageModel is None or SFTTrainer is None or TrainingArguments is None:
        print_error("Error: Dependencias de Unsloth no están disponibles")
        return 1
    
    # Verificar datasets
    train_path = paths['datasets'] / "ready4hire_dataset_train.jsonl"
    val_path = paths['datasets'] / "ready4hire_dataset_val.jsonl"
    
    if not train_path.exists() or not val_path.exists():
        print_error("Datasets no encontrados")
        print_warning("Ejecuta primero: python3 scripts/1_data/step3_create_dataset.py")
        return 1
    
    try:
        # Cargar datos
        print_section("Cargando datasets")
        train_data, val_data = load_dataset(train_path, val_path)
        
        # Cargar modelo
        print_section(f"Cargando modelo base: {args.model}")
        print_warning("Esto puede tomar varios minutos...")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        print_success("Modelo cargado")
        
        # Configurar LoRA
        print_section("Configurando LoRA")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        print_success("LoRA configurado")
        
        # Preparar datos
        print_section("Preparando datos")
        train_texts = [format_prompt(ex) for ex in train_data]
        val_texts = [format_prompt(ex) for ex in val_data]
        
        print_success(f"{len(train_texts)} ejemplos training formateados")
        print_success(f"{len(val_texts)} ejemplos validation formateados")
        
        # Configurar entrenamiento
        print_section("Configurando entrenamiento")
        
        output_dir = paths['models'] / "ready4hire-finetuned"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            output_dir=str(output_dir),
            report_to="none",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
        )
        
        print_info(f"Epochs: {args.epochs}")
        print_info(f"Batch size: {args.batch_size}")
        print_info(f"Learning rate: {args.learning_rate}")
        print_info(f"Output: {output_dir}")
        
        # Crear trainer
        print_section("Creando trainer")
        
        from datasets import Dataset
        
        train_dataset = Dataset.from_dict({"text": train_texts})
        eval_dataset = Dataset.from_dict({"text": val_texts})
        
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
        
        print_success("Trainer creado")
        
        # Entrenar
        print_header("INICIANDO ENTRENAMIENTO")
        print_warning("⏰ Esto puede tomar 30-120 minutos")
        print_warning("☕ Es un buen momento para un café...")
        print()
        
        trainer.train()
        
        print_success("Entrenamiento completado")
        
        # Guardar
        print_section("Guardando modelo")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        print_success(f"Modelo guardado en: {output_dir}")
        
        # Exportar a GGUF
        print_section("Exportando a GGUF para Ollama")
        
        model.save_pretrained_gguf(
            str(output_dir),
            tokenizer,
            quantization_method="q4_k_m"
        )
        
        gguf_file = list(output_dir.glob("*.gguf"))[0]
        print_success(f"GGUF exportado: {gguf_file.name}")
        
        print_header("✅ PASO 2.1 COMPLETADO")
        print_info("Próximo paso: python3 scripts/3_deployment/step1_import_to_ollama.py")
        
        return 0
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
