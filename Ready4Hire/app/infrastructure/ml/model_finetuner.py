"""
Model Fine-Tuner - Script para fine-tunear modelos LLM con Ollama.

Permite entrenar llama3.2:3b con datos de evaluaciones de Ready4Hire
para crear un modelo especializado: ready4hire-llama3.2:3b
"""
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile

logger = logging.getLogger(__name__)


class ModelFineTuner:
    """
    Fine-tunea modelos LLM usando Ollama.
    
    Características:
    - Crea Modelfile para Ollama
    - Ejecuta fine-tuning con datos de entrenamiento
    - Valida modelo resultante
    - Gestiona versiones del modelo
    """
    
    def __init__(
        self,
        base_model: str = "llama3.2:3b",
        finetuned_model_name: str = "ready4hire-llama3.2:3b",
        ollama_bin: str = "ollama"
    ):
        """
        Inicializa el fine-tuner.
        
        Args:
            base_model: Modelo base a fine-tunear
            finetuned_model_name: Nombre del modelo fine-tuneado
            ollama_bin: Path al binario de Ollama
        """
        self.base_model = base_model
        self.finetuned_model_name = finetuned_model_name
        self.ollama_bin = ollama_bin
        
        logger.info(f"ModelFineTuner initialized: {base_model} → {finetuned_model_name}")
    
    def create_modelfile(
        self,
        dataset_path: str,
        output_path: str = "Modelfile.ready4hire",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        num_ctx: int = 4096
    ) -> str:
        """
        Crea Modelfile para Ollama con configuración de fine-tuning.
        
        Args:
            dataset_path: Ruta al dataset JSONL de entrenamiento
            output_path: Donde guardar el Modelfile
            system_prompt: System prompt custom (opcional)
            temperature: Temperatura del modelo
            num_ctx: Context window size
        
        Returns:
            Path del Modelfile creado
        """
        if system_prompt is None:
            system_prompt = """Eres un evaluador experto de entrevistas técnicas para Ready4Hire.

Tu especialidad es evaluar respuestas de candidatos en entrevistas técnicas y de soft skills,
proporcionando scores precisos, justificaciones detalladas y feedback constructivo.

Siempre respondes en formato JSON con la estructura exacta requerida."""
        
        modelfile_content = f"""# Ready4Hire Fine-tuned Model
FROM {self.base_model}

# System prompt especializado
SYSTEM \"\"\"{system_prompt}\"\"\"

# Parámetros optimizados para evaluación
PARAMETER temperature {temperature}
PARAMETER num_ctx {num_ctx}
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# Adapter con datos de entrenamiento
ADAPTER {Path(dataset_path).absolute()}
"""
        
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile created: {output_file}")
        return str(output_file)
    
    def finetune(
        self,
        dataset_train_path: str,
        dataset_val_path: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ) -> Dict[str, Any]:
        """
        Ejecuta fine-tuning del modelo con Ollama.
        
        NOTA: Ollama actualmente NO soporta fine-tuning directo desde CLI.
        Esta función prepara todo para usar con herramientas externas como:
        - llama.cpp
        - Unsloth
        - Axolotl
        
        Para fine-tuning real, usar:
        1. Exportar modelo base con `ollama show --modelfile`
        2. Usar herramienta externa para fine-tune
        3. Importar modelo fine-tuneado con `ollama create`
        
        Args:
            dataset_train_path: Dataset de entrenamiento
            dataset_val_path: Dataset de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño de batch
            learning_rate: Learning rate
        
        Returns:
            Dict con resultados y paths
        """
        logger.warning(
            "Ollama no soporta fine-tuning directo. "
            "Esta función crea archivos necesarios para fine-tuning manual."
        )
        
        # Crear Modelfile
        modelfile_path = self.create_modelfile(
            dataset_path=dataset_train_path,
            output_path="Modelfile.ready4hire"
        )
        
        # Preparar config de entrenamiento
        training_config = {
            "base_model": self.base_model,
            "finetuned_model": self.finetuned_model_name,
            "dataset_train": dataset_train_path,
            "dataset_val": dataset_val_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "modelfile": modelfile_path
        }
        
        # Guardar config
        config_path = Path("finetune_config.json")
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        logger.info(f"Fine-tuning config saved: {config_path}")
        
        return {
            "status": "prepared",
            "modelfile": modelfile_path,
            "config": str(config_path),
            "next_steps": [
                "1. Export base model: ollama show --modelfile llama3.2:3b > base_model.txt",
                "2. Use external tool (llama.cpp, Unsloth, etc.) for fine-tuning",
                "3. Import fine-tuned model: ollama create ready4hire-llama3.2:3b -f Modelfile.ready4hire"
            ]
        }
    
    def create_from_modelfile(self, modelfile_path: str) -> bool:
        """
        Crea modelo en Ollama desde Modelfile.
        
        Usa: `ollama create <name> -f <modelfile>`
        
        Args:
            modelfile_path: Path al Modelfile
        
        Returns:
            True si éxito, False si error
        """
        try:
            cmd = [
                self.ollama_bin,
                "create",
                self.finetuned_model_name,
                "-f",
                modelfile_path
            ]
            
            logger.info(f"Creating model: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos
            )
            
            if result.returncode == 0:
                logger.info(f"Model created successfully: {self.finetuned_model_name}")
                return True
            else:
                logger.error(f"Model creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return False
    
    def validate_model(self, test_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Valida que el modelo fine-tuneado funcione correctamente.
        
        Args:
            test_prompt: Prompt de prueba (opcional)
        
        Returns:
            Dict con resultado de validación
        """
        if test_prompt is None:
            test_prompt = """Evalúa esta respuesta:

**Pregunta:** ¿Qué es Docker?
**Respuesta:** Docker es una plataforma de contenedores que permite empaquetar aplicaciones.
**Conceptos esperados:** contenedores, virtualización
**Keywords:** docker, container

Responde en JSON."""
        
        try:
            # Intentar generar con el modelo
            cmd = [
                self.ollama_bin,
                "run",
                self.finetuned_model_name,
                test_prompt
            ]
            
            logger.info("Validating model...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                response = result.stdout
                
                # Intentar parsear como JSON
                try:
                    json.loads(response)
                    json_valid = True
                except:
                    json_valid = False
                
                return {
                    "status": "success",
                    "model_exists": True,
                    "response_generated": True,
                    "json_valid": json_valid,
                    "response_sample": response[:200]
                }
            else:
                return {
                    "status": "error",
                    "model_exists": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            return {
                "status": "error",
                "model_exists": False,
                "error": str(e)
            }
    
    def list_models(self) -> list:
        """
        Lista modelos disponibles en Ollama.
        
        Returns:
            Lista de nombres de modelos
        """
        try:
            result = subprocess.run(
                [self.ollama_bin, "list"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = [line.split()[0] for line in lines if line.strip()]
                return models
            return []
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def model_exists(self, model_name: Optional[str] = None) -> bool:
        """
        Verifica si un modelo existe en Ollama.
        
        Args:
            model_name: Nombre del modelo (default: finetuned_model_name)
        
        Returns:
            True si existe, False si no
        """
        if model_name is None:
            model_name = self.finetuned_model_name
        
        models = self.list_models()
        return any(model_name in model for model in models)
    
    def get_training_guide(self) -> str:
        """
        Retorna guía paso a paso para fine-tuning manual.
        
        Returns:
            String con instrucciones completas
        """
        guide = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           READY4HIRE - FINE-TUNING GUIDE                             ║
╚══════════════════════════════════════════════════════════════════════╝

📋 PASOS PARA FINE-TUNEAR EL MODELO

1️⃣ PREPARAR DATOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # Recopilar evaluaciones (automático con TrainingDataCollector)
   # Generar dataset (automático con DatasetGenerator)
   
   Resultado: data/training/ready4hire_dataset_train.jsonl

2️⃣ EXPORTAR MODELO BASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ollama show --modelfile {self.base_model} > base_model.txt

3️⃣ FINE-TUNE CON HERRAMIENTA EXTERNA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Opción A: Unsloth (recomendado - más rápido)
   
   pip install unsloth
   python3 scripts/finetune_with_unsloth.py
   
   Opción B: llama.cpp
   
   ./llama.cpp/finetune \\
     --model-base base_model.gguf \\
     --train-data data/training/ready4hire_dataset_train.jsonl \\
     --output ready4hire_model.gguf

4️⃣ IMPORTAR MODELO A OLLAMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # Crear Modelfile
   python3 -c "
   from app.infrastructure.ml.model_finetuner import ModelFineTuner
   ft = ModelFineTuner()
   ft.create_modelfile('data/training/ready4hire_dataset_train.jsonl')
   "
   
   # Importar modelo
   ollama create {self.finetuned_model_name} -f Modelfile.ready4hire

5️⃣ VALIDAR MODELO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   python3 -c "
   from app.infrastructure.ml.model_finetuner import ModelFineTuner
   ft = ModelFineTuner()
   print(ft.validate_model())
   "

6️⃣ USAR MODELO FINE-TUNEADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # Configurar en EvaluationService
   service = EvaluationService(
       model="{self.finetuned_model_name}",
       enable_cache=True
   )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 RECURSOS
   - Unsloth: https://github.com/unslothai/unsloth
   - llama.cpp: https://github.com/ggerganov/llama.cpp
   - Ollama docs: https://github.com/ollama/ollama/blob/main/docs/modelfile.md

"""
        return guide
