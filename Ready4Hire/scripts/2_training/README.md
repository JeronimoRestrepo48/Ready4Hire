# 🤖 Fase 2: Fine-Tuning

Scripts para entrenar el modelo con Unsloth.

## ⚠️ Requisitos

- GPU con CUDA
- 8GB+ VRAM (16GB recomendado)
- 16GB+ RAM
- `pip install unsloth`

## Script

### step1_finetune_model.py
**Entrena el modelo con los datos generados**

```bash
# Instalar Unsloth primero
pip install unsloth

# Fine-tuning
python3 step1_finetune_model.py \
    --epochs 3 \
    --batch-size 2 \
    --learning-rate 2e-4
```

**¿Qué hace?**
- Carga modelo base (llama-3-8b-bnb-4bit)
- Aplica LoRA para entrenamiento eficiente
- Entrena con dataset generado en Fase 1
- Exporta a formato GGUF para Ollama

**Parámetros opcionales**:
- `--model`: Modelo base (default: unsloth/llama-3-8b-bnb-4bit)
- `--epochs`: Número de epochs (default: 3)
- `--batch-size`: Tamaño de batch (default: 2)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--max-seq-length`: Longitud máxima (default: 2048)

**Output**: `models/ready4hire-finetuned/`

## ⏱️ Tiempo estimado: 30-120 minutos

## 💡 Sin GPU?

Si no tienes GPU, puedes:
1. Usar el modelo base sin fine-tuning
2. Usar Google Colab con GPU gratuita
3. Saltar esta fase y continuar con modelo base

## ➡️ Próximo paso
```bash
cd ../3_deployment
python3 step1_import_to_ollama.py
```
