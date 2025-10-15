# 🚀 Guía Rápida: Fine-tuning en Google Colab

## ¿Por qué hacer fine-tuning?

**Modelo base (actual)**: 66.7% accuracy  
**Modelo fine-tuned (esperado)**: >80% accuracy

El modelo base funciona, pero el fine-tuning lo especializa en evaluar entrevistas técnicas y soft skills.

---

## Pasos para Fine-tuning en Colab:

### 1. Abrir Colab
1. Ve a: https://colab.research.google.com/
2. File → Open Notebook → Upload
3. Sube: `scripts/2_training/COLAB_FINETUNE.ipynb`

### 2. Activar GPU
1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU** (gratis)
3. Save

### 3. Subir datasets
Sube estos 2 archivos desde tu computadora:
```
app/datasets/ready4hire_dataset_train.jsonl
app/datasets/ready4hire_dataset_val.jsonl
```

O usa Google Drive:
- Copia los archivos a tu Drive
- En Colab: ejecuta la celda de montar Drive
- Ajusta las rutas en la celda de "Cargar dataset"

### 4. Ejecutar todas las celdas
1. Runtime → Run all
2. Espera 30-60 minutos
3. El modelo se entrenará con tus datos

### 5. Descargar modelo
1. Al finalizar, descarga el archivo `.gguf` generado
2. Cópialo a: `models/ready4hire-finetuned/`
3. Ejecuta: `python3 scripts/3_deployment/step1_import_to_ollama.py`

### 6. Re-testear
```bash
python3 scripts/4_testing/step1_test_model.py --model ready4hire:latest --compare --base-model llama3.2:3b
```

---

## ¿Sin tiempo para fine-tuning?

### Opción 1: Generar más datos demo
Más datos = mejor entrenamiento:

```bash
# Generar 500 evaluaciones (en lugar de 20)
python3 scripts/1_data/step1_generate_demo_data.py --num-samples 500

# Re-convertir y crear dataset
python3 scripts/1_data/step2_convert_to_training.py
python3 scripts/1_data/step3_create_dataset.py --min-score 7.0

# Luego hacer fine-tuning en Colab con más datos
```

### Opción 2: Usar modelo base actual
El modelo base funciona y puede usarse en producción:
- ✅ 66.7% accuracy (aceptable para MVP)
- ✅ Diferencia bien respuestas muy buenas vs muy malas
- ⚠️ Menos preciso en casos intermedios

---

## Próximos pasos con modelo actual:

### 1. Configurar aplicación
```bash
# Crear/editar .env
echo "MODEL_NAME=ready4hire:latest" > .env
echo "OLLAMA_HOST=http://localhost:11434" >> .env
```

### 2. Iniciar servidor
```bash
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Abrir navegador
```
http://localhost:8000
```

### 4. Probar entrevista
1. Selecciona rol y modo (Technical/Soft Skills)
2. Responde preguntas
3. Recibe evaluaciones del modelo

---

## Recomendación

🎯 **Para producción seria**: Hacer fine-tuning en Colab  
⚡ **Para pruebas rápidas**: Usar modelo base actual

¿Qué prefieres hacer?
