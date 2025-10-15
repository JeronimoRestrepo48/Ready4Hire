# 🚀 Fase 3: Deployment

Scripts para importar el modelo a Ollama.

## Script

### step1_import_to_ollama.py
**Importa el modelo fine-tuned a Ollama**

```bash
python3 step1_import_to_ollama.py --model-name ready4hire:latest
```

**¿Qué hace?**
- Busca el modelo GGUF en `models/ready4hire-finetuned/`
- Crea Modelfile con configuración optimizada
- Importa el modelo a Ollama
- Verifica la instalación
- Hace test rápido

**Parámetros opcionales**:
- `--model-dir`: Directorio del modelo (default: models/ready4hire-finetuned)
- `--model-name`: Nombre en Ollama (default: ready4hire:latest)
- `--skip-test`: Saltar prueba del modelo

**Output**: Modelo disponible en Ollama

## Verificar instalación

```bash
# Listar modelos
ollama list

# Probar el modelo
ollama run ready4hire:latest

# Dentro del chat:
>>> Evalúa esta respuesta: "JavaScript usa var para variables..."
```

## ⏱️ Tiempo estimado: ~2 minutos

## ➡️ Próximo paso
```bash
cd ../4_testing
python3 step1_test_model.py
```
