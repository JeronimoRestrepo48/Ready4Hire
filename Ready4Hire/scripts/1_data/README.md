# 📊 Fase 1: Preparación de Datos

Scripts para generar y preparar datos de entrenamiento.

## Scripts en orden de ejecución

### 1️⃣ step1_generate_demo_data.py
**Genera evaluaciones sintéticas**

```bash
python3 step1_generate_demo_data.py --num-samples 200
```

**¿Qué hace?**
- Crea 200 evaluaciones de demostración
- 60% técnicas, 40% soft skills
- 70% buenas respuestas, 30% malas

**Output**: `logs/audit_log.jsonl`

---

### 2️⃣ step2_convert_to_training.py
**Convierte a formato TrainingExample**

```bash
python3 step2_convert_to_training.py
```

**¿Qué hace?**
- Lee `audit_log.jsonl`
- Convierte al formato `TrainingExample`
- Agrega metadata necesaria

**Output**: `data/training/evaluations.jsonl`

---

### 3️⃣ step3_create_dataset.py
**Crea dataset en formato Alpaca**

```bash
python3 step3_create_dataset.py --min-score 7.0
```

**¿Qué hace?**
- Filtra ejemplos por score (>= 7.0)
- Balancea categorías
- Divide en train (80%) y validation (20%)
- Convierte a formato Alpaca

**Output**: 
- `app/datasets/ready4hire_dataset_train.jsonl`
- `app/datasets/ready4hire_dataset_val.jsonl`

---

## ⏱️ Tiempo estimado: ~3 minutos

## ➡️ Próximo paso
```bash
cd ../2_training
python3 step1_finetune_model.py
```
