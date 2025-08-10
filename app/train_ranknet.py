# train_ranknet.py
"""
Entrenador de RankNet para Ready4Hire
=====================================

¿Qué es?
---------
Script para entrenar un modelo de ranking profundo (RankNet) que aprende a priorizar preguntas relevantes y personalizadas para cada usuario, usando los datasets de preguntas técnicas, soft skills y (opcionalmente) interacciones reales de usuarios.

¿Para quién es?
---------------
- Técnicos: pueden modificar la arquitectura, hiperparámetros y lógica de generación de pares para fine-tuning avanzado.
- No técnicos: simplemente ejecutan el script para mejorar la inteligencia del simulador, sin necesidad de entender deep learning.

¿Cómo funciona?
----------------
1. Carga los datasets de preguntas técnicas y blandas, y las interacciones de fine-tuning si existen.
2. Genera pares de entrenamiento: para cada pregunta, crea ejemplos de respuesta correcta e incorrecta.
3. Extrae embeddings de pregunta y respuesta esperada usando SentenceTransformers.
4. Entrena RankNet para que aprenda a distinguir respuestas buenas de malas, optimizando el ranking de preguntas.
5. Guarda el modelo entrenado para ser usado automáticamente en el pipeline de Ready4Hire.

Ejemplo de uso rápido:
----------------------
    python3 app/train_ranknet.py

Autor: JeronimoRestrepo48
Licencia: MIT
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# --- Configuración ---
DATA_DIR = Path(__file__).parent / "datasets"
TECH_PATH = DATA_DIR / "tech_questions.jsonl"
SOFT_PATH = DATA_DIR / "soft_skills.jsonl"
FINETUNE_PATH = DATA_DIR / "finetune_interactions.jsonl"
MODEL_SAVE_PATH = DATA_DIR / "ranknet_model.pt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

# --- RankNet ---
class RankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_pairs_from_finetune(finetune):
    # Si no hay interacciones, genera pares sintéticos de tech y soft skills
    pairs = []
    # Cargar tech y soft para pares correctos/incorrectos
    tech = load_jsonl(TECH_PATH)
    soft = load_jsonl(SOFT_PATH)
    # Pares correctos: pregunta y respuesta esperada
    for i in range(len(tech)):
        for j in range(len(tech)):
            if i != j:
                a = tech[i]
                b = tech[j]
                # a es correcta, b es incorrecta (respuesta de b a pregunta de a)
                qa = {"question": a["question"], "answer": a["answer"], "correct": True}
                qb = {"question": a["question"], "answer": b["answer"], "correct": False}
                pairs.append((qa, qb, 1))
    for i in range(len(soft)):
        for j in range(len(soft)):
            if i != j:
                a = soft[i]
                b = soft[j]
                qa = {"question": a["scenario"], "answer": a["expected"], "correct": True}
                qb = {"question": a["scenario"], "answer": b["expected"], "correct": False}
                pairs.append((qa, qb, 1))
    return pairs

def embed_texts(model, texts):
    return model.encode(texts, convert_to_tensor=True)

def extract_features(q, model):
    # Usa embeddings de pregunta y respuesta esperada
    q_text = q.get("question") or q.get("scenario")
    a_text = q.get("answer") or q.get("expected")
    q_emb = model.encode([q_text], convert_to_tensor=True)[0]
    a_emb = model.encode([a_text], convert_to_tensor=True)[0]
    return torch.cat([q_emb, a_emb], dim=0)

def main():
    print("Cargando datos y modelo de embeddings...")
    tech = load_jsonl(TECH_PATH)
    soft = load_jsonl(SOFT_PATH)
    finetune = load_jsonl(FINETUNE_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL)
    # MiniLM-L6-v2 produce embeddings de 384 dimensiones
    input_dim = 384 * 2
    print(f"Dimensión de entrada: {input_dim}")
    print("Construyendo pares de entrenamiento...")
    pairs = build_pairs_from_finetune(finetune)
    print(f"Total de pares: {len(pairs)}")
    X1, X2, y = [], [], []
    for a, b, label in tqdm(pairs):
        X1.append(extract_features(a, model))
        X2.append(extract_features(b, model))
        y.append(label)
    X1 = torch.stack(X1)
    X2 = torch.stack(X2)
    y = torch.tensor(y, dtype=torch.float32)
    print("Entrenando RankNet...")
    ranknet = RankNet(input_dim)
    optimizer = optim.Adam(ranknet.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        ranknet.train()
        total_loss = 0
        for i in range(0, len(X1), BATCH_SIZE):
            x1b = X1[i:i+BATCH_SIZE]
            x2b = X2[i:i+BATCH_SIZE]
            yb = y[i:i+BATCH_SIZE]
            s1 = ranknet(x1b).squeeze()
            s2 = ranknet(x2b).squeeze()
            pred = s1 - s2
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")
    print(f"Guardando modelo en {MODEL_SAVE_PATH}")
    torch.save(ranknet.state_dict(), MODEL_SAVE_PATH)
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
