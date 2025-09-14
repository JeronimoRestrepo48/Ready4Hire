import torch
import numpy as np
from app.embeddings.embeddings_manager import EmbeddingsManager

# Cargar el gestor de embeddings y el modelo RankNet
emb_mgr = EmbeddingsManager()

# Cargar modelo RankNet entrenado
class RankNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

def rank_questions_with_model(user_context, candidate_questions, model_path, emb_mgr):
    # Embedding del contexto
    context_emb = emb_mgr.model.encode([user_context])[0]
    # Embedding de cada pregunta candidata
    question_embs = [emb_mgr.model.encode([(q.get('question') or q.get('scenario'))])[0] for q in candidate_questions]
    # Concatenar contexto + pregunta
    features = np.array([np.concatenate([context_emb, qemb]) for qemb in question_embs])
    # Cargar modelo
    model = RankNet(input_dim=features.shape[1])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(features, dtype=torch.float32)).numpy().flatten()
    # Ordenar preguntas por score
    idx = np.argsort(scores)[::-1]
    return [candidate_questions[i] for i in idx]

# Ejemplo de uso en tu pipeline:
# user_context = "Quiero prepararme para DevOps nivel senior"
# candidates = emb_mgr.advanced_question_selector(user_context, history=[], top_k=10, technical=True)
# ranked = rank_questions_with_model(user_context, candidates, 'ranknet.pth', emb_mgr)
# print(ranked)
