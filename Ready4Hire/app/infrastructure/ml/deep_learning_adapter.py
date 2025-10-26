"""
Deep Learning Adaptive Learning System
Sistema de aprendizaje adaptativo con redes neuronales
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """Estado de aprendizaje del candidato"""

    user_id: str
    skill_vector: np.ndarray  # Vector de habilidades [N_skills]
    knowledge_matrix: np.ndarray  # Matriz de conocimiento [N_topics x N_concepts]
    learning_rate: float  # Tasa de aprendizaje del candidato
    mastery_level: float  # Nivel de maestría general (0-1)
    next_recommended_topics: List[str]


class SkillEvolutionNN(nn.Module):
    """
    Red Neuronal para modelar evolución de habilidades.

    Arquitectura:
    - Input: estado actual + pregunta respondida + resultado
    - Hidden layers: 2 capas con dropout
    - Output: nuevo estado de habilidades

    Permite predecir cómo evolucionan las habilidades
    del candidato tras cada interacción.
    """

    def __init__(self, n_skills: int = 50, n_topics: int = 30, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        self.n_skills = n_skills
        self.n_topics = n_topics

        # Input: [skill_vector, question_features, result_features]
        input_dim = n_skills + n_topics + 10  # +10 para features de resultado

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Skill update branch
        self.skill_updater = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_skills),
            nn.Tanh(),  # Output: cambio en skills (-1 a +1)
        )

        # Mastery predictor branch
        self.mastery_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # Output: nivel de maestría (0-1)
        )

    def forward(
        self, skill_vector: torch.Tensor, question_features: torch.Tensor, result_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            skill_vector: [batch, n_skills]
            question_features: [batch, n_topics]
            result_features: [batch, 10]

        Returns:
            (skill_delta, mastery_level)
        """
        # Concatenar inputs
        x = torch.cat([skill_vector, question_features, result_features], dim=1)

        # Encode
        encoded = self.encoder(x)

        # Predict updates
        skill_delta = self.skill_updater(encoded)
        mastery = self.mastery_predictor(encoded)

        return skill_delta, mastery


class DeepLearningAdapter:
    """
    Sistema de aprendizaje adaptativo con Deep Learning.

    Funcionalidad:
    1. Modela el estado de conocimiento del candidato
    2. Predice evolución de habilidades tras cada respuesta
    3. Recomienda siguiente tema óptimo para maximizar aprendizaje
    4. Detecta plateau y sugiere cambios de estrategia
    5. Personaliza dificultad dinámicamente

    Benefits:
    - Aprendizaje más eficiente
    - Detección temprana de dificultades
    - Personalización extrema
    - Predicción de performance futuro
    """

    def __init__(self, n_skills: int = 50, n_topics: int = 30, model_path: Optional[Path] = None):
        self.n_skills = n_skills
        self.n_topics = n_topics

        self.model = SkillEvolutionNN(n_skills, n_topics)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.model_path = model_path

        # Cargar modelo pre-entrenado si existe
        if model_path and model_path.exists():
            self._load_model()

        # Skill y topic mappings (mock - en producción desde config)
        self.skill_names = [f"skill_{i}" for i in range(n_skills)]
        self.topic_names = [f"topic_{i}" for i in range(n_topics)]

    def initialize_user_state(self, user_id: str, role: str) -> LearningState:
        """
        Inicializa el estado de aprendizaje de un nuevo usuario.

        Args:
            user_id: ID del usuario
            role: Rol del usuario (para inicialización específica)

        Returns:
            LearningState inicial
        """
        # Inicialización con pequeños valores aleatorios
        skill_vector = np.random.randn(self.n_skills) * 0.1
        knowledge_matrix = np.random.randn(self.n_topics, 10) * 0.1

        return LearningState(
            user_id=user_id,
            skill_vector=skill_vector,
            knowledge_matrix=knowledge_matrix,
            learning_rate=1.0,  # Máximo al inicio
            mastery_level=0.0,  # Ninguna maestría al inicio
            next_recommended_topics=self._get_starter_topics(role),
        )

    def update_state(self, current_state: LearningState, question_data: Dict, answer_data: Dict) -> LearningState:
        """
        Actualiza el estado tras una respuesta.

        Args:
            current_state: Estado actual
            question_data: Datos de la pregunta
            answer_data: Datos de la respuesta

        Returns:
            Nuevo estado actualizado
        """
        self.model.eval()

        with torch.no_grad():
            # Preparar features
            skill_tensor = torch.FloatTensor(current_state.skill_vector).unsqueeze(0)

            question_features = self._encode_question(question_data)
            result_features = self._encode_result(answer_data)

            # Predecir cambios
            skill_delta, new_mastery = self.model(skill_tensor, question_features, result_features)

            # Aplicar actualización
            new_skill_vector = current_state.skill_vector + skill_delta.numpy()[0] * 0.1

            # Clip a rango válido
            new_skill_vector = np.clip(new_skill_vector, -3, 3)

        # Actualizar knowledge matrix (heurística simple)
        new_knowledge_matrix = current_state.knowledge_matrix.copy()
        topic_idx = self._topic_to_index(question_data.get("topic", ""))
        if 0 <= topic_idx < self.n_topics:
            score = answer_data.get("score", 0) / 10.0  # Normalizar
            new_knowledge_matrix[topic_idx] += score * 0.1

        # Calcular nueva learning rate (decay si score alto consistente)
        learning_rate = current_state.learning_rate
        if answer_data.get("score", 0) >= 8.0:
            learning_rate = max(0.5, learning_rate * 0.95)
        else:
            learning_rate = min(1.0, learning_rate * 1.05)

        # Recomendar próximos topics
        next_topics = self._recommend_next_topics(new_skill_vector, new_knowledge_matrix, question_data.get("role", ""))

        return LearningState(
            user_id=current_state.user_id,
            skill_vector=new_skill_vector,
            knowledge_matrix=new_knowledge_matrix,
            learning_rate=learning_rate,
            mastery_level=float(new_mastery.item()),
            next_recommended_topics=next_topics,
        )

    def predict_performance(self, state: LearningState, question_data: Dict) -> float:
        """
        Predice el performance esperado en una pregunta.

        Args:
            state: Estado actual
            question_data: Pregunta a predecir

        Returns:
            Score esperado (0-10)
        """
        self.model.eval()

        with torch.no_grad():
            skill_tensor = torch.FloatTensor(state.skill_vector).unsqueeze(0)
            question_features = self._encode_question(question_data)

            # Mock result para predicción
            mock_result = torch.zeros(1, 10)

            _, predicted_mastery = self.model(skill_tensor, question_features, mock_result)

            # Convertir mastery a score esperado
            expected_score = predicted_mastery.item() * 10.0

        return expected_score

    def detect_plateau(self, state: LearningState, recent_scores: List[float], window: int = 5) -> bool:
        """
        Detecta si el candidato está en plateau (no mejora).

        Args:
            state: Estado actual
            recent_scores: Scores recientes
            window: Ventana para analizar

        Returns:
            True si está en plateau
        """
        if len(recent_scores) < window:
            return False

        recent = recent_scores[-window:]

        # Plateau = varianza baja y no mejora
        variance = np.var(recent)
        trend = np.polyfit(range(len(recent)), recent, 1)[0]  # Pendiente

        is_plateau = variance < 0.5 and abs(trend) < 0.1

        return is_plateau

    def recommend_intervention(self, state: LearningState, recent_scores: List[float]) -> str:
        """
        Recomienda intervención si es necesario.

        Args:
            state: Estado actual
            recent_scores: Scores recientes

        Returns:
            Mensaje de intervención recomendada
        """
        if self.detect_plateau(state, recent_scores):
            return "🔄 Detectamos un plateau en tu aprendizaje. Te recomendamos cambiar de tema o intentar un enfoque diferente."

        if state.learning_rate < 0.6:
            return "🎯 ¡Excelente progreso! Estás cerca de dominar este tema. Te sugerimos avanzar al siguiente nivel."

        avg_score = np.mean(recent_scores[-5:]) if len(recent_scores) >= 5 else 0

        if avg_score < 6.0:
            return "💡 Parece que este tema es desafiante. ¿Quieres revisar algunos conceptos básicos primero?"

        return ""

    def _encode_question(self, question_data: Dict) -> torch.Tensor:
        """Codifica pregunta a vector de features"""
        topic_idx = self._topic_to_index(question_data.get("topic", ""))

        # One-hot encoding del topic
        topic_vec = torch.zeros(1, self.n_topics)
        if 0 <= topic_idx < self.n_topics:
            topic_vec[0, topic_idx] = 1.0

        return topic_vec

    def _encode_result(self, answer_data: Dict) -> torch.Tensor:
        """Codifica resultado a vector de features"""
        features = torch.zeros(1, 10)

        # Feature 0: Score normalizado
        features[0, 0] = answer_data.get("score", 0) / 10.0

        # Feature 1: Si es correcto
        features[0, 1] = 1.0 if answer_data.get("is_correct", False) else 0.0

        # Feature 2: Tiempo normalizado (log)
        time_taken = answer_data.get("time_taken", 60)
        features[0, 2] = min(1.0, np.log(time_taken + 1) / 10.0)

        # Feature 3: Hints usados
        features[0, 3] = min(1.0, answer_data.get("hints_used", 0) / 3.0)

        # Feature 4-9: Reservadas para futuras features

        return features

    def _topic_to_index(self, topic: str) -> int:
        """Convierte topic a índice"""
        try:
            return self.topic_names.index(topic)
        except ValueError:
            # Topic desconocido, mapear a hash consistente
            return hash(topic) % self.n_topics

    def _get_starter_topics(self, role: str) -> List[str]:
        """Obtiene topics iniciales recomendados por rol"""
        # Mock - en producción desde DB
        starter_map = {
            "Software Engineer": ["Programming Basics", "Data Structures", "Algorithms"],
            "Data Scientist": ["Statistics", "Python", "Machine Learning Basics"],
            "Product Manager": ["Product Strategy", "User Research", "Metrics"],
        }

        return starter_map.get(role, ["General Knowledge"])

    def _recommend_next_topics(self, skill_vector: np.ndarray, knowledge_matrix: np.ndarray, role: str) -> List[str]:
        """Recomienda próximos topics basado en estado"""
        # Calcular "gap" para cada topic (cuánto falta por aprender)
        topic_gaps = []

        for i in range(self.n_topics):
            knowledge_level = np.mean(knowledge_matrix[i])
            skill_readiness = skill_vector[i % len(skill_vector)]

            # Gap = potencial de aprendizaje
            gap = (1.0 - knowledge_level) * (1.0 + skill_readiness)
            topic_gaps.append((gap, self.topic_names[i]))

        # Ordenar por gap descendente
        topic_gaps.sort(reverse=True)

        # Retornar top 3
        return [topic for _, topic in topic_gaps[:3]]

    def train_step(
        self,
        skill_vectors: torch.Tensor,
        question_features: torch.Tensor,
        result_features: torch.Tensor,
        target_skill_deltas: torch.Tensor,
        target_mastery: torch.Tensor,
    ) -> float:
        """
        Paso de entrenamiento del modelo.

        Args:
            Batches de datos de entrenamiento

        Returns:
            Loss del batch
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward
        pred_skill_delta, pred_mastery = self.model(skill_vectors, question_features, result_features)

        # Loss
        skill_loss = self.criterion(pred_skill_delta, target_skill_deltas)
        mastery_loss = self.criterion(pred_mastery, target_mastery)

        total_loss = skill_loss + mastery_loss

        # Backward
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def save_model(self, path: Optional[Path] = None) -> None:
        """Guarda modelo en disco"""
        save_path = path or self.model_path
        if not save_path:
            logger.warning("No save path specified for DL model")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

        logger.info(f"DL model saved to {save_path}")

    def _load_model(self) -> None:
        """Carga modelo desde disco"""
        if not self.model_path or not self.model_path.exists():
            return

        try:
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"DL model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading DL model: {e}")


# Factory
_dl_adapter = None


def get_dl_adapter() -> DeepLearningAdapter:
    """Obtiene instancia singleton del adaptador de DL"""
    global _dl_adapter

    if _dl_adapter is None:
        from app.config import settings

        model_path = Path(settings.DATA_PATH) / "models" / "adaptive_learning.pt"
        _dl_adapter = DeepLearningAdapter(model_path=model_path)

    return _dl_adapter
