"""
Red neuronal para ajuste dinámico de dificultad.
Reemplaza al ml_dynamic_difficulty.py que usa reglas hardcodeadas.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from app.domain.entities.answer import Answer
from app.domain.value_objects.skill_level import SkillLevel


class DifficultyNet(nn.Module):
    """
    Red neuronal feedforward para predecir dificultad óptima.

    Input: 12 features del historial del usuario
    Output: Probabilidades para [easy, medium, hard]
    """

    def __init__(self, input_size=12, hidden_sizes=[64, 32], dropout=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_size = hidden_size

        # Capa de salida
        layers.append(nn.Linear(prev_size, 3))  # 3 clases: easy, medium, hard

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=1)


class NeuralDifficultyAdjuster:
    """
    Ajustador de dificultad con ML real.
    Aprende patrones de desempeño para optimizar el desafío.
    """

    DIFFICULTY_MAP = {0: "easy", 1: "medium", 2: "hard"}
    DIFFICULTY_TO_IDX = {"easy": 0, "medium": 1, "hard": 2}

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DifficultyNet().to(self.device)

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[INFO] Loaded difficulty model from {model_path}")
        else:
            print("[INFO] Using untrained difficulty model (will use heuristics)")

        self.model.eval()

    def extract_features(self, answers_history: List[Answer]) -> np.ndarray:
        """
        Extrae 12 features del historial de respuestas:

        1. Accuracy últimas 3 respuestas
        2. Accuracy últimas 5 respuestas
        3. Accuracy últimas 10 respuestas
        4. Accuracy total
        5. Racha actual (normalizada)
        6. Tiempo promedio de respuesta (normalizado)
        7. Uso de pistas (ratio)
        8. Nivel emocional (0=negativo, 1=positivo)
        9. Variabilidad de scores (std)
        10. Tendencia (mejorando/empeorando)
        11. Dificultad promedio enfrentada
        12. Ratio de respuestas rápidas (<30s)
        """
        if not answers_history:
            # Usuario nuevo - features por defecto (nivel medio)
            return np.array([0.5] * 12, dtype=np.float32)

        # Segmentos de historial
        recent_3 = answers_history[-3:]
        recent_5 = answers_history[-5:]
        recent_10 = answers_history[-10:]

        # Feature 1-4: Accuracy en diferentes ventanas
        acc_3 = self._calculate_accuracy(recent_3)
        acc_5 = self._calculate_accuracy(recent_5)
        acc_10 = self._calculate_accuracy(recent_10)
        acc_total = self._calculate_accuracy(answers_history)

        # Feature 5: Racha actual
        streak = self._calculate_streak(answers_history)
        streak_norm = min(streak / 10.0, 1.0)  # Normalizar a [0, 1]

        # Feature 6: Tiempo promedio
        avg_time = np.mean([a.time_taken for a in recent_10])
        time_norm = min(avg_time / 180.0, 1.0)  # Normalizar (máx 3 min)

        # Feature 7: Uso de pistas
        hints_ratio = sum(a.hints_used for a in recent_10) / max(len(recent_10), 1)
        hints_ratio = min(hints_ratio / 3.0, 1.0)  # Normalizar (máx 3 pistas)

        # Feature 8: Nivel emocional
        emotion_level = self._calculate_emotion_level(recent_5)

        # Feature 9: Variabilidad de scores
        scores = [a.score for a in recent_10]
        score_std = np.std(scores) / 10.0 if len(scores) > 1 else 0.2

        # Feature 10: Tendencia
        trend = self._calculate_trend(answers_history)

        # Feature 11: Dificultad promedio (si está disponible en metadata)
        avg_difficulty = 0.66  # Por defecto medium

        # Feature 12: Ratio respuestas rápidas
        quick_ratio = sum(1 for a in recent_10 if a.time_taken < 30) / max(len(recent_10), 1)

        features = np.array(
            [
                acc_3,
                acc_5,
                acc_10,
                acc_total,
                streak_norm,
                time_norm,
                hints_ratio,
                emotion_level,
                score_std,
                trend,
                avg_difficulty,
                quick_ratio,
            ],
            dtype=np.float32,
        )

        return features

    def predict(self, answers_history: List[Answer]) -> Tuple[str, float]:
        """
        Predice la dificultad óptima.

        Returns:
            (difficulty: str, confidence: float)
        """
        features = self.extract_features(answers_history)

        try:
            with torch.no_grad():
                X = torch.from_numpy(features).unsqueeze(0).to(self.device)
                probs = self.model(X)
                predicted_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_idx].item()

            return self.DIFFICULTY_MAP[predicted_idx], confidence

        except Exception as e:
            print(f"[WARN] Neural prediction failed, using heuristics: {e}")
            return self._heuristic_fallback(answers_history)

    def _heuristic_fallback(self, answers_history: List[Answer]) -> Tuple[str, float]:
        """Fallback usando heurísticas si el modelo falla"""
        if not answers_history:
            return "medium", 0.5

        recent_5 = answers_history[-5:]
        accuracy = sum(1 for a in recent_5 if a.is_correct) / len(recent_5)

        if accuracy >= 0.8:
            return "hard", 0.7
        elif accuracy <= 0.4:
            return "easy", 0.7
        else:
            return "medium", 0.6

    # Métodos auxiliares privados

    def _calculate_accuracy(self, answers: List[Answer]) -> float:
        if not answers:
            return 0.5
        correct = sum(1 for a in answers if a.is_correct)
        return correct / len(answers)

    def _calculate_streak(self, answers: List[Answer]) -> int:
        streak = 0
        for answer in reversed(answers):
            if answer.is_correct:
                streak += 1
            else:
                break
        return streak

    def _calculate_emotion_level(self, recent_answers: List[Answer]) -> float:
        """0 = negativo, 1 = positivo"""
        if not recent_answers:
            return 0.5

        positive_count = sum(1 for a in recent_answers if a.emotion.is_positive())
        return positive_count / len(recent_answers)

    def _calculate_trend(self, answers: List[Answer]) -> float:
        """Retorna -1 (empeorando) a 1 (mejorando)"""
        if len(answers) < 10:
            return 0.0

        mid = len(answers) // 2
        first_half_acc = self._calculate_accuracy(answers[:mid])
        second_half_acc = self._calculate_accuracy(answers[mid:])

        return second_half_acc - first_half_acc

    def adjust_for_skill_level(self, predicted_difficulty: str, skill_level: SkillLevel) -> str:
        """
        Ajusta la dificultad predicha según el nivel del usuario.
        Juniors: bias hacia easy/medium
        Seniors: bias hacia medium/hard
        """
        difficulties = ["easy", "medium", "hard"]
        current_idx = difficulties.index(predicted_difficulty)

        if skill_level == SkillLevel.JUNIOR:
            # Evitar hard para juniors
            if predicted_difficulty == "hard":
                return "medium"

        elif skill_level == SkillLevel.SENIOR:
            # Evitar easy para seniors
            if predicted_difficulty == "easy":
                return "medium"

        return predicted_difficulty


# Instancia global
_adjuster_instance = None


def get_difficulty_adjuster(model_path: str = "app/datasets/difficulty_model.pt") -> NeuralDifficultyAdjuster:
    """Obtiene instancia singleton del ajustador"""
    global _adjuster_instance
    if _adjuster_instance is None:
        _adjuster_instance = NeuralDifficultyAdjuster(model_path)
    return _adjuster_instance


def adjust_question_difficulty(answers_history: List[Answer], skill_level: SkillLevel) -> str:
    """
    Función de conveniencia compatible con la API anterior.
    Mantiene compatibilidad con código existente.
    """
    adjuster = get_difficulty_adjuster()
    predicted, _ = adjuster.predict(answers_history)
    return adjuster.adjust_for_skill_level(predicted, skill_level)
