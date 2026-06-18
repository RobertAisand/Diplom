"""segment/adapter/encoder/transformer_encoder.py — реализация EmbeddingPort.

Адаптер оператора векторизации Φ (выр. 2.22–2.23): прогоняет текст через
transformer-энкодер и нормирует результат на единичную гиперсферу.
РЕКОНСТРУКЦИЯ. Требует пакет sentence-transformers (pip install sentence-transformers).
"""

from __future__ import annotations

import numpy as np


class TransformerEncoder:
    """Реализация порта EmbeddingPort через sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Импорт внутри конструктора: тяжёлая зависимость не нужна ядру.
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Возвращает нормированное представление v̂ = Φ(x)/‖Φ(x)‖₂ (выр. 2.23)."""
        v = self._model.encode(text, convert_to_numpy=True).astype(np.float64)
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return v
        return v / norm