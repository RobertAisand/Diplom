"""segment/usecase/segment_inference.py — сценарий микро-уровня (Use Case).

Принимает допущенный мезо-фильтром контекст, векторизует его и выполняет
инференс разреженной сети Сегмента, возвращая результат и отчёт о разреженности.
РЕКОНСТРУКЦИЯ: в главах сценарий упомянут (SegmentInferenceUseCase), листинга нет.
"""

from __future__ import annotations

import numpy as np

from segment.usecase.ports import EmbeddingPort


class SegmentInferenceUseCase:
    """Сценарий инференса Сегмента (оператор Φ_micro во взаимодействии с Φ_meso)."""

    def __init__(self, network, encoder: EmbeddingPort) -> None:
        self._network = network    # реализация InferencePort (SegmentNetwork)
        self._encoder = encoder    # реализация EmbeddingPort

    def run(self, admitted_fragments: list[str]):
        """Векторизует допущенные фрагменты, усредняет в контекст и инферит."""
        if not admitted_fragments:
            return None
        vectors = np.vstack([self._encoder.embed(f) for f in admitted_fragments])
        context = vectors.mean(axis=0)         # агрегированное представление контекста
        return self._network.infer(context)