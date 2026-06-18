"""segment/usecase/ports.py — порты слоя доменных сценариев.

Согласно правилу зависимостей (подраздел 3.5.1), сценарии объявляют абстрактные
интерфейсы (порты) и не знают их конкретных реализаций; связывание портов с
адаптерами производится в композиционном корне main.py (инверсия зависимостей).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingPort(Protocol):
    """Порт оператора векторизации Φ (выражения 2.22–2.23).

    Реализуется адаптером энкодера. Возвращает нормированное представление
    v̂ = Φ(x)/‖Φ(x)‖₂ на единичной гиперсфере (‖v̂‖₂ = 1).
    """

    def embed(self, text: str) -> np.ndarray:
        ...


class InferencePort(Protocol):
    """Порт нейросети инференса Сегмента (микро-уровень Φ_micro).

    Реализуется адаптером keras_model. Выполняет прямой проход разреженного
    суб-графа со встроенными слоями-затворами и возвращает результат вместе с
    отчётом о фактической разреженности ρ_micro (выражение 2.46).
    """

    def infer(self, context: list[str]) -> "InferenceResult":
        ...


class InferenceResult(Protocol):
    """Результат инференса: выход сети и отчёт о разреженности."""

    output_text: str
    rho_micro: float
    active_units: int
    total_units: int