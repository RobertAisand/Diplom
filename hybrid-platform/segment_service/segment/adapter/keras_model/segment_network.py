"""segment/adapter/keras_model/segment_network.py — нейросеть Сегмента.

Реализация порта InferencePort: прямой проход доменной сети со встроенными
слоями-затворами (микро-уровень Φ_micro). Сеть строится средствами Functional
API Keras (подраздел 3.6.3): между плотными слоями вставлены STEHardGate,
обеспечивающие динамическую разреженность.

РЕКОНСТРУКЦИЯ: в главах приведена лишь тестовая заглушка (3.6.4), полной сети
Сегмента нет. Архитектура ниже — минимальный пример «энкодер-классификатор» с
затворами; замените на свою целевую сеть.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

from segment.adapter.keras_model.gates import STEHardGate


@dataclass
class InferenceResult:
    output_text: str
    rho_micro: float
    active_units: int
    total_units: int


class SegmentNetwork:
    """Нейросеть доменного Сегмента с микро-уровневыми затворами."""

    def __init__(self, model: keras.Model, gate_layers: list[STEHardGate]) -> None:
        self._model = model
        self._gates = gate_layers

    # ------------------------------------------------------------------ #
    # Фабрики                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, input_dim: int = 512, hidden: int = 256,
              n_classes: int = 8) -> "SegmentNetwork":
        """Строит сеть на Functional API: Dense → STEHardGate → Dense (выход)."""
        inp = keras.Input(shape=(input_dim,), name="context")
        h = keras.layers.Dense(hidden, activation="relu", name="dense_1")(inp)
        gate = STEHardGate(units=hidden, gate_bias_init=-0.3, name="micro_gate")
        h = gate(h)                                  # прореживание микро-уровня
        out = keras.layers.Dense(n_classes, activation="softmax", name="head")(h)
        model = keras.Model(inputs=inp, outputs=out, name="segment_network")
        return cls(model=model, gate_layers=[gate])

    @classmethod
    def load(cls, path: str) -> "SegmentNetwork":
        """Загружает обученные веса (.keras). Кастомные слои регистрируются."""
        model = keras.models.load_model(
            path, custom_objects={"STEHardGate": STEHardGate}
        )
        gates = [l for l in model.layers if isinstance(l, STEHardGate)]
        return cls(model=model, gate_layers=gates)

    # ------------------------------------------------------------------ #
    # Порт InferencePort                                                 #
    # ------------------------------------------------------------------ #
    def infer(self, context_vectors: np.ndarray) -> InferenceResult:
        """Прямой проход разреженного суб-графа + отчёт о разреженности."""
        x = tf.constant(np.atleast_2d(context_vectors).astype(np.float32))
        logits = self._model(x, training=False)
        cls_id = int(tf.argmax(logits[0]).numpy())

        # Отчёт о фактической разреженности по первому затвору (выр. 2.46).
        rho, active, total = self._gates[0].last_sparsity(
            self._model.get_layer("dense_1")(x), epsilon=0.0
        )
        return InferenceResult(
            output_text=f"class_{cls_id}",
            rho_micro=rho, active_units=active, total_units=total,
        )