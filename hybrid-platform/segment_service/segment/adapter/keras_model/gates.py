"""segment/adapter/keras_model/gates.py — кастомные слои-затворы микро-уровня.

Реализация оператора Φ_micro (подразделы 3.6.1–3.6.2): дифференцируемые
сигмоидальные затворы средствами Functional API Keras.

  SigmoidGate  — мягкий затвор g(h) ∈ (0,1): модулирует активации, но не
                 подавляет их до строгого нуля (выражения 2.33–2.37);
  STEHardGate  — жёсткий затвор g_hard ∈ {0,1} с рассогласованием прямого и
                 обратного проходов методом Straight-Through Estimator
                 (выражения 2.50–2.56): прямой проход — жёсткий порог
                 (строгая разреженность), обратный — градиент сигмоиды
                 (сохранение обучаемости).

Слои принадлежат слою адаптеров (подраздел 3.1.2): математическое определение
затвора принадлежит ядру, а конкретный класс на тензорах TensorFlow есть
инфраструктурная деталь периферии.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Мягкий сигмоидальный затвор (подраздел 3.6.1, выражения 2.33–2.37)
# ---------------------------------------------------------------------------
class SigmoidGate(keras.layers.Layer):
    """Мягкий затвор: g(h) = σ(W_g·h + b_g) ∈ (0,1)ᵐ, h̃ = g(h) ⊙ a(h).

    Полностью дифференцируем: все операции (matmul, bias, sigmoid, Адамар)
    имеют зарегистрированные правила дифференцирования, поэтому градиент
    распространяется штатно, без дополнительных приёмов.
    """

    def __init__(self, units: int, gate_bias_init: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.gate_bias_init = float(gate_bias_init)

    def build(self, input_shape) -> None:
        in_dim = int(input_shape[-1])
        # Параметры затвора W_g, b_g (обучаемые).
        self.w_gate = self.add_weight(
            name="w_gate",
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_gate = self.add_weight(
            name="b_gate",
            shape=(self.units,),
            initializer=keras.initializers.Constant(self.gate_bias_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Несущая активация a(h) подаётся как inputs; решение затвора по тем же h.
        z = tf.matmul(inputs, self.w_gate) + self.b_gate   # предактивация
        g = tf.sigmoid(z)                                   # g(h) ∈ (0,1)
        return g * inputs                                   # произведение Адамара

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "gate_bias_init": self.gate_bias_init})
        return cfg


# ---------------------------------------------------------------------------
# STE-операция: жёсткий порог вперёд, градиент сигмоиды назад
# (подраздел 3.6.2, выражения 2.53–2.55)
# ---------------------------------------------------------------------------
@tf.custom_gradient
def _ste_hard_threshold(z: tf.Tensor):
    """Прямой проход — жёсткий порог 𝟙[z ≥ 0] ∈ {0,1}; обратный — σ′(z).

    Декоратор tf.custom_gradient задаёт градиент операции отдельно от её
    прямого определения, чем реализуется рассогласование проходов — существо STE.
    """
    # Прямой проход: 𝟙[σ(z) ≥ 0.5] ⇔ 𝟙[z ≥ 0] (монотонность сигмоиды).
    g_hard = tf.cast(z >= 0.0, dtype=z.dtype)

    def grad_fn(upstream):
        # Обратный проход: σ′(z) = σ(z)(1 − σ(z)) вместо нулевой производной порога.
        sigma = tf.sigmoid(z)
        sigma_prime = sigma * (1.0 - sigma)   # σ′(z) — выр. 2.54
        return upstream * sigma_prime         # цепное правило: ∂L/∂z

    return g_hard, grad_fn


class STEHardGate(keras.layers.Layer):
    """Жёсткий затвор с STE-рассогласованием проходов (выражения 2.50–2.56).

    Прямой проход:  g_hard(h) = 𝟙[ W_g·h + b_g ≥ 0 ] ∈ {0,1}ᵐ,  h̃ = g_hard ⊙ a(h)
                    — строгое подавление узлов и реальная разреженность графа.
    Обратный проход: градиент через сигмоидную суррогатную функцию — параметры
                     W_g, b_g обучаемы штатными средствами автодифференцирования.
    """

    def __init__(self, units: int, gate_bias_init: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.gate_bias_init = float(gate_bias_init)

    def build(self, input_shape) -> None:
        in_dim = int(input_shape[-1])
        self.w_gate = self.add_weight(
            name="w_gate",
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_gate = self.add_weight(
            name="b_gate",
            shape=(self.units,),
            initializer=keras.initializers.Constant(self.gate_bias_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w_gate) + self.b_gate
        g_hard = _ste_hard_threshold(z)         # {0,1}: жёсткое решение
        return g_hard * inputs                  # подавленные узлы → строгий ноль

    def last_sparsity(self, inputs, epsilon: float = 0.0):
        """Доля фактически активных узлов ρ_micro для батча (выражения 2.45–2.46).

        Узел активен ⇔ g_hard > epsilon. Возвращает (rho_micro, active, total).
        """
        z = tf.matmul(inputs, self.w_gate) + self.b_gate
        g_hard = tf.cast(z >= 0.0, dtype=z.dtype)
        active = tf.reduce_sum(tf.cast(g_hard > epsilon, tf.float32))
        total = tf.cast(tf.size(g_hard), tf.float32)
        rho = float(active / total)
        return rho, int(active.numpy()), int(total.numpy())

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "gate_bias_init": self.gate_bias_init})
        return cfg