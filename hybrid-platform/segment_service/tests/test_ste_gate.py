"""tests/test_ste_gate.py — верификация свойств STEHardGate (листинг 3.6.4).

Тест проверяет рассогласование прямого и обратного проходов, составляющее
существо метода Straight-Through Estimator. Прохождение теста есть
эксплуатационное подтверждение того, что выражения (2.50–2.55) Главы 2
корректно реализованы в коде слоя.

Требует TensorFlow.  Запуск:  pytest tests/test_ste_gate.py -v
"""

import numpy as np
import tensorflow as tf

from segment.adapter.keras_model.gates import STEHardGate, _ste_hard_threshold


def test_forward_pass_is_hard_threshold() -> None:
    """Свойство блокировки прямого прохода: g_hard ∈ {0,1}, узлы с z<0
    строго подавлены до нуля (выражение 2.50)."""
    z = tf.constant([[-2.0, -0.1, 0.0, 0.1, 2.0]])
    g_hard = _ste_hard_threshold(z).numpy()
    assert set(g_hard.flatten()) <= {0.0, 1.0}
    expected = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(g_hard, expected)


def test_backward_pass_is_sigmoid_gradient() -> None:
    """Свойство проброса градиента: где жёсткий порог имеет нулевую
    производную, градиент через STE равен σ′(z) (выражение 2.54)."""
    z = tf.Variable([[-2.0, -0.1, 0.0, 0.1, 2.0]])
    with tf.GradientTape() as tape:
        g_hard = _ste_hard_threshold(z)
        loss = tf.reduce_sum(g_hard)   # ∂L/∂g_hard ≡ 1 ⇒ ∂L/∂z = σ′(z)
    grad_actual = tape.gradient(loss, z).numpy()
    sigma = 1.0 / (1.0 + np.exp(-z.numpy()))
    grad_expected = sigma * (1.0 - sigma)
    np.testing.assert_allclose(grad_actual, grad_expected, rtol=1e-6)
    # Градиент строго ненулевой везде, в том числе при z<0 — эффект проброса.
    assert np.all(grad_actual > 0.0)


def test_gate_trains_through_ste() -> None:
    """Интегральная проверка: параметры STEHardGate обновляются в обучении."""
    units = 4
    gate = STEHardGate(units=units, gate_bias_init=0.0)
    x = tf.random.normal((32, units))
    _ = gate(x)  # build
    before = [w.numpy().copy() for w in gate.trainable_weights]
    opt = tf.keras.optimizers.SGD(0.1)
    for _ in range(5):
        with tf.GradientTape() as tape:
            out = gate(x)
            loss = tf.reduce_mean(tf.square(out - x))
        grads = tape.gradient(loss, gate.trainable_weights)
        opt.apply_gradients(zip(grads, gate.trainable_weights))
    after = [w.numpy() for w in gate.trainable_weights]
    # Хотя бы один параметр изменился — STE пропускает градиент.
    assert any(not np.allclose(b, a) for b, a in zip(before, after))