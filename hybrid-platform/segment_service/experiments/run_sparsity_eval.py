"""experiments/run_sparsity_eval.py — эксперимент 4.2.2: степень прореживания.

Измеряет коэффициент разреженности ρ_micro (выр. 2.46) и степень прореживания
Sparsity Ratio S = 1 − ρ_micro (выр. 4.7–4.8) на STEHardGate по тестовой
выборке, а также среднюю степень прореживания S̄ (выр. 4.9).

Требует TensorFlow.  Запуск:  python -m experiments.run_sparsity_eval
"""

from __future__ import annotations

import csv
import os

import numpy as np
import tensorflow as tf

from experiments.make_dataset import make_dataset
from segment.adapter.keras_model.gates import STEHardGate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DIM = 512
GATE_UNITS = 512


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Генерация выборки…")
    X, y, _ = make_dataset(dim=DIM)
    X = X.astype(np.float32)

    # Затвор инициализирован со смещением, дающим частичное закрытие узлов.
    gate = STEHardGate(units=GATE_UNITS, gate_bias_init=-0.0)
    _ = gate(tf.zeros((1, DIM)))  # build

    batch = 2048
    rhos, actives, totals = [], [], []
    for i in range(0, len(X), batch):
        chunk = tf.constant(X[i:i + batch])
        rho, active, total = gate.last_sparsity(chunk, epsilon=0.0)  # выр. 2.45–2.46
        rhos.append(rho); actives.append(active); totals.append(total)

    mean_rho = float(np.mean(rhos))          # средний ρ_micro
    mean_S = 1.0 - mean_rho                   # средняя степень прореживания (4.9)

    csv_path = os.path.join(RESULTS_DIR, "sparsity_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_index", "rho_micro", "sparsity_ratio", "active", "total"])
        for idx, (r, a, t) in enumerate(zip(rhos, actives, totals)):
            w.writerow([idx, f"{r:.6f}", f"{1.0 - r:.6f}", a, t])

    print(f"\nСохранено: {csv_path}")
    print(f"Средний ρ_micro = {mean_rho:.4f}  (выр. 2.46)")
    print(f"Средняя степень прореживания S̄ = {mean_S:.4f}  (выр. 4.8–4.9)")
    print(f"То есть STE-затвор подавляет ≈ {mean_S*100:.1f}% узлов слоя.")


if __name__ == "__main__":
    main()