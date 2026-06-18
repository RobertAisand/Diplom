"""experiments/run_latency_bench.py — эксперимент 4.2.3: временные метрики.

Измеряет латентность мезо-уровня (векторизация + расстояние Махаланобиса) на
одиночных запросах и вычисляет:
    t_avg  (4.10),  p95 и p99  (4.11–4.12).

Замер ведётся в установившемся режиме после прогрева (warm-up), первый «холодный»
запрос исключается (условия измерения, раздел 4.1.3).

Запуск:  python -m experiments.run_latency_bench
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from experiments.make_dataset import make_dataset
from segment.entity.synaptic_profile import SynapticProfile

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_MEASURE = 20_000   # число замеряемых запросов
N_WARMUP = 1_000     # прогрев


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y, stats = make_dataset()
    profile = SynapticProfile.from_domain_statistics(*stats[0])

    # Прогрев (исключается из измерения).
    for i in range(N_WARMUP):
        profile.mahalanobis_distance(X[i])

    # Измерение латентности по одному запросу (на горячем пути, выр. 4.13: t_meso).
    times = np.empty(N_MEASURE, dtype=np.float64)
    for i in range(N_MEASURE):
        v = X[N_WARMUP + i]
        t0 = time.perf_counter()
        profile.mahalanobis_distance(v)
        times[i] = (time.perf_counter() - t0) * 1e6  # микросекунды

    t_avg = float(np.mean(times))                       # (4.10)
    p95 = float(np.quantile(times, 0.95))               # (4.12)
    p99 = float(np.quantile(times, 0.99))               # (4.12)

    csv_path = os.path.join(RESULTS_DIR, "latency_bench.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value_microseconds"])
        w.writerow(["t_avg", f"{t_avg:.4f}"])
        w.writerow(["p95", f"{p95:.4f}"])
        w.writerow(["p99", f"{p99:.4f}"])

    print(f"\nСохранено: {csv_path}")
    print(f"Латентность мезо-уровня (расстояние Махаланобиса, d=512):")
    print(f"  t_avg = {t_avg:.2f} мкс   (выр. 4.10)")
    print(f"  p95   = {p95:.2f} мкс     (выр. 4.12)")
    print(f"  p99   = {p99:.2f} мкс     (выр. 4.12)")
    print(f"  хвост p99 − t_avg = {p99 - t_avg:.2f} мкс")


if __name__ == "__main__":
    main()