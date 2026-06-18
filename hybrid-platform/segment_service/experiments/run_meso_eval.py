"""experiments/run_meso_eval.py — эксперимент 4.2.1: качество фильтрации мезо-уровня.

Строит синаптические профили 5 доменов из истинной статистики (μk, Σk),
прогоняет валидационную выборку через решающее правило Махаланобиса (выр. 2.29)
и вычисляет матрицу ошибок и метрики качества для каждого Сегмента и каждого
порога τd из сетки:

    Accuracy  (4.1),  Precision (4.2),  Recall (4.3),  F1 (4.4),
    макроусреднение (4.5),  оптимальный τd* = argmax F1 (4.6).

Результат сохраняется в experiments/results/meso_eval.csv и печатается сводка.
Запуск:  python -m experiments.run_meso_eval
"""

from __future__ import annotations

import csv
import os

import numpy as np

from experiments.make_dataset import make_dataset
from segment.entity.synaptic_profile import SynapticProfile

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_profiles(profiles_stats) -> list[SynapticProfile]:
    """Из истинной доменной статистики строит синаптические профили (фаза
    инициализации, подраздел 3.5.2): однократное обращение Σk → Σk⁻¹."""
    return [
        SynapticProfile.from_domain_statistics(centroid=mu, covariance=sigma)
        for (mu, sigma) in profiles_stats
    ]


def distance_matrix(profiles: list[SynapticProfile], X: np.ndarray) -> np.ndarray:
    """Матрица расстояний Махаланобиса (N × K): δ[i,k] = δk_mah(X[i])."""
    n, k = X.shape[0], len(profiles)
    D = np.empty((n, k), dtype=np.float64)
    for j, prof in enumerate(profiles):
        diff = X - prof.mu                       # (N, d)
        weighted = diff @ prof.sigma_inv         # (N, d), векторизованно
        q = np.einsum("ij,ij->i", diff, weighted)  # квадратичная форма построчно
        q = np.clip(q, 0.0, None)
        D[:, j] = np.sqrt(q)
    return D


def metrics_for_threshold(D: np.ndarray, y: np.ndarray, tau: float, k_domains: int):
    """Для порога τd считает TP/FP/FN/TN и метрики по каждому домену.

    Фрагмент с истинной меткой k допускается в Сегмент k ⇔ δ[i,k] ≤ τd.
    Метка == k_domains означает фон (не принадлежит ни одному домену).
    """
    rows = []
    for k in range(k_domains):
        admitted = D[:, k] <= tau           # решение правила (2.29) для Сегмента k
        relevant = (y == k)                 # истинная принадлежность домену k
        tp = int(np.sum(admitted & relevant))
        fp = int(np.sum(admitted & ~relevant))
        fn = int(np.sum(~admitted & relevant))
        tn = int(np.sum(~admitted & ~relevant))
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)        # (4.1)
        pre = tp / max(tp + fp, 1)                          # (4.2)
        rec = tp / max(tp + fn, 1)                          # (4.3)
        f1 = (2 * pre * rec / (pre + rec)) if (pre + rec) > 0 else 0.0  # (4.4)
        rows.append(dict(domain=k, tau=tau, tp=tp, fp=fp, fn=fn, tn=tn,
                         accuracy=acc, precision=pre, recall=rec, f1=f1))
    return rows


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Генерация выборки (раздел 4.1.2)…")
    X, y, stats = make_dataset()
    k_domains = len(stats)
    profiles = build_profiles(stats)

    print("Вычисление матрицы расстояний Махаланобиса…")
    D = distance_matrix(profiles, X)

    # Сетка порогов τd: от медианы расстояний до 90-го перцентиля.
    taus = np.quantile(D, np.linspace(0.01, 0.6, 40))

    all_rows = []
    macro_f1_by_tau = []
    for tau in taus:
        rows = metrics_for_threshold(D, y, float(tau), k_domains)
        all_rows.extend(rows)
        macro_f1 = float(np.mean([r["f1"] for r in rows]))     # макроусреднение (4.5)
        macro_f1_by_tau.append((float(tau), macro_f1))

    # Оптимальный порог τd* = argmax макро-F1 (4.6).
    tau_star, best_f1 = max(macro_f1_by_tau, key=lambda t: t[1])

    # Сохранение полной развёртки.
    csv_path = os.path.join(RESULTS_DIR, "meso_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nРазвёртка по τd сохранена: {csv_path}")
    print(f"Оптимальный порог τd* = {tau_star:.4f}, макро-F1 = {best_f1:.4f}  (выр. 4.6)\n")

    # Сводка по доменам в оптимуме.
    print(f"{'домен':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for r in metrics_for_threshold(D, y, tau_star, k_domains):
        print(f"{r['domain']:>6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f}")


if __name__ == "__main__":
    main()