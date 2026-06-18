"""experiments/make_dataset.py — генерация синтетического тестового датасета.

Воспроизводит структуру тестового потока из раздела 4.1.2: 5 доменных
подмножеств с анизотропными ковариациями (центроиды μk и матрицы Σk) плюс
фоновое (шумовое) подмножество для проверки резервной роли Базового сегмента.

Анизотропия ковариаций принципиальна: она создаёт условия, в которых евклидова
метрика не разделяет перекрывающиеся домены, а расстояние Махаланобиса —
разделяет (геометрическая интерпретация подраздела 2.3.4).

Размерность d = 512, объём выборки N = 100 000 (раздел 4.1.2). Векторы
нормируются на единичную сферу (‖v̂‖₂ = 1, выражение 2.23).
"""

from __future__ import annotations

import numpy as np

DIM = 512                 # размерность пространства представлений d (выр. 2.22)
N_TOTAL = 100_000         # полный объём валидационной выборки (раздел 4.1.2)
N_DOMAINS = 5             # медицина, химия, мол.биология, юриспруденция, ИТ
FRACTION_BACKGROUND = 0.2 # доля фонового (шумового) подмножества
SEED = 42                 # фиксация генератора для воспроизводимости


def _random_anisotropic_covariance(rng: np.random.Generator, d: int,
                                    cond: float = 30.0) -> np.ndarray:
    """Случайная положительно определённая анизотропная ковариация Σk.

    Генерируется как QᵀΛQ со случайным ортогональным Q и спектром Λ с заданным
    числом обусловленности cond — это и задаёт вытянутость доменного облака.
    """
    # Случайная ортогональная матрица через QR-разложение.
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    # Спектр: собственные значения от 1 до cond (анизотропия).
    eigs = np.linspace(1.0, cond, d)
    rng.shuffle(eigs)
    sigma = (q * eigs) @ q.T
    # Симметризация (страховка от ошибок округления).
    return 0.5 * (sigma + sigma.T)


def make_dataset(dim: int = DIM, n_total: int = N_TOTAL,
                 n_domains: int = N_DOMAINS,
                 frac_bg: float = FRACTION_BACKGROUND,
                 seed: int = SEED):
    """Возвращает (X, y, profiles_stats).

    X — массив (n_total, dim) нормированных представлений;
    y — метки доменов 0..n_domains-1, метка n_domains = фон (Базовый);
    profiles_stats — список (mu_k, Sigma_k) истинной статистики доменов,
        из которой строятся синаптические профили мезо-фильтра.
    """
    rng = np.random.default_rng(seed)

    # Истинные центроиды доменов: разнесены на гиперсфере.
    centroids = rng.standard_normal((n_domains, dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    # Сближаем пару доменов (медицина/химия), чтобы их облака перекрывались
    # и проверяли разрешающую способность Махаланобиса.
    centroids[1] = 0.6 * centroids[0] + 0.4 * centroids[1]
    centroids[1] /= np.linalg.norm(centroids[1])

    covariances = [_random_anisotropic_covariance(rng, dim) for _ in range(n_domains)]

    n_bg = int(n_total * frac_bg)
    n_dom_total = n_total - n_bg
    per_domain = n_dom_total // n_domains

    X_parts, y_parts = [], []
    profiles_stats = []
    for k in range(n_domains):
        mu = centroids[k]
        sigma = covariances[k]
        # Масштабируем ковариацию, чтобы облако было компактным относительно
        # расстояний между центроидами.
        sigma_scaled = sigma * 1e-3
        samples = rng.multivariate_normal(mu, sigma_scaled, size=per_domain)
        X_parts.append(samples)
        y_parts.append(np.full(per_domain, k, dtype=np.int64))
        profiles_stats.append((mu.copy(), sigma_scaled.copy()))

    # Фон: широкий изотропный шум, не принадлежащий ни одному домену.
    bg = rng.standard_normal((n_bg, dim)) * 0.5
    X_parts.append(bg)
    y_parts.append(np.full(n_bg, n_domains, dtype=np.int64))  # метка фона

    X = np.vstack(X_parts).astype(np.float64)
    y = np.concatenate(y_parts)

    # Нормировка на единичную гиперсферу (выражение 2.23).
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms

    # Перемешивание (гетерогенный поток).
    perm = rng.permutation(len(X))
    return X[perm], y[perm], profiles_stats


if __name__ == "__main__":
    X, y, stats = make_dataset()
    print(f"X: {X.shape}, y: {y.shape}, доменов: {len(stats)} + фон")
    print(f"норма первого вектора: {np.linalg.norm(X[0]):.6f} (ожидается ≈ 1.0)")
    for k, (mu, sigma) in enumerate(stats):
        print(f"  домен {k}: ‖μ‖={np.linalg.norm(mu):.3f}, "
              f"след(Σ)={np.trace(sigma):.4f}")