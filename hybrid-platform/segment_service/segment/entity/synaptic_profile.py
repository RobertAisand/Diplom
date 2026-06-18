"""segment/entity/synaptic_profile.py — сущность синаптического профиля (ядро).

Синаптический профиль домена есть, согласно подразделу 2.3.3, параметрическое
описание доменного контекста ⟨μk, Σk⟩, относительно которого мезо-фильтр
оценивает контекстную близость запроса:
    μk  — эталонный доменный вектор (центроид), выражение (2.24);
    Σk  — ковариационная матрица доменного облака, выражение (2.25).

Метрика мезо-уровня — расстояние Махаланобиса (выражение 2.28) — требует не
самой Σk, а её обращения Σk⁻¹. Обращение есть операция кубической сложности
O(d³); поэтому, согласно подразделу 3.5.2, оно выполняется ОДНОКРАТНО в фазе
инициализации (фабрика from_domain_statistics) и кэшируется, а на горячем пути
остаются лишь операции O(d²) и O(d).

Модуль импортирует только стандартную библиотеку и NumPy (правило зависимостей,
подраздел 3.5.1): NumPy используется как средство эффективного исполнения
элементарных операций линейной алгебры, а не как источник готовой метрики.
"""

from __future__ import annotations

import numpy as np

from segment.entity.exceptions import (
    DimensionMismatchError,
    ProfileNotPositiveDefiniteError,
    SingularCovarianceError,
)


class SynapticProfile:
    """Синаптический профиль Сегмента ⟨μk, Σk⁻¹⟩.

    Хранится ОБРАЩЁННАЯ ковариационная матрица: расстояние Махаланобиса
    (2.28) требует Σk⁻¹, и обращение выполняется однократно при построении
    профиля, а не на каждом запросе (подраздел 3.5.2).
    """

    def __init__(self, mu: np.ndarray, sigma_inv: np.ndarray) -> None:
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma_inv = np.asarray(sigma_inv, dtype=np.float64)
        self.dim = int(self.mu.shape[0])
        if self.sigma_inv.shape != (self.dim, self.dim):
            raise DimensionMismatchError(
                f"Σk⁻¹ должна быть {self.dim}×{self.dim}, получена {self.sigma_inv.shape}"
            )

    # ------------------------------------------------------------------ #
    # Фабрика: фаза инициализации (подраздел 3.5.2)                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_domain_statistics(
        cls,
        centroid: np.ndarray,
        covariance: np.ndarray,
        ridge: float = 1e-6,
    ) -> "SynapticProfile":
        """Строит профиль из доменной статистики ⟨μk, Σk⟩.

        centroid   — эталонный доменный вектор μk (выражение 2.24);
        covariance — ковариационная матрица Σk (выражение 2.25);
        ridge      — регуляризатор (добавка λ·I к диагонали Σk), гарантирующий
                     обратимость при близкой к вырожденной матрице.

        Здесь, в фазе инициализации, выполняется единственное затратное
        обращение Σk → Σk⁻¹ (O(d³)); результат кэшируется в поле sigma_inv.
        Вырожденность Σk диагностируется явно (подраздел 3.5.2): профиль с
        необратимой ковариацией не допускается к запуску.
        """
        mu = np.asarray(centroid, dtype=np.float64)
        sigma = np.asarray(covariance, dtype=np.float64)
        d = mu.shape[0]
        if sigma.shape != (d, d):
            raise DimensionMismatchError(
                f"Σk должна быть {d}×{d}, получена {sigma.shape}"
            )
        # Регуляризация диагонали для устойчивого обращения.
        sigma_reg = sigma + ridge * np.eye(d, dtype=np.float64)
        try:
            sigma_inv = np.linalg.inv(sigma_reg)
        except np.linalg.LinAlgError as exc:  # матрица сингулярна
            raise SingularCovarianceError(
                "ковариационная матрица Σk вырождена и не допускает обращения"
            ) from exc
        return cls(mu=mu, sigma_inv=sigma_inv)

    # ------------------------------------------------------------------ #
    # Метрика и решающее правило: горячий путь (подраздел 3.5.3)          #
    # ------------------------------------------------------------------ #
    def mahalanobis_distance(self, v: np.ndarray) -> float:
        """Расстояние Махаланобиса δk_mah(v) до центроида домена (выр. 2.28):

            δk_mah(v) = √[ (v − μk)ᵀ · Σk⁻¹ · (v − μk) ].

        Каждый шаг формулы выписан явно (методологическое требование 3.5.3).
        Сложность O(d²): доминирует умножение матрицы Σk⁻¹ на вектор.
        """
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (self.dim,):
            raise DimensionMismatchError(
                f"ожидался вектор размерности {self.dim}, получена форма {v.shape}"
            )
        # Шаг 1 — центрирование: delta = v − μk.
        delta = v - self.mu
        # Шаг 2 — квадратичная форма q = deltaᵀ · Σk⁻¹ · delta.
        weighted = self.sigma_inv @ delta      # Σk⁻¹ · delta, O(d²)
        q = float(delta @ weighted)            # deltaᵀ · (·), O(d)
        if q < 0.0:
            raise ProfileNotPositiveDefiniteError(
                "квадратичная форма отрицательна: Σk⁻¹ не положительно определена"
            )
        # Шаг 3 — извлечение корня.
        return float(np.sqrt(q))

    def is_context_admissible(self, v: np.ndarray, threshold: float) -> bool:
        """Решающее правило изоляции контекста (выражение 2.29):

            контекст допустим  ⇔  δk_mah(v) ≤ τd.
        """
        return self.mahalanobis_distance(v) <= threshold