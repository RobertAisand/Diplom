"""tests/test_mahalanobis.py — верификация метрики мезо-уровня (выр. 2.28–2.29).

Проверяет, что реализация расстояния Махаланобиса в SynapticProfile корректно
воплощает выражение (2.28) и решающее правило (2.29). Прохождение теста есть
эксплуатационное подтверждение соответствия кода математике Главы 2.

Запуск:  pytest tests/test_mahalanobis.py -v
"""

import numpy as np
import pytest

from segment.entity.exceptions import DimensionMismatchError
from segment.entity.synaptic_profile import SynapticProfile


def test_distance_to_centroid_is_zero():
    """В центроиде домена расстояние Махаланобиса равно нулю."""
    d = 8
    mu = np.zeros(d)
    sigma = np.eye(d)
    prof = SynapticProfile.from_domain_statistics(mu, sigma)
    assert prof.mahalanobis_distance(mu) == pytest.approx(0.0, abs=1e-9)


def test_identity_covariance_equals_euclidean():
    """При Σ = I расстояние Махаланобиса совпадает с евклидовым (выр. 2.27–2.28)."""
    d = 16
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(d)
    prof = SynapticProfile.from_domain_statistics(mu, np.eye(d))
    v = rng.standard_normal(d)
    expected = float(np.linalg.norm(v - mu))
    # Допуск учитывает регуляризатор ridge=1e-6, добавляемый к диагонали Σ
    # при обращении (вносит отклонение порядка 1e-5 относительно чистого Σ=I).
    assert prof.mahalanobis_distance(v) == pytest.approx(expected, rel=1e-4)


def test_anisotropy_changes_distance():
    """Анизотропная Σ растягивает/сжимает направления: равные по евклиду
    отклонения дают разное расстояние Махаланобиса (геом. смысл 2.3.4)."""
    mu = np.zeros(2)
    sigma = np.array([[4.0, 0.0], [0.0, 0.25]])  # вытянуто по оси x
    prof = SynapticProfile.from_domain_statistics(mu, sigma)
    # Отклонение вдоль большого разброса (x) ближе, чем поперёк (y).
    d_along = prof.mahalanobis_distance(np.array([1.0, 0.0]))
    d_across = prof.mahalanobis_distance(np.array([0.0, 1.0]))
    assert d_across > d_along


def test_decision_rule_admits_and_rejects():
    """Решающее правило (2.29): δk_mah ≤ τd ⇒ допуск; иначе отсечение."""
    d = 4
    prof = SynapticProfile.from_domain_statistics(np.zeros(d), np.eye(d))
    near = np.array([0.1, 0.0, 0.0, 0.0])
    far = np.array([5.0, 5.0, 5.0, 5.0])
    assert prof.is_context_admissible(near, threshold=1.0) is True
    assert prof.is_context_admissible(far, threshold=1.0) is False


def test_dimension_mismatch_raises():
    """Несовпадение размерности входа и профиля диагностируется явно."""
    prof = SynapticProfile.from_domain_statistics(np.zeros(4), np.eye(4))
    with pytest.raises(DimensionMismatchError):
        prof.mahalanobis_distance(np.zeros(3))