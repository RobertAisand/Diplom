"""segment/usecase/meso_filter.py — сценарий мезо-уровня (Use Case).

Реализует оператор Φ_meso (подраздел 2.3.5): семантическую верификацию решения
макро-уровня и пофрагментную изоляцию контекста гетерогенного запроса. Для
каждого фрагмента применяется решающее правило Махаланобиса (выражения
2.28–2.29): фрагмент допускается в Сегмент ⇔ δk_mah(v) ≤ τd.

Сценарий зависит от сущности профиля (ядро) и порта энкодера (абстракция);
конкретная реализация энкодера внедряется композиционным корнем (подраздел 3.5.1).
"""

from __future__ import annotations

from segment.entity.domain_context import FragmentVerdict, MesoVerdict
from segment.entity.synaptic_profile import SynapticProfile
from segment.usecase.ports import EmbeddingPort


class MesoFilterUseCase:
    """Сценарий мезо-фильтрации: изоляция контекста по расстоянию Махаланобиса."""

    def __init__(
        self,
        profile: SynapticProfile,
        encoder: EmbeddingPort,
        threshold: float,
    ) -> None:
        self._profile = profile
        self._encoder = encoder
        self._threshold = threshold   # порог контекстной близости τd (выр. 2.29)

    def filter_request(self, fragments: list[str]) -> MesoVerdict:
        """Применяет решающее правило Махаланобиса к каждому фрагменту запроса,
        разграничивая допущенный и изолированный контекст.

        Разбиение текста на фрагменты выполнено вызывающим адаптером; сценарий
        оперирует уже выделенными фрагментами.
        """
        verdicts: list[FragmentVerdict] = []
        admitted: list[str] = []
        isolated: list[str] = []

        for fragment in fragments:
            # Векторизация фрагмента через порт энкодера (выр. 2.22–2.23).
            v = self._encoder.embed(fragment)
            # Решающее правило Махаланобиса (выр. 2.28–2.29).
            distance = self._profile.mahalanobis_distance(v)
            is_admitted = distance <= self._threshold

            verdicts.append(
                FragmentVerdict(text=fragment, distance=distance, admitted=is_admitted)
            )
            (admitted if is_admitted else isolated).append(fragment)

        return MesoVerdict(
            admitted_fragments=admitted,
            isolated_fragments=isolated,
            fragment_verdicts=verdicts,
        )