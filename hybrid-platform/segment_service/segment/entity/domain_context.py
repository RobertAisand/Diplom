"""segment/entity/domain_context.py — изолированный семантический контекст.

Результат работы мезо-уровня (подраздел 2.3.5): пофрагментный вердикт о допуске
либо изоляции контекста. Сущности ядра, не зависящие от тензорных библиотек.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FragmentVerdict:
    """Вердикт мезо-фильтра по одному фрагменту запроса."""

    text: str           # текст фрагмента
    distance: float     # расстояние Махаланобиса δk_mah фрагмента (выр. 2.28)
    admitted: bool      # допущен ли фрагмент: δk_mah ≤ τd (выр. 2.29)


@dataclass(frozen=True)
class MesoVerdict:
    """Совокупный результат мезо-фильтрации запроса для одного Сегмента."""

    admitted_fragments: list[str]              # допущенные к инференсу фрагменты
    isolated_fragments: list[str]              # изолированные (чуждые домену)
    fragment_verdicts: list[FragmentVerdict]   # подробный вердикт по фрагментам

    @property
    def is_fully_isolated(self) -> bool:
        """Истина, если ни один фрагмент не допущен: запрос целиком чужд домену
        и подлежит перенаправлению в Базовый сегмент (тотальность Φ_meso)."""
        return len(self.admitted_fragments) == 0