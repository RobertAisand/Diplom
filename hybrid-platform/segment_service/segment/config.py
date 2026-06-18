"""segment/config.py — конфигурация доменного Сегмента.

Считывается из переменных окружения (подраздел 3.5.1): один и тот же образ
сервиса разворачивается для разных доменов различием лишь конфигурации.
РЕКОНСТРУКЦИЯ: в главах класс упомянут (SegmentConfig.from_environment), но не
приведён листингом.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentConfig:
    segment_id: str            # идентификатор Сегмента (MED / CHEM / BASE …)
    meso_threshold: float      # порог контекстной близости τd (выр. 2.29)
    encoder_model: str         # имя модели энкодера
    network_weights_path: str  # путь к весам нейросети Сегмента (.keras)
    profile_path: str          # путь к synaptic_profile.npz (μk, Σk)
    listen_address: str        # адрес gRPC-сервера
    max_workers: int           # размер пула воркеров gRPC

    @classmethod
    def from_environment(cls) -> "SegmentConfig":
        return cls(
            segment_id=os.environ.get("SEGMENT_ID", "MED"),
            meso_threshold=float(os.environ.get("MESO_THRESHOLD", "12.0")),
            encoder_model=os.environ.get("ENCODER_MODEL", "all-MiniLM-L6-v2"),
            network_weights_path=os.environ.get(
                "NETWORK_WEIGHTS", "artifacts/segment_network.keras"),
            profile_path=os.environ.get(
                "PROFILE_PATH", "artifacts/synaptic_profile.npz"),
            listen_address=os.environ.get("LISTEN_ADDRESS", "[::]:50051"),
            max_workers=int(os.environ.get("MAX_WORKERS", "8")),
        )

    def load_centroid(self) -> np.ndarray:
        """Загружает эталонный доменный вектор μk (выр. 2.24) из профиля."""
        with np.load(self.profile_path) as data:
            return data["mu"]

    def load_covariance(self) -> np.ndarray:
        """Загружает ковариационную матрицу Σk (выр. 2.25) из профиля."""
        with np.load(self.profile_path) as data:
            return data["sigma"]