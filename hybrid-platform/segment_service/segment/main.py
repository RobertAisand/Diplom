"""main.py — точка входа Python-сервиса доменного Сегмента (подраздел 3.5.1).

Композиционный корень (Composition Root): единственное место, где конкретные
адаптеры инстанцируются и внедряются в сценарии как реализации портов. Здесь же
исполняется фаза инициализации — однократное вычисление и кэширование величин
горячего пути (Σk⁻¹).
"""

import logging
from concurrent import futures

import grpc

from segment.config import SegmentConfig
from segment.entity.synaptic_profile import SynapticProfile
from segment.usecase.meso_filter import MesoFilterUseCase
from segment.usecase.segment_inference import SegmentInferenceUseCase
from segment.adapter.encoder.transformer_encoder import TransformerEncoder
from segment.adapter.keras_model.segment_network import SegmentNetwork
from segment.adapter.grpc_server.cascade_servicer import CascadeServicer
from api.proto import cascade_pb2_grpc


def build_servicer(cfg: SegmentConfig) -> CascadeServicer:
    """Фаза инициализации: сборка сервиса из конфигурации домена."""
    # Синаптический профиль ⟨μk, Σk⟩; здесь однократно кэшируется Σk⁻¹ (3.5.2).
    profile = SynapticProfile.from_domain_statistics(
        centroid=cfg.load_centroid(),
        covariance=cfg.load_covariance(),
    )
    # Адаптеры (реализации портов).
    encoder = TransformerEncoder(model_name=cfg.encoder_model)
    network = SegmentNetwork.load(cfg.network_weights_path)
    # Внедрение зависимостей в сценарии.
    meso = MesoFilterUseCase(
        profile=profile, encoder=encoder, threshold=cfg.meso_threshold,
    )
    inference = SegmentInferenceUseCase(network=network, encoder=encoder)
    return CascadeServicer(
        meso=meso, inference=inference,
        segment_id=cfg.segment_id, threshold=cfg.meso_threshold,
    )


def serve(cfg: SegmentConfig) -> None:
    """Фаза обслуживания: запуск gRPC-сервера и приём потоков от диспетчера."""
    servicer = build_servicer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg.max_workers))
    cascade_pb2_grpc.add_CascadeProcessorServicer_to_server(servicer, server)
    server.add_insecure_port(cfg.listen_address)
    logging.info("Сегмент %s запущен на %s", cfg.segment_id, cfg.listen_address)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve(SegmentConfig.from_environment())