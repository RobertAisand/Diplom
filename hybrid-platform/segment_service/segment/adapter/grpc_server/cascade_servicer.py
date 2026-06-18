"""segment/adapter/grpc_server/cascade_servicer.py — gRPC-сервер Сегмента.

Принимает контракт SegmentRequest от Go-диспетчера (подраздел 3.1.3), исполняет
сценарии мезо- и микро-уровней и возвращает SegmentResponse. Транслирует между
доменной моделью и моделью протокола; типы Protobuf заперты в периферийном слое.
РЕКОНСТРУКЦИЯ: в главах класс упомянут, листинга нет.
"""

from __future__ import annotations

from api.proto import cascade_pb2, cascade_pb2_grpc


class CascadeServicer(cascade_pb2_grpc.CascadeProcessorServicer):
    def __init__(self, meso, inference, segment_id: str, threshold: float) -> None:
        self._meso = meso             # MesoFilterUseCase
        self._inference = inference   # SegmentInferenceUseCase
        self._segment_id = segment_id
        self._threshold = threshold

    def ProcessSegment(self, request, context):
        # Разбиение текста на фрагменты (упрощённо — по предложениям).
        fragments = [s.strip() for s in request.request.text.split(".") if s.strip()]

        # Мезо-уровень: пофрагментная изоляция (Φ_meso, выр. 2.29).
        verdict = self._meso.filter_request(fragments)

        meso_msg = cascade_pb2.MesoVerdict(
            context_admitted=not verdict.is_fully_isolated,
            mahalanobis_distance=(verdict.fragment_verdicts[0].distance
                                  if verdict.fragment_verdicts else 0.0),
            threshold=self._threshold,
            routed_to_base=verdict.is_fully_isolated,
        )

        # Если всё изолировано — в Базовый сегмент, инференс не выполняется.
        if verdict.is_fully_isolated:
            return cascade_pb2.SegmentResponse(
                request_id=request.request.request_id,
                segment_id=self._segment_id,
                output_text="",
                meso=meso_msg,
                sparsity=cascade_pb2.SparsityReport(),
            )

        # Микро-уровень: инференс разреженной сети (Φ_micro).
        result = self._inference.run(verdict.admitted_fragments)
        sparsity_msg = cascade_pb2.SparsityReport(
            rho_micro=result.rho_micro,
            active_units=result.active_units,
            total_units=result.total_units,
        )
        return cascade_pb2.SegmentResponse(
            request_id=request.request.request_id,
            segment_id=self._segment_id,
            output_text=result.output_text,
            meso=meso_msg,
            sparsity=sparsity_msg,
        )