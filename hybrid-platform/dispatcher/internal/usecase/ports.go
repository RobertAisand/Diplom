package usecase

import "dispatcher/internal/entity"

// IngestQueue — порт неблокирующей очереди приёма (выр. 2.18).
type IngestQueue interface {
	Pop() (entity.Request, bool)
}

// InferenceGateway — порт обращения к Сегменту (реализуется gRPC-клиентом).
type InferenceGateway interface {
	ProcessSegment(req entity.Request, segID entity.SegmentID,
		score float64, cohort []entity.SegmentID) (entity.SegmentResponse, error)
}

// Vectorizer — порт оператора векторизации (выр. 2.22–2.23).
type Vectorizer interface {
	Vectorize(text string) ([]float64, error)
}
