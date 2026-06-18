package usecase

import "dispatcher/internal/entity"

// Scorer — оператор скоринга σ(qt): формирует ℕ_act по пороговому правилу (выр. 2.21).
type Scorer struct {
	Segments  map[entity.SegmentID]entity.Segment
	Threshold float64 // порог активации ϑ макро-уровня
}

// Score оценивает запрос против всех Сегментов и формирует непустое ℕ_act.
func (s Scorer) Score(v []float64) *entity.ActivationSet {
	act := entity.NewActivationSet()
	for id, seg := range s.Segments {
		if id == entity.SegmentBase {
			continue
		}
		admit, dist, err := seg.IsAdmissibleContext(v, s.Threshold)
		if err == nil && admit {
			// Чем меньше расстояние, тем выше оценка скоринга.
			act.Add(id, -dist)
		}
	}
	act.EnsureNonEmpty() // тотальность: резерв — Базовый сегмент
	return act
}
