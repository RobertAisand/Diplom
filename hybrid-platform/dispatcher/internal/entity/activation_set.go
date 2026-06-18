package entity

// ActivationSet — множество активируемых Сегментов ℕ_act с оценками скоринга.
type ActivationSet struct {
	scores map[SegmentID]float64
}

func NewActivationSet() *ActivationSet {
	return &ActivationSet{scores: make(map[SegmentID]float64)}
}

func (a *ActivationSet) Add(id SegmentID, score float64) { a.scores[id] = score }
func (a *ActivationSet) IsEmpty() bool                   { return len(a.scores) == 0 }
func (a *ActivationSet) Members() map[SegmentID]float64  { return a.scores }

// EnsureNonEmpty гарантирует тотальность: при пустом множестве активируется
// Базовый сегмент (резервная маршрутизация, подраздел 2.3.5).
func (a *ActivationSet) EnsureNonEmpty() {
	if a.IsEmpty() {
		a.Add(SegmentBase, 0.0)
	}
}
