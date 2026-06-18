package usecase

import (
	"sync"

	"dispatcher/internal/entity"
)

// Dispatcher — сценарий диспетчеризации (Алгоритм 2.1): конкурентный конвейер
// «поставщик — пул воркеров» с конкурентным скорингом и тотальной маршрутизацией.
type Dispatcher struct {
	Queue      IngestQueue
	Vectorizer Vectorizer
	Scorer     Scorer
	Gateway    InferenceGateway
	Workers    int
}

// Run запускает пул воркеров. Каждый запрос, активировавший несколько
// Сегментов, порождает несколько вызовов ProcessSegment — по одному на элемент
// ℕ_act (следствие подраздела 2.1.3).
func (d Dispatcher) Run() {
	var wg sync.WaitGroup
	for w := 0; w < d.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				req, ok := d.Queue.Pop()
				if !ok {
					return
				}
				d.processOne(req)
			}
		}()
	}
	wg.Wait()
}

func (d Dispatcher) processOne(req entity.Request) {
	v, err := d.Vectorizer.Vectorize(req.Text)
	if err != nil {
		return
	}
	act := d.Scorer.Score(v) // Φ_macro: формирование ℕ_act

	cohort := make([]entity.SegmentID, 0, len(act.Members()))
	for id := range act.Members() {
		cohort = append(cohort, id)
	}
	// Маршрутизация в каналы { Ck }: независимые вызовы ProcessSegment.
	for id, score := range act.Members() {
		_, _ = d.Gateway.ProcessSegment(req, id, score, cohort)
	}
}