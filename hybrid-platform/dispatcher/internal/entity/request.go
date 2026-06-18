package entity

import "time"

// Request — запрос qt = ⟨xt, μt, τt⟩ (выр. 2.16).
type Request struct {
	ID        string
	Text      string    // текстовое тело xt
	Meta      []float64 // вектор метаданных μt
	ArrivalAt time.Time // отметка поступления τt
}
