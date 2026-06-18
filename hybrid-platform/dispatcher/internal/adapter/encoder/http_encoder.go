// Адаптер оператора векторизации с делегированием внешнему сервису-энкодеру.
// РЕКОНСТРУКЦИЯ. Здесь — заглушка детерминированной векторизации; замените на
// реальный HTTP/gRPC-вызов энкодера.
package encoder

import (
	"hash/fnv"
	"math"
)

type HTTPEncoder struct {
	Dim int
}

func NewHTTPEncoder(dim int) *HTTPEncoder { return &HTTPEncoder{Dim: dim} }

// Vectorize возвращает нормированный вектор (‖v̂‖₂ = 1, выр. 2.23).
func (e *HTTPEncoder) Vectorize(text string) ([]float64, error) {
	v := make([]float64, e.Dim)
	h := fnv.New64a()
	_, _ = h.Write([]byte(text))
	seed := h.Sum64()
	var norm float64
	for i := range v {
		seed = seed*6364136223846793005 + 1442695040888963407
		v[i] = float64(int64(seed>>11)) / float64(1<<53)
		norm += v[i] * v[i]
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
	return v, nil
}
