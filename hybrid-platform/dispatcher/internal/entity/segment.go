package entity

import "math"

// SegmentID — идентификатор доменного Сегмента (отдельный тип исключает
// смешение с произвольной строкой на этапе компиляции).
type SegmentID string

const (
	SegmentMedical  SegmentID = "MED"  // Сегмент медицины ℕ_М
	SegmentChemical SegmentID = "CHEM" // Сегмент химии и фармакологии ℕ_Х
	SegmentBase     SegmentID = "BASE" // Базовый общеязыковой сегмент ℕ_Б
)

// CascadeLevel — уровень каскадной фильтрации (выр. 2.7).
type CascadeLevel uint8

const (
	LevelMacro CascadeLevel = iota
	LevelMeso
	LevelMicro
)

// SynapticProfile — синаптический профиль ⟨μk, Σk⁻¹⟩.
// Хранится ОБРАЩЁННАЯ матрица: обращение (O(d³)) выполняется однократно.
type SynapticProfile struct {
	Dim           int       // размерность d (выр. 2.22)
	Centroid      []float64 // μk (выр. 2.24), длина = Dim
	InvCovariance []float64 // Σk⁻¹ (выр. 2.25), d×d в row-major
}

// MahalanobisDistance вычисляет δk_mah (выр. 2.28) элементарной арифметикой.
func (sp SynapticProfile) MahalanobisDistance(v []float64) (float64, error) {
	if len(v) != sp.Dim {
		return 0, ErrDimensionMismatch
	}
	diff := make([]float64, sp.Dim)
	for i := 0; i < sp.Dim; i++ {
		diff[i] = v[i] - sp.Centroid[i]
	}
	u := make([]float64, sp.Dim)
	for i := 0; i < sp.Dim; i++ {
		var acc float64
		for j := 0; j < sp.Dim; j++ {
			acc += sp.InvCovariance[i*sp.Dim+j] * diff[j]
		}
		u[i] = acc
	}
	var q float64
	for i := 0; i < sp.Dim; i++ {
		q += diff[i] * u[i]
	}
	if q < 0 {
		return 0, ErrProfileNotPositiveDefinite
	}
	return math.Sqrt(q), nil
}

// Segment — доменный Сегмент ℕk (выр. 2.3).
type Segment struct {
	ID      SegmentID
	Domain  string
	Profile SynapticProfile
}

// IsAdmissibleContext применяет решающее правило (выр. 2.29): δk_mah(v) ≤ τd.
func (s Segment) IsAdmissibleContext(v []float64, tau float64) (bool, float64, error) {
	dist, err := s.Profile.MahalanobisDistance(v)
	if err != nil {
		return false, 0, err
	}
	return dist <= tau, dist, nil
}

// SegmentResponse — доменный образ ответа Сегмента (очищен от транспорта).
type SegmentResponse struct {
	RequestID  string
	SegmentID  SegmentID
	OutputText string
}
