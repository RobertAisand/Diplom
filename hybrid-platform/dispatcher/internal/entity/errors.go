// Package entity — независимое ядро (Entities). Импортирует только стандартную
// библиотеку (правило зависимостей, подраздел 3.2.2).
package entity

import "errors"

var (
	ErrDimensionMismatch          = errors.New("размерность вектора не совпадает с профилем")
	ErrProfileNotPositiveDefinite = errors.New("Σk⁻¹ не положительно определена")
	ErrEmptyActivationSet         = errors.New("множество активируемых Сегментов пусто")
)
