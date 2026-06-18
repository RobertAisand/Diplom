// Адаптер неблокирующей очереди поверх буферизированного канала (выр. 2.18).
package queue

import "dispatcher/internal/entity"

type BoundedChannel struct {
	ch chan entity.Request
}

func NewBoundedChannel(capacity int) *BoundedChannel {
	return &BoundedChannel{ch: make(chan entity.Request, capacity)}
}

// Push — неблокирующая запись: при переполнении запрос отбрасывается (асимметрия).
func (b *BoundedChannel) Push(req entity.Request) bool {
	select {
	case b.ch <- req:
		return true
	default:
		return false
	}
}

func (b *BoundedChannel) Pop() (entity.Request, bool) {
	req, ok := <-b.ch
	return req, ok
}

func (b *BoundedChannel) Close() { close(b.ch) }