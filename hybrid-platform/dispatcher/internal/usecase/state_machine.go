package usecase

// State — состояние жизненного цикла запроса (конечный автомат 𝒜).
type State uint8

const (
	StateReceived State = iota
	StateScored
	StateDispatched
	StateCompleted
	StateRejected
)

// Transition — таблица допустимых переходов (единая точка контроля).
var transitions = map[State][]State{
	StateReceived:   {StateScored, StateRejected},
	StateScored:     {StateDispatched, StateRejected},
	StateDispatched: {StateCompleted},
}

func CanTransition(from, to State) bool {
	for _, s := range transitions[from] {
		if s == to {
			return true
		}
	}
	return false
}
