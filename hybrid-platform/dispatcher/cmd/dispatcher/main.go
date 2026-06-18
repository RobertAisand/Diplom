// main.go — точка входа Go-диспетчера + композиционный корень (composition root).
// Единственное место связывания слоёв (правило зависимостей, подраздел 3.1.2).
package main

import (
	"log"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"dispatcher/internal/adapter/encoder"
	"dispatcher/internal/adapter/grpcclient"
	"dispatcher/internal/adapter/queue"
	"dispatcher/internal/entity"
	"dispatcher/internal/usecase"
)

const dim = 512

func main() {
	// Множество Сегментов ℕ (профили загружаются из конфигурации/артефактов).
	segments := map[entity.SegmentID]entity.Segment{
		entity.SegmentMedical:  {ID: entity.SegmentMedical, Domain: "медицина"},
		entity.SegmentChemical: {ID: entity.SegmentChemical, Domain: "химия"},
	}

	// Адреса Сегментов (соответствуют docker-compose).
	addrs := map[entity.SegmentID]string{
		entity.SegmentMedical:  "localhost:50051",
		entity.SegmentChemical: "localhost:50052",
	}

	gateway, err := grpcclient.NewSegmentClient(addrs,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("не удалось подключиться к Сегментам: %v", err)
	}

	q := queue.NewBoundedChannel(1024)
	enc := encoder.NewHTTPEncoder(dim)
	scorer := usecase.Scorer{Segments: segments, Threshold: 12.0}

	d := usecase.Dispatcher{
		Queue:      q,
		Vectorizer: enc,
		Scorer:     scorer,
		Gateway:    gateway,
		Workers:    8,
	}

	// Демонстрационный запрос (в реальной системе — приём из сети).
	q.Push(entity.Request{ID: "demo-1", Text: "пример медицинского запроса"})

	log.Println("диспетчер запущен")
	d.Run()
}
