// gRPC-клиент Сегментов: реализация порта InferenceGateway. Локализует
// трансляцию между доменной моделью и моделью протокола.
package grpcclient

import (
	"context"
	"time"

	"google.golang.org/grpc"

	pb "dispatcher/api/proto"
	"dispatcher/internal/entity"
)

type SegmentClient struct {
	clients map[entity.SegmentID]pb.CascadeProcessorClient
}

// NewSegmentClient устанавливает соединения с адресами Сегментов.
func NewSegmentClient(addrs map[entity.SegmentID]string,
	opts ...grpc.DialOption) (*SegmentClient, error) {
	clients := make(map[entity.SegmentID]pb.CascadeProcessorClient)
	for id, addr := range addrs {
		conn, err := grpc.Dial(addr, opts...)
		if err != nil {
			return nil, err
		}
		clients[id] = pb.NewCascadeProcessorClient(conn)
	}
	return &SegmentClient{clients: clients}, nil
}

func (c *SegmentClient) ProcessSegment(req entity.Request, segID entity.SegmentID,
	score float64, cohort []entity.SegmentID) (entity.SegmentResponse, error) {

	cohortStr := make([]string, len(cohort))
	for i, id := range cohort {
		cohortStr[i] = string(id)
	}
	msg := &pb.SegmentRequest{
		Request: &pb.Request{
			RequestId: req.ID,
			Text:      req.Text,
			Meta:      req.Meta,
			ArrivalTs: req.ArrivalAt.UnixNano(),
		},
		Activation: &pb.Activation{SegmentId: string(segID), MacroScore: score},
		Cohort:     cohortStr,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.clients[segID].ProcessSegment(ctx, msg)
	if err != nil {
		return entity.SegmentResponse{}, err
	}
	return entity.SegmentResponse{
		RequestID:  resp.RequestId,
		SegmentID:  entity.SegmentID(resp.SegmentId),
		OutputText: resp.OutputText,
	}, nil
}
