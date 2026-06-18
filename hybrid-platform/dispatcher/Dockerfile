# Многоэтапная сборка Go-диспетчера.
FROM golang:1.22-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" \
    -o /out/dispatcher ./cmd/dispatcher

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /out/dispatcher /dispatcher
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/dispatcher"]