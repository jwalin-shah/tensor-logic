.PHONY: build test fmt fmt-check vet clean ci

build:
	go build ./...

test:
	go test -v -race ./...

fmt:
	gofmt -w .

fmt-check:
	@unformatted=$$(gofmt -l .); if [ -n "$$unformatted" ]; then echo "unformatted files:"; echo "$$unformatted"; exit 1; fi; echo "✓ fmt ok"

vet:
	go vet ./...

ci: fmt-check vet test

clean:
	go clean
