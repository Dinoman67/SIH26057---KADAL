.PHONY: all help install build run start dev clean test

# Default target
all: run

help:
	@echo "KADAL Makefile Commands:"
	@echo "  make run        - Build frontend (if needed) and start unified app (port 8000)"
	@echo "  make dev        - Start backend and frontend dev servers concurrently"
	@echo "  make build      - Build frontend production bundle (frontend/dist)"
	@echo "  make install    - Install Python and Node.js dependencies"
	@echo "  make test       - Run backend test suite"
	@echo "  make clean      - Remove build artifacts and temporary cache files"

install: .venv/bin/python
	cd frontend && npm install

.venv/bin/python:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

build:
	cd frontend && npm install && npm run build

run start:
	./start.sh

dev:
	./start.sh --dev

test: .venv/bin/python
	.venv/bin/python -m pytest tests/ -q

clean:
	rm -rf frontend/dist frontend/node_modules/.vite
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
