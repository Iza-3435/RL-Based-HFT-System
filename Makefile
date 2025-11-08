.PHONY: help install test lint format type-check clean run docker-build docker-up

help:
	@echo "HFT Network Optimizer - Production Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make setup          Full development setup"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         Format code with black"
	@echo "  make lint           Lint with ruff"
	@echo "  make type-check     Type check with mypy"
	@echo "  make quality        Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-cov       Run tests with coverage"
	@echo ""
	@echo "Running:"
	@echo "  make run            Run in production mode"
	@echo "  make run-fast       Run in fast mode"
	@echo "  make run-dev        Run in development mode"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean cache and temp files"

install:
	pip install -r requirements.txt

setup: install
	pre-commit install
	mkdir -p logs data config
	cp -n .env.example .env || true

format:
	black .

lint:
	ruff check . --fix

type-check:
	mypy src/

quality: format lint type-check
	@echo "✓ All quality checks passed"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

run:
	python main.py --mode production --log-level normal

run-fast:
	python main.py --mode fast --duration 120 --log-level verbose

run-dev:
	python main.py --config config/development.yaml --log-level verbose

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f hft-system

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	@echo "✓ Cleaned"
