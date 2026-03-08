.PHONY: help install dev test lint format docker-up docker-down train clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt
	pre-commit install

dev: ## Start backend API in development mode (hot reload)
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend: ## Start Streamlit frontend
	cd frontend && API_BASE_URL=http://localhost:8000 streamlit run app.py

test: ## Run all tests with coverage
	cd backend && pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

lint: ## Run all linters
	black --check backend/ ml_pipeline/
	isort --check-only backend/ ml_pipeline/
	flake8 backend/ ml_pipeline/ --max-line-length=120

format: ## Auto-format code
	black backend/ ml_pipeline/
	isort backend/ ml_pipeline/

docker-up: ## Start all services with Docker Compose
	docker-compose up --build -d
	@echo "✅ Services started:"
	@echo "   API:      http://localhost:8000"
	@echo "   Docs:     http://localhost:8000/docs"
	@echo "   Frontend: http://localhost:8501"

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

train: ## Train the model (requires dataset)
	@echo "Usage: make train DATASET=/path/to/plantvillage"
	python ml_pipeline/train.py --dataset $(DATASET) --output models/ --epochs-phase1 10 --epochs-phase2 20

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf backend/.coverage backend/htmlcov backend/coverage.xml
	rm -rf .pytest_cache

setup-env: ## Create .env from template
	cp .env.example .env
	@echo "✅ .env created. Please edit it with your values."
