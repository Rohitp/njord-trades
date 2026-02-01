.PHONY: dev db-up db-down db-logs migrate-exec migrate-diff migrate-down test test-cov lint lint-fix sync

# Run FastAPI dev server with reload
dev:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --log-level warning

# Database commands (run from project root)
db-up:
	podman compose -f ../docker-compose.yml up -d postgres

db-down:
	podman compose -f ../docker-compose.yml down

db-logs:
	podman compose -f ../docker-compose.yml logs -f postgres

# Alembic migrations
migrate-exec:
	uv run alembic upgrade head

migrate-diff:
	@echo "Enter migration message:"; \
	read msg; \
	uv run alembic revision --autogenerate -m "$$msg"

migrate-down:
	uv run alembic downgrade -1

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=term-missing

# Linting
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

lint-fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

# Sync dependencies
sync:
	uv sync
