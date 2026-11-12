.PHONY: install lint test db init-db ingest serve eval

install:
	uv sync --extra observability

lint:
	uv run ruff check .

test:
	uv run pytest

db:
	docker compose up -d db

init-db:
	uv run assistente-docs init-db

ingest:
	uv run assistente-docs ingest

serve:
	uv run uvicorn assistente_docs.api.app:app --reload

eval:
	uv run assistente-docs eval
