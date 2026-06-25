.PHONY: install lint format typecheck test pre-commit

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy packages/boltzgen_design packages/dataset/src packages/bbb_models/src

test:
	uv run pytest -q

pre-commit:
	uv run pre-commit run --all-files
