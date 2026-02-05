.PHONY: all install run lint format types-check tests help

UV          := uv
PYTHONPATH  := .
HOST        := 127.0.0.1
PORT        := 8000

# Default target
all: help

# -----------------------------
# Core dev workflow
# -----------------------------

## Install Python deps into .venv (editable, CLI, etc.)
install:
	$(UV) sync

## Run the FastAPI server (reload)
run:
	PYTHONPATH=$(PYTHONPATH) $(UV) run uvicorn src.server:app --reload --host $(HOST) --port $(PORT)

## Lint Python (ruff)
lint:
	$(UV) run ruff check src --exclude web --exclude @web

## Format Python (ruff format)
format:
	$(UV) run ruff format src --exclude web --exclude @web

## Type check Python (pyright)
types-check:
	$(UV) run pyright

check: lint format types-check tests


# -----------------------------
# Utilities
# -----------------------------

## Run all Python tests (if tests/ exists)
tests:
	PYTHONPATH=$(PYTHONPATH) $(UV) run pytest -ra ./tests

# -----------------------------
# Help
# -----------------------------

help:
	@echo ""
	@echo "Available make targets:"
	@echo "  install          Install Python dependencies into .venv"
	@echo "  run              Run the FastAPI server (HOST/PORT overridable)"
	@echo "  lint             Lint code with ruff"
	@echo "  format           Format code with ruff"
	@echo "  types-check      (Placeholder) Type checking"
	@echo "  tests            Run all tests (pytest)"
	@echo ""
	@echo "Overridable variables:"
	@echo "  HOST=$(HOST)  PORT=$(PORT)"
	@echo "  Example: make run HOST=0.0.0.0 PORT=8000"
