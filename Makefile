.PHONY: all install run lint format types-check tests help

UV          := uv
PYTHONPATH  := .

# Default target
all: help

# -----------------------------
# Core dev workflow
# -----------------------------

## Install Python deps into .venv (editable, CLI, etc.)
install:
	$(UV) sync

## Run the main script
run:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python src/main.py

## Lint Python (ruff)
lint:
	$(UV) run ruff check

## Format Python (ruff format)
format:
	$(UV) run ruff format ./src

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
	@echo "  run              Run the main script"
	@echo "  lint             Lint code with ruff"
	@echo "  format           Format code with ruff"
	@echo "  types-check      (Placeholder) Type checking"
	@echo "  tests            Run all tests (pytest)"
	@echo ""
