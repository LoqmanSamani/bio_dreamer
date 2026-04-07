# BioDreamer Makefile
#
# Common project commands for development, training, testing, and deployment.
# This file centralises frequently used commands so developers don't need to
# remember long CLI invocations.
#
# Key targets:
#   make install       — Install Python dependencies + frontend deps
#   make dev           — Start backend + frontend in development mode
#   make test          — Run full test suite (unit + integration)
#   make train-world   — Launch world model training (specify MODULE=protein_dreamer etc.)
#   make train-policy  — Launch RL policy training
#   make docker-up     — Build and start all services via docker-compose
#   make lint          — Run linters (ruff, mypy, eslint)
#   make docs          — Build documentation
#   make docs-serve    — Live-reload docs dev server

.PHONY: docs docs-serve

docs:
	mkdocs build

docs-serve:
	mkdocs serve
