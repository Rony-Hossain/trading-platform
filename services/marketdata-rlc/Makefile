.PHONY: dev run test migrate lint

PYTHON ?= python

dev:
	uvicorn app.main:app --reload --port 8080

run:
	uvicorn app.main:app --port 8080

test:
	pytest -q

migrate:
	bash scripts/migrate.sh

lint:
	$(PYTHON) -m compileall app
