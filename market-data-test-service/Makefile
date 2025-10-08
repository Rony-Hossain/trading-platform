SHELL := /bin/bash
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
ARTIFACT_DIR ?= artifacts

.PHONY: install test bdd contract perf lint trace ci clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

test:
	mkdir -p $(ARTIFACT_DIR)/viztraces
	pytest tests/market_data tests/data_quality \
		--maxfail=1 --disable-warnings \
		--junitxml=$(ARTIFACT_DIR)/pytest-junit.xml \
		--html=$(ARTIFACT_DIR)/pytest-report.html \
		--self-contained-html

bdd:
	mkdir -p $(ARTIFACT_DIR)/viztraces
	pytest tests/bdd \
		--junitxml=$(ARTIFACT_DIR)/bdd-junit.xml \
		--disable-warnings

contract:
	bash scripts/run-schemathesis.sh

perf:
	k6 run tests/perf/k6-price-smoke.js
	k6 run tests/perf/k6-ws-smoke.js

lint:
	pre-commit run --all-files

trace:
	mkdir -p $(ARTIFACT_DIR)/viztraces
	pytest -m profile --disable-warnings

ci: test bdd contract

clean:
	rm -rf $(ARTIFACT_DIR) .pytest_cache __pycache__ .coverage htmlcov
