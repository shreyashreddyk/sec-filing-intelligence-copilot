.PHONY: venv
venv:
	python3.12 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip

.PHONY: install-dev
install-dev:
	.venv/bin/pip install -e ".[dev]"

.PHONY: test
test:
	.venv/bin/python -m pytest

.PHONY: smoke
smoke:
	.venv/bin/python -m pytest tests/unit/test_smoke.py

.PHONY: lint
lint:
	@echo "Lint target placeholder. Add Ruff or equivalent when implementation begins."
