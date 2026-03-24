.PHONY: test
test:
	python -m pytest

.PHONY: smoke
smoke:
	python -m pytest tests/unit/test_smoke.py

.PHONY: lint
lint:
	@echo "Lint target placeholder. Add Ruff or equivalent when implementation begins."
