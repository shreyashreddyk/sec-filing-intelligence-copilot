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

.PHONY: ingest
ingest:
	.venv/bin/python -m sec_copilot.ingest.cli run --companies-config configs/companies.yaml

.PHONY: ingest-sample
ingest-sample:
	.venv/bin/python -m sec_copilot.ingest.cli run --companies-config configs/companies.yaml --company NVDA --annual-limit 1 --quarterly-limit 0

.PHONY: test-live
test-live:
	.venv/bin/python -m pytest -m live_sec

.PHONY: index
index:
	.venv/bin/python -m sec_copilot.retrieval.cli index

.PHONY: retrieve
retrieve:
	.venv/bin/python -m sec_copilot.retrieval.cli retrieve --question "What does NVIDIA say about export controls?" --ticker NVDA --form-type 10-K --debug

.PHONY: answer-mock
answer-mock:
	.venv/bin/python -m sec_copilot.retrieval.cli answer --question "What does NVIDIA say about export controls?" --ticker NVDA --form-type 10-K --provider mock --debug

.PHONY: compare-v3
compare-v3:
	.venv/bin/python scripts/compare_retrieval_modes.py --include-generation --write-csv

.PHONY: walkthrough-v2
walkthrough-v2:
	.venv/bin/python scripts/v2_cpu_walkthrough.py
