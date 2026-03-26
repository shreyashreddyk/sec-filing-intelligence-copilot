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

.PHONY: eval-smoke
eval-smoke:
	.venv/bin/python -m sec_copilot.eval.cli run --subset ci_smoke --mode full --provider reference --score-backend deterministic --output-dir artifacts/evals/ci_smoke_latest

.PHONY: eval-full
eval-full:
	.venv/bin/python -m sec_copilot.eval.cli run --subset full --mode full --provider reference --score-backend deterministic --output-dir artifacts/evals/full_latest

.PHONY: serve-api
serve-api:
	.venv/bin/python -m uvicorn sec_copilot.api.app:app --reload

.PHONY: serve-ui
serve-ui:
	SEC_COPILOT_UI_BACKEND_URL=http://127.0.0.1:8000 .venv/bin/python -m streamlit run src/sec_copilot/frontend/streamlit_app.py --server.port 8501
