PYTHON ?= .venv/bin/python
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
UI_HOST ?= 0.0.0.0
UI_PORT ?= 8501
UI_BACKEND_URL ?= http://127.0.0.1:8000

.PHONY: venv
venv:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip

.PHONY: install-dev
install-dev:
	.venv/bin/pip install -e ".[dev]"

.PHONY: test
test:
	$(PYTHON) -m pytest

.PHONY: smoke
smoke:
	$(PYTHON) -m pytest tests/unit/test_smoke.py

.PHONY: lint
lint:
	@echo "Lint target placeholder. Add Ruff or equivalent when implementation begins."

.PHONY: ingest
ingest:
	$(PYTHON) -m sec_copilot.ingest.cli run --companies-config configs/companies.yaml

.PHONY: ingest-sample
ingest-sample:
	$(PYTHON) -m sec_copilot.ingest.cli run --companies-config configs/companies.yaml --company NVDA --annual-limit 1 --quarterly-limit 0

.PHONY: test-live
test-live:
	$(PYTHON) -m pytest -m live_sec

.PHONY: index
index:
	$(PYTHON) -m sec_copilot.retrieval.cli index

.PHONY: retrieve
retrieve:
	$(PYTHON) -m sec_copilot.retrieval.cli retrieve --question "What does NVIDIA say about export controls?" --ticker NVDA --form-type 10-K --debug

.PHONY: answer-mock
answer-mock:
	$(PYTHON) -m sec_copilot.retrieval.cli answer --question "What does NVIDIA say about export controls?" --ticker NVDA --form-type 10-K --provider mock --debug

.PHONY: eval-smoke
eval-smoke:
	$(PYTHON) -m sec_copilot.eval.cli run --subset ci_smoke --mode full --provider reference --score-backend deterministic --output-dir artifacts/evals/ci_smoke_latest

.PHONY: eval-full
eval-full:
	$(PYTHON) -m sec_copilot.eval.cli run --subset full --mode full --provider reference --score-backend deterministic --output-dir artifacts/evals/full_latest

.PHONY: serve-api
serve-api:
	$(PYTHON) -m uvicorn sec_copilot.api.app:admin_app --reload

.PHONY: serve-ui
serve-ui:
	SEC_COPILOT_UI_BACKEND_URL=http://127.0.0.1:8000 $(PYTHON) -m streamlit run src/sec_copilot/frontend/streamlit_app.py --server.port 8501

.PHONY: serve-api-public
serve-api-public:
	SEC_COPILOT_ENV=production SEC_COPILOT_ENABLE_ADMIN_ROUTES=false $(PYTHON) -m uvicorn sec_copilot.api.app:public_app --host $(API_HOST) --port $(API_PORT)

.PHONY: serve-api-admin
serve-api-admin:
	SEC_COPILOT_ENV=production SEC_COPILOT_ENABLE_ADMIN_ROUTES=true $(PYTHON) -m uvicorn sec_copilot.api.app:admin_app --host $(API_HOST) --port $(API_PORT)

.PHONY: serve-ui-container
serve-ui-container:
	SEC_COPILOT_UI_BACKEND_URL=$(UI_BACKEND_URL) $(PYTHON) -m streamlit run src/sec_copilot/frontend/streamlit_app.py --server.address $(UI_HOST) --server.port $(UI_PORT)
