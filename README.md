# SEC Filing Intelligence Copilot

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/ui-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/container-Docker-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/orchestrator-Kubernetes-326CE5?logo=kubernetes&logoColor=white)
![Grafana](https://img.shields.io/badge/observability-Grafana-F46800?logo=grafana&logoColor=white)
![CI](https://img.shields.io/badge/ci-GitHub%20Actions-2088FF)
![RAG Evaluation](https://img.shields.io/badge/focus-RAG%20Evaluation-6A5ACD)

[![Watch Demo](https://img.shields.io/badge/demo-Watch%20YouTube%20Demo-red?logo=youtube&logoColor=white)](https://youtu.be/5gisz5Mcxpg)

Production-oriented RAG system for SEC filings that ingests live 10-K and 10-Q reports, retrieves evidence with hybrid search, reranks results, and returns grounded answers with explicit citations.

This project is designed to show recruiter-relevant engineering signals: modular AI system design, typed API contracts, evidence-first answer generation, evaluation discipline, and CI-visible regression checks.

## Overview

SEC filings are long, dense, and difficult to compare quickly. This system turns them into a queryable, source-grounded research workflow for a focused semiconductor company universe:

- NVIDIA
- AMD
- Intel
- Broadcom
- Qualcomm

The application supports live corpus bootstrap from SEC EDGAR, section-aware processing, hybrid retrieval, reranking, grounded answer generation, and offline evaluation with tracked artifacts.

## Why This Project Stands Out

- Real ingestion pipeline instead of a static demo corpus only
- Hybrid retrieval stack combining dense search and BM25-style lexical matching
- Cross-encoder reranking before answer generation
- Citation-enforced answers tied back to real retrieved chunks
- Explicit abstention behavior when support is weak or coverage is missing
- Typed FastAPI backend with health, readiness, coverage, ingest, retrieval, query, and eval surfaces
- Split Docker images, GKE manifests, and Prometheus metrics that make the deployment story visible without pretending the API is already fully stateless
- Reproducible evaluation artifacts stored under `artifacts/`
- CI workflow that runs tests, executes an eval smoke subset, uploads eval outputs, and fails on regression

## System Architecture

The system is separated into ingestion, processing, retrieval, reranking, generation, serving, and evaluation so each layer stays inspectable and testable.

```mermaid
flowchart LR
    UI[Streamlit UI] -->|GET /health| API[FastAPI Backend]
    UI -->|GET /build-info| API
    UI -->|POST /ingest/run| API
    UI -->|POST /query| API
    UI -->|POST /retrieve/debug| API

    API --> INGEST[Ingestion Pipeline]
    INGEST --> SEC[SEC EDGAR]
    INGEST --> DATA[(data/processed)]
    INGEST --> CHROMA[(artifacts/chroma)]

    API --> RET[Hybrid Retrieval]
    RET --> BM25[BM25 Retriever]
    RET --> DENSE[Dense Retriever]
    RET --> RERANK[Cross-Encoder Reranker]
    API --> GEN[Grounded Answer Pipeline]
    GEN --> PROMPTS[Versioned Prompts]
    GEN --> DATA

    EVAL[Eval CLI] -.uses.-> FIXTURE[(tests/fixtures/eval_corpus)]
    EVAL -.writes outputs.-> EVA[(artifacts/evals)]
```

## Tech Stack

- Language: Python 3.12
- Backend: FastAPI, Uvicorn, Pydantic
- Frontend: Streamlit
- Vector store: Chroma
- Embeddings and reranking: Sentence Transformers, CrossEncoder
- LLM provider integration: OpenAI with mock fallback
- Parsing and ingestion: Requests, BeautifulSoup4
- Evaluation: custom deterministic metrics plus optional Ragas
- Testing and CI: Pytest, GitHub Actions

## Production-Oriented Features

### API surface

The backend exposes typed endpoints for both user-facing and operator workflows.

Public query-serving surface:

- `GET /health`
- `GET /build-info`
- `POST /query`
- `POST /retrieve/debug`

Admin-only surface for local bootstrap or internal operations:

- `POST /ingest/run`
- `POST /eval/run`

Local development uses the admin-capable FastAPI app. Container-safe public startup uses the default FastAPI app with `SEC_COPILOT_ENABLE_ADMIN_ROUTES=false`, which keeps the public surface while preserving the stable `sec_copilot.api.app:app` import path for Docker.

### Retrieval and answer pipeline

- Live filing ingestion from SEC EDGAR with configurable company and form scope
- Chunked processed corpus persisted locally for indexing and retrieval
- Hybrid retrieval using dense vectors and lexical retrieval
- Cross-encoder reranking for higher-quality evidence selection
- Grounded answer pipeline that returns citations mapped to retrieved chunks
- Coverage-aware behavior for missing corpus scope or stale index state

### Reliability and engineering discipline

- Typed request and response models across the API layer
- Health, readiness, build-info, and Prometheus metrics endpoints for operational visibility
- Readiness and coverage checks before query execution
- Mock-provider fallback when `OPENAI_API_KEY` is unavailable
- Unit and integration tests covering ingestion, retrieval, API contracts, prompts, evaluation, and frontend helpers

## Evaluation And Validation

Latest validated workspace-local deterministic smoke run:

- Command: `make eval-smoke`
- Report: `artifacts/evals/ci_smoke_latest/report.md`
- Results: `artifacts/evals/ci_smoke_latest/results.json`
- Retrieval `hit_rate@4`: `1.0`
- Retrieval `recall@4`: `1.0`
- Retrieval `mrr`: `0.9167`
- Citation validity rate: `1.0`
- Abstention accuracy: `1.0`

These metrics come from the checked-in eval artifacts produced by the current workspace and reflect the existing gold-set smoke subset, not a hand-written claim.

Optional richer local evaluation is also supported through the OpenAI + Ragas path, with outputs already present under `artifacts/evals/ci_smoke_openai_ragas_latest/` and `artifacts/evals/ci_smoke_openai_ragas_retry/`.

CI coverage is visible in [`.github/workflows/ci.yml`](.github/workflows/ci.yml), where the pipeline:

- installs the package
- runs `pytest`
- runs the eval smoke subset
- uploads eval artifacts
- fails the workflow if the eval smoke run regresses

## Benchmark Highlights

Latest validated local deterministic smoke run in this workspace:

- command: `make eval-smoke`
- local output: `artifacts/evals/ci_smoke_latest/report.md`
- retrieval: `hit_rate@4=1.0`, `recall@4=1.0`, `mrr=0.9167`
- answer safety: `citation_validity_rate=1.0`, `abstention_accuracy=1.0`
- answer-quality proxies: `context_precision_proxy=0.6146`, `response_relevancy_proxy=0.7230`, `faithfulness_proxy=0.6931`

Optional richer local OpenAI plus Ragas run:

- command: `.venv/bin/python -m sec_copilot.eval.cli run --subset ci_smoke --mode full --provider openai --score-backend both --output-dir artifacts/evals/ci_smoke_openai_ragas_retry --fail-on-thresholds false`
- local output: `artifacts/evals/ci_smoke_openai_ragas_retry/report.md`
- default Ragas evaluator: `gpt-4.1-mini`
- Ragas: `ragas_faithfulness=0.9667`, `ragas_response_relevancy=0.7841`, `ragas_context_precision=0.5556`

These numbers are local validation results tied to the commands above, the tracked gold set, and the current workspace outputs.

## Artifacts In This Repo

This repository includes concrete implementation evidence beyond source code:

- Eval report: [`artifacts/evals/ci_smoke_latest/report.md`](artifacts/evals/ci_smoke_latest/report.md)
- Eval results JSON: [`artifacts/evals/ci_smoke_latest/results.json`](artifacts/evals/ci_smoke_latest/results.json)
- Indexed coverage snapshot: [`artifacts/chroma/sec_semis_v1_hybrid_v3.coverage.json`](artifacts/chroma/sec_semis_v1_hybrid_v3.coverage.json)
- CI workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
- Gold eval dataset and fixtures under [`tests/fixtures`](tests/fixtures)

The current indexed coverage artifact shows a fresh local index over the five-company universe with `10` documents and `1611` chunks across `10-K` and `10-Q` filings.

## Validated GKE Path

The repo now includes the exact deployment path that was validated live on GKE:

- build and push the API and UI images with [`cloudbuild.yaml`](cloudbuild.yaml) into Artifact Registry
- deploy the CPU-first overlay at [`k8s/overlays/gke-cpu-fallback`](k8s/overlays/gke-cpu-fallback)
- keep the UI public through GKE Ingress and the API internal behind `ClusterIP`
- create the shared runtime PVC through the quota-safer `pd-standard-rwo` StorageClass in the GKE overlay
- bootstrap the first live corpus with `kubectl create job --from=cronjob/sec-copilot-corpus-refresh ...`
- restart the API after refresh so the serving pod reloads the rebuilt corpus and Chroma state
- expose FastAPI `/metrics` through Grafana-ready service annotations in the GKE overlay

The full operator walkthrough, including the failure/debug path that was fixed during the first rollout, lives in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Quickstart

### 1. Set up the environment

```bash
make venv
source .venv/bin/activate
make install-dev
cp .env.example .env
```

Edit `.env` before running the live workflow. Set `SEC_USER_AGENT` to your real name and contact email for SEC access. `HF_TOKEN` is optional and only needed for gated or private Hugging Face model downloads.

### 2. Start the backend

```bash
make serve-api
```

This local development target starts the admin-capable FastAPI app so the UI can bootstrap filings with `/ingest/run`.

### 3. Start the UI

```bash
make serve-ui
```

Open the local Streamlit URL shown in the terminal, usually `http://127.0.0.1:8501`.

### 4. Bootstrap the corpus

Use the UI ingest flow or run the CLI:

```bash
make ingest
```

For a smaller sample:

```bash
make ingest-sample
```

### 5. Ask a first question

Recommended starter question:

```text
What export control risks does NVIDIA describe?
```

Useful validation commands:

```bash
make eval-smoke
make test
```

## Deployment

### Local container test

The repo uses separate images because the FastAPI API owns model loading, dense retrieval, reranking, and optional GPU execution, while the Streamlit UI is a thin CPU-only client.

Both images stay stateless by default. Corpus artifacts, Chroma persistence, eval outputs, and model caches belong on mounted runtime paths rather than inside image layers.

Build the images:

```bash
docker build -f Dockerfile.api -t sec-copilot-api:local .
docker build -f Dockerfile.ui -t sec-copilot-ui:local .
```

The Dockerfiles pin `torch==2.4.1` directly and use [`constraints/docker-runtime.txt`](constraints/docker-runtime.txt) to hold the validated deployment-only `sentence-transformers` and `transformers` versions that passed the first GKE rollout.

Create a shared network and run the API with mounted state:

```bash
docker network create sec-copilot || true
docker run --rm --name sec-copilot-api --network sec-copilot -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" -v "$(pwd)/artifacts:/app/artifacts" sec-copilot-api:local
```

Verify the public API surface:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

Run the UI against the API container:

```bash
docker run --rm --name sec-copilot-ui --network sec-copilot -p 8501:8501 -e SEC_COPILOT_UI_BACKEND_URL=http://sec-copilot-api:8000 sec-copilot-ui:local
```

The API image defaults to `SEC_COPILOT_ENABLE_ADMIN_ROUTES=false`, so public container startup keeps ingest and eval routes out of the internet-facing surface. On Docker hosts with NVIDIA runtime support, add `--gpus all` to the API container when you want CUDA-backed Torch inside the container.

### GKE deployment overview

The deployment story in this repo is based on plain Kubernetes manifests under `k8s/base/`, GKE-focused overlays under `k8s/overlays/`, and a two-image Cloud Build flow in [`cloudbuild.yaml`](cloudbuild.yaml).

Current validated GKE posture:

- the UI is the public entrypoint through GKE Ingress
- the API stays internal behind a `ClusterIP` service
- the API and refresh workflow share one PVC-backed runtime state volume
- the refresh CronJob is present but suspended by default for safe manual rollout
- [`k8s/overlays/gke-cpu-fallback`](k8s/overlays/gke-cpu-fallback) is the recommended first live deployment path
- the GKE overlay creates `pd-standard-rwo`, a CSI-backed HDD StorageClass that avoids assuming SSD quota is available
- the standalone [`k8s/base/corpus-refresh-job.yaml`](k8s/base/corpus-refresh-job.yaml) file is a reference template, but the first live bootstrap should come from `kubectl create job --from=cronjob/sec-copilot-corpus-refresh ...` so the job inherits overlay image rewrites

Build and push the images from Cloud Shell:

```bash
export REGION="us-west1"
export AR_REPO="sec-copilot"
export IMAGE_TAG="$(git rev-parse --short HEAD)"

gcloud builds submit \
  --region="$REGION" \
  --config=cloudbuild.yaml \
  --substitutions=_REGION="$REGION",_AR_REPO="$AR_REPO",_IMAGE_TAG="$IMAGE_TAG"
```

Apply the first live CPU fallback deployment:

```bash
kubectl create namespace sec-copilot --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic sec-copilot-secrets \
  -n sec-copilot \
  --from-literal=SEC_USER_AGENT="YOUR_NAME your@email.com" \
  --from-literal=OPENAI_API_KEY="YOUR_OPENAI_KEY" \
  --from-literal=HF_TOKEN="YOUR_OPTIONAL_HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -k k8s/overlays/gke-cpu-fallback
kubectl create job \
  --from=cronjob/sec-copilot-corpus-refresh \
  sec-copilot-corpus-refresh-manual-$(date +%s) \
  -n sec-copilot
kubectl rollout restart deployment/sec-copilot-api -n sec-copilot
kubectl rollout status deployment/sec-copilot-api -n sec-copilot
kubectl get ingress sec-copilot-ui -n sec-copilot
```

That bootstrap-plus-restart sequence is required today because the API loads the processed corpus and Chroma state on startup rather than hot-reloading them at runtime.

When GPU capacity is available later, switch the API back to the GPU-targeted overlay:

```bash
kubectl apply -k k8s/overlays/gke-student
kubectl rollout status deployment/sec-copilot-api -n sec-copilot
```

For the full command-by-command walkthrough, including the exact quota, security-context, and runtime failures we hit and fixed during the first rollout, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

### Grafana Cloud integration overview

The repo is instrumented for Grafana Cloud, but it does not claim a live Grafana stack or hosted dashboard URL yet.

What already exists in the codebase:

- `GET /metrics` on the FastAPI app
- route-level request counters and latency histograms in `src/sec_copilot/api/metrics.py`
- separate query, retrieval, ingest, and eval timing metrics
- query error and abstention counters that are useful for grounded-RAG operations
- GKE overlay annotations on the internal API service for Grafana/Alloy scraping

The intended hosted path is:

1. use Grafana Cloud Kubernetes Monitoring to collect cluster-level signals from the GKE cluster
2. scrape the FastAPI `/metrics` endpoint from inside the cluster
3. correlate pod health and resource usage with API request latency, failure, and abstention behavior

Relevant runtime env vars for deployment prep:

- `SEC_COPILOT_ENV`
- `SEC_COPILOT_ENABLE_ADMIN_ROUTES`
- `SEC_COPILOT_LOG_LEVEL`
- `SEC_COPILOT_PROJECT_ROOT`
- `SEC_COPILOT_DATA_DIR`
- `SEC_COPILOT_CHROMA_DIR`
- `SEC_COPILOT_OPENAI_MODEL`
- `OPENAI_API_KEY`
- `HF_TOKEN`
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `SENTENCE_TRANSFORMERS_HOME`
- `SEC_COPILOT_UI_BACKEND_URL`

## Scaling

### Current scaling behavior

The repo is deployment-oriented, but the serving tier is not yet a truly stateless web application.

Current behavior in the Kubernetes manifests:

- `k8s/base/ui-hpa.yaml` allows the UI to scale from `1` to `3` replicas
- `k8s/base/api-hpa.yaml` exists as the future API scaling interface
- `k8s/overlays/gke-common/api-hpa-cap-patch.yaml` caps the API at `1` replica in both the CPU fallback and GPU overlays
- the API and corpus refresh workflow share a `ReadWriteOnce` PVC for processed corpus and Chroma state

That means the truthful current deployment model is one public API replica plus independently scalable UI replicas.

### Why the UI scales more easily than the API

The UI is easier to scale because it is mostly a stateless HTTP client:

- it does not own the processed corpus or Chroma persistence directory
- it does not load embedding or reranker models
- it does not need the shared PVC mounted into the pod
- extra UI replicas mainly add request-handling capacity for Streamlit sessions

The API is heavier because it owns retrieval, reranking, answer generation dependencies, and startup-time loading of mounted runtime state.

### What must change for true stateless API scaling

The API HPA only becomes a real scale-out lever after the current local-state boundary is externalized more deliberately.

That future step requires:

- moving processed corpus artifacts to shared durable storage with a versioned handoff
- replacing embedded local Chroma with a service-oriented or safely shared index layer
- removing the startup-only reload contract that currently requires an API rollout after each refresh
- keeping refresh as a separate batch workflow instead of sharing write-heavy work with public serving pods
- load-testing multi-replica startup and concurrent query behavior before lifting the current one-replica cap

## Monitoring

### App metrics

The FastAPI layer already exposes Prometheus-compatible metrics at `GET /metrics`. The current app-level metrics are:

- `sec_copilot_http_requests_total`
- `sec_copilot_http_request_duration_seconds`
- `sec_copilot_query_duration_seconds`
- `sec_copilot_retrieval_debug_duration_seconds`
- `sec_copilot_ingest_duration_seconds`
- `sec_copilot_eval_duration_seconds`
- `sec_copilot_query_errors_total`
- `sec_copilot_query_abstentions_total`

These metrics are designed to make grounded-RAG behavior visible, not just generic uptime.

### Cluster metrics

Grafana Cloud Kubernetes Monitoring should provide the cluster-level view around those app metrics:

- pod CPU and memory usage for the API, UI, and refresh workloads
- node pressure and scheduling behavior for the API workload in either CPU fallback or future GPU mode
- readiness, liveness, and restart signals
- HPA activity for the UI and the capped API HPA object
- CronJob and one-off Job status for corpus refresh
- scrape visibility for the annotated `sec-copilot-api` service on `/metrics`

### What the Grafana dashboard should show

For this repo, the most useful dashboard is a combined application and cluster view:

- API request rate, p50, and p95 latency from the HTTP and query histograms
- query failure counts and abstention trends
- UI and API pod health, restart count, and resource usage
- HPA behavior, especially whether only the UI is scaling while the API stays fixed at one replica
- refresh-job activity alongside any API rollout needed to load the new corpus state

There is no committed dashboard JSON in the repo yet. The honest current story is a Grafana-ready metrics surface plus Kubernetes manifests that give Grafana Cloud enough structure to visualize serving health and capacity once the collector side is connected.

## Repository Structure

```text
artifacts/            Eval outputs, index metadata, and local retrieval artifacts
configs/              Company universe, retrieval settings, prompts, and eval config
data/                 Downloaded and processed filing data
src/sec_copilot/      Ingestion, retrieval, reranking, generation, API, and frontend code
tests/                Unit tests, integration tests, fixtures, and gold eval data
Makefile              Common local development and validation commands
```

## Recommended Git Commands

These commands commit and push the tracked README changes without including the local-only `docs/` trail:

```bash
git status --short
git add README.md DEPLOYMENT.md cloudbuild.yaml Dockerfile.api Dockerfile.ui constraints/ k8s/
git commit -m "Harden GKE deployment path"
git push origin main
```

## Current Limitations / Next Improvements

- Filing scope is currently focused on `10-K` and `10-Q`; `8-K` support is not implemented yet
- A validated GKE deployment path exists, but the repo does not claim a permanently operated public demo environment or committed Grafana Cloud collector install yet
- Ingest and query operations are synchronous today rather than background-job driven
- The evaluation corpus is intentionally smaller than the full live research surface
- The current company universe is focused on one sector to keep retrieval quality and evaluation tractable
- The serving stack is still local-disk oriented: processed chunks live under `data/`, Chroma persists under `artifacts/chroma`, and the API loads the processed corpus into memory on startup
- Horizontal Kubernetes scaling is not ready yet because multi-replica API pods would need shared state or externalized storage for the corpus and vector index
- Grafana Cloud collector configuration is still an operator step rather than a committed in-repo deployment artifact

## What makes this production-ready?

- clear separation between ingestion, retrieval, reranking, generation, serving, and evaluation
- grounded-answer behavior instead of generic LLM summarization
- measurable retrieval and answer-quality validation
- CI integration that treats evaluation as part of engineering quality
- thoughtful attention to typed interfaces, artifact visibility, and reproducibility

## Full Package Surface

Core runtime dependencies currently used in this repository:

- `beautifulsoup4` for SEC filing parsing
- `chromadb` for local vector storage and retrieval indexing
- `fastapi` for the typed backend API
- `numpy` for numeric and metric utilities
- `openai` for live answer generation and optional richer eval scoring
- `pydantic` for typed request, response, and artifact contracts
- `PyYAML` for company, retrieval, prompt, and eval configuration loading
- `python-dotenv` for local environment loading across CLI, API, and UI entrypoints
- `requests` for SEC EDGAR ingestion and HTTP access
- `sentence-transformers` for embeddings and cross-encoder reranking
- `streamlit` for the local portfolio UI
- `tiktoken` for token-aware chunking and prompt packing behavior
- `torch` for local model execution
- `transformers` for the underlying model stack
- `uvicorn` for serving the FastAPI application

Development and validation dependencies:

- `pytest` for unit and integration coverage
- `httpx` for API and contract-oriented tests

Optional evaluation dependencies:

- `ragas` for richer answer-quality evaluation
- `langchain-openai` for the optional Ragas-backed evaluator path

Build system dependencies:

- `setuptools`
- `wheel`

## Additional Engineering Signals

Some of the strongest production-oriented choices in this codebase are not just the model stack, but the operational and evaluation details around it:

- Citation traceability is preserved end to end. The system keeps stable `document_id` and `chunk_id` identifiers so answers can be traced back to real filing chunks instead of loose text snippets.
- The retrieval stack separates public citation units from embedding-only subchunks. That lets the index optimize retrieval quality without losing human-readable evidence boundaries.
- The service distinguishes readiness from coverage. A backend can be healthy but still correctly refuse a query if the requested ticker, form type, or filing-date scope is not indexed.
- Unsupported answers fail closed. Invalid citations, weak evidence, no hits, missing coverage, and reranker unavailability all map to explicit failure semantics instead of silently returning a fluent but weak answer.
- Evaluation includes both answerable and unanswerable examples. The tracked fixture dataset is designed to test fact lookup, cross-period comparison, multi-document synthesis, and abstention behavior.
- CI is wired to quality gates, not just tests. The workflow runs an eval smoke subset, uploads artifacts, and only then fails the job on regression so debugging evidence is still preserved.
- The current tracked eval set is intentionally reproducible. The fixture corpus uses six annual filings across NVIDIA, AMD, and Intel with a 12-example gold set and an 8-example `ci_smoke` subset.
- SEC ingestion is handled conservatively. Request behavior is centralized, a valid SEC user agent is required, and caching reduces unnecessary network traffic.
- Runtime behavior is environment-aware. `.env` is loaded consistently across CLI, API, and UI entrypoints, `HF_TOKEN` remains optional, and embedding device selection resolves automatically across `cuda`, `mps`, and `cpu`.
- The repo keeps retrieval, reranking, generation, serving, and evaluation as separate modules, which makes debugging, regression analysis, and future upgrades much easier than in a monolithic pipeline.
