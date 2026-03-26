# SEC Filing Intelligence Copilot

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/ui-Streamlit-FF4B4B)
![CI](https://img.shields.io/badge/ci-GitHub%20Actions-2088FF)
![RAG Evaluation](https://img.shields.io/badge/focus-RAG%20Evaluation-6A5ACD)

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

The backend exposes typed endpoints for both user-facing and operational workflows:

- `GET /health`
- `GET /build-info`
- `POST /ingest/run`
- `POST /query`
- `POST /retrieve/debug`
- `POST /eval/run`

### Retrieval and answer pipeline

- Live filing ingestion from SEC EDGAR with configurable company and form scope
- Chunked processed corpus persisted locally for indexing and retrieval
- Hybrid retrieval using dense vectors and lexical retrieval
- Cross-encoder reranking for higher-quality evidence selection
- Grounded answer pipeline that returns citations mapped to retrieved chunks
- Coverage-aware behavior for missing corpus scope or stale index state

### Reliability and engineering discipline

- Typed request and response models across the API layer
- Health and build-info endpoints for operational visibility
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

## Repository Structure

```text
artifacts/            Eval outputs, index metadata, and local retrieval artifacts
configs/              Company universe, retrieval settings, prompts, and eval config
data/                 Downloaded and processed filing data
src/sec_copilot/      Ingestion, retrieval, reranking, generation, API, and frontend code
tests/                Unit tests, integration tests, fixtures, and gold eval data
Makefile              Common local development and validation commands
```

## Current Limitations / Next Improvements

- Filing scope is currently focused on `10-K` and `10-Q`; `8-K` support is not implemented yet
- The primary demo flow is local-first rather than deployed as a public hosted application
- Ingest and query operations are synchronous today rather than background-job driven
- The evaluation corpus is intentionally smaller than the full live research surface
- The current company universe is focused on one sector to keep retrieval quality and evaluation tractable

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
