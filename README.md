# SEC Filing Intelligence Copilot

SEC Filing Intelligence Copilot is a source-grounded analyst assistant for comparing and querying SEC filings across a focused company universe. The project is designed as a production-style retrieval-augmented generation system and as a learning artifact that documents architectural decisions, tradeoffs, evaluation strategy, and likely failure modes.

## Product framing

Core use cases:
- What changed in NVIDIA's risk factors from the last 10-K to the current one?
- What do AMD and Intel say about supply chain risk, and how do the disclosures differ?
- Which recent 8-Ks mention restructuring or layoffs?
- Summarize Microsoft's AI infrastructure risk statements and cite the exact passages.

The initial corpus is intentionally narrow:
- Companies: NVIDIA, AMD, Intel, Broadcom, Qualcomm
- V1 filing scope: latest 2 annual 10-Ks and latest 4 quarterly 10-Qs per company
- Later scope: recent 8-Ks and optional XBRL company-facts enrichment

## Why this project exists

This repository is meant to show:
- production-oriented AI system design
- modular retrieval and ranking architecture
- explicit handling of grounding, citations, and abstention
- measurable evaluation with regression gates
- documentation quality that helps a reviewer understand both the system and the learning process

SEC filings are a strong RAG domain because they are high-value, public, structured, and large enough that search quality, chunk design, and citation fidelity matter.

## Planned versions

- V0: repository scaffold, configs, docs, smoke tests
- V1: SEC ingestion, parsing, metadata normalization, section-aware chunking
- V2: dense retrieval baseline with Chroma and sentence-transformer embeddings
- V3: BM25 plus dense hybrid retrieval, reciprocal rank fusion, cross-encoder reranking
- V4: citation-enforced answers, abstention behavior, offline evaluation harness
- V5: FastAPI backend with typed contracts
- V6: CI regression gates and optional frontend polish

## Planned stack

- Python 3.11+
- LangChain for baseline orchestration
- Chroma for vector storage
- Sentence Transformers bi-encoder embeddings
- `rank_bm25` for lexical retrieval
- Sentence Transformers `CrossEncoder` for reranking
- Ragas plus custom retrieval metrics for evaluation
- FastAPI for serving
- GitHub Actions for CI

## Repository layout

```text
configs/              Centralized YAML configs for companies, prompts, and retrieval defaults
docs/                 Project charter, system design, eval plan, failure analysis, progress logs
src/sec_copilot/      Python package with clear module boundaries
tests/                Unit, integration, and regression test suites
.github/workflows/    CI automation
```

## Local development

The repository targets Python 3.11 or later. The current local interpreter should be checked with:

```bash
python3 --version
```

Recommended setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
make test
```

## Environment and SEC access

When ingestion is implemented, SEC requests should:
- send a descriptive `User-Agent`
- respect SEC fair-access guidance
- stay within the published request rate limits
- preserve filing metadata needed for traceable citations

See [docs/07_references.md](docs/07_references.md) for the primary sources that inform this project.
