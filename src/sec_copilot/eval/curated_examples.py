"""Curated local examples for deterministic retrieval and reranking comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import EmbeddingSubchunk
from sec_copilot.schemas.ingestion import ChunkRecord
from sec_copilot.schemas.retrieval import RetrievedChunk


@dataclass(frozen=True)
class CuratedQueryExample:
    """One curated query plus the expected relevant chunk and improvement mode."""

    name: str
    query: str
    tickers: tuple[str, ...]
    form_types: tuple[str, ...]
    expected_relevant_chunk_id: str
    improvement_mode: str


def curated_examples() -> tuple[CuratedQueryExample, ...]:
    return (
        CuratedQueryExample(
            name="semantic_baseline",
            query="What does NVIDIA say about AI infrastructure and accelerated computing?",
            tickers=("NVDA",),
            form_types=("10-K",),
            expected_relevant_chunk_id="nvda_ai_platform",
            improvement_mode="dense_baseline",
        ),
        CuratedQueryExample(
            name="bm25_help_h20",
            query="What H20 license restrictions does NVIDIA describe?",
            tickers=("NVDA",),
            form_types=("10-K",),
            expected_relevant_chunk_id="nvda_h20_license",
            improvement_mode="bm25_help",
        ),
        CuratedQueryExample(
            name="rerank_help_energy_capital",
            query="What constraints on energy and capital could slow NVIDIA AI infrastructure deployments?",
            tickers=("NVDA",),
            form_types=("10-K",),
            expected_relevant_chunk_id="nvda_energy_capital_risk",
            improvement_mode="rerank_help",
        ),
    )


def build_curated_store() -> ProcessedChunkStore:
    chunks = {
        "nvda_ai_platform": _build_chunk(
            chunk_id="nvda_ai_platform",
            section_title="Business",
            text=(
                "NVIDIA is a data center scale AI infrastructure company. "
                "Its CUDA software, Blackwell systems, and accelerated computing platforms support AI training and inference."
            ),
        ),
        "nvda_ai_marketing": _build_chunk(
            chunk_id="nvda_ai_marketing",
            section_title="Business",
            text=(
                "All major cloud service providers and enterprises use NVIDIA AI infrastructure and computing platforms "
                "to support customer deployments and deliver AI services."
            ),
        ),
        "nvda_export_controls_general": _build_chunk(
            chunk_id="nvda_export_controls_general",
            section_title="Risk Factors",
            text=(
                "Government actions such as export controls, tariffs, and changing policies in China and other regions "
                "may adversely affect demand for NVIDIA products."
            ),
        ),
        "nvda_h20_license": _build_chunk(
            chunk_id="nvda_h20_license",
            section_title="Business",
            text=(
                "In April 2025, the USG required a license for export to China of H20 integrated circuits and similar chips. "
                "These H20 licensing restrictions reduced demand and created excess inventory and purchase obligations."
            ),
        ),
        "nvda_energy_capital_risk": _build_chunk(
            chunk_id="nvda_energy_capital_risk",
            section_title="Risk Factors",
            text=(
                "The availability of data centers, energy, and capital to support the buildout of NVIDIA AI infrastructure "
                "by customers and partners is crucial, and shortages could delay deployments or reduce adoption."
            ),
        ),
        "nvda_responsible_ai": _build_chunk(
            chunk_id="nvda_responsible_ai",
            section_title="Business",
            text=(
                "Compliance with cybersecurity, climate change, and responsible use of AI requirements could increase costs "
                "and affect NVIDIA's competitive position."
            ),
        ),
        "nvda_purchase_commitments": _build_chunk(
            chunk_id="nvda_purchase_commitments",
            section_title="Business",
            text=(
                "NVIDIA may place non-cancellable inventory orders and prepaid manufacturing commitments in advance "
                "to secure future supply and capacity."
            ),
        ),
    }
    return ProcessedChunkStore(chunks)


class CuratedEmbeddingAdapter:
    """Deterministic embedding adapter for offline hybrid-retrieval tests and comparisons."""

    model_name = "curated-embedding-v1"
    model_revision = None
    max_seq_length = 256
    embedding_dimension = 4
    requested_device = "cpu"
    resolved_device = "cpu"
    sentence_transformers_version = "curated"
    torch_version = "curated"
    transformers_version = "curated"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(any(token in lowered for token in ("ai", "infrastructure", "deployments", "cloud"))),
                    float(any(token in lowered for token in ("export", "license", "tariff", "china", "government"))),
                    float(any(token in lowered for token in ("accelerated", "cuda", "blackwell", "training", "inference"))),
                    float(any(token in lowered for token in ("risk", "demand", "supply", "obligations", "inventory"))),
                ]
            )
        return vectors

    def build_subchunks(self, chunk: ChunkRecord) -> tuple[EmbeddingSubchunk, ...]:
        return (
            EmbeddingSubchunk(
                subchunk_id=f"{chunk.chunk_id}__emb_0000",
                parent_chunk_id=chunk.chunk_id,
                subchunk_index=0,
                subchunk_count=1,
                text=chunk.text,
                ticker=chunk.ticker,
                form_type=chunk.form_type,
                filing_date=chunk.filing_date,
                accession_number=chunk.accession_number,
                document_id=chunk.document_id,
                section_title=chunk.section_title,
                source_url=chunk.source_url,
            ),
        )


class CuratedReranker:
    """Deterministic reranker that rewards exact support terms for curated queries."""

    def rerank(self, question: str, candidates: tuple[RetrievedChunk, ...]) -> tuple[RetrievedChunk, ...]:
        lowered_question = question.lower()
        rescored = []
        for chunk in candidates:
            score = _curated_rerank_score(lowered_question, chunk.text.lower())
            rescored.append(
                chunk.model_copy(
                    update={
                        "rerank_score": score,
                    }
                )
            )
        ordered = sorted(rescored, key=lambda chunk: (-(chunk.rerank_score or 0.0), chunk.chunk_id))
        return tuple(
            chunk.model_copy(update={"rerank_rank": index + 1})
            for index, chunk in enumerate(ordered)
        )


def build_curated_config(tmp_path: Path):
    from sec_copilot.config import load_retrieval_config

    config = load_retrieval_config("configs/retrieval.yaml")
    return config.model_copy(
        update={
            "index": config.index.model_copy(
                update={
                    "persist_directory": str(tmp_path / "chroma"),
                    "collection_name": "curated_hybrid_v3",
                }
            )
        }
    )


def _curated_rerank_score(question: str, text: str) -> float:
    score = 0.1
    if "h20" in question and "h20" in text:
        score += 0.8
    if "energy" in question and "energy" in text:
        score += 0.6
    if "capital" in question and "capital" in text:
        score += 0.6
    if "export" in question and "export" in text:
        score += 0.4
    if "license" in question and "license" in text:
        score += 0.4
    if "ai infrastructure" in question and "ai infrastructure" in text:
        score += 0.3
    if "accelerated computing" in question and "accelerated computing" in text:
        score += 0.3
    return min(score, 0.99)


def _build_chunk(*, chunk_id: str, section_title: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        schema_version="chunk_record.v1",
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        company_name="NVIDIA",
        ticker="NVDA",
        cik="0001045810",
        form_type="10-K",
        filing_date="2026-02-25",
        report_date="2026-01-25",
        accession_number="0001045810-26-000021",
        source_url=f"https://example.com/{chunk_id}",
        filing_index_url="https://example.com/index",
        source_kind="primary_document",
        raw_path=f"data/raw/{chunk_id}.txt",
        section_key=section_title.lower().replace(" ", "_"),
        section_title=section_title,
        section_order=1,
        item_number=None,
        parser_strategy="item_headers",
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        token_start=0,
        token_end=max(1, len(text.split())),
        token_count=max(1, len(text.split())),
        content_hash=f"hash-{chunk_id}",
        text=text,
    )


__all__ = [
    "CuratedEmbeddingAdapter",
    "CuratedQueryExample",
    "CuratedReranker",
    "build_curated_config",
    "build_curated_store",
    "curated_examples",
]
