from __future__ import annotations

from pathlib import Path

from sec_copilot.config import load_retrieval_config
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import EmbeddingSubchunk
from sec_copilot.schemas import ChunkRecord


class FakeEmbeddingAdapter:
    model_name = "fake-embedding-v1"
    model_revision = None
    max_seq_length = 256
    embedding_dimension = 4
    requested_device = "auto"
    resolved_device = "cpu"
    sentence_transformers_version = "fake"
    torch_version = "fake"
    transformers_version = "fake"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0,
                    float("ai" in lowered or "accelerated" in lowered or "data center" in lowered),
                    2.0 * float("export" in lowered or "license" in lowered or "china" in lowered),
                    float("risk" in lowered or "competition" in lowered or "inventory" in lowered),
                ]
            )
        return vectors

    def build_subchunks(self, chunk: ChunkRecord) -> tuple[EmbeddingSubchunk, ...]:
        parts = [part.strip() for part in chunk.text.split("||") if part.strip()]
        if not parts:
            parts = [chunk.text]
        subchunks = [
            EmbeddingSubchunk(
                subchunk_id=f"{chunk.chunk_id}__emb_{index:04d}",
                parent_chunk_id=chunk.chunk_id,
                subchunk_index=index,
                subchunk_count=len(parts),
                text=part,
                ticker=chunk.ticker,
                form_type=chunk.form_type,
                filing_date=chunk.filing_date,
                accession_number=chunk.accession_number,
                document_id=chunk.document_id,
                section_title=chunk.section_title,
                source_url=chunk.source_url,
            )
            for index, part in enumerate(parts)
        ]
        return tuple(subchunks)


class FakeEmbeddingAdapterV2(FakeEmbeddingAdapter):
    model_name = "fake-embedding-v2"


def build_chunk(
    *,
    chunk_id: str,
    document_id: str,
    ticker: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
    section_title: str,
    text: str,
) -> ChunkRecord:
    return ChunkRecord(
        schema_version="chunk_record.v1",
        chunk_id=chunk_id,
        document_id=document_id,
        company_name=ticker,
        ticker=ticker,
        cik="0000000000",
        form_type=form_type,
        filing_date=filing_date,
        report_date=filing_date,
        accession_number=accession_number,
        source_url=f"https://example.com/{accession_number}",
        filing_index_url=f"https://example.com/{accession_number}/index",
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
        token_end=1,
        token_count=1,
        content_hash=f"hash-{chunk_id}",
        text=text,
    )


def build_store() -> ProcessedChunkStore:
    chunks = {
        "nvda_business": build_chunk(
            chunk_id="nvda_business",
            document_id="doc_nvda_business",
            ticker="NVDA",
            form_type="10-K",
            filing_date="2026-02-25",
            accession_number="0001045810-26-000021",
            section_title="Business",
            text="NVIDIA builds AI infrastructure and accelerated computing systems for data centers.",
        ),
        "nvda_risk": build_chunk(
            chunk_id="nvda_risk",
            document_id="doc_nvda_risk",
            ticker="NVDA",
            form_type="10-K",
            filing_date="2026-02-25",
            accession_number="0001045810-26-000021",
            section_title="Risk Factors",
            text="General competition risk.||Export controls in China may require licenses and disrupt sales.",
        ),
        "amd_q": build_chunk(
            chunk_id="amd_q",
            document_id="doc_amd_q",
            ticker="AMD",
            form_type="10-Q",
            filing_date="2025-10-29",
            accession_number="0000002488-25-000099",
            section_title="Risk Factors",
            text="Inventory risk and competition affect quarterly demand.",
        ),
    }
    return ProcessedChunkStore(chunks)


def build_config(tmp_path: Path):
    config = load_retrieval_config("configs/retrieval.yaml")
    return config.model_copy(
        update={
            "index": config.index.model_copy(
                update={
                    "persist_directory": str(tmp_path / "chroma"),
                    "collection_name": "test_dense_v2",
                }
            )
        }
    )
