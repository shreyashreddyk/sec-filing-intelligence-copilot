from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot.retrieval.bm25 import tokenize_bm25_text
from sec_copilot.retrieval.indexer import ChromaIndexManager, DenseIndexError
from sec_copilot.retrieval.retriever import DenseRetriever
from sec_copilot.schemas import QueryRequest, RetrievalFilters
from tests.unit.retrieval_helpers import CuratedEmbeddingAdapter, CuratedEmbeddingAdapterV2, build_config, build_store


def test_index_build_is_idempotent_and_dense_rollup_works(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)
    adapter = CuratedEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)

    first = index_manager.build(store, mode="rebuild")
    second = index_manager.build(store, mode="upsert")

    assert first.embedding_subchunk_count == len(store)
    assert second.embedding_subchunk_count == len(store)
    assert second.stale_id_count == 0
    metadata = index_manager.load_build_metadata()
    assert metadata is not None
    assert metadata.requested_embedding_device == "cpu"
    assert metadata.resolved_embedding_device == "cpu"
    assert index_manager.get_collection().count() == len(store)

    retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())
    result = retriever.retrieve(
        "What does NVIDIA say about AI infrastructure and accelerated computing?",
        RetrievalFilters(tickers=["nvda"], form_types=["10-k"]),
    )

    assert result.results[0].chunk_id == "nvda_ai_platform"
    assert result.results[0].best_subchunk_id == "nvda_ai_platform__emb_0000"
    assert result.results[0].dense_rank == 1


def test_upsert_detects_model_mismatch_and_requires_rebuild(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)

    ChromaIndexManager(config, CuratedEmbeddingAdapter()).build(store, mode="rebuild")

    with pytest.raises(DenseIndexError):
        ChromaIndexManager(config, CuratedEmbeddingAdapterV2()).build(store, mode="upsert")


def test_retrieval_filters_normalize_case_and_apply_date_boundaries(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)
    adapter = CuratedEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())

    result = retriever.retrieve(
        "What does NVIDIA say about AI infrastructure?",
        RetrievalFilters(
            tickers=["nvda"],
            form_types=["10-k"],
            filing_date_from="2026-02-25",
            filing_date_to="2026-02-25",
        ),
    )
    assert result.results

    no_match_filters = RetrievalFilters(
        tickers=["nvda"],
        form_types=["10-k"],
        filing_date_from="2026-02-26",
    )
    empty = retriever.retrieve("What does NVIDIA say about AI infrastructure?", no_match_filters)
    assert empty.results == ()


def test_bm25_tokenization_preserves_sec_terms_and_hyphen_parts() -> None:
    tokens = tokenize_bm25_text("Item 1A H20 export-license AI infrastructure NVDA")
    assert "item" in tokens
    assert "1a" in tokens
    assert "h20" in tokens
    assert "export-license" in tokens
    assert "export" in tokens
    assert "license" in tokens
    assert "ai" in tokens
    assert "infrastructure" in tokens
    assert "nvda" in tokens


def test_query_request_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        QueryRequest(question="   ")
