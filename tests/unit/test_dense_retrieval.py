from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot.retrieval.indexer import ChromaIndexManager, DenseIndexError
from sec_copilot.retrieval.retriever import DenseRetriever
from sec_copilot.schemas import QueryRequest, RetrievalFilters
from tests.unit.retrieval_helpers import FakeEmbeddingAdapter, FakeEmbeddingAdapterV2, build_config, build_store


def test_index_build_is_idempotent_and_parent_child_mapping_works(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)
    adapter = FakeEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)

    first = index_manager.build(store, mode="rebuild")
    second = index_manager.build(store, mode="upsert")

    assert first.embedding_subchunk_count == 4
    assert second.embedding_subchunk_count == 4
    assert second.stale_id_count == 0
    assert first.requested_embedding_device == "auto"
    assert first.resolved_embedding_device == "cpu"
    metadata = index_manager.load_build_metadata()
    assert metadata is not None
    assert metadata.requested_embedding_device == "auto"
    assert metadata.resolved_embedding_device == "cpu"
    assert metadata.torch_version == "fake"
    assert index_manager.get_collection().count() == 4

    retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())
    outcome = retriever.retrieve(
        QueryRequest(
            question="What export control risks does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["nvda"], form_types=["10-k"]),
            debug=True,
        )
    )

    assert outcome.status == "ok"
    assert outcome.results[0].chunk_id == "nvda_risk"
    assert outcome.results[0].best_subchunk_id.endswith("__emb_0001")
    assert outcome.results[0].score > 0.20


def test_upsert_detects_model_mismatch_and_requires_rebuild(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)

    ChromaIndexManager(config, FakeEmbeddingAdapter()).build(store, mode="rebuild")

    with pytest.raises(DenseIndexError):
        ChromaIndexManager(config, FakeEmbeddingAdapterV2()).build(store, mode="upsert")


def test_retrieval_filters_normalize_case_and_apply_date_boundaries(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)
    adapter = FakeEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())

    outcome = retriever.retrieve(
        QueryRequest(
            question="What does the company say about AI infrastructure?",
            filters=RetrievalFilters(
                tickers=["nvda"],
                form_types=["10-k"],
                filing_date_from="2026-02-25",
                filing_date_to="2026-02-25",
            ),
        )
    )

    assert outcome.status == "ok"
    assert [chunk.chunk_id for chunk in outcome.results] == ["nvda_business", "nvda_risk"]

    no_match = retriever.retrieve(
        QueryRequest(
            question="What does the company say about AI infrastructure?",
            filters=RetrievalFilters(
                tickers=["nvda"],
                form_types=["10-k"],
                filing_date_from="2026-02-26",
            ),
        )
    )

    assert no_match.status == "no_results"
    assert no_match.reason_code == "filters_excluded_all"
