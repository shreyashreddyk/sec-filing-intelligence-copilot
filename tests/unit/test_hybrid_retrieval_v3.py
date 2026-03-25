from __future__ import annotations

from pathlib import Path

from sec_copilot.eval.curated_examples import curated_examples
from sec_copilot.retrieval.bm25 import BM25Retriever
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever, HybridRetriever
from sec_copilot.schemas import QueryRequest, RetrievalFilters
from tests.unit.retrieval_helpers import CuratedEmbeddingAdapter, build_config, build_reranker, build_store


def _build_stack(tmp_path: Path):
    store = build_store()
    config = build_config(tmp_path)
    adapter = CuratedEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    dense_retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())
    hybrid_retriever = HybridRetriever(config, store, dense_retriever, BM25Retriever(store), build_reranker())
    return config, dense_retriever, hybrid_retriever


def _rank_of(chunk_ids: list[str], target: str) -> int | None:
    try:
        return chunk_ids.index(target) + 1
    except ValueError:
        return None


def test_bm25_improvement_moves_expected_chunk_higher(tmp_path: Path) -> None:
    examples = {example.name: example for example in curated_examples()}
    example = examples["bm25_help_h20"]
    config, dense_retriever, hybrid_retriever = _build_stack(tmp_path)

    filters = RetrievalFilters(tickers=list(example.tickers), form_types=list(example.form_types))
    dense_results = dense_retriever.retrieve(example.query, filters, top_k=config.retrieval.dense_top_k)
    hybrid_results = hybrid_retriever.retrieve(QueryRequest(question=example.query, filters=filters))

    dense_rank = _rank_of([chunk.chunk_id for chunk in dense_results.results], example.expected_relevant_chunk_id)
    hybrid_rank = _rank_of([chunk.chunk_id for chunk in hybrid_results.retrieved_chunks], example.expected_relevant_chunk_id)

    assert dense_rank is not None
    assert hybrid_rank is not None
    assert hybrid_rank < dense_rank


def test_reranking_improvement_moves_expected_chunk_higher(tmp_path: Path) -> None:
    examples = {example.name: example for example in curated_examples()}
    example = examples["rerank_help_energy_capital"]
    _, dense_retriever, hybrid_retriever = _build_stack(tmp_path)

    filters = RetrievalFilters(tickers=list(example.tickers), form_types=list(example.form_types))
    dense_results = dense_retriever.retrieve(example.query, filters)
    hybrid_results = hybrid_retriever.retrieve(QueryRequest(question=example.query, filters=filters))

    dense_rank = _rank_of([chunk.chunk_id for chunk in dense_results.results], example.expected_relevant_chunk_id)
    rerank_rank = _rank_of([chunk.chunk_id for chunk in hybrid_results.retrieved_chunks], example.expected_relevant_chunk_id)

    assert dense_rank is not None
    assert rerank_rank is not None
    assert rerank_rank < dense_rank


def test_retrieval_response_exposes_stage_counts_and_rerank_fields(tmp_path: Path) -> None:
    _, _, hybrid_retriever = _build_stack(tmp_path)

    response = hybrid_retriever.retrieve(
        QueryRequest(
            question="What H20 license restrictions does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
        )
    ).to_response()

    assert response.stage_counts.filtered_parent_count > 0
    assert response.stage_counts.fused_candidate_count > 0
    assert response.reranker_applied is True
    assert any(chunk.rrf_score is not None for chunk in response.retrieved_chunks)
    assert any(chunk.rerank_rank is not None for chunk in response.retrieved_chunks)
