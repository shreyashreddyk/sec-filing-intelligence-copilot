"""Dense, BM25, fusion, and hybrid retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter

from sec_copilot.config.retrieval import RetrievalConfig
from sec_copilot.rerank.cross_encoder import Reranker, RerankerUnavailableError
from sec_copilot.retrieval.bm25 import BM25Retriever
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import EmbeddingAdapter
from sec_copilot.retrieval.filters import build_chroma_where
from sec_copilot.retrieval.fusion import fuse_with_rrf
from sec_copilot.schemas.retrieval import (
    QueryRequest,
    ReasonCode,
    RetrievedChunk,
    RetrievalFilters,
    RetrievalResponse,
    RetrievalStageCounts,
)
from sec_copilot.utils.logging import log_query_event


@dataclass(frozen=True)
class DenseRetrievalResult:
    """Dense parent-candidate results plus query-stage counts."""

    results: tuple[RetrievedChunk, ...]
    subchunk_hit_count: int
    parent_candidate_count: int
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class RetrievalTimingBreakdown:
    """Latency summary for retrieval stages."""

    dense_ms: float
    bm25_ms: float
    fusion_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True)
class HybridRetrievalOutcome:
    """Internal retrieval outcome shared by retrieval and answer flows."""

    reason_code: ReasonCode
    retrieved_chunks: tuple[RetrievedChunk, ...]
    stage_counts: RetrievalStageCounts
    reranker_applied: bool
    reranker_skipped_reason: str | None = None
    timings: RetrievalTimingBreakdown = RetrievalTimingBreakdown(
        dense_ms=0.0,
        bm25_ms=0.0,
        fusion_ms=0.0,
        rerank_ms=0.0,
        total_ms=0.0,
    )

    def to_response(self) -> RetrievalResponse:
        return RetrievalResponse(
            reason_code=self.reason_code,
            retrieved_chunks=list(self.retrieved_chunks),
            stage_counts=self.stage_counts,
            reranker_applied=self.reranker_applied,
            reranker_skipped_reason=self.reranker_skipped_reason,
        )


class DenseRetriever:
    """Dense parent-chunk retriever backed by Chroma subchunk vectors."""

    def __init__(self, config: RetrievalConfig, adapter: EmbeddingAdapter, store: ProcessedChunkStore, collection) -> None:
        self.config = config
        self.adapter = adapter
        self.store = store
        self.collection = collection

    def retrieve(self, question: str, filters: RetrievalFilters, top_k: int | None = None) -> DenseRetrievalResult:
        started_at = perf_counter()
        if len(self.store) == 0:
            return DenseRetrievalResult(
                results=(),
                subchunk_hit_count=0,
                parent_candidate_count=0,
                elapsed_ms=0.0,
            )

        dense_top_k = top_k or self.config.retrieval.dense_top_k
        where = build_chroma_where(filters)
        query_embedding = self.adapter.embed_texts([question])[0]
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.retrieval.dense_subchunk_top_k,
            where=where,
            include=["distances", "metadatas"],
        )

        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        collapsed: dict[str, RetrievedChunk] = {}

        for subchunk_id, raw_distance, metadata in zip(ids, distances, metadatas):
            if not metadata:
                continue
            parent_chunk_id = metadata["parent_chunk_id"]
            parent_chunk = self.store.get(parent_chunk_id)
            if parent_chunk is None:
                continue

            score = 1.0 - float(raw_distance)
            current = collapsed.get(parent_chunk_id)
            if current is not None and score <= (current.dense_score or float("-inf")):
                continue

            collapsed[parent_chunk_id] = RetrievedChunk(
                chunk_id=parent_chunk.chunk_id,
                document_id=parent_chunk.document_id,
                ticker=parent_chunk.ticker,
                company_name=parent_chunk.company_name,
                form_type=parent_chunk.form_type,
                filing_date=date.fromisoformat(parent_chunk.filing_date),
                accession_number=parent_chunk.accession_number,
                section_title=parent_chunk.section_title,
                source_url=parent_chunk.source_url,
                text=parent_chunk.text,
                dense_score=score,
                dense_raw_distance=float(raw_distance),
                best_subchunk_id=subchunk_id,
            )

        ordered = sorted(
            collapsed.values(),
            key=lambda chunk: (
                -(chunk.dense_score or float("-inf")),
                chunk.dense_raw_distance if chunk.dense_raw_distance is not None else float("inf"),
                chunk.chunk_id,
            ),
        )[:dense_top_k]
        ranked = tuple(
            chunk.model_copy(update={"dense_rank": index + 1})
            for index, chunk in enumerate(ordered)
        )
        return DenseRetrievalResult(
            results=ranked,
            subchunk_hit_count=len(ids),
            parent_candidate_count=len(collapsed),
            elapsed_ms=(perf_counter() - started_at) * 1000.0,
        )


class HybridRetriever:
    """Hybrid retrieval with dense roll-up, BM25, RRF, and optional reranking."""

    def __init__(
        self,
        config: RetrievalConfig,
        store: ProcessedChunkStore,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

    def retrieve(self, request: QueryRequest) -> HybridRetrievalOutcome:
        started_at = perf_counter()
        filtered_parent_chunks = self.store.filtered_values(request.filters)
        if not filtered_parent_chunks:
            stage_counts = RetrievalStageCounts(
                filtered_parent_count=0,
                dense_subchunk_hit_count=0,
                dense_parent_candidate_count=0,
                bm25_candidate_count=0,
                fused_candidate_count=0,
                reranked_candidate_count=0,
            )
            outcome = HybridRetrievalOutcome(
                reason_code="filters_excluded_all_chunks",
                retrieved_chunks=(),
                stage_counts=stage_counts,
                reranker_applied=False,
                reranker_skipped_reason="filters_excluded_all_chunks",
                timings=RetrievalTimingBreakdown(
                    dense_ms=0.0,
                    bm25_ms=0.0,
                    fusion_ms=0.0,
                    rerank_ms=0.0,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )
            self._log_retrieval(request, outcome)
            return outcome

        dense_result = self.dense_retriever.retrieve(request.question, request.filters)
        bm25_started_at = perf_counter()
        bm25_result = self.bm25_retriever.retrieve(
            request.question,
            request.filters,
            self.config.retrieval.bm25_top_k,
        )
        bm25_elapsed_ms = (perf_counter() - bm25_started_at) * 1000.0

        if not dense_result.results and not bm25_result.results:
            stage_counts = RetrievalStageCounts(
                filtered_parent_count=len(filtered_parent_chunks),
                dense_subchunk_hit_count=dense_result.subchunk_hit_count,
                dense_parent_candidate_count=dense_result.parent_candidate_count,
                bm25_candidate_count=bm25_result.candidate_count,
                fused_candidate_count=0,
                reranked_candidate_count=0,
            )
            outcome = HybridRetrievalOutcome(
                reason_code="no_hits",
                retrieved_chunks=(),
                stage_counts=stage_counts,
                reranker_applied=False,
                reranker_skipped_reason="no_hits",
                timings=RetrievalTimingBreakdown(
                    dense_ms=dense_result.elapsed_ms,
                    bm25_ms=bm25_elapsed_ms,
                    fusion_ms=0.0,
                    rerank_ms=0.0,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )
            self._log_retrieval(request, outcome)
            return outcome

        fusion_started_at = perf_counter()
        fused_results = fuse_with_rrf(
            dense_results=dense_result.results,
            bm25_results=bm25_result.results,
            rrf_k=self.config.fusion.rrf_k,
            top_k=self.config.retrieval.fused_top_k_before_rerank,
        )
        fusion_elapsed_ms = (perf_counter() - fusion_started_at) * 1000.0

        reason_code: ReasonCode = "ok"
        reranker_applied = False
        reranker_skipped_reason: str | None = None
        final_results = fused_results
        rerank_elapsed_ms = 0.0

        if self.config.reranking.enabled:
            if self.reranker is None:
                reason_code = "reranker_unavailable"
                reranker_skipped_reason = "reranker_not_configured"
            else:
                try:
                    rerank_started_at = perf_counter()
                    final_results = self.reranker.rerank(request.question, fused_results)
                    rerank_elapsed_ms = (perf_counter() - rerank_started_at) * 1000.0
                    reranker_applied = True
                except RerankerUnavailableError:
                    reason_code = "reranker_unavailable"
                    reranker_skipped_reason = "reranker_unavailable"
                    final_results = fused_results
        else:
            reranker_skipped_reason = "disabled_by_config"

        stage_counts = RetrievalStageCounts(
            filtered_parent_count=len(filtered_parent_chunks),
            dense_subchunk_hit_count=dense_result.subchunk_hit_count,
            dense_parent_candidate_count=dense_result.parent_candidate_count,
            bm25_candidate_count=bm25_result.candidate_count,
            fused_candidate_count=len(fused_results),
            reranked_candidate_count=len(final_results) if reranker_applied else 0,
        )
        outcome = HybridRetrievalOutcome(
            reason_code=reason_code,
            retrieved_chunks=final_results,
            stage_counts=stage_counts,
            reranker_applied=reranker_applied,
            reranker_skipped_reason=reranker_skipped_reason,
            timings=RetrievalTimingBreakdown(
                dense_ms=dense_result.elapsed_ms,
                bm25_ms=bm25_elapsed_ms,
                fusion_ms=fusion_elapsed_ms,
                rerank_ms=rerank_elapsed_ms,
                total_ms=(perf_counter() - started_at) * 1000.0,
            ),
        )
        self._log_retrieval(request, outcome)
        return outcome

    def _log_retrieval(self, request: QueryRequest, outcome: HybridRetrievalOutcome) -> None:
        log_query_event(
            {
                "query_text": request.question,
                "filters": request.filters.model_dump(mode="json"),
                "filtered_parent_count": outcome.stage_counts.filtered_parent_count,
                "dense_subchunk_hit_count": outcome.stage_counts.dense_subchunk_hit_count,
                "dense_parent_candidate_count": outcome.stage_counts.dense_parent_candidate_count,
                "bm25_candidate_count": outcome.stage_counts.bm25_candidate_count,
                "fused_candidate_count": outcome.stage_counts.fused_candidate_count,
                "reranked_candidate_count": outcome.stage_counts.reranked_candidate_count,
                "final_context_chunk_ids": [],
                "prompt_name": None,
                "prompt_version": None,
                "abstained": None,
                "reason_code": outcome.reason_code,
                "reranker_applied": outcome.reranker_applied,
                "reranker_skipped_reason": outcome.reranker_skipped_reason,
            }
        )


__all__ = [
    "DenseRetrievalResult",
    "DenseRetriever",
    "HybridRetrievalOutcome",
    "HybridRetriever",
    "RetrievalTimingBreakdown",
]
