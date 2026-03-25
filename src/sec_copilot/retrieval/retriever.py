"""Dense retrieval over Chroma with parent-chunk collapse and typed outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from sec_copilot.config.retrieval import RetrievalConfig
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import EmbeddingAdapter
from sec_copilot.retrieval.filters import build_chroma_where
from sec_copilot.schemas.retrieval import DebugRetrieval, QueryRequest, RetrievedChunk


@dataclass(frozen=True)
class DenseRetrievalOutcome:
    """Result of one dense-retrieval call before generation."""

    status: Literal["ok", "no_results", "weak_results"]
    reason_code: Literal["none", "filters_excluded_all", "no_retrieval_hits", "low_similarity"]
    results: tuple[RetrievedChunk, ...]
    debug: DebugRetrieval


class DenseRetriever:
    """Dense parent-chunk retriever backed by Chroma subchunk vectors."""

    def __init__(self, config: RetrievalConfig, adapter: EmbeddingAdapter, store: ProcessedChunkStore, collection) -> None:
        self.config = config
        self.adapter = adapter
        self.store = store
        self.collection = collection

    def retrieve(self, request: QueryRequest) -> DenseRetrievalOutcome:
        if len(self.store) == 0:
            return self._empty_outcome(
                status="no_results",
                reason_code="no_retrieval_hits",
                retrieval_top_k=request.retrieval_top_k or self.config.retrieval.retrieval_top_k,
            )

        if not self.store.has_matches(request.filters):
            return self._empty_outcome(
                status="no_results",
                reason_code="filters_excluded_all",
                retrieval_top_k=request.retrieval_top_k or self.config.retrieval.retrieval_top_k,
            )

        retrieval_top_k = request.retrieval_top_k or self.config.retrieval.retrieval_top_k
        where = build_chroma_where(request.filters)
        query_embedding = self.adapter.embed_texts([request.question])[0]
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.retrieval.retrieval_subchunk_top_k,
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
            if current is not None and score <= current.score:
                continue

            collapsed[parent_chunk_id] = RetrievedChunk(
                rank=1,
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
                score=score,
                raw_distance=float(raw_distance),
                best_subchunk_id=subchunk_id,
            )

        if not collapsed:
            return self._empty_outcome(
                status="no_results",
                reason_code="no_retrieval_hits",
                retrieval_top_k=retrieval_top_k,
            )

        ordered = sorted(
            collapsed.values(),
            key=lambda chunk: (-chunk.score, chunk.raw_distance, chunk.chunk_id),
        )[:retrieval_top_k]
        ranked = tuple(
            chunk.model_copy(update={"rank": index + 1})
            for index, chunk in enumerate(ordered)
        )
        debug = DebugRetrieval(
            retrieval_subchunk_top_k=self.config.retrieval.retrieval_subchunk_top_k,
            retrieval_top_k=retrieval_top_k,
            results=list(ranked),
        )
        if ranked and ranked[0].score < self.config.retrieval.weak_score_threshold:
            return DenseRetrievalOutcome(
                status="weak_results",
                reason_code="low_similarity",
                results=ranked,
                debug=debug,
            )

        return DenseRetrievalOutcome(
            status="ok",
            reason_code="none",
            results=ranked,
            debug=debug,
        )

    def _empty_outcome(
        self,
        *,
        status: Literal["no_results", "weak_results"],
        reason_code: Literal["filters_excluded_all", "no_retrieval_hits", "low_similarity"],
        retrieval_top_k: int,
    ) -> DenseRetrievalOutcome:
        return DenseRetrievalOutcome(
            status=status,
            reason_code=reason_code,
            results=(),
            debug=DebugRetrieval(
                retrieval_subchunk_top_k=self.config.retrieval.retrieval_subchunk_top_k,
                retrieval_top_k=retrieval_top_k,
                results=[],
            ),
        )


__all__ = ["DenseRetrievalOutcome", "DenseRetriever"]
