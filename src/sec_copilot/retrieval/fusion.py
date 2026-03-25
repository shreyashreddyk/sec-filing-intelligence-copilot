"""Reciprocal rank fusion helpers for dense and BM25 retrieval."""

from __future__ import annotations

from sec_copilot.schemas.retrieval import RetrievedChunk


def fuse_with_rrf(
    *,
    dense_results: tuple[RetrievedChunk, ...],
    bm25_results: tuple[RetrievedChunk, ...],
    rrf_k: int,
    top_k: int,
) -> tuple[RetrievedChunk, ...]:
    """Fuse parent-chunk candidates with reciprocal rank fusion."""

    merged: dict[str, RetrievedChunk] = {}
    for result in dense_results:
        merged[result.chunk_id] = result
    for result in bm25_results:
        existing = merged.get(result.chunk_id)
        if existing is None:
            merged[result.chunk_id] = result
            continue
        merged[result.chunk_id] = existing.model_copy(
            update={
                "bm25_rank": result.bm25_rank,
                "bm25_score": result.bm25_score,
            }
        )

    fused: list[RetrievedChunk] = []
    for chunk_id, result in merged.items():
        rrf_score = 0.0
        if result.dense_rank is not None:
            rrf_score += 1.0 / (rrf_k + result.dense_rank)
        if result.bm25_rank is not None:
            rrf_score += 1.0 / (rrf_k + result.bm25_rank)
        fused.append(result.model_copy(update={"rrf_score": rrf_score}))

    ordered = sorted(
        fused,
        key=lambda chunk: (
            -(chunk.rrf_score or 0.0),
            _best_available_rank(chunk),
            chunk.dense_rank if chunk.dense_rank is not None else 10**9,
            chunk.bm25_rank if chunk.bm25_rank is not None else 10**9,
            chunk.chunk_id,
        ),
    )
    return tuple(ordered[:top_k])


def _best_available_rank(chunk: RetrievedChunk) -> int:
    candidate_ranks = [rank for rank in (chunk.dense_rank, chunk.bm25_rank) if rank is not None]
    if not candidate_ranks:
        return 10**9
    return min(candidate_ranks)


__all__ = ["fuse_with_rrf"]
