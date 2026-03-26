from __future__ import annotations

from sec_copilot.eval.offline_runtime import TokenOverlapReranker
from sec_copilot.schemas.retrieval import RetrievedChunk


def _chunk(*, chunk_id: str, rrf_score: float, dense_rank: int, bm25_rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"{chunk_id}_doc",
        ticker="NVDA",
        company_name="NVIDIA",
        form_type="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        section_title="Business",
        source_url="https://example.com",
        text="ai infrastructure accelerated computing hopper blackwell",
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        rrf_score=rrf_score,
    )


def test_token_overlap_reranker_prefers_higher_rrf_for_tied_scores() -> None:
    reranker = TokenOverlapReranker(rerank_top_k=4)
    weaker = _chunk(
        chunk_id="sec_0001045810_000104581026000021_s02_c0006",
        rrf_score=0.030330882352941176,
        dense_rank=4,
        bm25_rank=8,
    )
    stronger = _chunk(
        chunk_id="sec_0001045810_000104581026000021_s07_c0008",
        rrf_score=0.031746031746031744,
        dense_rank=3,
        bm25_rank=3,
    )

    reranked = reranker.rerank(
        "Across NVIDIA, AMD, and Intel, which products are positioned around AI infrastructure or AI compute in this fixture corpus?",
        (weaker, stronger),
    )

    assert [chunk.chunk_id for chunk in reranked] == [
        "sec_0001045810_000104581026000021_s07_c0008",
        "sec_0001045810_000104581026000021_s02_c0006",
    ]
