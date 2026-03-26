from __future__ import annotations

from sec_copilot.config import load_prompt_catalog, load_retrieval_config
from sec_copilot.eval.offline_runtime import build_offline_eval_runtime
from sec_copilot.schemas.retrieval import QueryRequest, RetrievalFilters


def test_three_company_synthesis_prompt_keeps_nvidia_ai_chunk() -> None:
    config = load_retrieval_config("configs/retrieval.yaml")
    prompts = load_prompt_catalog("configs/prompts.yaml")
    runtime = build_offline_eval_runtime(
        config=config,
        prompt_catalog=prompts,
        corpus_path="tests/fixtures/eval_corpus",
        persist_directory="artifacts/test_prompt_regression_chroma",
        collection_name="test_prompt_regression_three_company_synthesis",
    )
    request = QueryRequest(
        question="Across NVIDIA, AMD, and Intel, which products are positioned around AI infrastructure or AI compute in this fixture corpus?",
        filters=RetrievalFilters(
            tickers=["NVDA", "AMD", "INTC"],
            form_types=["10-K"],
            filing_date_from="2026-01-01",
            filing_date_to="2026-12-31",
        ),
        debug=True,
    )

    retrieval = runtime.retriever.retrieve(request)
    retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieval.retrieved_chunks]
    prompt = runtime.prompt_builder.build(
        question=request.question,
        retrieved_chunks=list(retrieval.retrieved_chunks),
    )

    assert "sec_0001045810_000104581026000021_s07_c0008" in retrieved_chunk_ids
    assert "sec_0001045810_000104581026000021_s07_c0008" in prompt.context_chunk_ids
    assert "sec_0001045810_000104581026000021_s02_c0006" not in prompt.context_chunk_ids
    assert {"NVDA", "AMD", "INTC"} == {
        chunk.ticker for chunk in retrieval.retrieved_chunks[:4]
    }


def test_three_company_synthesis_prefers_stronger_nvidia_tie_break() -> None:
    config = load_retrieval_config("configs/retrieval.yaml")
    prompts = load_prompt_catalog("configs/prompts.yaml")
    runtime = build_offline_eval_runtime(
        config=config,
        prompt_catalog=prompts,
        corpus_path="tests/fixtures/eval_corpus",
        persist_directory="artifacts/test_prompt_regression_order_chroma",
        collection_name="test_prompt_regression_three_company_order",
    )
    request = QueryRequest(
        question="Across NVIDIA, AMD, and Intel, which products are positioned around AI infrastructure or AI compute in this fixture corpus?",
        filters=RetrievalFilters(
            tickers=["NVDA", "AMD", "INTC"],
            form_types=["10-K"],
            filing_date_from="2026-01-01",
            filing_date_to="2026-12-31",
        ),
        debug=True,
    )

    retrieval = runtime.retriever.retrieve(request)
    positions = {
        chunk.chunk_id: index
        for index, chunk in enumerate(retrieval.retrieved_chunks)
    }

    assert positions["sec_0001045810_000104581026000021_s07_c0008"] < positions[
        "sec_0001045810_000104581026000021_s02_c0006"
    ]
