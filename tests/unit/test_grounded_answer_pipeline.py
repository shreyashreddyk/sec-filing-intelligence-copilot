from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot.config import load_prompt_catalog
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.generation.prompts import GroundedPromptBuilder
from sec_copilot.generation.providers import LLMProvider
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever
from sec_copilot.schemas import ProviderAnswer, QueryRequest, RetrievalFilters
from tests.unit.retrieval_helpers import FakeEmbeddingAdapter, build_config, build_store


class StaticProvider(LLMProvider):
    name = "static"

    def __init__(self, answer: ProviderAnswer) -> None:
        self.answer = answer
        self.called = False

    def generate(self, prompt) -> ProviderAnswer:
        self.called = True
        return self.answer


def _build_pipeline(tmp_path: Path, provider: LLMProvider) -> GroundedAnswerPipeline:
    store = build_store()
    config = build_config(tmp_path)
    adapter = FakeEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())
    prompt_catalog = load_prompt_catalog("configs/prompts.yaml")
    prompt_builder = GroundedPromptBuilder(config.prompting, prompt_catalog.get("grounded_answer_baseline"))
    return GroundedAnswerPipeline(retriever, prompt_builder, provider)


def test_prompt_builder_includes_chunk_ids_and_metadata(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)
    adapter = FakeEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())
    prompt_catalog = load_prompt_catalog("configs/prompts.yaml")
    prompt_builder = GroundedPromptBuilder(config.prompting, prompt_catalog.get("grounded_answer_baseline"))

    outcome = retriever.retrieve(
        QueryRequest(
            question="What export control risks does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
        )
    )
    prompt = prompt_builder.build(question="What export control risks does NVIDIA describe?", retrieved_chunks=list(outcome.results))
    rendered = "\n".join(message.content for message in prompt.messages)

    assert "chunk_id: nvda_risk" in rendered
    assert "ticker: NVDA" in rendered
    assert "form_type: 10-K" in rendered
    assert "source_url:" in rendered
    assert prompt.context_chunk_ids == ["nvda_risk", "nvda_business"]


def test_answer_pipeline_enforces_citation_subset_and_serializes_response(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer_text="NVIDIA says export controls in China may require licenses and disrupt sales.",
            citation_chunk_ids=["nvda_risk"],
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What export control risks does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
            debug=True,
        )
    )

    assert response.status == "ok"
    assert {citation.citation_id for citation in response.citations} <= {
        chunk.chunk_id for chunk in response.retrieval_debug.results
    }
    assert response.model_dump(mode="json")["citations"][0]["citation_id"] == "nvda_risk"
    assert provider.called is True


def test_answer_pipeline_invalid_citations_return_typed_failure(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer_text="Unsupported answer text.",
            citation_chunk_ids=["unknown_chunk"],
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What export control risks does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
        )
    )

    assert response.status == "invalid_grounding"
    assert response.reason_code == "invalid_citations"
    assert response.citations == []


def test_answer_pipeline_no_results_path_is_typed_and_does_not_call_provider(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer_text="This should never be used.",
            citation_chunk_ids=["nvda_risk"],
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What export control risks does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["QCOM"]),
        )
    )

    assert response.status == "no_results"
    assert response.reason_code == "filters_excluded_all"
    assert response.citations == []
    assert provider.called is False


def test_schema_validation_rejects_invalid_date_ordering() -> None:
    with pytest.raises(ValueError):
        RetrievalFilters(filing_date_from="2026-02-26", filing_date_to="2026-02-25")
