from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot.config import load_prompt_catalog
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.generation.prompts import GroundedPromptBuilder, PromptManager
from sec_copilot.generation.providers import LLMProvider
from sec_copilot.retrieval.bm25 import BM25Retriever
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever, HybridRetriever
from sec_copilot.schemas import ProviderAnswer, QueryRequest, RetrievalFilters
from tests.unit.retrieval_helpers import CuratedEmbeddingAdapter, build_config, build_reranker, build_store


class StaticProvider(LLMProvider):
    name = "static"

    def __init__(self, answer: ProviderAnswer) -> None:
        self.answer = answer
        self.called = False

    def generate(self, prompt) -> ProviderAnswer:
        self.called = True
        return self.answer


def _build_pipeline(tmp_path: Path, provider: LLMProvider, *, reranker=None) -> GroundedAnswerPipeline:
    store = build_store()
    config = build_config(tmp_path)
    adapter = CuratedEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    dense_retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())
    hybrid_retriever = HybridRetriever(
        config,
        store,
        dense_retriever,
        BM25Retriever(store),
        reranker or build_reranker(),
    )
    prompt_catalog = load_prompt_catalog("configs/prompts.yaml")
    prompt_builder = GroundedPromptBuilder(
        config.retrieval,
        config.prompting,
        PromptManager(prompt_catalog).get_prompt(
            config.prompting.prompt_name,
            expected_version=config.prompting.prompt_version,
        ),
    )
    return GroundedAnswerPipeline(config, hybrid_retriever, prompt_builder, provider)


def test_prompt_builder_includes_chunk_ids_and_metadata(tmp_path: Path) -> None:
    store = build_store()
    config = build_config(tmp_path)
    adapter = CuratedEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    dense_retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())
    hybrid_retriever = HybridRetriever(config, store, dense_retriever, BM25Retriever(store), build_reranker())
    prompt_catalog = load_prompt_catalog("configs/prompts.yaml")
    prompt_builder = GroundedPromptBuilder(
        config.retrieval,
        config.prompting,
        PromptManager(prompt_catalog).get_prompt(
            config.prompting.prompt_name,
            expected_version=config.prompting.prompt_version,
        ),
    )

    retrieval = hybrid_retriever.retrieve(
        QueryRequest(
            question="What H20 export license restrictions does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
        )
    )
    prompt = prompt_builder.build(
        question="What H20 export license restrictions does NVIDIA describe?",
        retrieved_chunks=list(retrieval.retrieved_chunks),
    )
    rendered = "\n".join(message.content for message in prompt.messages)

    assert "chunk_id: nvda_h20_license" in rendered
    assert "ticker: NVDA" in rendered
    assert "form_type: 10-K" in rendered
    assert "source_url:" in rendered
    assert "nvda_h20_license" in prompt.context_chunk_ids


def test_answer_pipeline_enforces_final_context_citation_subset_and_serializes_response(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer="NVIDIA says H20 exports to China require licenses and reduced demand.",
            citation_chunk_ids=["nvda_h20_license", "nvda_export_controls_general"],
            abstained=False,
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What H20 export license restrictions does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
            debug=True,
        )
    )

    assert response.reason_code == "ok"
    assert response.abstained is False
    assert {citation.citation_id for citation in response.citations} <= {
        chunk.chunk_id for chunk in response.retrieved_chunks
    }
    assert response.model_dump(mode="json")["citations"][0]["citation_id"] == "nvda_h20_license"
    assert provider.called is True


def test_answer_pipeline_invalid_citations_return_abstention(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer="Unsupported answer text.",
            citation_chunk_ids=["unknown_chunk"],
            abstained=False,
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What H20 export license restrictions does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
        )
    )

    assert response.abstained is True
    assert response.reason_code == "invalid_citations"
    assert response.citations == []


def test_answer_pipeline_model_abstention_is_preserved(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer="The evidence is not sufficient for a grounded answer.",
            citation_chunk_ids=[],
            abstained=True,
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What H20 export license restrictions does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
        )
    )

    assert response.abstained is True
    assert response.reason_code == "model_abstained"


def test_answer_pipeline_filters_excluded_path_is_typed_and_does_not_call_provider(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer="This should never be used.",
            citation_chunk_ids=["nvda_h20_license"],
            abstained=False,
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What H20 export license restrictions does NVIDIA describe?",
            filters=RetrievalFilters(tickers=["QCOM"]),
        )
    )

    assert response.abstained is True
    assert response.reason_code == "filters_excluded_all_chunks"
    assert provider.called is False


def test_answer_pipeline_abstains_when_too_few_strong_supporting_chunks(tmp_path: Path) -> None:
    provider = StaticProvider(
        ProviderAnswer(
            answer="This should never be used.",
            citation_chunk_ids=["nvda_ai_platform"],
            abstained=False,
        )
    )
    pipeline = _build_pipeline(tmp_path, provider)

    response = pipeline.answer(
        QueryRequest(
            question="What does NVIDIA say about AI infrastructure and accelerated computing?",
            filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
        )
    )

    assert response.abstained is True
    assert response.reason_code == "insufficient_supporting_chunks"
    assert provider.called is False


def test_schema_validation_rejects_invalid_date_ordering() -> None:
    with pytest.raises(ValueError):
        RetrievalFilters(filing_date_from="2026-02-26", filing_date_to="2026-02-25")
