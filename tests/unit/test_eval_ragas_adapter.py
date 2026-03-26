from __future__ import annotations

import sys
from types import ModuleType

import pytest

from sec_copilot.eval.ragas_adapter import RagasUnavailableError, _normalize_runtime_error, score_with_ragas
from sec_copilot.eval.schemas import EvalRagasConfig


def test_score_with_ragas_uses_explicit_evaluator_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, api_key: str | None = None) -> None:
            captured["api_key"] = api_key

    class FakeOpenAIEmbeddings:
        def __init__(self, model: str, api_key: str | None = None) -> None:
            captured["embedding_model"] = model
            captured["embedding_api_key"] = api_key

    class FakeEmbeddingsWrapper:
        def __init__(self, wrapped: object) -> None:
            captured["embeddings_wrapped"] = wrapped

    class FakeEvaluationDataset:
        @classmethod
        def from_list(cls, rows: list[dict[str, object]]) -> list[dict[str, object]]:
            captured["dataset_rows"] = rows
            return rows

    class FakeAnswerRelevancy:
        def __init__(self, strictness: int = 3) -> None:
            captured["answer_relevancy_strictness"] = strictness
            self.strictness = strictness

    fake_faithfulness = object()
    fake_context_precision = object()

    def fake_llm_factory(model: str, provider: str = "openai", client: object | None = None, **kwargs: object) -> object:
        captured["llm_factory_model"] = model
        captured["llm_factory_provider"] = provider
        captured["llm_factory_client"] = client
        captured["llm_factory_kwargs"] = kwargs
        return object()

    class FakeDataFrame:
        def to_dict(self, orient: str = "records") -> list[dict[str, float]]:
            assert orient == "records"
            return [{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7}]

    class FakeScored:
        def to_pandas(self) -> FakeDataFrame:
            return FakeDataFrame()

    def fake_evaluate(*, dataset: object, metrics: list[object], llm: object, embeddings: object, show_progress: bool, raise_exceptions: bool) -> FakeScored:
        captured["evaluate_dataset"] = dataset
        captured["evaluate_metrics"] = metrics
        captured["evaluate_show_progress"] = show_progress
        captured["evaluate_raise_exceptions"] = raise_exceptions
        return FakeScored()

    ragas_module = ModuleType("ragas")
    ragas_module.evaluate = fake_evaluate
    ragas_dataset_module = ModuleType("ragas.dataset_schema")
    ragas_dataset_module.EvaluationDataset = FakeEvaluationDataset
    ragas_embeddings_module = ModuleType("ragas.embeddings")
    ragas_embeddings_module.LangchainEmbeddingsWrapper = FakeEmbeddingsWrapper
    ragas_llms_module = ModuleType("ragas.llms")
    ragas_llms_module.llm_factory = fake_llm_factory
    ragas_metrics_module = ModuleType("ragas.metrics")
    ragas_metrics_module.AnswerRelevancy = FakeAnswerRelevancy
    ragas_metrics_module.faithfulness = fake_faithfulness
    ragas_metrics_module.context_precision = fake_context_precision
    langchain_openai_module = ModuleType("langchain_openai")
    langchain_openai_module.OpenAIEmbeddings = FakeOpenAIEmbeddings
    openai_module = ModuleType("openai")
    openai_module.OpenAI = FakeOpenAI

    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.dataset_schema", ragas_dataset_module)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", ragas_embeddings_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics_module)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_module)
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    result = score_with_ragas(
        [{"user_input": "q", "response": "a", "reference": "r", "retrieved_contexts": ["c"], "reference_contexts": ["c"]}],
        ragas_config=EvalRagasConfig(
            model_name="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
            max_completion_tokens=4096,
            answer_relevancy_strictness=1,
        ),
        api_key="test-key",
        score_backend="both",
    )

    assert captured["llm_factory_model"] == "gpt-4.1-mini"
    assert captured["embedding_model"] == "text-embedding-3-small"
    assert captured["answer_relevancy_strictness"] == 1
    assert captured["llm_factory_kwargs"] == {"max_tokens": 4096}
    metrics = captured["evaluate_metrics"]
    assert isinstance(metrics, list)
    assert len(metrics) == 3
    assert metrics[0] is fake_faithfulness
    assert metrics[2] is fake_context_precision
    assert getattr(metrics[1], "strictness", None) == 1
    assert result == [
        {
            "ragas_faithfulness": 0.9,
            "ragas_response_relevancy": 0.8,
            "ragas_context_precision": 0.7,
        }
    ]


def test_normalize_runtime_error_handles_token_limit() -> None:
    message = _normalize_runtime_error(
        Exception("The output is incomplete due to a max_tokens length limit."),
        EvalRagasConfig(model_name="gpt-4.1-mini", max_completion_tokens=4096),
        "both",
    )

    assert "evaluator model exhausted its completion budget" in message
    assert "model=gpt-4.1-mini" in message
    assert "score_backend=both" in message
