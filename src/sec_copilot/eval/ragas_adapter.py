"""Optional Ragas scoring adapter for richer local answer evaluation."""

from __future__ import annotations

from typing import Any


class RagasUnavailableError(RuntimeError):
    """Raised when Ragas scoring was requested but cannot run."""


def score_with_ragas(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    embedding_model: str = "text-embedding-3-small",
    api_key: str | None = None,
) -> list[dict[str, float | None]]:
    """Score eligible answer rows with Ragas.

    The adapter is intentionally lazy-imported so CI and base installs do not
    require the optional Ragas dependency.
    """

    try:
        from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
        from openai import OpenAI
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RagasUnavailableError(f"Ragas is unavailable: {exc}") from exc

    if not rows:
        return []

    client = OpenAI(api_key=api_key)
    llm = llm_factory(
        model_name,
        provider="openai",
        client=client,
        max_tokens=4096,
    )
    embeddings = LangchainEmbeddingsWrapper(
        LangchainOpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )
    )
    dataset = EvaluationDataset.from_list(rows)
    try:
        scored = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=True,
        )
    except Exception as exc:  # pragma: no cover - optional runtime path
        raise RagasUnavailableError(f"Ragas scoring failed: {exc}") from exc

    result_rows = scored.to_pandas().to_dict(orient="records")
    normalized: list[dict[str, float | None]] = []
    for row in result_rows:
        normalized.append(
            {
                "ragas_faithfulness": row.get("faithfulness"),
                "ragas_response_relevancy": row.get("answer_relevancy"),
                "ragas_context_precision": row.get("context_precision"),
            }
        )
    return normalized


__all__ = ["RagasUnavailableError", "score_with_ragas"]
