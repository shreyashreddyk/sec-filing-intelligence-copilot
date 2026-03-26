"""Optional Ragas scoring adapter for richer local answer evaluation."""

from __future__ import annotations

from typing import Any

from sec_copilot.eval.schemas import EvalRagasConfig


class RagasUnavailableError(RuntimeError):
    """Raised when Ragas scoring was requested but cannot run."""


def score_with_ragas(
    rows: list[dict[str, Any]],
    *,
    ragas_config: EvalRagasConfig,
    api_key: str | None = None,
    score_backend: str = "ragas",
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
        from ragas.metrics import AnswerRelevancy, context_precision, faithfulness
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RagasUnavailableError(
            f"Ragas is unavailable (score_backend={score_backend}, model={ragas_config.model_name}): {exc}"
        ) from exc

    if not rows:
        return []

    # We already use the Ragas-native Instructor-based llm_factory path here.
    # The noisy "requested 3, got 1" warning comes from upstream Ragas prompt
    # handling, not from wrapping LangChain ChatOpenAI in this repository.
    client = OpenAI(api_key=api_key)
    llm_kwargs: dict[str, Any] = {
        "provider": "openai",
        "client": client,
        "max_tokens": ragas_config.max_completion_tokens,
    }
    if ragas_config.reasoning_effort is not None:
        llm_kwargs["reasoning_effort"] = ragas_config.reasoning_effort
    llm = llm_factory(
        ragas_config.model_name,
        provider="openai",
        client=client,
        **{key: value for key, value in llm_kwargs.items() if key not in {"provider", "client"}},
    )
    embeddings = LangchainEmbeddingsWrapper(
        LangchainOpenAIEmbeddings(
            model=ragas_config.embedding_model,
            api_key=api_key,
        )
    )
    dataset = EvaluationDataset.from_list(rows)
    answer_relevancy = AnswerRelevancy(strictness=ragas_config.answer_relevancy_strictness)
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
        message = _normalize_runtime_error(exc, ragas_config, score_backend)
        raise RagasUnavailableError(message) from exc

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


def _normalize_runtime_error(exc: Exception, ragas_config: EvalRagasConfig, score_backend: str) -> str:
    raw_message = str(exc)
    prefix = (
        "Ragas scoring failed "
        f"(score_backend={score_backend}, model={ragas_config.model_name}, "
        f"max_completion_tokens={ragas_config.max_completion_tokens})"
    )
    lowered = raw_message.lower()
    if "max_tokens length limit" in lowered or "finish_reason='length'" in raw_message:
        return (
            f"{prefix}: evaluator model exhausted its completion budget before returning valid structured output."
        )
    return f"{prefix}: {raw_message}"


__all__ = ["RagasUnavailableError", "score_with_ragas"]
