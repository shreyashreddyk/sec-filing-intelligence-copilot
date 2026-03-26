"""Deterministic retrieval and answer metrics for offline evaluation."""

from __future__ import annotations

from sec_copilot.retrieval.bm25 import tokenize_bm25_text


def recall_at_k(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str], k: int) -> float | None:
    """Return recall@k for answerable examples, else None."""

    if not gold_chunk_ids:
        return None
    gold = set(gold_chunk_ids)
    retrieved = set(retrieved_chunk_ids[:k])
    return len(gold & retrieved) / len(gold)


def hit_rate_at_k(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str], k: int) -> float | None:
    """Return hit-rate@k for answerable examples, else None."""

    if not gold_chunk_ids:
        return None
    gold = set(gold_chunk_ids)
    return 1.0 if gold & set(retrieved_chunk_ids[:k]) else 0.0


def mean_reciprocal_rank(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> float | None:
    """Return reciprocal rank of the first gold hit, else None for unanswerable examples."""

    if not gold_chunk_ids:
        return None
    gold = set(gold_chunk_ids)
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold:
            return 1.0 / index
    return 0.0


def citation_validity(
    *,
    abstained: bool,
    citation_chunk_ids: list[str],
    final_context_chunk_ids: list[str],
    reason_code: str,
) -> float:
    """Check whether citations obey the final-context contract."""

    if abstained:
        return 1.0 if not citation_chunk_ids and reason_code != "invalid_citations" else 0.0
    if not citation_chunk_ids:
        return 0.0
    allowed = set(final_context_chunk_ids)
    return 1.0 if all(chunk_id in allowed for chunk_id in citation_chunk_ids) else 0.0


def abstention_accuracy(*, expected_abstention: bool, actual_abstention: bool) -> float:
    """Return 1.0 when abstention behavior matches the gold label."""

    return 1.0 if expected_abstention == actual_abstention else 0.0


def context_precision_proxy(
    *,
    gold_chunk_ids: list[str],
    final_context_chunk_ids: list[str],
    expected_abstention: bool,
) -> float:
    """Estimate how much of the final prompt context is relevant."""

    if expected_abstention:
        return 1.0 if not final_context_chunk_ids else 0.0
    if not final_context_chunk_ids:
        return 0.0
    gold = set(gold_chunk_ids)
    return len(gold & set(final_context_chunk_ids)) / len(final_context_chunk_ids)


def response_relevancy_proxy(
    *,
    answer: str,
    reference_key_points: list[str],
    abstained: bool,
) -> float | None:
    """Estimate whether the answer covers the labeled key points."""

    if abstained:
        return None
    if not reference_key_points:
        return None
    answer_tokens = _token_set(answer)
    per_key_point = []
    for key_point in reference_key_points:
        key_point_tokens = _token_set(key_point)
        per_key_point.append(len(answer_tokens & key_point_tokens) / max(1, len(key_point_tokens)))
    return _mean(per_key_point)


def faithfulness_proxy(
    *,
    answer: str,
    citation_chunk_ids: list[str],
    chunk_text_by_id: dict[str, str],
    abstained: bool,
) -> float | None:
    """Estimate answer grounding against cited support only."""

    if abstained:
        return None
    if not citation_chunk_ids:
        return 0.0
    support_text = " ".join(chunk_text_by_id.get(chunk_id, "") for chunk_id in citation_chunk_ids).strip()
    answer_tokens = _token_set(answer)
    support_tokens = _token_set(support_text)
    return len(answer_tokens & support_tokens) / max(1, len(answer_tokens))


def aggregate_metric(values: list[float | None]) -> tuple[int, float | None]:
    """Return eligible-example count plus the arithmetic mean."""

    eligible = [value for value in values if value is not None]
    if not eligible:
        return 0, None
    return len(eligible), _mean(eligible)


def _token_set(text: str) -> set[str]:
    return set(tokenize_bm25_text(text))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


__all__ = [
    "abstention_accuracy",
    "aggregate_metric",
    "citation_validity",
    "context_precision_proxy",
    "faithfulness_proxy",
    "hit_rate_at_k",
    "mean_reciprocal_rank",
    "recall_at_k",
    "response_relevancy_proxy",
]
