from __future__ import annotations

from sec_copilot.eval.metrics import (
    abstention_accuracy,
    aggregate_metric,
    citation_validity,
    context_precision_proxy,
    faithfulness_proxy,
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    response_relevancy_proxy,
)


def test_retrieval_metrics_handle_single_and_multi_gold_examples() -> None:
    assert recall_at_k(["a"], ["a", "b"], 1) == 1.0
    assert hit_rate_at_k(["a"], ["b", "a"], 1) == 0.0
    assert mean_reciprocal_rank(["a"], ["b", "a"]) == 0.5

    assert recall_at_k(["a", "b"], ["b", "c", "a"], 2) == 0.5
    assert hit_rate_at_k(["a", "b"], ["c", "d"], 2) == 0.0
    assert mean_reciprocal_rank(["a", "b"], ["c", "b", "a"]) == 0.5


def test_retrieval_metrics_exclude_unanswerable_examples() -> None:
    assert recall_at_k([], ["a"], 1) is None
    assert hit_rate_at_k([], ["a"], 1) is None
    assert mean_reciprocal_rank([], ["a"]) is None


def test_answer_metrics_handle_missing_citations_and_abstentions() -> None:
    assert citation_validity(
        abstained=False,
        citation_chunk_ids=["a"],
        final_context_chunk_ids=["a", "b"],
        reason_code="ok",
    ) == 1.0
    assert citation_validity(
        abstained=False,
        citation_chunk_ids=[],
        final_context_chunk_ids=["a"],
        reason_code="ok",
    ) == 0.0
    assert citation_validity(
        abstained=True,
        citation_chunk_ids=[],
        final_context_chunk_ids=[],
        reason_code="model_abstained",
    ) == 1.0
    assert abstention_accuracy(expected_abstention=True, actual_abstention=False) == 0.0


def test_proxy_metrics_match_contract_shapes() -> None:
    assert context_precision_proxy(
        gold_chunk_ids=["a", "b"],
        final_context_chunk_ids=["a", "c"],
        expected_abstention=False,
    ) == 0.5
    assert context_precision_proxy(
        gold_chunk_ids=[],
        final_context_chunk_ids=[],
        expected_abstention=True,
    ) == 1.0

    relevancy = response_relevancy_proxy(
        answer="NVIDIA says Hopper and Blackwell drove AI demand.",
        reference_key_points=[
            "Hopper and Blackwell drove demand.",
            "The filing links growth to AI infrastructure demand.",
        ],
        abstained=False,
    )
    assert relevancy is not None
    assert relevancy > 0.4

    faithfulness = faithfulness_proxy(
        answer="AMD says EPYC and Instinct products drove data center demand.",
        citation_chunk_ids=["c1"],
        chunk_text_by_id={"c1": "AMD attributed growth to EPYC processors and Instinct GPU products for data center demand."},
        abstained=False,
    )
    assert faithfulness is not None
    assert faithfulness > 0.4


def test_aggregate_metric_reports_denominator() -> None:
    count, value = aggregate_metric([1.0, None, 0.0, 1.0])

    assert count == 3
    assert value == (2.0 / 3.0)
