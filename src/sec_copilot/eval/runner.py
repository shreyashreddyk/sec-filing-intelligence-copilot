"""Offline eval runner over the tracked fixture corpus and gold dataset."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path

from sec_copilot.config.retrieval import PromptCatalog, RetrievalConfig
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
from sec_copilot.eval.offline_runtime import build_offline_eval_runtime
from sec_copilot.eval.providers import build_eval_provider
from sec_copilot.eval.ragas_adapter import RagasUnavailableError, score_with_ragas
from sec_copilot.eval.schemas import (
    AnswerExampleResult,
    AnswerExecutionTrace,
    AnswerPayload,
    EvalConfig,
    EvalDataset,
    EvalExample,
    EvalMode,
    EvalProviderName,
    EvalRagasConfig,
    EvalRunResult,
    EvalScoreBackend,
    EvalSectionResult,
    MetricAggregate,
    RetrievalExampleResult,
    ThresholdCheck,
)
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.schemas.retrieval import QueryRequest


def run_eval(
    *,
    eval_config: EvalConfig,
    retrieval_config: RetrievalConfig,
    prompt_catalog: PromptCatalog,
    dataset: EvalDataset,
    dataset_path: str | Path,
    corpus_path: str | Path,
    ragas_config: EvalRagasConfig,
    subset: str,
    mode: EvalMode,
    provider: EvalProviderName | None,
    score_backend: EvalScoreBackend | None,
    output_dir: str | Path,
) -> EvalRunResult:
    """Run the offline eval suite and return the machine-readable results."""

    started_at = datetime.now(UTC)
    selected_examples = _select_examples(dataset, subset)
    if not selected_examples:
        raise ValueError(f"Subset {subset!r} selected zero eval examples.")

    retrieval_results: list[RetrievalExampleResult] = []
    answer_results: list[AnswerExampleResult] = []
    warnings: list[str] = []
    errors: list[str] = []
    corpus_store = ProcessedChunkStore.load(corpus_path)

    try:
        runtime = build_offline_eval_runtime(
            config=retrieval_config,
            prompt_catalog=prompt_catalog,
            corpus_path=corpus_path,
            persist_directory=Path(output_dir) / "chroma",
            collection_name=f"sec_eval_{started_at.strftime('%Y%m%d%H%M%S')}",
        )
        store = runtime.store

        ragas_rows: list[dict[str, object]] = []
        ragas_target_indices: list[int] = []

        for example in selected_examples:
            request = QueryRequest(question=example.question, filters=example.filters, debug=True)
            retrieval_outcome = None

            if mode in {"retrieval", "full"}:
                retrieval_outcome = runtime.retriever.retrieve(request)
                retrieval_results.append(_build_retrieval_result(example, retrieval_outcome, eval_config.retrieval_ks))

            if mode in {"answer", "full"}:
                if retrieval_outcome is None:
                    retrieval_outcome = runtime.retriever.retrieve(request)

                current_provider = build_eval_provider(
                    provider or eval_config.default_provider,
                    example=example,
                    openai_model=retrieval_config.provider.openai_model,
                )
                pipeline = GroundedAnswerPipeline(
                    runtime.config,
                    runtime.retriever,
                    runtime.prompt_builder,
                    current_provider,
                )
                response, used_retrieval, prompt = pipeline.answer_with_trace(request, retrieval_outcome)
                trace = AnswerExecutionTrace(
                    provider_name=getattr(current_provider, "name", provider or eval_config.default_provider),
                    retrieval_reason_code=used_retrieval.reason_code,
                    stage_counts=used_retrieval.stage_counts,
                    reranker_applied=used_retrieval.reranker_applied,
                    reranker_skipped_reason=used_retrieval.reranker_skipped_reason,
                    prompt_name=prompt.prompt_name if prompt is not None else None,
                    prompt_version=prompt.prompt_version if prompt is not None else None,
                    retrieved_chunk_ids=[chunk.chunk_id for chunk in used_retrieval.retrieved_chunks],
                    final_context_chunk_ids=prompt.context_chunk_ids if prompt is not None else [],
                    truncated_chunk_ids=prompt.truncated_chunk_ids if prompt is not None else [],
                    used_context_tokens=prompt.used_context_tokens if prompt is not None else 0,
                    response=response,
                )
                answer_result = _build_answer_result(example, trace, store)
                answer_results.append(answer_result)

                if _ragas_eligible(example, answer_result, provider, score_backend):
                    ragas_rows.append(
                        {
                            "user_input": example.question,
                            "response": answer_result.response.answer,
                            "reference": example.reference_answer,
                            "retrieved_contexts": [
                                store.get(chunk_id).text
                                for chunk_id in answer_result.final_context_chunk_ids
                                if store.get(chunk_id) is not None
                            ],
                            "reference_contexts": [
                                store.get(chunk_id).text
                                for chunk_id in example.gold_chunk_ids
                                if store.get(chunk_id) is not None
                            ],
                        }
                    )
                    ragas_target_indices.append(len(answer_results) - 1)

        if score_backend in {"ragas", "both"}:
            try:
                ragas_scores = score_with_ragas(
                    ragas_rows,
                    ragas_config=ragas_config,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    score_backend=score_backend or "ragas",
                )
            except RagasUnavailableError as exc:
                if score_backend == "ragas":
                    raise
                warnings.append(str(exc))
                ragas_scores = []
            for target_index, ragas_score in zip(ragas_target_indices, ragas_scores, strict=False):
                merged = answer_results[target_index].ragas | ragas_score
                answer_results[target_index] = answer_results[target_index].model_copy(update={"ragas": merged})

    except Exception as exc:
        errors.append(str(exc))

    completed_at = datetime.now(UTC)
    status = "execution_failed" if errors else "passed"

    retrieval_section = _build_retrieval_section(retrieval_results, executed=mode in {"retrieval", "full"})
    answer_section = _build_answer_section(answer_results, executed=mode in {"answer", "full"})
    result = EvalRunResult(
        schema_version="sec_eval_results.v1",
        run_id=started_at.strftime("%Y%m%dT%H%M%SZ"),
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        subset=subset,
        mode=mode,
        provider=provider,
        score_backend=score_backend,
        paths={},
        dataset={
            "path": str(dataset_path),
            "schema_version": dataset.schema_version,
            "dataset_name": dataset.dataset_name,
            "total_examples": len(dataset.examples),
            "selected_example_count": len(selected_examples),
            "selected_example_ids": [example.example_id for example in selected_examples],
            "fingerprint": _file_fingerprint(dataset_path),
        },
        corpus={
            "path": str(corpus_path),
            "document_count": len({chunk.document_id for chunk in corpus_store.values()}),
            "chunk_count": len(corpus_store),
            "fingerprint": _corpus_fingerprint(corpus_path),
        },
        config={
            "dataset_path": eval_config.dataset_path,
            "corpus_path": eval_config.corpus_path,
            "subset": subset,
            "mode": mode,
            "provider": provider,
            "score_backend": score_backend,
            "ragas": ragas_config.model_dump(mode="json"),
            "retrieval_ks": eval_config.retrieval_ks,
            "thresholds": eval_config.thresholds.model_dump(mode="json"),
        },
        summary={
            "dataset_example_count": len(dataset.examples),
            "selected_example_count": len(selected_examples),
            "retrieval_example_count": len(retrieval_results),
            "answer_example_count": len(answer_results),
            "warning_count": len(warnings),
            "error_count": len(errors),
        },
        retrieval=retrieval_section,
        answer=answer_section,
        thresholds={},
        warnings=warnings,
        errors=errors,
    )
    threshold_payload = _evaluate_thresholds(result, eval_config)
    if result.status == "passed" and not threshold_payload["overall_passed"]:
        result = result.model_copy(update={"status": "threshold_failed"})
    return result.model_copy(update={"thresholds": threshold_payload})


def _select_examples(dataset: EvalDataset, subset: str) -> list[EvalExample]:
    if subset == "full":
        return list(dataset.examples)
    return [example for example in dataset.examples if subset in example.tags]


def _build_retrieval_result(
    example: EvalExample,
    retrieval_outcome,
    ks: list[int],
) -> RetrievalExampleResult:
    retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieval_outcome.retrieved_chunks]
    metrics = {
        **{f"recall@{k}": recall_at_k(example.gold_chunk_ids, retrieved_chunk_ids, k) for k in ks},
        **{f"hit_rate@{k}": hit_rate_at_k(example.gold_chunk_ids, retrieved_chunk_ids, k) for k in ks},
        "mrr": mean_reciprocal_rank(example.gold_chunk_ids, retrieved_chunk_ids),
    }
    return RetrievalExampleResult(
        example_id=example.example_id,
        category=example.category,
        question=example.question,
        expected_abstention=example.expected_abstention,
        filters=example.filters,
        gold_chunk_ids=list(example.gold_chunk_ids),
        reason_code=retrieval_outcome.reason_code,
        retrieved_chunk_ids=retrieved_chunk_ids,
        stage_counts=retrieval_outcome.stage_counts,
        reranker_applied=retrieval_outcome.reranker_applied,
        reranker_skipped_reason=retrieval_outcome.reranker_skipped_reason,
        metrics=metrics,
    )


def _build_answer_result(example: EvalExample, trace: AnswerExecutionTrace, store) -> AnswerExampleResult:
    citation_chunk_ids = [citation.citation_id for citation in trace.response.citations]
    chunk_text_by_id = {
        chunk_id: store.get(chunk_id).text
        for chunk_id in citation_chunk_ids
        if store.get(chunk_id) is not None
    }
    metrics = {
        "citation_validity": citation_validity(
            abstained=trace.response.abstained,
            citation_chunk_ids=citation_chunk_ids,
            final_context_chunk_ids=trace.final_context_chunk_ids,
            reason_code=trace.response.reason_code,
        ),
        "abstention_accuracy": abstention_accuracy(
            expected_abstention=example.expected_abstention,
            actual_abstention=trace.response.abstained,
        ),
        "context_precision_proxy": context_precision_proxy(
            gold_chunk_ids=example.gold_chunk_ids,
            final_context_chunk_ids=trace.final_context_chunk_ids,
            expected_abstention=example.expected_abstention,
        ),
        "response_relevancy_proxy": response_relevancy_proxy(
            answer=trace.response.answer,
            reference_key_points=example.reference_key_points,
            abstained=trace.response.abstained,
        ),
        "faithfulness_proxy": faithfulness_proxy(
            answer=trace.response.answer,
            citation_chunk_ids=citation_chunk_ids,
            chunk_text_by_id=chunk_text_by_id,
            abstained=trace.response.abstained,
        ),
    }
    return AnswerExampleResult(
        example_id=example.example_id,
        category=example.category,
        question=example.question,
        expected_abstention=example.expected_abstention,
        filters=example.filters,
        reference_answer=example.reference_answer,
        reference_key_points=list(example.reference_key_points),
        gold_chunk_ids=list(example.gold_chunk_ids),
        required_citation_chunk_ids=list(example.required_citation_chunk_ids),
        response=AnswerPayload(
            answer=trace.response.answer,
            abstained=trace.response.abstained,
            reason_code=trace.response.reason_code,
            citation_chunk_ids=citation_chunk_ids,
        ),
        retrieval_reason_code=trace.retrieval_reason_code,
        retrieved_chunk_ids=list(trace.retrieved_chunk_ids),
        final_context_chunk_ids=list(trace.final_context_chunk_ids),
        truncated_chunk_ids=list(trace.truncated_chunk_ids),
        stage_counts=trace.stage_counts,
        metrics=metrics,
        ragas={
            "ragas_faithfulness": None,
            "ragas_response_relevancy": None,
            "ragas_context_precision": None,
        },
    )


def _build_retrieval_section(results: list[RetrievalExampleResult], *, executed: bool) -> EvalSectionResult:
    return EvalSectionResult(
        executed=executed,
        metrics_overall=_aggregate_retrieval_metrics(results),
        metrics_by_category={
            category: _aggregate_retrieval_metrics([result for result in results if result.category == category])
            for category in sorted({result.category for result in results})
        },
        examples=[result.model_dump(mode="json") for result in results],
    )


def _build_answer_section(results: list[AnswerExampleResult], *, executed: bool) -> EvalSectionResult:
    return EvalSectionResult(
        executed=executed,
        metrics_overall=_aggregate_answer_metrics(results),
        metrics_by_category={
            category: _aggregate_answer_metrics([result for result in results if result.category == category])
            for category in sorted({result.category for result in results})
        },
        examples=[result.model_dump(mode="json") for result in results],
    )


def _aggregate_retrieval_metrics(results: list[RetrievalExampleResult]) -> MetricAggregate:
    metric_names = sorted({name for result in results for name in result.metrics})
    values: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for metric_name in metric_names:
        count, value = aggregate_metric([result.metrics.get(metric_name) for result in results])
        values[metric_name] = value
        counts[metric_name] = count
    primary_count = max(counts.values(), default=0)
    return MetricAggregate(
        eligible_example_count=primary_count,
        eligible_example_count_by_metric=counts,
        values=values,
    )


def _aggregate_answer_metrics(results: list[AnswerExampleResult]) -> MetricAggregate:
    metric_groups = {
        "citation_validity_rate": [result.metrics.get("citation_validity") for result in results],
        "abstention_accuracy": [result.metrics.get("abstention_accuracy") for result in results],
        "context_precision_proxy": [result.metrics.get("context_precision_proxy") for result in results],
        "response_relevancy_proxy": [result.metrics.get("response_relevancy_proxy") for result in results],
        "faithfulness_proxy": [result.metrics.get("faithfulness_proxy") for result in results],
        "ragas_faithfulness": [result.ragas.get("ragas_faithfulness") for result in results],
        "ragas_response_relevancy": [result.ragas.get("ragas_response_relevancy") for result in results],
        "ragas_context_precision": [result.ragas.get("ragas_context_precision") for result in results],
    }
    values: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for metric_name, metric_values in metric_groups.items():
        count, value = aggregate_metric(metric_values)
        values[metric_name] = value
        counts[metric_name] = count
    primary_count = max(counts.values(), default=0)
    return MetricAggregate(
        eligible_example_count=primary_count,
        eligible_example_count_by_metric=counts,
        values=values,
    )


def _evaluate_thresholds(result: EvalRunResult, eval_config: EvalConfig) -> dict[str, object]:
    payload = result.model_dump(mode="json")
    blocking_checks = [
        _run_threshold_check(payload, threshold.name, threshold.metric_path, threshold.operator, threshold.value, True)
        for threshold in eval_config.thresholds.blocking
    ]
    non_blocking_checks = [
        _run_threshold_check(payload, threshold.name, threshold.metric_path, threshold.operator, threshold.value, False)
        for threshold in eval_config.thresholds.non_blocking
    ]
    overall_passed = all(check.passed for check in blocking_checks)
    return {
        "blocking": [check.model_dump(mode="json") for check in blocking_checks],
        "non_blocking": [check.model_dump(mode="json") for check in non_blocking_checks],
        "overall_passed": overall_passed,
    }


def _run_threshold_check(
    payload: dict[str, object],
    name: str,
    metric_path: str,
    operator: str,
    expected_value: float,
    blocking: bool,
) -> ThresholdCheck:
    actual_value = _lookup_metric_path(payload, metric_path)
    if not isinstance(actual_value, (float, int)):
        passed = False
        normalized_actual = None
    else:
        normalized_actual = float(actual_value)
        if operator == "gte":
            passed = normalized_actual >= expected_value
        else:
            passed = abs(normalized_actual - expected_value) <= 1e-9
    return ThresholdCheck(
        name=name,
        metric_path=metric_path,
        operator=operator,
        expected_value=expected_value,
        actual_value=normalized_actual,
        blocking=blocking,
        passed=passed,
    )


def _lookup_metric_path(payload: dict[str, object], metric_path: str) -> object:
    current: object = payload
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _ragas_eligible(
    example: EvalExample,
    answer_result: AnswerExampleResult,
    provider: EvalProviderName | None,
    score_backend: EvalScoreBackend | None,
) -> bool:
    return bool(
        score_backend in {"ragas", "both"}
        and provider == "openai"
        and not example.expected_abstention
        and not answer_result.response.abstained
        and example.reference_answer
        and answer_result.final_context_chunk_ids
    )


def _file_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _corpus_fingerprint(corpus_path: str | Path) -> str:
    chunk_paths = sorted((Path(corpus_path) / "processed" / "chunks").glob("**/*.jsonl"))
    payload = hashlib.sha256()
    for chunk_path in chunk_paths:
        payload.update(chunk_path.read_bytes())
    return payload.hexdigest()


__all__ = ["run_eval"]
