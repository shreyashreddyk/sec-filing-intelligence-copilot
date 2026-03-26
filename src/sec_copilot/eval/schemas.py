"""Typed internal contracts for offline evaluation datasets, config, and results."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sec_copilot.schemas.retrieval import QueryResponse, ReasonCode, RetrievalFilters, RetrievalStageCounts


EvalCategory = Literal[
    "fact_lookup",
    "cross_period_comparison",
    "multi_document_synthesis",
    "unanswerable",
]
EvalMode = Literal["retrieval", "answer", "full"]
EvalProviderName = Literal["reference", "mock", "openai"]
EvalScoreBackend = Literal["deterministic", "ragas", "both"]
EvalStatus = Literal["passed", "threshold_failed", "execution_failed", "config_error"]
ThresholdOperator = Literal["gte", "eq"]


class EvalExample(BaseModel):
    """One gold evaluation example over the tracked fixture corpus."""

    model_config = ConfigDict(extra="forbid")

    example_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: EvalCategory
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    expected_abstention: bool
    gold_chunk_ids: list[str] = Field(default_factory=list)
    required_citation_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    reference_key_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("example_id")
    @classmethod
    def _normalize_example_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("example_id must be non-empty")
        if any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in normalized):
            raise ValueError("example_id must contain only lowercase letters, digits, and underscores")
        return normalized

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must be non-empty")
        return normalized

    @field_validator("gold_chunk_ids", "required_citation_chunk_ids", mode="after")
    @classmethod
    def _dedupe_chunk_ids(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            chunk_id = item.strip()
            if not chunk_id:
                raise ValueError("chunk ID values must be non-empty")
            if chunk_id not in normalized:
                normalized.append(chunk_id)
        return normalized

    @field_validator("reference_answer")
    @classmethod
    def _normalize_reference_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("reference_key_points", mode="after")
    @classmethod
    def _normalize_reference_key_points(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            key_point = item.strip()
            if not key_point:
                raise ValueError("reference_key_points must not contain empty values")
            if key_point not in normalized:
                normalized.append(key_point)
        return normalized

    @field_validator("tags", mode="after")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            tag = item.strip().lower()
            if not tag:
                raise ValueError("tags must not contain empty values")
            if any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in tag):
                raise ValueError("tags must contain only lowercase letters, digits, and underscores")
            if tag not in normalized:
                normalized.append(tag)
        return normalized

    @model_validator(mode="after")
    def _validate_answerability_contract(self) -> "EvalExample":
        gold_set = set(self.gold_chunk_ids)
        citation_set = set(self.required_citation_chunk_ids)
        if not citation_set.issubset(gold_set):
            raise ValueError("required_citation_chunk_ids must be a subset of gold_chunk_ids")

        if self.expected_abstention:
            if self.gold_chunk_ids:
                raise ValueError("unanswerable examples must have empty gold_chunk_ids")
            if self.required_citation_chunk_ids:
                raise ValueError("unanswerable examples must have empty required_citation_chunk_ids")
            if self.reference_answer is not None:
                raise ValueError("unanswerable examples must omit reference_answer")
            if self.reference_key_points:
                raise ValueError("unanswerable examples must have empty reference_key_points")
        else:
            if not self.gold_chunk_ids:
                raise ValueError("answerable examples must define gold_chunk_ids")
            if not self.required_citation_chunk_ids:
                raise ValueError("answerable examples must define required_citation_chunk_ids")
            if self.reference_answer is None:
                raise ValueError("answerable examples must define reference_answer")
            if not self.reference_key_points:
                raise ValueError("answerable examples must define reference_key_points")
        return self


class EvalDataset(BaseModel):
    """Tracked gold dataset contract for SEC filing QA."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sec_eval_dataset.v1"]
    dataset_name: str = Field(min_length=1)
    description: str | None = None
    examples: list[EvalExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_examples(self) -> "EvalDataset":
        example_ids = [example.example_id for example in self.examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("example_id values must be unique")
        return self


class EvalThreshold(BaseModel):
    """One configurable threshold check over an aggregate metric."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    metric_path: str = Field(min_length=1)
    operator: ThresholdOperator
    value: float


class EvalThresholdGroups(BaseModel):
    """Blocking and non-blocking threshold definitions."""

    model_config = ConfigDict(extra="forbid")

    blocking: list[EvalThreshold] = Field(default_factory=list)
    non_blocking: list[EvalThreshold] = Field(default_factory=list)


class EvalRagasConfig(BaseModel):
    """Optional richer local Ragas scoring configuration."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="gpt-4.1-mini", min_length=1)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    max_completion_tokens: int = Field(default=4096, ge=256)
    answer_relevancy_strictness: int = Field(default=1, ge=1)
    reasoning_effort: str | None = Field(default=None, min_length=1)

    @field_validator("model_name", "embedding_model", "reasoning_effort")
    @classmethod
    def _normalize_string_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return normalized


class EvalConfig(BaseModel):
    """Offline evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sec_eval_config.v1"]
    dataset_path: str = Field(min_length=1)
    corpus_path: str = Field(min_length=1)
    default_subset: str = Field(min_length=1)
    default_mode: EvalMode
    default_provider: EvalProviderName
    default_score_backend: EvalScoreBackend
    retrieval_ks: list[int] = Field(default_factory=lambda: [1, 2, 4])
    output_root: str = Field(min_length=1)
    ragas: EvalRagasConfig = Field(default_factory=EvalRagasConfig)
    thresholds: EvalThresholdGroups

    @field_validator("retrieval_ks", mode="after")
    @classmethod
    def _validate_retrieval_ks(cls, value: list[int]) -> list[int]:
        normalized = sorted(dict.fromkeys(int(item) for item in value))
        if not normalized:
            raise ValueError("retrieval_ks must contain at least one k value")
        if any(item <= 0 for item in normalized):
            raise ValueError("retrieval_ks must contain only positive integers")
        return normalized


class MetricAggregate(BaseModel):
    """Aggregate metrics plus the denominator used."""

    model_config = ConfigDict(extra="forbid")

    eligible_example_count: int = Field(ge=0)
    eligible_example_count_by_metric: dict[str, int] = Field(default_factory=dict)
    values: dict[str, float | None] = Field(default_factory=dict)


class ThresholdCheck(BaseModel):
    """One executed threshold check."""

    model_config = ConfigDict(extra="forbid")

    name: str
    metric_path: str
    operator: ThresholdOperator
    expected_value: float
    actual_value: float | None = None
    blocking: bool
    passed: bool


class AnswerExecutionTrace(BaseModel):
    """Internal trace assembled around one grounded-answer execution."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str = Field(min_length=1)
    retrieval_reason_code: ReasonCode
    stage_counts: RetrievalStageCounts
    reranker_applied: bool
    reranker_skipped_reason: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    final_context_chunk_ids: list[str] = Field(default_factory=list)
    truncated_chunk_ids: list[str] = Field(default_factory=list)
    used_context_tokens: int = Field(default=0, ge=0)
    response: QueryResponse


class RetrievalExampleResult(BaseModel):
    """Per-example retrieval evaluation result."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    category: EvalCategory
    question: str
    expected_abstention: bool
    filters: RetrievalFilters
    gold_chunk_ids: list[str] = Field(default_factory=list)
    reason_code: ReasonCode
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    stage_counts: RetrievalStageCounts
    reranker_applied: bool
    reranker_skipped_reason: str | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AnswerPayload(BaseModel):
    """Compact answer payload persisted in eval artifacts."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    abstained: bool
    reason_code: ReasonCode
    citation_chunk_ids: list[str] = Field(default_factory=list)


class AnswerExampleResult(BaseModel):
    """Per-example answer evaluation result."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    category: EvalCategory
    question: str
    expected_abstention: bool
    filters: RetrievalFilters
    reference_answer: str | None = None
    reference_key_points: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    required_citation_chunk_ids: list[str] = Field(default_factory=list)
    response: AnswerPayload
    retrieval_reason_code: ReasonCode
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    final_context_chunk_ids: list[str] = Field(default_factory=list)
    truncated_chunk_ids: list[str] = Field(default_factory=list)
    stage_counts: RetrievalStageCounts
    metrics: dict[str, float | None] = Field(default_factory=dict)
    ragas: dict[str, float | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class EvalSectionResult(BaseModel):
    """Top-level retrieval or answer section in eval results."""

    model_config = ConfigDict(extra="forbid")

    executed: bool
    metrics_overall: MetricAggregate = Field(default_factory=lambda: MetricAggregate(eligible_example_count=0))
    metrics_by_category: dict[str, MetricAggregate] = Field(default_factory=dict)
    examples: list[dict] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    """Machine-readable top-level eval artifact contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sec_eval_results.v1"]
    run_id: str
    started_at: datetime
    completed_at: datetime
    status: EvalStatus
    subset: str
    mode: EvalMode
    provider: EvalProviderName | None = None
    score_backend: EvalScoreBackend | None = None
    paths: dict[str, str] = Field(default_factory=dict)
    dataset: dict[str, object] = Field(default_factory=dict)
    corpus: dict[str, object] = Field(default_factory=dict)
    config: dict[str, object] = Field(default_factory=dict)
    summary: dict[str, object] = Field(default_factory=dict)
    retrieval: EvalSectionResult
    answer: EvalSectionResult
    thresholds: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "AnswerExampleResult",
    "AnswerExecutionTrace",
    "AnswerPayload",
    "EvalCategory",
    "EvalConfig",
    "EvalDataset",
    "EvalExample",
    "EvalMode",
    "EvalProviderName",
    "EvalRagasConfig",
    "EvalRunResult",
    "EvalScoreBackend",
    "EvalSectionResult",
    "EvalThreshold",
    "EvalThresholdGroups",
    "MetricAggregate",
    "RetrievalExampleResult",
    "ThresholdCheck",
]
