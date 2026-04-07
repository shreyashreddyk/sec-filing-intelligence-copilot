"""Typed public request and response models for the V5 FastAPI layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sec_copilot.api.coverage import (
    CoverageState,
    GlobalCoverageStatus,
    IndexStatus,
    IndexedScope,
    MissingScope,
    RequestCoverageStatus,
    TargetScope,
)
from sec_copilot.eval.schemas import EvalRunResult, EvalScoreBackend
from sec_copilot.retrieval.indexer import IndexBuildMetadata, IndexBuildResult
from sec_copilot.schemas.retrieval import Citation, QueryRequest, QueryResponse, RetrievedChunk, RetrievalStageCounts
from sec_copilot.utils.normalization import normalize_form_type, normalize_ticker


ProviderMode = Literal["openai", "mock"]
EvalProviderMode = Literal["reference", "mock", "openai"]


class RetrievalTimings(BaseModel):
    """Latency breakdown for retrieval-only execution."""

    model_config = ConfigDict(extra="forbid")

    dense_ms: float = Field(ge=0.0)
    bm25_ms: float = Field(ge=0.0)
    fusion_ms: float = Field(ge=0.0)
    rerank_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class QueryTimings(BaseModel):
    """Latency breakdown for grounded answer execution."""

    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalTimings
    prompt_build_ms: float = Field(ge=0.0)
    generation_ms: float = Field(ge=0.0)
    citation_validation_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    """Fast health and readiness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service: str
    version: str
    retrieve_ready: bool
    query_ready: bool
    index_status: IndexStatus
    last_index_refresh_at: datetime | None = None
    last_ingest_completed_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)


class ReadinessResponse(BaseModel):
    """Compact readiness response for Kubernetes probes."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"]
    service: str
    version: str
    retrieve_ready: bool
    query_ready: bool
    index_status: IndexStatus
    last_index_refresh_at: datetime | None = None
    last_ingest_completed_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)


class BuildInfoResponse(BaseModel):
    """Detailed build, runtime, and corpus state response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service: str
    version: str
    retrieve_ready: bool
    query_ready: bool
    configured_provider: ProviderMode
    effective_provider: ProviderMode
    provider_fallback_enabled: bool
    provider_fallback_active: bool
    provider_fallback_reason: str | None = None
    prompt_name: str
    prompt_version: str
    collection_name: str
    persist_directory: str
    coverage_status: GlobalCoverageStatus
    target_scope: TargetScope
    indexed_scope: IndexedScope
    index_status: IndexStatus
    processed_corpus_fingerprint: str | None = None
    indexed_corpus_fingerprint: str | None = None
    index_build_metadata: IndexBuildMetadata | None = None
    last_ingest_completed_at: datetime | None = None
    last_index_refresh_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)


class ServiceNotReadyResponse(BaseModel):
    """Typed failure response for not-ready query and retrieval endpoints."""

    model_config = ConfigDict(extra="forbid")

    error_type: Literal["service_not_ready"] = "service_not_ready"
    message: str
    retrieve_ready: bool
    query_ready: bool
    index_status: IndexStatus
    coverage_status: GlobalCoverageStatus
    indexed_scope: IndexedScope
    last_ingest_completed_at: datetime | None = None
    last_index_refresh_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)


class CoverageFailureResponse(BaseModel):
    """Typed failure response for strict request-scope coverage misses."""

    model_config = ConfigDict(extra="forbid")

    error_type: Literal["coverage_error"] = "coverage_error"
    message: str
    coverage_status: RequestCoverageStatus
    indexed_scope: IndexedScope
    missing_scope: MissingScope
    last_index_refresh_at: datetime | None = None


class QuerySuccessResponse(BaseModel):
    """Successful grounded-answer response."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    abstained: bool
    reason_code: str
    coverage_status: RequestCoverageStatus
    indexed_scope: IndexedScope
    missing_scope: MissingScope
    last_index_refresh_at: datetime | None = None
    timings: QueryTimings

    @classmethod
    def from_query_response(
        cls,
        response: QueryResponse,
        *,
        coverage_status: RequestCoverageStatus,
        indexed_scope: IndexedScope,
        missing_scope: MissingScope,
        last_index_refresh_at: datetime | None,
        timings: QueryTimings,
    ) -> "QuerySuccessResponse":
        return cls(
            answer=response.answer,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            abstained=response.abstained,
            reason_code=response.reason_code,
            coverage_status=coverage_status,
            indexed_scope=indexed_scope,
            missing_scope=missing_scope,
            last_index_refresh_at=last_index_refresh_at,
            timings=timings,
        )


class RetrievalDebugResponse(BaseModel):
    """Successful retrieval-debug response."""

    model_config = ConfigDict(extra="forbid")

    reason_code: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    stage_counts: RetrievalStageCounts
    reranker_applied: bool
    reranker_skipped_reason: str | None = None
    coverage_status: RequestCoverageStatus
    indexed_scope: IndexedScope
    missing_scope: MissingScope
    last_index_refresh_at: datetime | None = None
    timings: RetrievalTimings


class CompanyResultModel(BaseModel):
    """Pydantic mirror of one ingestion company result."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    status: str
    requested_filings: int
    successful_filings: int
    failed_filings: int
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None


class FilingResultModel(BaseModel):
    """Pydantic mirror of one ingestion filing result."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    form_type: str
    accession_number: str
    document_id: str | None
    status: str
    chunk_path: str | None = None
    manifest_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None


class RunIssueModel(BaseModel):
    """Pydantic mirror of one ingestion warning or error."""

    model_config = ConfigDict(extra="forbid")

    level: str
    code: str
    message: str
    ticker: str | None = None
    document_id: str | None = None
    form_type: str | None = None
    accession_number: str | None = None


class RunSummaryModel(BaseModel):
    """Pydantic mirror of the top-level ingestion run summary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str
    companies_config_path: str
    requested_companies: int
    attempted_companies: int
    successful_companies: int
    failed_companies: int
    requested_filings: int
    successful_filings: int
    failed_filings: int
    warning_count: int
    error_count: int
    annual_limit: int
    quarterly_limit: int
    company_results: list[CompanyResultModel] = Field(default_factory=list)
    filing_results: list[FilingResultModel] = Field(default_factory=list)
    warnings: list[RunIssueModel] = Field(default_factory=list)
    errors: list[RunIssueModel] = Field(default_factory=list)


class IngestRunRequest(BaseModel):
    """Admin request for ingestion plus index refresh."""

    model_config = ConfigDict(extra="forbid")

    companies: list[str] = Field(default_factory=list)
    form_types: list[str] = Field(default_factory=lambda: ["10-K", "10-Q"])
    annual_limit: int = Field(default=2, ge=0)
    quarterly_limit: int = Field(default=4, ge=0)
    force_refresh: bool = False
    user_agent: str | None = None
    index_mode: Literal["rebuild", "upsert"] = "rebuild"

    @field_validator("companies", mode="before")
    @classmethod
    def _normalize_companies(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [normalize_ticker(str(item)) for item in value]

    @field_validator("form_types", mode="before")
    @classmethod
    def _normalize_form_types(cls, value: object) -> list[str]:
        if value is None:
            return ["10-K", "10-Q"]
        if isinstance(value, str):
            value = [value]
        return [normalize_form_type(str(item)) for item in value]


class IngestRunTimings(BaseModel):
    """Latency summary for ingestion and index refresh."""

    model_config = ConfigDict(extra="forbid")

    ingest_ms: float = Field(ge=0.0)
    index_ms: float = Field(ge=0.0)
    refresh_state_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class IngestRunResponse(BaseModel):
    """Typed admin response for ingestion plus index refresh."""

    model_config = ConfigDict(extra="forbid")

    run_summary: RunSummaryModel
    index_build: IndexBuildResult | None = None
    coverage_state: CoverageState
    timings: IngestRunTimings


class EvalRunRequest(BaseModel):
    """Admin request for one offline eval run."""

    model_config = ConfigDict(extra="forbid")

    subset: str | None = None
    mode: Literal["retrieval", "answer", "full"] | None = None
    provider: EvalProviderMode | None = None
    score_backend: EvalScoreBackend | None = None
    output_dir: str | None = None
    fail_on_thresholds: bool = True


class EvalRunTimings(BaseModel):
    """Latency summary for eval execution and artifact writing."""

    model_config = ConfigDict(extra="forbid")

    total_ms: float = Field(ge=0.0)


class EvalRunResponse(BaseModel):
    """Typed admin response for one eval run."""

    model_config = ConfigDict(extra="forbid")

    result: EvalRunResult
    timings: EvalRunTimings


__all__ = [
    "BuildInfoResponse",
    "CompanyResultModel",
    "CoverageFailureResponse",
    "EvalRunRequest",
    "EvalRunResponse",
    "EvalRunTimings",
    "HealthResponse",
    "IngestRunRequest",
    "IngestRunResponse",
    "IngestRunTimings",
    "ProviderMode",
    "QueryRequest",
    "QuerySuccessResponse",
    "QueryTimings",
    "RetrievalDebugResponse",
    "RetrievalTimings",
    "RunIssueModel",
    "RunSummaryModel",
    "ServiceNotReadyResponse",
]
