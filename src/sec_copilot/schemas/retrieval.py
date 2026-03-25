"""Pydantic schema boundaries for V3 retrieval, reranking, and grounded answers."""

from __future__ import annotations

from datetime import date
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sec_copilot.utils.normalization import normalize_form_type, normalize_ticker


ReasonCode: TypeAlias = Literal[
    "ok",
    "no_hits",
    "filters_excluded_all_chunks",
    "weak_support",
    "insufficient_supporting_chunks",
    "model_abstained",
    "invalid_citations",
    "reranker_unavailable",
]


class RetrievalFilters(BaseModel):
    """Normalized metadata filters applied before dense and BM25 retrieval."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    form_types: list[str] = Field(default_factory=list)
    filing_date_from: date | None = None
    filing_date_to: date | None = None

    @field_validator("tickers", mode="before")
    @classmethod
    def _coerce_tickers(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [normalize_ticker(str(item)) for item in value]

    @field_validator("form_types", mode="before")
    @classmethod
    def _coerce_form_types(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [normalize_form_type(str(item)) for item in value]

    @model_validator(mode="after")
    def _validate_date_range(self) -> "RetrievalFilters":
        if self.filing_date_from and self.filing_date_to and self.filing_date_from > self.filing_date_to:
            raise ValueError("filing_date_from must be on or before filing_date_to")
        self.tickers = list(dict.fromkeys(self.tickers))
        self.form_types = list(dict.fromkeys(self.form_types))
        return self


class QueryRequest(BaseModel):
    """Typed query contract for V3 retrieval and grounded answer generation."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must be non-empty")
        return normalized


class RetrievedChunk(BaseModel):
    """Parent-chunk retrieval artifact shared across debug and query responses."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    form_type: str = Field(min_length=1)
    filing_date: date
    accession_number: str = Field(min_length=1)
    section_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    text: str = Field(min_length=1)
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    dense_raw_distance: float | None = None
    best_subchunk_id: str | None = None
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, value: str) -> str:
        return normalize_form_type(value)


class Citation(BaseModel):
    """Citation metadata returned with a grounded answer."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=1, description="Stable citation ID. Always equals the parent chunk_id.")
    ticker: str = Field(min_length=1)
    form_type: str = Field(min_length=1)
    filing_date: date
    accession_number: str = Field(min_length=1)
    section_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    snippet: str = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, value: str) -> str:
        return normalize_form_type(value)


class RetrievalStageCounts(BaseModel):
    """Stage-by-stage candidate counts for retrieval debugging."""

    model_config = ConfigDict(extra="forbid")

    filtered_parent_count: int = Field(ge=0)
    dense_subchunk_hit_count: int = Field(ge=0)
    dense_parent_candidate_count: int = Field(ge=0)
    bm25_candidate_count: int = Field(ge=0)
    fused_candidate_count: int = Field(ge=0)
    reranked_candidate_count: int = Field(ge=0)


class RetrievalResponse(BaseModel):
    """Public retrieval-debug response returned by the retrieval CLI or API."""

    model_config = ConfigDict(extra="forbid")

    reason_code: ReasonCode
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    stage_counts: RetrievalStageCounts
    reranker_applied: bool
    reranker_skipped_reason: str | None = None


class ProviderAnswer(BaseModel):
    """Strict structured-output contract returned by provider implementations."""

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    citation_chunk_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    notes: str | None = None

    @field_validator("answer")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        return value.strip()

    @field_validator("citation_chunk_ids", mode="after")
    @classmethod
    def _normalize_citation_chunk_ids(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            clean = item.strip()
            if not clean:
                raise ValueError("citation_chunk_ids must not contain empty values")
            if clean not in normalized:
                normalized.append(clean)
        return normalized

    @model_validator(mode="after")
    def _validate_shape(self) -> "ProviderAnswer":
        if not self.abstained and not self.answer:
            raise ValueError("answer must be non-empty when abstained is false")
        return self


class QueryResponse(BaseModel):
    """Typed V3 grounded-answer payload returned by the answer pipeline."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    reason_code: ReasonCode


__all__ = [
    "Citation",
    "ProviderAnswer",
    "QueryRequest",
    "QueryResponse",
    "ReasonCode",
    "RetrievedChunk",
    "RetrievalFilters",
    "RetrievalResponse",
    "RetrievalStageCounts",
]
