"""Pydantic schema boundaries for V2 retrieval and grounded answers."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sec_copilot.utils.normalization import normalize_form_type, normalize_ticker


class RetrievalFilters(BaseModel):
    """Normalized metadata filters applied before dense retrieval."""

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
    """Typed query contract for retrieval and grounded answer generation."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    prompt_top_n: int | None = Field(default=None, ge=1, le=20)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must be non-empty")
        return normalized


class RetrievedChunk(BaseModel):
    """Collapsed parent-chunk result returned to callers."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
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
    score: float
    raw_distance: float
    best_subchunk_id: str = Field(min_length=1)

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

    citation_id: str = Field(min_length=1, description="Stable citation ID. Equals the parent chunk_id in V2.")
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


class DebugRetrieval(BaseModel):
    """Structured retrieval-debug payload for inspection and CLI output."""

    model_config = ConfigDict(extra="forbid")

    distance_space: Literal["cosine"] = "cosine"
    score_semantics: Literal["score = 1.0 - raw_distance"] = "score = 1.0 - raw_distance"
    pre_llm_retrieval_only: bool = True
    retrieval_subchunk_top_k: int = Field(ge=1)
    retrieval_top_k: int = Field(ge=1)
    results: list[RetrievedChunk] = Field(default_factory=list)


class ProviderAnswer(BaseModel):
    """Strict structured-output contract returned by provider implementations."""

    model_config = ConfigDict(extra="forbid")

    answer_text: str = Field(min_length=1)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("answer_text")
    @classmethod
    def _validate_answer_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer_text must be non-empty")
        return normalized

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


class AnswerResponse(BaseModel):
    """Typed grounded-answer payload returned by the V2 answer pipeline."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "no_results", "weak_results", "invalid_grounding"]
    reason_code: Literal[
        "none",
        "filters_excluded_all",
        "no_retrieval_hits",
        "low_similarity",
        "invalid_citations",
    ]
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_debug: DebugRetrieval | None = None
    provider_name: str | None = None
    prompt_version: str = Field(min_length=1)


__all__ = [
    "AnswerResponse",
    "Citation",
    "DebugRetrieval",
    "ProviderAnswer",
    "QueryRequest",
    "RetrievedChunk",
    "RetrievalFilters",
]
