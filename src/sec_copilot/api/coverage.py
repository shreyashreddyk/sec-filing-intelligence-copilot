"""Coverage-state models and helpers for the V5 API layer."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.schemas.retrieval import RetrievalFilters
from sec_copilot.utils.io import write_json


IndexStatus = Literal["missing", "stale", "fresh"]
GlobalCoverageStatus = Literal["uninitialized", "partial", "full"]
RequestCoverageStatus = Literal["covered", "partially_covered", "uncovered"]


class TargetScope(BaseModel):
    """Admin-requested intended corpus scope."""

    model_config = ConfigDict(extra="forbid")

    companies: list[str] = Field(default_factory=list)
    form_types: list[str] = Field(default_factory=list)
    annual_limit: int = Field(default=0, ge=0)
    quarterly_limit: int = Field(default=0, ge=0)


class IndexedScopeEntry(BaseModel):
    """Coverage summary for one ticker and form-type slice."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    form_type: str
    filing_count: int = Field(ge=0)
    filing_date_from: date | None = None
    filing_date_to: date | None = None


class IndexedScope(BaseModel):
    """Actual indexed coverage summary."""

    model_config = ConfigDict(extra="forbid")

    companies: list[str] = Field(default_factory=list)
    form_types: list[str] = Field(default_factory=list)
    entries: list[IndexedScopeEntry] = Field(default_factory=list)
    document_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)


class MissingDateRange(BaseModel):
    """One uncovered request date segment for a given scope slice."""

    model_config = ConfigDict(extra="forbid")

    ticker: str | None = None
    form_type: str | None = None
    filing_date_from: date | None = None
    filing_date_to: date | None = None


class MissingScopePair(BaseModel):
    """One uncovered ticker and form-type combination."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    form_type: str


class MissingScope(BaseModel):
    """Request-scoped uncovered dimensions."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    form_types: list[str] = Field(default_factory=list)
    pairs: list[MissingScopePair] = Field(default_factory=list)
    date_ranges: list[MissingDateRange] = Field(default_factory=list)


class CoverageState(BaseModel):
    """Persisted global coverage and freshness state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["coverage_state.v1"] = "coverage_state.v1"
    target_scope: TargetScope
    indexed_scope: IndexedScope
    coverage_status: GlobalCoverageStatus
    last_ingest_run_id: str | None = None
    last_ingest_completed_at: datetime | None = None
    last_index_refresh_at: datetime | None = None
    processed_corpus_fingerprint: str | None = None
    indexed_corpus_fingerprint: str | None = None
    index_status: IndexStatus = "missing"


class RequestCoverageAssessment(BaseModel):
    """Request-scoped coverage evaluation over the indexed corpus."""

    model_config = ConfigDict(extra="forbid")

    coverage_status: RequestCoverageStatus
    indexed_scope: IndexedScope
    missing_scope: MissingScope


def coverage_state_path(persist_directory: str | Path, collection_name: str) -> Path:
    """Return the persisted coverage-state artifact path."""

    return Path(persist_directory) / f"{collection_name}.coverage.json"


def load_coverage_state(path: str | Path) -> CoverageState | None:
    """Load a coverage-state artifact when present."""

    coverage_path = Path(path)
    if not coverage_path.exists():
        return None
    return CoverageState.model_validate_json(coverage_path.read_text(encoding="utf-8"))


def write_coverage_state(path: str | Path, state: CoverageState) -> None:
    """Persist the coverage-state artifact."""

    write_json(Path(path), state)


def build_indexed_scope(
    store: ProcessedChunkStore,
    *,
    filters: RetrievalFilters | None = None,
    ignore_request_dates: bool = False,
) -> IndexedScope:
    """Summarize indexed coverage for the whole store or a filtered slice."""

    effective_filters = filters
    if ignore_request_dates and filters is not None:
        effective_filters = RetrievalFilters(
            tickers=list(filters.tickers),
            form_types=list(filters.form_types),
        )

    chunks = store.filtered_values(effective_filters) if effective_filters is not None else store.values()
    documents_by_scope: dict[tuple[str, str], set[str]] = {}
    filing_dates_by_scope: dict[tuple[str, str], list[date]] = {}

    for chunk in chunks:
        scope_key = (chunk.ticker, chunk.form_type)
        documents_by_scope.setdefault(scope_key, set()).add(chunk.document_id)
        filing_dates_by_scope.setdefault(scope_key, []).append(date.fromisoformat(chunk.filing_date))

    entries = [
        IndexedScopeEntry(
            ticker=ticker,
            form_type=form_type,
            filing_count=len(documents_by_scope[(ticker, form_type)]),
            filing_date_from=min(filing_dates_by_scope[(ticker, form_type)]),
            filing_date_to=max(filing_dates_by_scope[(ticker, form_type)]),
        )
        for ticker, form_type in sorted(documents_by_scope)
    ]
    return IndexedScope(
        companies=sorted({entry.ticker for entry in entries}),
        form_types=sorted({entry.form_type for entry in entries}),
        entries=entries,
        document_count=len({chunk.document_id for chunk in chunks}),
        chunk_count=len(chunks),
    )


def build_coverage_state(
    *,
    target_scope: TargetScope,
    indexed_scope: IndexedScope,
    last_ingest_run_id: str | None,
    last_ingest_completed_at: datetime | None,
    last_index_refresh_at: datetime | None,
    processed_corpus_fingerprint: str | None,
    indexed_corpus_fingerprint: str | None,
    index_status: IndexStatus,
) -> CoverageState:
    """Construct the persisted coverage-state snapshot."""

    coverage_status: GlobalCoverageStatus = "uninitialized"
    if index_status == "fresh" and indexed_scope.document_count > 0:
        missing_expected_scope = False
        expected_by_form = {
            "10-K": target_scope.annual_limit if "10-K" in target_scope.form_types else 0,
            "10-Q": target_scope.quarterly_limit if "10-Q" in target_scope.form_types else 0,
        }
        actual_counts = {
            (entry.ticker, entry.form_type): entry.filing_count
            for entry in indexed_scope.entries
        }
        for ticker in target_scope.companies:
            for form_type, expected_count in expected_by_form.items():
                if expected_count <= 0:
                    continue
                if actual_counts.get((ticker, form_type), 0) < expected_count:
                    missing_expected_scope = True
                    break
            if missing_expected_scope:
                break
        coverage_status = "partial" if missing_expected_scope else "full"

    return CoverageState(
        target_scope=target_scope,
        indexed_scope=indexed_scope,
        coverage_status=coverage_status,
        last_ingest_run_id=last_ingest_run_id,
        last_ingest_completed_at=last_ingest_completed_at,
        last_index_refresh_at=last_index_refresh_at,
        processed_corpus_fingerprint=processed_corpus_fingerprint,
        indexed_corpus_fingerprint=indexed_corpus_fingerprint,
        index_status=index_status,
    )


def assess_request_coverage(store: ProcessedChunkStore, filters: RetrievalFilters) -> RequestCoverageAssessment:
    """Evaluate request-scoped indexed coverage under strict V5 semantics."""

    indexed_scope = build_indexed_scope(store, filters=filters, ignore_request_dates=True)
    missing_tickers = [ticker for ticker in filters.tickers if ticker not in indexed_scope.companies]
    missing_form_types = [form_type for form_type in filters.form_types if form_type not in indexed_scope.form_types]
    available_pairs = {(entry.ticker, entry.form_type) for entry in indexed_scope.entries}
    missing_pairs = [
        MissingScopePair(ticker=ticker, form_type=form_type)
        for ticker in filters.tickers
        for form_type in filters.form_types
        if (ticker, form_type) not in available_pairs
    ]
    missing_date_ranges = _missing_date_ranges(indexed_scope, filters)
    missing_scope = MissingScope(
        tickers=missing_tickers,
        form_types=missing_form_types,
        pairs=missing_pairs,
        date_ranges=missing_date_ranges,
    )

    date_filtered_matches = store.filtered_values(filters)
    if missing_scope.tickers or missing_scope.form_types or missing_scope.pairs or missing_scope.date_ranges:
        coverage_status: RequestCoverageStatus = "partially_covered" if date_filtered_matches else "uncovered"
    else:
        coverage_status = "covered"

    return RequestCoverageAssessment(
        coverage_status=coverage_status,
        indexed_scope=indexed_scope,
        missing_scope=missing_scope,
    )


def coerce_coverage_state_from_snapshot(path: str | Path) -> CoverageState | None:
    """Load a JSON snapshot using the canonical coverage-state schema."""

    try:
        return load_coverage_state(path)
    except Exception:
        return None


def latest_ingest_snapshot(data_dir: str | Path) -> tuple[str | None, datetime | None]:
    """Return the latest completed ingestion run identifier and timestamp."""

    summaries_dir = Path(data_dir) / "processed" / "run_summaries"
    latest_run_id: str | None = None
    latest_completed_at: datetime | None = None
    for path in sorted(summaries_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        completed_at_raw = payload.get("completed_at")
        if not isinstance(completed_at_raw, str) or not completed_at_raw:
            continue
        try:
            completed_at = datetime.fromisoformat(completed_at_raw)
        except ValueError:
            continue
        if latest_completed_at is None or completed_at > latest_completed_at:
            latest_completed_at = completed_at
            latest_run_id = str(payload.get("run_id") or path.stem)
    return latest_run_id, latest_completed_at


def _missing_date_ranges(indexed_scope: IndexedScope, filters: RetrievalFilters) -> list[MissingDateRange]:
    if filters.filing_date_from is None and filters.filing_date_to is None:
        return []
    if not indexed_scope.entries:
        return [
            MissingDateRange(
                filing_date_from=filters.filing_date_from,
                filing_date_to=filters.filing_date_to,
            )
        ]

    date_ranges: list[MissingDateRange] = []
    requested_tickers = filters.tickers or [None]
    requested_forms = filters.form_types or [None]

    for ticker in requested_tickers:
        for form_type in requested_forms:
            matching_entries = [
                entry
                for entry in indexed_scope.entries
                if (ticker is None or entry.ticker == ticker)
                and (form_type is None or entry.form_type == form_type)
            ]
            if not matching_entries:
                continue
            scope_start = min(entry.filing_date_from for entry in matching_entries if entry.filing_date_from is not None)
            scope_end = max(entry.filing_date_to for entry in matching_entries if entry.filing_date_to is not None)

            if filters.filing_date_from is not None and filters.filing_date_from < scope_start:
                date_ranges.append(
                    MissingDateRange(
                        ticker=ticker,
                        form_type=form_type,
                        filing_date_from=filters.filing_date_from,
                        filing_date_to=min(filters.filing_date_to or scope_start, scope_start - timedelta(days=1)),
                    )
                )
            if filters.filing_date_to is not None and filters.filing_date_to > scope_end:
                date_ranges.append(
                    MissingDateRange(
                        ticker=ticker,
                        form_type=form_type,
                        filing_date_from=max(filters.filing_date_from or scope_end, scope_end + timedelta(days=1)),
                        filing_date_to=filters.filing_date_to,
                    )
                )

    deduped: list[MissingDateRange] = []
    seen: set[tuple[str | None, str | None, date | None, date | None]] = set()
    for item in date_ranges:
        key = (item.ticker, item.form_type, item.filing_date_from, item.filing_date_to)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


__all__ = [
    "CoverageState",
    "GlobalCoverageStatus",
    "IndexStatus",
    "IndexedScope",
    "IndexedScopeEntry",
    "MissingDateRange",
    "MissingScopePair",
    "MissingScope",
    "RequestCoverageAssessment",
    "RequestCoverageStatus",
    "TargetScope",
    "assess_request_coverage",
    "build_coverage_state",
    "build_indexed_scope",
    "coerce_coverage_state_from_snapshot",
    "coverage_state_path",
    "latest_ingest_snapshot",
    "load_coverage_state",
    "write_coverage_state",
]
