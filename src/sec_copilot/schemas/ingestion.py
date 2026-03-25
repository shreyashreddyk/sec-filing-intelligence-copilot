"""Typed models for the SEC ingestion and chunk-manifest pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompanyConfig:
    """Validated company configuration used for ingestion."""

    name: str
    ticker: str
    cik: str
    enabled: bool = True
    sector: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class CompanyUniverse:
    """Top-level company universe configuration."""

    universe_name: str
    companies: tuple[CompanyConfig, ...]
    sector: str | None = None
    notes: str | None = None

    def enabled_companies(self) -> tuple[CompanyConfig, ...]:
        return tuple(company for company in self.companies if company.enabled)


@dataclass(frozen=True)
class FilingRecord:
    """Normalized filing metadata selected from SEC submissions payloads."""

    company_name: str
    ticker: str
    cik: str
    form_type: str
    filing_date: str
    report_date: str | None
    acceptance_datetime: str | None
    accession_number: str
    accession_no_dash: str
    primary_document: str | None
    primary_doc_description: str | None
    filing_index_url: str
    filing_metadata_url: str


@dataclass(frozen=True)
class DownloadedFiling:
    """Resolved source artifact used for parsing a filing."""

    filing: FilingRecord
    source_url: str
    source_kind: str
    raw_path: str
    raw_text: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedSection:
    """A section extracted from a normalized filing document."""

    section_key: str
    section_title: str
    section_order: int
    item_number: str | None
    text: str
    char_start: int
    char_end: int
    parser_strategy: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedDocument:
    """The parser output for a filing artifact."""

    text: str
    parser_strategy: str
    warnings: tuple[str, ...]
    sections: tuple[ParsedSection, ...]


@dataclass(frozen=True)
class ChunkRecord:
    """A stored chunk record that later retrieval can index."""

    schema_version: str
    chunk_id: str
    document_id: str
    company_name: str
    ticker: str
    cik: str
    form_type: str
    filing_date: str
    report_date: str | None
    accession_number: str
    source_url: str
    filing_index_url: str
    source_kind: str
    raw_path: str
    section_key: str
    section_title: str
    section_order: int
    item_number: str | None
    parser_strategy: str
    chunk_index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_count: int
    content_hash: str
    text: str


@dataclass(frozen=True)
class SectionSummary:
    """Manifest summary for one parsed section."""

    section_key: str
    section_title: str
    section_order: int
    item_number: str | None
    char_start: int
    char_end: int
    warnings: tuple[str, ...]
    chunk_count: int


@dataclass(frozen=True)
class FilingManifest:
    """Per-filing manifest describing processed outputs and warnings."""

    schema_version: str
    document_id: str
    company_name: str
    ticker: str
    cik: str
    form_type: str
    filing_date: str
    report_date: str | None
    accession_number: str
    filing_index_url: str
    source_url: str
    source_kind: str
    raw_path: str
    chunk_path: str
    parser_version: str
    chunker_version: str
    parser_strategy: str
    parser_warnings: tuple[str, ...]
    chunker_config: dict[str, str | int]
    section_count: int
    chunk_count: int
    sections: tuple[SectionSummary, ...]


@dataclass(frozen=True)
class RunIssue:
    """Structured warning or error recorded in a run summary."""

    level: str
    code: str
    message: str
    ticker: str | None = None
    document_id: str | None = None
    form_type: str | None = None
    accession_number: str | None = None


@dataclass(frozen=True)
class FilingResult:
    """Result row for one filing in a run summary."""

    ticker: str
    form_type: str
    accession_number: str
    document_id: str | None
    status: str
    chunk_path: str | None
    manifest_path: str | None
    warnings: tuple[str, ...] = ()
    error_code: str | None = None


@dataclass(frozen=True)
class CompanyResult:
    """Result row for one company in a run summary."""

    ticker: str
    status: str
    requested_filings: int
    successful_filings: int
    failed_filings: int
    warnings: tuple[str, ...] = ()
    error_code: str | None = None


@dataclass
class RunSummary:
    """Top-level summary artifact for an ingestion run."""

    schema_version: str
    run_id: str
    started_at: str
    completed_at: str | None
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
    company_results: list[CompanyResult] = field(default_factory=list)
    filing_results: list[FilingResult] = field(default_factory=list)
    warnings: list[RunIssue] = field(default_factory=list)
    errors: list[RunIssue] = field(default_factory=list)
