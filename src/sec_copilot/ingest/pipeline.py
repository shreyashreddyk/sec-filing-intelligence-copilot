"""End-to-end V1 SEC ingestion pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sec_copilot.config import CompanyConfigError, load_company_universe
from sec_copilot.ingest.constants import (
    CHUNKER_VERSION,
    ERROR_CONFIG,
    ERROR_DOWNLOAD,
    ERROR_NETWORK,
    ERROR_PARSE,
    ERROR_PRIMARY_DOC,
    ERROR_WRITE,
    MANIFEST_SCHEMA_VERSION,
    MIN_USEFUL_TEXT_CHARS,
    PARSER_VERSION,
    RUN_SUMMARY_SCHEMA_VERSION,
    TARGET_FORMS,
    WARNING_PRIMARY_DOC_UNUSABLE,
)
from sec_copilot.ingest.downloader import (
    download_full_submission_text,
    download_preferred_source,
    fetch_submission_payload,
    raw_metadata_path,
    select_target_filings,
)
from sec_copilot.ingest.sec_client import SecClient, SecClientPreflightError, SecRequestError, validate_user_agent
from sec_copilot.processing.chunker import chunk_filing
from sec_copilot.processing.parser import parse_filing
from sec_copilot.schemas.ingestion import (
    CompanyConfig,
    CompanyResult,
    FilingManifest,
    FilingResult,
    RunIssue,
    RunSummary,
)
from sec_copilot.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class IngestionConfig:
    """Runtime configuration for one ingestion run."""

    companies_config: Path
    data_dir: Path
    user_agent: str
    annual_limit: int = 2
    quarterly_limit: int = 4
    companies: tuple[str, ...] = ()
    form_types: tuple[str, ...] = TARGET_FORMS
    force_refresh: bool = False


class IngestionPreflightError(RuntimeError):
    """Raised when ingestion cannot start safely."""


def run_ingestion(config: IngestionConfig) -> RunSummary:
    """Run the V1 ingestion pipeline and write summary artifacts."""

    _preflight(config)
    universe = load_company_universe(config.companies_config)
    companies = _select_companies(universe.enabled_companies(), config.companies)
    client = SecClient(user_agent=config.user_agent)

    started_at = _timestamp()
    run_id = f"ingest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    summary = RunSummary(
        schema_version=RUN_SUMMARY_SCHEMA_VERSION,
        run_id=run_id,
        started_at=started_at,
        completed_at=None,
        status="running",
        companies_config_path=str(config.companies_config),
        requested_companies=len(companies),
        attempted_companies=0,
        successful_companies=0,
        failed_companies=0,
        requested_filings=0,
        successful_filings=0,
        failed_filings=0,
        warning_count=0,
        error_count=0,
        annual_limit=config.annual_limit,
        quarterly_limit=config.quarterly_limit,
    )

    for company in companies:
        summary.attempted_companies += 1
        company_requested_filings = 0
        company_successful_filings = 0
        company_failed_filings = 0
        company_warning_codes: list[str] = []
        company_error_code: str | None = None

        try:
            submissions_payload = fetch_submission_payload(client, company, config.data_dir, config.force_refresh)
            selected_filings, selection_issues = select_target_filings(
                client=client,
                company=company,
                submissions_payload=submissions_payload,
                data_dir=config.data_dir,
                annual_limit=config.annual_limit,
                quarterly_limit=config.quarterly_limit,
                form_types=config.form_types,
                force_refresh=config.force_refresh,
            )

            for issue in selection_issues:
                _record_issue(summary, issue)
                company_warning_codes.append(issue.code)

            company_requested_filings = len(selected_filings)
            summary.requested_filings += company_requested_filings

            for filing in selected_filings:
                try:
                    chunk_path, manifest_path, warning_codes = _process_filing(
                        client=client,
                        company=company,
                        filing=filing,
                        config=config,
                    )
                    for warning_code in warning_codes:
                        _record_issue(
                            summary,
                            RunIssue(
                                level="warning",
                                code=warning_code,
                                message=(
                                    f"{filing.ticker} {filing.form_type} {filing.accession_number} "
                                    f"reported warning {warning_code}"
                                ),
                                ticker=filing.ticker,
                                document_id=f"sec_{filing.cik}_{filing.accession_no_dash}",
                                form_type=filing.form_type,
                                accession_number=filing.accession_number,
                            ),
                        )
                        company_warning_codes.append(warning_code)
                    company_successful_filings += 1
                    summary.successful_filings += 1
                    summary.filing_results.append(
                        FilingResult(
                            ticker=filing.ticker,
                            form_type=filing.form_type,
                            accession_number=filing.accession_number,
                            document_id=f"sec_{filing.cik}_{filing.accession_no_dash}",
                            status="success",
                            chunk_path=chunk_path,
                            manifest_path=manifest_path,
                            warnings=tuple(warning_codes),
                            error_code=None,
                        )
                    )
                except SecRequestError as exc:
                    company_failed_filings += 1
                    summary.failed_filings += 1
                    company_error_code = ERROR_DOWNLOAD
                    _record_issue(
                        summary,
                        RunIssue(
                            level="error",
                            code=ERROR_DOWNLOAD,
                            message=str(exc),
                            ticker=filing.ticker,
                            document_id=f"sec_{filing.cik}_{filing.accession_no_dash}",
                            form_type=filing.form_type,
                            accession_number=filing.accession_number,
                        ),
                    )
                    summary.filing_results.append(
                        FilingResult(
                            ticker=filing.ticker,
                            form_type=filing.form_type,
                            accession_number=filing.accession_number,
                            document_id=f"sec_{filing.cik}_{filing.accession_no_dash}",
                            status="failed",
                            chunk_path=None,
                            manifest_path=None,
                            warnings=(),
                            error_code=ERROR_DOWNLOAD,
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive summary handling
                    company_failed_filings += 1
                    summary.failed_filings += 1
                    company_error_code = ERROR_PARSE
                    _record_issue(
                        summary,
                        RunIssue(
                            level="error",
                            code=ERROR_PARSE,
                            message=str(exc),
                            ticker=filing.ticker,
                            document_id=f"sec_{filing.cik}_{filing.accession_no_dash}",
                            form_type=filing.form_type,
                            accession_number=filing.accession_number,
                        ),
                    )
                    summary.filing_results.append(
                        FilingResult(
                            ticker=filing.ticker,
                            form_type=filing.form_type,
                            accession_number=filing.accession_number,
                            document_id=f"sec_{filing.cik}_{filing.accession_no_dash}",
                            status="failed",
                            chunk_path=None,
                            manifest_path=None,
                            warnings=(),
                            error_code=ERROR_PARSE,
                        )
                    )

            company_status = "success" if company_failed_filings == 0 else "failed"
            summary.company_results.append(
                CompanyResult(
                    ticker=company.ticker,
                    status=company_status,
                    requested_filings=company_requested_filings,
                    successful_filings=company_successful_filings,
                    failed_filings=company_failed_filings,
                    warnings=tuple(dict.fromkeys(company_warning_codes)),
                    error_code=company_error_code,
                )
            )
            if company_failed_filings:
                summary.failed_companies += 1
            else:
                summary.successful_companies += 1
        except CompanyConfigError as exc:
            summary.failed_companies += 1
            _record_issue(
                summary,
                RunIssue(level="error", code=ERROR_CONFIG, message=str(exc), ticker=company.ticker),
            )
            summary.company_results.append(
                CompanyResult(
                    ticker=company.ticker,
                    status="failed",
                    requested_filings=0,
                    successful_filings=0,
                    failed_filings=0,
                    warnings=(),
                    error_code=ERROR_CONFIG,
                )
            )
        except SecRequestError as exc:
            summary.failed_companies += 1
            _record_issue(
                summary,
                RunIssue(level="error", code=ERROR_NETWORK, message=str(exc), ticker=company.ticker),
            )
            summary.company_results.append(
                CompanyResult(
                    ticker=company.ticker,
                    status="failed",
                    requested_filings=0,
                    successful_filings=0,
                    failed_filings=0,
                    warnings=(),
                    error_code=ERROR_NETWORK,
                )
            )

    summary.status = "completed_with_errors" if summary.error_count else "success"
    summary.completed_at = _timestamp()
    summary_path = config.data_dir / "processed" / "run_summaries" / f"{summary.run_id}.json"
    write_json(summary_path, summary)
    return summary


def _preflight(config: IngestionConfig) -> None:
    try:
        validate_user_agent(config.user_agent)
    except SecClientPreflightError as exc:
        raise IngestionPreflightError(str(exc)) from exc

    if config.annual_limit < 0 or config.quarterly_limit < 0:
        raise IngestionPreflightError("Filing limits must be zero or positive integers")
    if set(config.form_types) - set(TARGET_FORMS):
        raise IngestionPreflightError(f"form_types must be a subset of {TARGET_FORMS}")

    for path in (
        config.data_dir / "raw" / "submissions",
        config.data_dir / "raw" / "filings",
        config.data_dir / "processed" / "chunks",
        config.data_dir / "processed" / "manifests",
        config.data_dir / "processed" / "run_summaries",
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IngestionPreflightError(f"Unable to create output directory {path}") from exc


def _select_companies(
    enabled_companies: tuple[CompanyConfig, ...],
    company_filters: tuple[str, ...],
) -> list[CompanyConfig]:
    if not company_filters:
        return list(enabled_companies)

    normalized_filters = {value.strip().upper() for value in company_filters}
    selected = [
        company
        for company in enabled_companies
        if company.ticker.upper() in normalized_filters or company.name.upper() in normalized_filters
    ]
    if not selected:
        raise IngestionPreflightError(f"No enabled companies matched filters {sorted(normalized_filters)}")
    return selected


def _process_filing(
    client: SecClient,
    company: CompanyConfig,
    filing,
    config: IngestionConfig,
) -> tuple[str, str, list[str]]:
    downloaded = download_preferred_source(client, filing, config.data_dir, config.force_refresh)
    parsed = parse_filing(downloaded.raw_text, filing.form_type)
    warning_codes = list(downloaded.warnings) + list(parsed.warnings)

    if downloaded.source_kind == "primary_document" and len(parsed.text) < MIN_USEFUL_TEXT_CHARS:
        warning_codes.append(WARNING_PRIMARY_DOC_UNUSABLE)
        downloaded = download_full_submission_text(client, filing, config.data_dir, config.force_refresh)
        parsed = parse_filing(downloaded.raw_text, filing.form_type)
        warning_codes = list(dict.fromkeys(warning_codes + list(downloaded.warnings) + list(parsed.warnings)))

    if len(parsed.text) < MIN_USEFUL_TEXT_CHARS or not parsed.sections:
        raise RuntimeError(
            f"Unable to parse useful text for {filing.ticker} {filing.form_type} {filing.accession_number}"
        )

    document_id = f"sec_{filing.cik}_{filing.accession_no_dash}"
    metadata_path = raw_metadata_path(config.data_dir, filing)
    write_json(
        metadata_path,
        {
            "document_id": document_id,
            "company_name": company.name,
            "ticker": filing.ticker,
            "cik": filing.cik,
            "form_type": filing.form_type,
            "filing_date": filing.filing_date,
            "report_date": filing.report_date,
            "accession_number": filing.accession_number,
            "filing_index_url": filing.filing_index_url,
            "filing_metadata_url": filing.filing_metadata_url,
            "source_url": downloaded.source_url,
            "source_kind": downloaded.source_kind,
            "raw_path": downloaded.raw_path,
        },
    )

    chunk_records, section_summaries, chunker_config = chunk_filing(downloaded, parsed)
    chunk_path = (
        config.data_dir
        / "processed"
        / "chunks"
        / filing.ticker
        / filing.form_type
        / f"{filing.accession_no_dash}.jsonl"
    )
    write_jsonl(chunk_path, list(chunk_records))

    manifest = FilingManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        document_id=document_id,
        company_name=company.name,
        ticker=filing.ticker,
        cik=filing.cik,
        form_type=filing.form_type,
        filing_date=filing.filing_date,
        report_date=filing.report_date,
        accession_number=filing.accession_number,
        filing_index_url=filing.filing_index_url,
        source_url=downloaded.source_url,
        source_kind=downloaded.source_kind,
        raw_path=downloaded.raw_path,
        chunk_path=chunk_path.as_posix(),
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        parser_strategy=parsed.parser_strategy,
        parser_warnings=tuple(dict.fromkeys(warning_codes)),
        chunker_config=chunker_config,
        section_count=len(section_summaries),
        chunk_count=len(chunk_records),
        sections=section_summaries,
    )
    manifest_path = (
        config.data_dir
        / "processed"
        / "manifests"
        / filing.ticker
        / filing.form_type
        / f"{filing.accession_no_dash}.json"
    )
    write_json(manifest_path, manifest)
    return chunk_path.as_posix(), manifest_path.as_posix(), list(dict.fromkeys(warning_codes))


def _record_issue(summary: RunSummary, issue: RunIssue) -> None:
    if issue.level == "warning":
        summary.warnings.append(issue)
        summary.warning_count += 1
    else:
        summary.errors.append(issue)
        summary.error_count += 1


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
