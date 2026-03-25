"""SEC submissions handling, filing selection, and raw artifact downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sec_copilot.ingest.constants import (
    ANNUAL_FORM,
    PRIMARY_DOCUMENT_EXTENSIONS,
    QUARTERLY_FORM,
    TARGET_FORMS,
    WARNING_FEWER_FILINGS,
    WARNING_MISSING_FILING_DATE,
    WARNING_PRIMARY_DOC_UNUSABLE,
    WARNING_UNSUPPORTED_PRIMARY_DOC_EXTENSION,
)
from sec_copilot.ingest.sec_client import SecClient, SecRequestError
from sec_copilot.schemas.ingestion import CompanyConfig, DownloadedFiling, FilingRecord, RunIssue


def fetch_submission_payload(
    client: SecClient,
    company: CompanyConfig,
    data_dir: Path,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch the primary submissions payload for one company."""

    cache_path = data_dir / "raw" / "submissions" / f"CIK{company.cik}.json"
    url = f"https://data.sec.gov/submissions/CIK{company.cik}.json"
    return client.get_json(url, cache_path=cache_path, force_refresh=force_refresh)


def select_target_filings(
    client: SecClient,
    company: CompanyConfig,
    submissions_payload: dict[str, Any],
    data_dir: Path,
    annual_limit: int,
    quarterly_limit: int,
    form_types: tuple[str, ...] = TARGET_FORMS,
    force_refresh: bool = False,
) -> tuple[list[FilingRecord], list[RunIssue]]:
    """Select deterministic V1 filing targets from submissions metadata."""

    requested_limits = {
        ANNUAL_FORM: annual_limit if ANNUAL_FORM in form_types else 0,
        QUARTERLY_FORM: quarterly_limit if QUARTERLY_FORM in form_types else 0,
    }

    all_rows = _extract_rows(submissions_payload)
    selected, issues = _select_from_rows(company, all_rows, requested_limits)
    if _has_enough(selected, requested_limits):
        return selected, issues

    for older_file in submissions_payload.get("filings", {}).get("files", []):
        name = older_file.get("name")
        if not isinstance(name, str) or not name:
            continue

        older_url = f"https://data.sec.gov/submissions/{name}"
        cache_path = data_dir / "raw" / "submissions" / name
        older_payload = client.get_json(older_url, cache_path=cache_path, force_refresh=force_refresh)
        all_rows.extend(_extract_rows(older_payload))
        selected, issues = _select_from_rows(company, all_rows, requested_limits)
        if _has_enough(selected, requested_limits):
            break

    return selected, issues


def download_preferred_source(
    client: SecClient,
    filing: FilingRecord,
    data_dir: Path,
    force_refresh: bool = False,
) -> DownloadedFiling:
    """Download the preferred primary filing artifact or fall back when necessary."""

    warnings: list[str] = []
    primary_document = filing.primary_document
    if primary_document:
        extension = Path(primary_document).suffix.lower()
        if extension in PRIMARY_DOCUMENT_EXTENSIONS:
            source_url = _primary_document_url(filing)
            raw_path = _raw_filing_path(data_dir, filing, primary_document)
            try:
                raw_text = client.get_text(source_url, cache_path=raw_path, force_refresh=force_refresh)
                return DownloadedFiling(
                    filing=filing,
                    source_url=source_url,
                    source_kind="primary_document",
                    raw_path=raw_path.as_posix(),
                    raw_text=raw_text,
                    warnings=tuple(warnings),
                )
            except SecRequestError:
                warnings.append(WARNING_PRIMARY_DOC_UNUSABLE)
        else:
            warnings.append(WARNING_UNSUPPORTED_PRIMARY_DOC_EXTENSION)

    fallback = download_full_submission_text(client, filing, data_dir, force_refresh=force_refresh)
    return DownloadedFiling(
        filing=fallback.filing,
        source_url=fallback.source_url,
        source_kind=fallback.source_kind,
        raw_path=fallback.raw_path,
        raw_text=fallback.raw_text,
        warnings=tuple(warnings) + fallback.warnings,
    )


def download_full_submission_text(
    client: SecClient,
    filing: FilingRecord,
    data_dir: Path,
    force_refresh: bool = False,
) -> DownloadedFiling:
    """Download the full submission text for a filing."""

    source_url = _full_submission_text_url(filing)
    raw_filename = f"{filing.accession_number}.txt"
    raw_path = _raw_filing_path(data_dir, filing, raw_filename)
    raw_text = client.get_text(source_url, cache_path=raw_path, force_refresh=force_refresh)
    return DownloadedFiling(
        filing=filing,
        source_url=source_url,
        source_kind="full_submission_text",
        raw_path=raw_path.as_posix(),
        raw_text=raw_text,
        warnings=(),
    )


def raw_metadata_path(data_dir: Path, filing: FilingRecord) -> Path:
    """Return the raw metadata sidecar path for one filing."""

    return _filing_directory(data_dir, filing) / "filing_metadata.json"


def _select_from_rows(
    company: CompanyConfig,
    rows: list[dict[str, Any]],
    requested_limits: dict[str, int],
) -> tuple[list[FilingRecord], list[RunIssue]]:
    issues: list[RunIssue] = []
    normalized_rows: list[dict[str, Any]] = []
    warned_missing_filing_date = False

    for row in rows:
        form_type = row.get("form")
        if form_type not in TARGET_FORMS:
            continue
        if requested_limits.get(form_type, 0) <= 0:
            continue
        filing_date = row.get("filingDate")
        if not filing_date:
            if not warned_missing_filing_date:
                issues.append(
                    RunIssue(
                        level="warning",
                        code=WARNING_MISSING_FILING_DATE,
                        message=f"Skipped {company.ticker} records missing filingDate",
                        ticker=company.ticker,
                    )
                )
                warned_missing_filing_date = True
            continue
        accession_number = row.get("accessionNumber")
        if not accession_number:
            continue
        normalized_rows.append(row)

    normalized_rows.sort(
        key=lambda row: (
            row.get("form"),
            row.get("filingDate") or "",
            row.get("acceptanceDateTime") or "",
            row.get("accessionNumber") or "",
        ),
        reverse=True,
    )

    deduped_rows: list[dict[str, Any]] = []
    seen_accessions: set[str] = set()
    for row in normalized_rows:
        accession_number = str(row["accessionNumber"])
        if accession_number in seen_accessions:
            continue
        seen_accessions.add(accession_number)
        deduped_rows.append(row)

    selected: list[FilingRecord] = []
    counts = {ANNUAL_FORM: 0, QUARTERLY_FORM: 0}
    for row in deduped_rows:
        form_type = str(row["form"])
        if counts[form_type] >= requested_limits[form_type]:
            continue
        selected.append(_to_filing_record(company, row))
        counts[form_type] += 1
        if _has_enough(selected, requested_limits):
            break

    for form_type, requested_count in requested_limits.items():
        if requested_count and counts[form_type] < requested_count:
            issues.append(
                RunIssue(
                    level="warning",
                    code=WARNING_FEWER_FILINGS,
                    message=(
                        f"{company.ticker} has only {counts[form_type]} {form_type} filings "
                        f"for requested limit {requested_count}"
                    ),
                    ticker=company.ticker,
                    form_type=form_type,
                )
            )

    return selected, issues


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    recent = payload.get("filings", {}).get("recent")
    if isinstance(recent, dict):
        return _columnar_mapping_to_rows(recent)
    if isinstance(payload, dict) and "accessionNumber" in payload:
        return _columnar_mapping_to_rows(payload)
    return []


def _columnar_mapping_to_rows(columnar_payload: dict[str, Any]) -> list[dict[str, Any]]:
    list_fields = {
        key: value
        for key, value in columnar_payload.items()
        if isinstance(value, list)
    }
    if not list_fields:
        return []
    expected_length = len(next(iter(list_fields.values())))
    rows: list[dict[str, Any]] = []
    for index in range(expected_length):
        row = {key: values[index] for key, values in list_fields.items()}
        rows.append(row)
    return rows


def _to_filing_record(company: CompanyConfig, row: dict[str, Any]) -> FilingRecord:
    accession_number = str(row["accessionNumber"])
    accession_no_dash = accession_number.replace("-", "")
    filing_directory = _archive_base_url(company.cik, accession_no_dash)
    return FilingRecord(
        company_name=company.name,
        ticker=company.ticker,
        cik=company.cik,
        form_type=str(row["form"]),
        filing_date=str(row["filingDate"]),
        report_date=_optional_string(row.get("reportDate")),
        acceptance_datetime=_optional_string(row.get("acceptanceDateTime")),
        accession_number=accession_number,
        accession_no_dash=accession_no_dash,
        primary_document=_optional_string(row.get("primaryDocument")),
        primary_doc_description=_optional_string(row.get("primaryDocDescription")),
        filing_index_url=f"{filing_directory}/",
        filing_metadata_url=f"{filing_directory}/index.json",
    )


def _archive_base_url(cik: str, accession_no_dash: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}"


def _primary_document_url(filing: FilingRecord) -> str:
    if not filing.primary_document:
        raise ValueError("primary_document is required for primary-document URL resolution")
    return f"{_archive_base_url(filing.cik, filing.accession_no_dash)}/{filing.primary_document}"


def _full_submission_text_url(filing: FilingRecord) -> str:
    return f"{_archive_base_url(filing.cik, filing.accession_no_dash)}/{filing.accession_number}.txt"


def _filing_directory(data_dir: Path, filing: FilingRecord) -> Path:
    return data_dir / "raw" / "filings" / filing.ticker / filing.form_type / filing.accession_no_dash


def _raw_filing_path(data_dir: Path, filing: FilingRecord, filename: str) -> Path:
    return _filing_directory(data_dir, filing) / filename


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _has_enough(selected_filings: list[FilingRecord], limits: dict[str, int]) -> bool:
    counts = {ANNUAL_FORM: 0, QUARTERLY_FORM: 0}
    for filing in selected_filings:
        counts[filing.form_type] += 1
    return all(counts[form_type] >= limit for form_type, limit in limits.items() if limit > 0)
