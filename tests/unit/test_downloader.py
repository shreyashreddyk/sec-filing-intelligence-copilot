from __future__ import annotations

import json
from pathlib import Path

from sec_copilot.config import load_company_universe
from sec_copilot.ingest.downloader import download_preferred_source, select_target_filings


class FakeSecClient:
    def __init__(self, payloads: dict[str, str]) -> None:
        self.payloads = payloads

    def get_json(self, url: str, cache_path: Path, force_refresh: bool = False):
        return json.loads(self.payloads[url])

    def get_text(self, url: str, cache_path: Path, force_refresh: bool = False) -> str:
        return self.payloads[url]


def _fixture_path(relative_path: str) -> Path:
    return Path("tests/fixtures/sec") / relative_path


def test_select_target_filings_fetches_older_pages_when_recent_is_insufficient() -> None:
    company = load_company_universe("configs/companies.yaml").companies[0]
    primary_payload = json.loads(_fixture_path("submissions/CIK0001045810.json").read_text(encoding="utf-8"))
    older_payload_text = _fixture_path("submissions/CIK0001045810-submissions-001.json").read_text(encoding="utf-8")
    client = FakeSecClient(
        {
            "https://data.sec.gov/submissions/CIK0001045810-submissions-001.json": older_payload_text,
        }
    )

    filings, issues = select_target_filings(
        client=client,
        company=company,
        submissions_payload=primary_payload,
        data_dir=Path("data"),
        annual_limit=2,
        quarterly_limit=2,
    )

    assert [filing.form_type for filing in filings] == ["10-Q", "10-Q", "10-K", "10-K"]
    assert [filing.accession_number for filing in filings][-1] == "0001045810-24-000099"
    issue_codes = {issue.code for issue in issues}
    assert "missing_filing_date" in issue_codes


def test_download_preferred_source_falls_back_when_primary_extension_is_not_supported(tmp_path: Path) -> None:
    company = load_company_universe("configs/companies.yaml").companies[0]
    filing = select_target_filings(
        client=FakeSecClient({}),
        company=company,
        submissions_payload={
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2025-02-20"],
                    "reportDate": ["2025-01-26"],
                    "acceptanceDateTime": ["2025-02-20T16:10:00.000Z"],
                    "accessionNumber": ["0001045810-25-000050"],
                    "primaryDocument": ["nvda-primary.xml"],
                    "primaryDocDescription": ["Form 10-K"]
                }
            }
        },
        data_dir=tmp_path,
        annual_limit=1,
        quarterly_limit=0,
    )[0][0]

    fallback_url = (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000050/"
        "0001045810-25-000050.txt"
    )
    client = FakeSecClient({fallback_url: "Fallback filing text"})
    downloaded = download_preferred_source(client, filing, tmp_path)

    assert downloaded.source_kind == "full_submission_text"
    assert "unsupported_primary_doc_extension" in downloaded.warnings
