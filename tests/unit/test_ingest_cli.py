from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv as real_load_dotenv

from sec_copilot.ingest.cli import main
from sec_copilot.schemas.ingestion import RunSummary


def _successful_summary() -> RunSummary:
    return RunSummary(
        schema_version="sec_ingest_run_summary.v1",
        run_id="ingest_test",
        started_at="2026-03-26T00:00:00Z",
        completed_at="2026-03-26T00:00:01Z",
        status="success",
        companies_config_path="configs/companies.yaml",
        requested_companies=1,
        attempted_companies=1,
        successful_companies=1,
        failed_companies=0,
        requested_filings=0,
        successful_filings=0,
        failed_filings=0,
        warning_count=0,
        error_count=0,
        annual_limit=1,
        quarterly_limit=0,
        company_results=[],
        filing_results=[],
        warnings=[],
        errors=[],
    )


def test_ingest_cli_loads_user_agent_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_ingestion(config):
        captured["user_agent"] = config.user_agent
        return _successful_summary()

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr("sec_copilot.ingest.cli.run_ingestion", fake_run_ingestion)
    (tmp_path / ".env").write_text('SEC_USER_AGENT="Local Tester local@example.com"\n', encoding="utf-8")
    monkeypatch.setattr(
        "sec_copilot.ingest.cli.load_dotenv",
        lambda: real_load_dotenv(dotenv_path=tmp_path / ".env"),
    )

    exit_code = main(
        [
            "run",
            "--companies-config",
            "configs/companies.yaml",
            "--data-dir",
            str(tmp_path / "data"),
            "--company",
            "NVDA",
            "--annual-limit",
            "1",
            "--quarterly-limit",
            "0",
        ]
    )

    assert exit_code == 0
    assert captured["user_agent"] == "Local Tester local@example.com"


def test_ingest_cli_prefers_explicit_user_agent_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_ingestion(config):
        captured["user_agent"] = config.user_agent
        return _successful_summary()

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr("sec_copilot.ingest.cli.run_ingestion", fake_run_ingestion)
    (tmp_path / ".env").write_text('SEC_USER_AGENT="Local Tester local@example.com"\n', encoding="utf-8")
    monkeypatch.setattr(
        "sec_copilot.ingest.cli.load_dotenv",
        lambda: real_load_dotenv(dotenv_path=tmp_path / ".env"),
    )

    exit_code = main(
        [
            "run",
            "--companies-config",
            "configs/companies.yaml",
            "--data-dir",
            str(tmp_path / "data"),
            "--company",
            "NVDA",
            "--annual-limit",
            "1",
            "--quarterly-limit",
            "0",
            "--user-agent",
            "Explicit Tester explicit@example.com",
        ]
    )

    assert exit_code == 0
    assert captured["user_agent"] == "Explicit Tester explicit@example.com"
