from __future__ import annotations

import json
from pathlib import Path

from sec_copilot.ingest.pipeline import IngestionConfig, run_ingestion


def _fixture_text(relative_path: str) -> str:
    return (Path("tests/fixtures/sec") / relative_path).read_text(encoding="utf-8")


class FakeSecClient:
    def __init__(self, user_agent: str, rate_limit_seconds: float = 1.0, timeout_seconds: float = 30.0, max_retries: int = 3) -> None:
        self.user_agent = user_agent

    def get_json(self, url: str, cache_path: Path, force_refresh: bool = False):
        payload = _fixture_payloads()[url]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(payload, encoding="utf-8")
        return json.loads(payload)

    def get_text(self, url: str, cache_path: Path, force_refresh: bool = False) -> str:
        payload = _fixture_payloads()[url]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(payload, encoding="utf-8")
        return payload


def _fixture_payloads() -> dict[str, str]:
    return {
        "https://data.sec.gov/submissions/CIK0001045810.json": _fixture_text("submissions/CIK0001045810.json"),
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000050/nvda-20250126x10k.htm": _fixture_text(
            "filings/nvda_10k_primary.html"
        ),
    }


def test_run_ingestion_writes_chunks_manifest_and_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sec_copilot.ingest.pipeline.SecClient", FakeSecClient)

    summary = run_ingestion(
        IngestionConfig(
            companies_config=Path("configs/companies.yaml"),
            data_dir=tmp_path,
            user_agent="Shreyash Reddy shreyash@acme.test",
            annual_limit=1,
            quarterly_limit=0,
            companies=("NVDA",),
        )
    )

    assert summary.status == "success"
    assert summary.successful_filings == 1
    chunk_path = tmp_path / "processed" / "chunks" / "NVDA" / "10-K" / "000104581025000050.jsonl"
    manifest_path = tmp_path / "processed" / "manifests" / "NVDA" / "10-K" / "000104581025000050.json"
    summary_path = tmp_path / "processed" / "run_summaries" / f"{summary.run_id}.json"

    assert chunk_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    chunk_rows = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert chunk_rows[0]["document_id"] == "sec_0001045810_000104581025000050"
    assert chunk_rows[0]["source_kind"] == "primary_document"
    assert manifest["chunk_count"] == len(chunk_rows)
    assert manifest["parser_version"] == "parser.v1"
    assert run_summary["successful_filings"] == 1
