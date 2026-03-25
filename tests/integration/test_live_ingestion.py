from __future__ import annotations

import os
from pathlib import Path

import pytest

from sec_copilot.ingest.pipeline import IngestionConfig, run_ingestion


@pytest.mark.live_sec
def test_live_ingestion_sample(tmp_path: Path) -> None:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        pytest.skip("SEC_USER_AGENT is required for live SEC ingestion")

    summary = run_ingestion(
        IngestionConfig(
            companies_config=Path("configs/companies.yaml"),
            data_dir=tmp_path,
            user_agent=user_agent,
            annual_limit=1,
            quarterly_limit=0,
            companies=("NVDA",),
        )
    )

    assert summary.successful_filings == 1
