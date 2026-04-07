from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot.config import CompanyConfigError, load_company_universe, normalize_cik


def test_normalize_cik_zero_pads_values() -> None:
    assert normalize_cik("1045810") == "0001045810"
    assert normalize_cik(2488) == "0000002488"


def test_load_company_universe_reads_current_config() -> None:
    universe = load_company_universe(Path("configs/companies.yaml"))
    assert universe.universe_name == "semis_v1"
    assert universe.companies[0].ticker == "NVDA"
    assert universe.companies[0].cik == "0001045810"


def test_load_company_universe_reads_sample_config() -> None:
    universe = load_company_universe(Path("configs/companies.sample.yaml"))

    assert universe.universe_name == "semis_sample"
    assert [company.ticker for company in universe.companies] == ["NVDA", "AMD", "INTC"]


def test_load_company_universe_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "companies.yaml"
    config_path.write_text(
        """
universe_name: test
companies:
  - name: Example
    ticker: EXM
    cik: "1234"
    unknown: value
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CompanyConfigError):
        load_company_universe(config_path)
