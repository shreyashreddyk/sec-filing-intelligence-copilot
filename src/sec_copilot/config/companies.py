"""Company configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sec_copilot.schemas.ingestion import CompanyConfig, CompanyUniverse


class CompanyConfigError(ValueError):
    """Raised when the companies config is invalid."""


_TOP_LEVEL_REQUIRED_FIELDS = {"universe_name", "companies"}
_TOP_LEVEL_OPTIONAL_FIELDS = {"sector", "notes"}
_COMPANY_REQUIRED_FIELDS = {"name", "ticker", "cik"}
_COMPANY_OPTIONAL_FIELDS = {"enabled", "sector", "notes"}


def normalize_cik(value: str | int) -> str:
    """Normalize a SEC CIK to a zero-padded 10-digit string."""

    normalized = str(value).strip()
    if not normalized.isdigit():
        raise CompanyConfigError(f"CIK must contain only digits, got {value!r}")
    if len(normalized) > 10:
        raise CompanyConfigError(f"CIK must be at most 10 digits, got {value!r}")
    return normalized.zfill(10)


def load_company_universe(path: str | Path) -> CompanyUniverse:
    """Load and validate the fixed company universe config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise CompanyConfigError("Top-level companies config must be a mapping")

    unknown_top_level = set(payload) - (_TOP_LEVEL_REQUIRED_FIELDS | _TOP_LEVEL_OPTIONAL_FIELDS)
    if unknown_top_level:
        raise CompanyConfigError(
            f"Unknown top-level fields in companies config: {sorted(unknown_top_level)}"
        )

    missing_top_level = _TOP_LEVEL_REQUIRED_FIELDS - set(payload)
    if missing_top_level:
        raise CompanyConfigError(
            f"Missing required top-level fields in companies config: {sorted(missing_top_level)}"
        )

    universe_name = _require_non_empty_string(payload["universe_name"], "universe_name")
    sector = _optional_string(payload.get("sector"), "sector")
    notes = _optional_string(payload.get("notes"), "notes")

    companies_payload = payload["companies"]
    if not isinstance(companies_payload, list) or not companies_payload:
        raise CompanyConfigError("companies must be a non-empty list")

    companies = tuple(_load_company(entry) for entry in companies_payload)
    return CompanyUniverse(
        universe_name=universe_name,
        companies=companies,
        sector=sector,
        notes=notes,
    )


def _load_company(entry: Any) -> CompanyConfig:
    if not isinstance(entry, dict):
        raise CompanyConfigError("Each company entry must be a mapping")

    unknown_fields = set(entry) - (_COMPANY_REQUIRED_FIELDS | _COMPANY_OPTIONAL_FIELDS)
    if unknown_fields:
        raise CompanyConfigError(
            f"Unknown company fields in companies config: {sorted(unknown_fields)}"
        )

    missing_fields = _COMPANY_REQUIRED_FIELDS - set(entry)
    if missing_fields:
        raise CompanyConfigError(
            f"Missing required company fields in companies config: {sorted(missing_fields)}"
        )

    name = _require_non_empty_string(entry["name"], "name")
    ticker = _require_non_empty_string(entry["ticker"], "ticker").upper()
    cik = normalize_cik(entry["cik"])
    enabled = _optional_bool(entry.get("enabled"), "enabled", default=True)
    sector = _optional_string(entry.get("sector"), "sector")
    notes = _optional_string(entry.get("notes"), "notes")

    return CompanyConfig(
        name=name,
        ticker=ticker,
        cik=cik,
        enabled=enabled,
        sector=sector,
        notes=notes,
    )


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompanyConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CompanyConfigError(f"{field_name} must be a non-empty string when provided")
    return value.strip()


def _optional_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise CompanyConfigError(f"{field_name} must be a boolean when provided")
    return value
