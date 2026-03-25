"""Normalization helpers shared across schemas and retrieval code."""

from __future__ import annotations

from datetime import date
import re


def normalize_ticker(value: str) -> str:
    """Normalize ticker values to an uppercase ASCII-ish representation."""

    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("Ticker values must be non-empty")
    if not re.fullmatch(r"[A-Z0-9.\-]+", normalized):
        raise ValueError(f"Unsupported ticker format: {value!r}")
    return normalized


def normalize_form_type(value: str) -> str:
    """Normalize common SEC form-type variants to a canonical representation."""

    compact = re.sub(r"[\s_]", "", value.strip().upper())
    if not compact:
        raise ValueError("Form type values must be non-empty")
    if "-" not in compact:
        compact = re.sub(r"^(\d+)([A-Z].*)$", r"\1-\2", compact)
    return compact


def filing_date_to_ordinal(value: date) -> int:
    """Convert a filing date into a filterable ordinal integer."""

    return value.toordinal()
