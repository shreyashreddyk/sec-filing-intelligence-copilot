"""Typed helper functions that keep Streamlit UI logic small and explicit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from sec_copilot.api.models import BuildInfoResponse, IngestRunRequest
from sec_copilot.config import load_company_universe
from sec_copilot.schemas.retrieval import QueryRequest


@dataclass(frozen=True)
class ScopeOptions:
    """Available filter options shown in the UI."""

    companies: tuple[str, ...]
    form_types: tuple[str, ...]
    source: Literal["indexed_scope", "target_scope", "config_fallback"]


def resolve_scope_options(
    build_info_result: BuildInfoResponse | object,
    *,
    companies_config_path: str | Path,
    fallback_form_types: tuple[str, ...] = ("10-K", "10-Q"),
) -> ScopeOptions:
    """Prefer indexed scope, then target scope, then the committed companies config."""

    if isinstance(build_info_result, BuildInfoResponse):
        if build_info_result.indexed_scope.companies or build_info_result.indexed_scope.form_types:
            companies = tuple(build_info_result.indexed_scope.companies)
            form_types = tuple(build_info_result.indexed_scope.form_types or fallback_form_types)
            return ScopeOptions(companies=companies, form_types=form_types, source="indexed_scope")
        if build_info_result.target_scope.companies or build_info_result.target_scope.form_types:
            companies = tuple(build_info_result.target_scope.companies)
            form_types = tuple(build_info_result.target_scope.form_types or fallback_form_types)
            return ScopeOptions(companies=companies, form_types=form_types, source="target_scope")

    universe = load_company_universe(companies_config_path)
    companies = tuple(company.ticker for company in universe.enabled_companies())
    return ScopeOptions(
        companies=companies,
        form_types=fallback_form_types,
        source="config_fallback",
    )


def configured_company_tickers(companies_config_path: str | Path) -> tuple[str, ...]:
    """Return enabled configured tickers for bootstrap controls."""

    universe = load_company_universe(companies_config_path)
    return tuple(company.ticker for company in universe.enabled_companies())


def build_query_request(
    *,
    question: str,
    tickers: list[str],
    form_types: list[str],
    use_date_filter: bool,
    filing_date_from: date | None,
    filing_date_to: date | None,
    debug: bool = False,
) -> QueryRequest:
    """Build a validated query request from Streamlit form state."""

    filters = {
        "tickers": tickers,
        "form_types": form_types,
        "filing_date_from": filing_date_from if use_date_filter else None,
        "filing_date_to": filing_date_to if use_date_filter else None,
    }
    return QueryRequest(
        question=question,
        filters=filters,
        debug=debug,
    )


def build_ingest_request(
    *,
    companies: list[str],
    form_types: list[str],
    annual_limit: int,
    quarterly_limit: int,
    force_refresh: bool,
    user_agent: str,
    index_mode: str,
) -> IngestRunRequest:
    """Build a validated ingest request from Streamlit form state."""

    return IngestRunRequest(
        companies=companies,
        form_types=form_types,
        annual_limit=annual_limit,
        quarterly_limit=quarterly_limit,
        force_refresh=force_refresh,
        user_agent=user_agent.strip() or None,
        index_mode=index_mode,
    )


def safe_json(raw_body: object | str | None) -> str:
    """Return a UI-friendly string representation for raw payloads."""

    if raw_body is None:
        return "None"
    if isinstance(raw_body, str):
        return raw_body
    import json

    return json.dumps(raw_body, indent=2, sort_keys=True, default=str)


__all__ = [
    "ScopeOptions",
    "build_ingest_request",
    "build_query_request",
    "configured_company_tickers",
    "resolve_scope_options",
    "safe_json",
]
