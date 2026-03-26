from __future__ import annotations

from datetime import date

import pytest

from sec_copilot.api.models import BuildInfoResponse
from sec_copilot.frontend.presenters import (
    build_ingest_request,
    build_query_request,
    configured_company_tickers,
    resolve_scope_options,
)
from sec_copilot.frontend.runtime import FrontendTimeouts, load_frontend_timeouts_from_env
from sec_copilot.frontend.starter_queries import STARTER_QUERIES


def test_resolve_scope_options_prefers_indexed_scope() -> None:
    build_info = BuildInfoResponse.model_validate(
        {
            "status": "ok",
            "service": "sec-live",
            "version": "0.1.0",
            "retrieve_ready": True,
            "query_ready": True,
            "configured_provider": "openai",
            "effective_provider": "mock",
            "provider_fallback_enabled": True,
            "provider_fallback_active": True,
            "provider_fallback_reason": "missing_openai_api_key",
            "prompt_name": "grounded_answer_v3",
            "prompt_version": "v3_hybrid_reranked_grounded",
            "collection_name": "sec_semis_v1",
            "persist_directory": "artifacts/chroma",
            "coverage_status": "full",
            "target_scope": {
                "companies": ["NVDA", "AMD", "INTC", "AVGO", "QCOM"],
                "form_types": ["10-K", "10-Q"],
                "annual_limit": 2,
                "quarterly_limit": 4,
            },
            "indexed_scope": {
                "companies": ["AMD", "INTC", "NVDA"],
                "form_types": ["10-K", "10-Q"],
                "entries": [],
                "document_count": 6,
                "chunk_count": 57,
            },
            "index_status": "fresh",
            "processed_corpus_fingerprint": "abc",
            "indexed_corpus_fingerprint": "abc",
            "index_build_metadata": None,
            "last_ingest_completed_at": None,
            "last_index_refresh_at": None,
            "warnings": [],
        }
    )

    options = resolve_scope_options(
        build_info,
        companies_config_path="configs/companies.yaml",
    )
    assert options.source == "indexed_scope"
    assert options.companies == ("AMD", "INTC", "NVDA")
    assert options.form_types == ("10-K", "10-Q")


def test_resolve_scope_options_uses_target_scope_before_config() -> None:
    build_info = BuildInfoResponse.model_validate(
        {
            "status": "ok",
            "service": "sec-live",
            "version": "0.1.0",
            "retrieve_ready": False,
            "query_ready": False,
            "configured_provider": "openai",
            "effective_provider": "mock",
            "provider_fallback_enabled": True,
            "provider_fallback_active": True,
            "provider_fallback_reason": "missing_openai_api_key",
            "prompt_name": "grounded_answer_v3",
            "prompt_version": "v3_hybrid_reranked_grounded",
            "collection_name": "sec_semis_v1",
            "persist_directory": "artifacts/chroma",
            "coverage_status": "uninitialized",
            "target_scope": {
                "companies": ["NVDA", "AMD", "INTC", "AVGO", "QCOM"],
                "form_types": ["10-K", "10-Q"],
                "annual_limit": 2,
                "quarterly_limit": 4,
            },
            "indexed_scope": {
                "companies": [],
                "form_types": [],
                "entries": [],
                "document_count": 0,
                "chunk_count": 0,
            },
            "index_status": "missing",
            "processed_corpus_fingerprint": None,
            "indexed_corpus_fingerprint": None,
            "index_build_metadata": None,
            "last_ingest_completed_at": None,
            "last_index_refresh_at": None,
            "warnings": [],
        }
    )

    options = resolve_scope_options(
        build_info,
        companies_config_path="configs/companies.yaml",
    )
    assert options.source == "target_scope"
    assert options.companies == ("NVDA", "AMD", "INTC", "AVGO", "QCOM")
    assert options.form_types == ("10-K", "10-Q")


def test_resolve_scope_options_falls_back_to_config() -> None:
    options = resolve_scope_options(
        object(),
        companies_config_path="configs/companies.yaml",
    )
    assert options.source == "config_fallback"
    assert options.companies == ("NVDA", "AMD", "INTC", "AVGO", "QCOM")
    assert options.form_types == ("10-K", "10-Q")


def test_configured_company_tickers_reads_live_config() -> None:
    assert configured_company_tickers("configs/companies.yaml") == ("NVDA", "AMD", "INTC", "AVGO", "QCOM")


def test_build_query_request_applies_optional_date_filters() -> None:
    request = build_query_request(
        question="What export control risks does NVIDIA describe?",
        tickers=["NVDA"],
        form_types=["10-K"],
        use_date_filter=True,
        filing_date_from=date(2025, 2, 26),
        filing_date_to=date(2026, 2, 25),
        debug=False,
    )
    assert request.filters.filing_date_from == date(2025, 2, 26)
    assert request.filters.filing_date_to == date(2026, 2, 25)


def test_build_ingest_request_normalizes_blank_user_agent() -> None:
    request = build_ingest_request(
        companies=["NVDA", "AMD"],
        form_types=["10-K", "10-Q"],
        annual_limit=1,
        quarterly_limit=1,
        force_refresh=False,
        user_agent="   ",
        index_mode="rebuild",
    )
    assert request.user_agent is None
    assert request.annual_limit == 1
    assert request.form_types == ["10-K", "10-Q"]


def test_starter_queries_match_live_app_defaults() -> None:
    assert len(STARTER_QUERIES) == 3
    questions = {query.question for query in STARTER_QUERIES}
    assert "What export control risks does NVIDIA describe?" in questions
    assert "What does NVIDIA say about AI infrastructure and accelerated computing?" in questions
    assert "What does AMD say about supply chain risk?" in questions


def test_load_frontend_timeouts_from_env_uses_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SEC_COPILOT_UI_STATUS_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SEC_COPILOT_UI_QUERY_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SEC_COPILOT_UI_RETRIEVE_DEBUG_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SEC_COPILOT_UI_INGEST_TIMEOUT_SECONDS", raising=False)

    assert load_frontend_timeouts_from_env() == FrontendTimeouts()


def test_load_frontend_timeouts_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SEC_COPILOT_UI_STATUS_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("SEC_COPILOT_UI_QUERY_TIMEOUT_SECONDS", "210")
    monkeypatch.setenv("SEC_COPILOT_UI_RETRIEVE_DEBUG_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("SEC_COPILOT_UI_INGEST_TIMEOUT_SECONDS", "1200")

    assert load_frontend_timeouts_from_env() == FrontendTimeouts(
        status_seconds=12.0,
        query_seconds=210.0,
        retrieve_debug_seconds=240.0,
        ingest_seconds=1200.0,
    )


def test_load_frontend_timeouts_from_env_rejects_non_positive_values(monkeypatch) -> None:
    monkeypatch.setenv("SEC_COPILOT_UI_QUERY_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError, match="SEC_COPILOT_UI_QUERY_TIMEOUT_SECONDS"):
        load_frontend_timeouts_from_env()
