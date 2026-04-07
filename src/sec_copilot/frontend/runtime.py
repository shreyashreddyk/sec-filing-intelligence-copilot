"""Runtime configuration helpers for the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass

from sec_copilot.config.runtime import read_env_positive_float, read_env_string


@dataclass(frozen=True)
class FrontendTimeouts:
    """Endpoint-specific backend timeouts used by the Streamlit client."""

    status_seconds: float = 10.0
    query_seconds: float = 180.0
    retrieve_debug_seconds: float = 180.0
    ingest_seconds: float = 900.0


def load_frontend_backend_url_from_env() -> str:
    """Load the backend base URL used by the Streamlit frontend."""

    return read_env_string("SEC_COPILOT_UI_BACKEND_URL", "http://127.0.0.1:8000") or "http://127.0.0.1:8000"


def load_frontend_timeouts_from_env() -> FrontendTimeouts:
    """Load Streamlit timeout configuration from environment variables."""

    return FrontendTimeouts(
        status_seconds=read_env_positive_float("SEC_COPILOT_UI_STATUS_TIMEOUT_SECONDS", 10.0),
        query_seconds=read_env_positive_float("SEC_COPILOT_UI_QUERY_TIMEOUT_SECONDS", 180.0),
        retrieve_debug_seconds=read_env_positive_float("SEC_COPILOT_UI_RETRIEVE_DEBUG_TIMEOUT_SECONDS", 180.0),
        ingest_seconds=read_env_positive_float("SEC_COPILOT_UI_INGEST_TIMEOUT_SECONDS", 900.0),
    )


__all__ = [
    "FrontendTimeouts",
    "load_frontend_backend_url_from_env",
    "load_frontend_timeouts_from_env",
]
