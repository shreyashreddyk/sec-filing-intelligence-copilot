"""Runtime configuration helpers for the Streamlit frontend."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendTimeouts:
    """Endpoint-specific backend timeouts used by the Streamlit client."""

    status_seconds: float = 10.0
    query_seconds: float = 180.0
    retrieve_debug_seconds: float = 180.0
    ingest_seconds: float = 900.0


def load_frontend_timeouts_from_env() -> FrontendTimeouts:
    """Load Streamlit timeout configuration from environment variables."""

    return FrontendTimeouts(
        status_seconds=_read_timeout_seconds("SEC_COPILOT_UI_STATUS_TIMEOUT_SECONDS", 10.0),
        query_seconds=_read_timeout_seconds("SEC_COPILOT_UI_QUERY_TIMEOUT_SECONDS", 180.0),
        retrieve_debug_seconds=_read_timeout_seconds("SEC_COPILOT_UI_RETRIEVE_DEBUG_TIMEOUT_SECONDS", 180.0),
        ingest_seconds=_read_timeout_seconds("SEC_COPILOT_UI_INGEST_TIMEOUT_SECONDS", 900.0),
    )


def _read_timeout_seconds(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0 seconds.")
    return value


__all__ = ["FrontendTimeouts", "load_frontend_timeouts_from_env"]
