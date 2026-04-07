"""Lightweight structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

from sec_copilot.config.runtime import read_env_string


LOGGER = logging.getLogger("sec_copilot.query")
API_LOGGER = logging.getLogger("sec_copilot.api")


def configure_logging(level: str | None = None) -> None:
    """Configure simple structured logging for local API runs."""

    target_level = (level or read_env_string("SEC_COPILOT_LOG_LEVEL", "INFO") or "INFO").upper()
    resolved_level = getattr(logging, target_level, logging.INFO)
    logging.basicConfig(level=resolved_level)
    logging.getLogger().setLevel(resolved_level)
    API_LOGGER.setLevel(resolved_level)
    LOGGER.setLevel(resolved_level)


def log_api_event(event: dict[str, Any]) -> None:
    """Emit one structured API event."""

    API_LOGGER.info(json.dumps(event, default=str, sort_keys=True))


def log_query_event(event: dict[str, Any]) -> None:
    """Emit one structured query event for lightweight debugging."""

    LOGGER.info(json.dumps(event, default=str, sort_keys=True))


__all__ = ["API_LOGGER", "LOGGER", "configure_logging", "log_api_event", "log_query_event"]
