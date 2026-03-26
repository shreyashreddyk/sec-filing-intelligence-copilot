"""Lightweight structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any


LOGGER = logging.getLogger("sec_copilot.query")
API_LOGGER = logging.getLogger("sec_copilot.api")


def configure_logging(level: str = "INFO") -> None:
    """Configure simple structured logging for local API runs."""

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def log_api_event(event: dict[str, Any]) -> None:
    """Emit one structured API event."""

    API_LOGGER.info(json.dumps(event, default=str, sort_keys=True))


def log_query_event(event: dict[str, Any]) -> None:
    """Emit one structured query event for lightweight debugging."""

    LOGGER.info(json.dumps(event, default=str, sort_keys=True))


__all__ = ["API_LOGGER", "LOGGER", "configure_logging", "log_api_event", "log_query_event"]
