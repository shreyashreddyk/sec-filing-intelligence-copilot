"""Lightweight structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any


LOGGER = logging.getLogger("sec_copilot.query")


def log_query_event(event: dict[str, Any]) -> None:
    """Emit one structured query event for lightweight debugging."""

    LOGGER.info(json.dumps(event, default=str, sort_keys=True))


__all__ = ["LOGGER", "log_query_event"]
