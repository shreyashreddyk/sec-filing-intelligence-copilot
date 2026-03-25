"""Single SEC HTTP client with fair-access behavior and cache-first helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from sec_copilot.ingest.constants import PLACEHOLDER_USER_AGENT


class SecClientError(RuntimeError):
    """Base SEC client error."""


class SecClientPreflightError(SecClientError):
    """Raised when the SEC client cannot be safely initialized."""


class SecRequestError(SecClientError):
    """Raised when a SEC request fails after retries."""


def validate_user_agent(user_agent: str | None) -> str:
    """Validate the configured SEC user agent."""

    normalized = (user_agent or "").strip()
    lowered = normalized.lower()
    if not normalized:
        raise SecClientPreflightError("SEC user agent is required")
    if normalized == PLACEHOLDER_USER_AGENT or "your name" in lowered or "example.com" in lowered:
        raise SecClientPreflightError("SEC user agent must not use the placeholder example")
    return normalized


class SecClient:
    """Cache-aware SEC client that centralizes request policy."""

    def __init__(
        self,
        user_agent: str,
        rate_limit_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._user_agent = validate_user_agent(user_agent)
        self._rate_limit_seconds = rate_limit_seconds
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._last_request_monotonic: float | None = None
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self._user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def get_json(self, url: str, cache_path: Path, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch JSON from SEC or load it from the local cache."""

        text = self.get_text(url, cache_path=cache_path, force_refresh=force_refresh)
        return json.loads(text)

    def get_text(self, url: str, cache_path: Path, force_refresh: bool = False) -> str:
        """Fetch text from SEC or load it from the local cache."""

        if cache_path.exists() and not force_refresh:
            return cache_path.read_text(encoding="utf-8")

        text = self._request_text(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return text

    def _request_text(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                self._sleep_for_rate_limit()
                response = self._session.get(url, timeout=self._timeout_seconds)
                self._last_request_monotonic = time.monotonic()

                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    if attempt == self._max_retries:
                        response.raise_for_status()
                    self._sleep_for_retry(response.headers.get("Retry-After"), attempt)
                    continue

                response.raise_for_status()
                response.encoding = response.encoding or "utf-8"
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
                self._sleep_for_retry(None, attempt)

        raise SecRequestError(f"Failed SEC request for {url}") from last_error

    def _sleep_for_rate_limit(self) -> None:
        if self._last_request_monotonic is None:
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        remaining = self._rate_limit_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _sleep_for_retry(self, retry_after: str | None, attempt: int) -> None:
        if retry_after and retry_after.isdigit():
            time.sleep(float(retry_after))
            return
        time.sleep(self._rate_limit_seconds * attempt)
