"""Helpers for Hugging Face Hub authentication configuration."""

from __future__ import annotations

import os


def resolve_huggingface_token() -> str | None:
    """Return the configured Hugging Face token when one is available."""

    token = os.getenv("HF_TOKEN", "").strip()
    return token or None


__all__ = ["resolve_huggingface_token"]
