"""Typed loading for offline evaluation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sec_copilot.eval.schemas import EvalConfig


class EvalConfigError(ValueError):
    """Raised when the eval configuration file is invalid."""


def load_eval_config(path: str | Path = "configs/eval.yaml") -> EvalConfig:
    """Load the eval configuration from YAML."""

    return EvalConfig.model_validate(_load_yaml(path))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise EvalConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise EvalConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise EvalConfigError(f"Config file must contain a top-level mapping: {config_path}")
    return payload


__all__ = ["EvalConfigError", "load_eval_config"]
