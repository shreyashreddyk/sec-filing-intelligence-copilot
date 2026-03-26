"""Typed loading for the tracked SEC filing QA eval dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sec_copilot.eval.schemas import EvalDataset


class EvalDatasetError(ValueError):
    """Raised when the eval dataset file is invalid."""


def load_eval_dataset(path: str | Path) -> EvalDataset:
    """Load the eval dataset from YAML."""

    return EvalDataset.model_validate(_load_yaml(path))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    try:
        payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise EvalDatasetError(f"Dataset file not found: {dataset_path}") from exc
    except yaml.YAMLError as exc:
        raise EvalDatasetError(f"Invalid YAML in {dataset_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise EvalDatasetError(f"Dataset file must contain a top-level mapping: {dataset_path}")
    return payload


__all__ = ["EvalDatasetError", "load_eval_dataset"]
