"""Evaluation layer package."""

from sec_copilot.eval.config import EvalConfigError, load_eval_config
from sec_copilot.eval.dataset import EvalDatasetError, load_eval_dataset
from sec_copilot.eval.schemas import (
    AnswerExampleResult,
    AnswerExecutionTrace,
    EvalConfig,
    EvalDataset,
    EvalExample,
    EvalRunResult,
    RetrievalExampleResult,
    ThresholdCheck,
)

__all__ = [
    "AnswerExampleResult",
    "AnswerExecutionTrace",
    "EvalConfig",
    "EvalConfigError",
    "EvalDataset",
    "EvalDatasetError",
    "EvalExample",
    "EvalRunResult",
    "RetrievalExampleResult",
    "ThresholdCheck",
    "load_eval_config",
    "load_eval_dataset",
]
