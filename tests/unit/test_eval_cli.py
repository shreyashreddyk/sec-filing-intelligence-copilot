from __future__ import annotations

import json
from pathlib import Path

from sec_copilot.eval.cli import _resolve_ragas_config, main
from sec_copilot.eval.schemas import EvalRagasConfig


def test_eval_cli_rejects_mock_provider_with_threshold_gating(tmp_path: Path) -> None:
    output_dir = tmp_path / "mock_eval"

    exit_code = main(
        [
            "--eval-config",
            "configs/eval.yaml",
            "run",
            "--mode",
            "full",
            "--provider",
            "mock",
            "--score-backend",
            "deterministic",
            "--output-dir",
            str(output_dir),
            "--fail-on-thresholds",
            "true",
        ]
    )

    assert exit_code == 2


def test_eval_cli_rejects_ragas_with_reference_provider(tmp_path: Path) -> None:
    output_dir = tmp_path / "bad_eval"

    exit_code = main(
        [
            "--eval-config",
            "configs/eval.yaml",
            "run",
            "--mode",
            "full",
            "--provider",
            "reference",
            "--score-backend",
            "both",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2


def test_eval_cli_smoke_subset_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "ci_smoke_eval"

    exit_code = main(
        [
            "--eval-config",
            "configs/eval.yaml",
            "--retrieval-config",
            "configs/retrieval.yaml",
            "--prompts-config",
            "configs/prompts.yaml",
            "run",
            "--subset",
            "ci_smoke",
            "--mode",
            "full",
            "--provider",
            "reference",
            "--score-backend",
            "deterministic",
            "--output-dir",
            str(output_dir),
        ]
    )

    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"

    assert exit_code == 0
    assert results_path.exists()
    assert report_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sec_eval_results.v1"
    assert payload["subset"] == "ci_smoke"
    assert payload["mode"] == "full"
    assert payload["provider"] == "reference"
    assert payload["thresholds"]["blocking"]
    assert payload["retrieval"]["executed"] is True
    assert payload["answer"]["executed"] is True


def test_resolve_ragas_config_applies_cli_overrides() -> None:
    base = EvalRagasConfig()
    resolved = {
        "ragas_model": "gpt-5-mini",
        "ragas_max_completion_tokens": 8192,
        "ragas_answer_relevancy_strictness": 2,
        "ragas_reasoning_effort": "low",
    }

    effective = _resolve_ragas_config(base, resolved)

    assert effective.model_name == "gpt-5-mini"
    assert effective.max_completion_tokens == 8192
    assert effective.answer_relevancy_strictness == 2
    assert effective.reasoning_effort == "low"
