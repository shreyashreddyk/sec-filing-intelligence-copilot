"""Artifact writing and Markdown report rendering for eval runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json

from sec_copilot.eval.schemas import EvalRunResult


def resolve_output_dir(output_root: str | Path, output_dir: str | Path | None = None) -> Path:
    """Resolve the output directory for one eval run."""

    if output_dir is not None:
        path = Path(output_dir)
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = Path(output_root) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_eval_artifacts(result: EvalRunResult, output_dir: str | Path) -> dict[str, str]:
    """Write machine-readable and Markdown eval artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "results.json"
    report_path = output_path / "report.md"

    updated = result.model_copy(
        update={
            "paths": {
                **result.paths,
                "output_dir": str(output_path),
                "results_json": str(result_path),
                "report_md": str(report_path),
            }
        }
    )

    result_path.write_text(
        json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown_report(updated), encoding="utf-8")
    return updated.paths


def render_markdown_report(result: EvalRunResult) -> str:
    """Render a stable human-readable summary of one eval run."""

    lines = [
        "# Eval Report",
        "",
        "## Run Summary",
        f"- run_id: `{result.run_id}`",
        f"- status: `{result.status}`",
        f"- subset: `{result.subset}`",
        f"- mode: `{result.mode}`",
        f"- provider: `{result.provider}`",
        f"- score_backend: `{result.score_backend}`",
        "",
        "## Blocking Threshold Summary",
    ]

    blocking_checks = result.thresholds.get("blocking", [])
    if blocking_checks:
        for check in blocking_checks:
            lines.append(
                f"- `{check['name']}`: passed={check['passed']} actual={check['actual_value']} target={check['operator']} {check['expected_value']}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Retrieval Metrics",
            *_render_metric_block(result.retrieval.metrics_overall),
            "",
            "## Answer Metrics",
            *_render_metric_block(result.answer.metrics_overall),
            "",
            "## Category Slices",
        ]
    )

    category_names = sorted(
        set(result.retrieval.metrics_by_category) | set(result.answer.metrics_by_category)
    )
    if category_names:
        for category in category_names:
            lines.append(f"### {category}")
            lines.append("- retrieval:")
            lines.extend(f"  - {line}" for line in _render_metric_block(result.retrieval.metrics_by_category.get(category)))
            lines.append("- answer:")
            lines.extend(f"  - {line}" for line in _render_metric_block(result.answer.metrics_by_category.get(category)))
    else:
        lines.append("- none")

    lines.extend(["", "## Failed Examples"])
    failed_examples = _failed_examples(result)
    if failed_examples:
        for example in failed_examples:
            lines.append(f"- `{example}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings and Errors"])
    if result.warnings:
        for warning in result.warnings:
            lines.append(f"- warning: {warning}")
    if result.errors:
        for error in result.errors:
            lines.append(f"- error: {error}")
    if not result.warnings and not result.errors:
        lines.append("- none")

    lines.append("")
    return "\n".join(lines)


def _render_metric_block(metric_aggregate) -> list[str]:
    if metric_aggregate is None:
        return ["none"]
    lines = [f"eligible_example_count={metric_aggregate.eligible_example_count}"]
    if metric_aggregate.values:
        for key, value in sorted(metric_aggregate.values.items()):
            lines.append(f"{key}={value}")
    else:
        lines.append("no metrics")
    return lines


def _failed_examples(result: EvalRunResult) -> list[str]:
    failed: list[str] = []
    for example in result.retrieval.examples:
        if example.get("errors"):
            failed.append(example["example_id"])
    for example in result.answer.examples:
        if example.get("errors") or example.get("metrics", {}).get("citation_validity") == 0.0:
            if example["example_id"] not in failed:
                failed.append(example["example_id"])
    return failed


__all__ = ["render_markdown_report", "resolve_output_dir", "write_eval_artifacts"]
