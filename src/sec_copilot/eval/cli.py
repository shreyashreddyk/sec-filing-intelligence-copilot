"""CLI for offline evaluation over the tracked SEC filing QA dataset."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from sec_copilot.config import load_prompt_catalog, load_retrieval_config
from sec_copilot.eval.artifacts import resolve_output_dir, write_eval_artifacts
from sec_copilot.eval.config import EvalConfigError, load_eval_config
from sec_copilot.eval.dataset import EvalDatasetError, load_eval_dataset
from sec_copilot.eval.runner import run_eval


def main(argv: list[str] | None = None) -> int:
    """Run the offline eval CLI."""

    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    try:
        eval_config = load_eval_config(args.eval_config)
        retrieval_config = load_retrieval_config(args.retrieval_config)
        prompt_catalog = load_prompt_catalog(args.prompts_config)
        dataset = load_eval_dataset(eval_config.dataset_path)
        resolved = _resolve_run_settings(args, eval_config)
        _validate_combinations(args, resolved)
        if not _select_count(dataset.examples, resolved["subset"]):
            raise ValueError(f"Subset {resolved['subset']!r} selected zero eval examples.")
    except (EvalConfigError, EvalDatasetError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    output_dir = resolve_output_dir(eval_config.output_root, resolved["output_dir"])
    result = run_eval(
        eval_config=eval_config,
        retrieval_config=retrieval_config,
        prompt_catalog=prompt_catalog,
        dataset=dataset,
        dataset_path=eval_config.dataset_path,
        corpus_path=eval_config.corpus_path,
        subset=resolved["subset"],
        mode=resolved["mode"],
        provider=resolved["provider"],
        score_backend=resolved["score_backend"],
        output_dir=output_dir,
    )
    paths = write_eval_artifacts(result, output_dir)
    print(f"results_json={paths['results_json']}")
    print(f"report_md={paths['report_md']}")

    if result.status == "execution_failed":
        return 3
    if resolved["fail_on_thresholds"] and result.status == "threshold_failed":
        return 1
    return 0


def _resolve_run_settings(args: argparse.Namespace, eval_config) -> dict[str, object]:
    mode = args.mode or eval_config.default_mode
    subset = args.subset or eval_config.default_subset
    provider = None if mode == "retrieval" else (args.provider or eval_config.default_provider)
    score_backend = None if mode == "retrieval" else (args.score_backend or eval_config.default_score_backend)
    fail_on_thresholds = True if args.fail_on_thresholds is None else args.fail_on_thresholds
    return {
        "subset": subset,
        "mode": mode,
        "provider": provider,
        "score_backend": score_backend,
        "output_dir": args.output_dir,
        "fail_on_thresholds": fail_on_thresholds,
    }


def _validate_combinations(args: argparse.Namespace, resolved: dict[str, object]) -> None:
    mode = resolved["mode"]
    provider = resolved["provider"]
    score_backend = resolved["score_backend"]
    fail_on_thresholds = bool(resolved["fail_on_thresholds"])

    if mode == "retrieval" and args.provider is not None:
        raise ValueError("mode=retrieval does not accept an explicit provider.")
    if mode == "retrieval" and args.score_backend in {"ragas", "both"}:
        raise ValueError("mode=retrieval does not support ragas scoring.")
    if provider in {"reference", "mock"} and score_backend in {"ragas", "both"}:
        raise ValueError("provider=reference or provider=mock does not support ragas scoring.")
    if provider == "mock" and fail_on_thresholds:
        raise ValueError("provider=mock cannot be used with fail-on-thresholds=true.")
    if provider == "openai" and mode in {"answer", "full"} and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai.")


def _select_count(examples, subset: str) -> int:
    if subset == "full":
        return len(examples)
    return sum(1 for example in examples if subset in example.tags)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline SEC filing eval CLI.")
    parser.add_argument("--eval-config", default="configs/eval.yaml", help="Path to eval config YAML.")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml", help="Path to retrieval config YAML.")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml", help="Path to prompts config YAML.")

    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run the offline eval harness.")
    run_parser.add_argument("--subset", help="Example subset tag to run, or 'full'.")
    run_parser.add_argument("--mode", choices=("retrieval", "answer", "full"), help="Eval execution mode.")
    run_parser.add_argument("--provider", choices=("reference", "mock", "openai"), help="Answer provider backend.")
    run_parser.add_argument(
        "--score-backend",
        choices=("deterministic", "ragas", "both"),
        help="Answer scoring backend.",
    )
    run_parser.add_argument("--output-dir", help="Optional explicit output directory.")
    run_parser.add_argument(
        "--fail-on-thresholds",
        type=_parse_bool,
        default=None,
        help="Whether blocking threshold failures should return exit code 1.",
    )
    return parser


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, yes, no, 1, 0")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
