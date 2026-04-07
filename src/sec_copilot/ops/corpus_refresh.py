"""Reusable corpus refresh workflow for deployed query-serving environments."""

from __future__ import annotations

import argparse
import shlex
import sys

from dotenv import load_dotenv

from sec_copilot.config import CorpusRefreshSettings, load_corpus_refresh_settings_from_env
from sec_copilot.ingest import cli as ingest_cli
from sec_copilot.retrieval import cli as retrieval_cli


def main(argv: list[str] | None = None) -> int:
    """Run the env-driven corpus refresh workflow."""

    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_corpus_refresh_settings_from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    ingest_args = build_ingest_args(settings)
    index_args = build_index_args(settings)

    print("Corpus refresh command sequence:")
    print(_format_shell_command("python", "-m", "sec_copilot.ingest.cli", *ingest_args))
    print(_format_shell_command("python", "-m", "sec_copilot.retrieval.cli", *index_args))

    if args.dry_run:
        return 0

    ingest_exit_code = ingest_cli.main(ingest_args)
    if ingest_exit_code != 0:
        return ingest_exit_code
    return retrieval_cli.main(index_args)


def build_ingest_args(settings: CorpusRefreshSettings) -> list[str]:
    """Build the existing ingest CLI arguments for one refresh run."""

    return [
        "run",
        "--companies-config",
        str(settings.companies_config_path),
        "--data-dir",
        str(settings.data_dir),
        "--form-types",
        *settings.form_types,
        "--annual-limit",
        str(settings.annual_limit),
        "--quarterly-limit",
        str(settings.quarterly_limit),
    ]


def build_index_args(settings: CorpusRefreshSettings) -> list[str]:
    """Build the existing retrieval CLI arguments for one refresh run."""

    return [
        "--data-dir",
        str(settings.data_dir),
        "--persist-directory",
        str(settings.chroma_dir),
        "index",
        "--mode",
        "rebuild",
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SEC ingest plus index rebuild for corpus refresh.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded ingest and index command sequence without executing it.",
    )
    return parser


def _format_shell_command(*parts: str) -> str:
    """Render one command line for operator-visible dry-run output."""

    return shlex.join(parts)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
