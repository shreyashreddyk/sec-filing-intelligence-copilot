"""CLI entrypoint for the SEC ingestion pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sec_copilot.ingest.constants import TARGET_FORMS
from sec_copilot.ingest.pipeline import IngestionConfig, IngestionPreflightError, run_ingestion


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 2

    try:
        config = IngestionConfig(
            companies_config=Path(args.companies_config),
            data_dir=Path(args.data_dir),
            user_agent=args.user_agent or os.getenv("SEC_USER_AGENT", ""),
            annual_limit=args.annual_limit,
            quarterly_limit=args.quarterly_limit,
            companies=tuple(args.company or ()),
            form_types=tuple(args.form_types),
            force_refresh=args.force_refresh,
        )
        summary = run_ingestion(config)
    except IngestionPreflightError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    print(f"Run ID: {summary.run_id}")
    print(f"Status: {summary.status}")
    print(f"Successful filings: {summary.successful_filings}")
    print(f"Failed filings: {summary.failed_filings}")
    print(f"Warnings: {summary.warning_count}")
    print(f"Errors: {summary.error_count}")
    return 0 if summary.error_count == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V1 SEC ingestion pipeline.")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run SEC ingestion")
    run_parser.add_argument(
        "--companies-config",
        default="configs/companies.yaml",
        help="Path to the fixed company universe YAML file.",
    )
    run_parser.add_argument(
        "--data-dir",
        default=os.getenv("SEC_COPILOT_DATA_DIR", "data"),
        help="Base data directory for raw and processed artifacts.",
    )
    run_parser.add_argument(
        "--company",
        action="append",
        help="Optional ticker or company name filter. Repeat to include multiple companies.",
    )
    run_parser.add_argument(
        "--form-types",
        nargs="+",
        default=list(TARGET_FORMS),
        choices=TARGET_FORMS,
        help="Subset of V1 form types to ingest.",
    )
    run_parser.add_argument("--annual-limit", type=int, default=2, help="Maximum exact 10-K filings per company.")
    run_parser.add_argument("--quarterly-limit", type=int, default=4, help="Maximum exact 10-Q filings per company.")
    run_parser.add_argument("--user-agent", help="Descriptive SEC user agent override.")
    run_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore local raw caches and refetch from SEC.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
