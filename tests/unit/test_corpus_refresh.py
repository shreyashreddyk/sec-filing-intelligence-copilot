from __future__ import annotations

from pathlib import Path

from sec_copilot.config.runtime import CorpusRefreshSettings
from sec_copilot.ops.corpus_refresh import build_index_args, build_ingest_args, main


def _settings(tmp_path: Path) -> CorpusRefreshSettings:
    return CorpusRefreshSettings(
        companies_config_path=tmp_path / "configs" / "companies.sample.yaml",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "artifacts" / "chroma",
        form_types=("10-K", "10-Q"),
        annual_limit=2,
        quarterly_limit=4,
    )


def test_build_ingest_args_uses_existing_cli_shape(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert build_ingest_args(settings) == [
        "run",
        "--companies-config",
        str(settings.companies_config_path),
        "--data-dir",
        str(settings.data_dir),
        "--form-types",
        "10-K",
        "10-Q",
        "--annual-limit",
        "2",
        "--quarterly-limit",
        "4",
    ]


def test_build_index_args_uses_existing_cli_shape(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert build_index_args(settings) == [
        "--data-dir",
        str(settings.data_dir),
        "--persist-directory",
        str(settings.chroma_dir),
        "index",
        "--mode",
        "rebuild",
    ]


def test_corpus_refresh_dry_run_prints_exact_command_sequence(tmp_path: Path, monkeypatch, capsys) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.load_corpus_refresh_settings_from_env", lambda: settings)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.ingest_cli.main", lambda argv: 0)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.retrieval_cli.main", lambda argv: 0)

    exit_code = main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Corpus refresh command sequence:" in captured.out
    assert (
        f"python -m sec_copilot.ingest.cli run --companies-config {settings.companies_config_path} "
        f"--data-dir {settings.data_dir} --form-types 10-K 10-Q --annual-limit 2 --quarterly-limit 4"
    ) in captured.out
    assert (
        f"python -m sec_copilot.retrieval.cli --data-dir {settings.data_dir} "
        f"--persist-directory {settings.chroma_dir} index --mode rebuild"
    ) in captured.out


def test_corpus_refresh_runs_ingest_then_index(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.load_corpus_refresh_settings_from_env", lambda: settings)

    def fake_ingest(argv: list[str]) -> int:
        calls.append(("ingest", argv))
        return 0

    def fake_index(argv: list[str]) -> int:
        calls.append(("index", argv))
        return 0

    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.ingest_cli.main", fake_ingest)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.retrieval_cli.main", fake_index)

    exit_code = main([])

    assert exit_code == 0
    assert calls == [
        ("ingest", build_ingest_args(settings)),
        ("index", build_index_args(settings)),
    ]


def test_corpus_refresh_stops_when_ingest_fails(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.load_corpus_refresh_settings_from_env", lambda: settings)

    def fake_ingest(argv: list[str]) -> int:
        calls.append("ingest")
        return 1

    def fake_index(argv: list[str]) -> int:
        calls.append("index")
        return 0

    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.ingest_cli.main", fake_ingest)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.retrieval_cli.main", fake_index)

    exit_code = main([])

    assert exit_code == 1
    assert calls == ["ingest"]


def test_corpus_refresh_propagates_index_failure(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.load_corpus_refresh_settings_from_env", lambda: settings)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.ingest_cli.main", lambda argv: 0)
    monkeypatch.setattr("sec_copilot.ops.corpus_refresh.retrieval_cli.main", lambda argv: 3)

    assert main([]) == 3


def test_corpus_refresh_reports_configuration_errors(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sec_copilot.ops.corpus_refresh.load_corpus_refresh_settings_from_env",
        lambda: (_ for _ in ()).throw(ValueError("bad refresh config")),
    )

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error: bad refresh config" in captured.err
