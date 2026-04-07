from __future__ import annotations

from sec_copilot.config.runtime import (
    default_project_root,
    load_api_runtime_settings_from_env,
    load_runtime_paths_from_env,
)
from sec_copilot.frontend.runtime import load_frontend_backend_url_from_env


def test_runtime_paths_default_to_repo_root(monkeypatch) -> None:
    monkeypatch.delenv("SEC_COPILOT_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("SEC_COPILOT_CONFIG_DIR", raising=False)
    monkeypatch.delenv("SEC_COPILOT_COMPANIES_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_RETRIEVAL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_PROMPTS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_EVAL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_DATA_DIR", raising=False)
    monkeypatch.delenv("SEC_COPILOT_CHROMA_DIR", raising=False)

    project_root = default_project_root()
    paths = load_runtime_paths_from_env()

    assert paths.project_root == project_root
    assert paths.config_dir == project_root / "configs"
    assert paths.data_dir == project_root / "data"
    assert paths.chroma_dir == project_root / "artifacts" / "chroma"
    assert paths.companies_config_path == project_root / "configs" / "companies.yaml"


def test_runtime_paths_honor_project_root_override(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "container-root"
    monkeypatch.setenv("SEC_COPILOT_PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("SEC_COPILOT_CONFIG_DIR", raising=False)
    monkeypatch.delenv("SEC_COPILOT_COMPANIES_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_RETRIEVAL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_PROMPTS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_EVAL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SEC_COPILOT_DATA_DIR", raising=False)
    monkeypatch.delenv("SEC_COPILOT_CHROMA_DIR", raising=False)

    paths = load_runtime_paths_from_env()

    assert default_project_root() == project_root.resolve()
    assert paths.project_root == project_root.resolve()
    assert paths.config_dir == project_root.resolve() / "configs"
    assert paths.data_dir == project_root.resolve() / "data"
    assert paths.chroma_dir == project_root.resolve() / "artifacts" / "chroma"


def test_api_runtime_settings_disable_admin_routes_in_production_by_default(monkeypatch) -> None:
    monkeypatch.setenv("SEC_COPILOT_ENV", "production")
    monkeypatch.delenv("SEC_COPILOT_ENABLE_ADMIN_ROUTES", raising=False)

    settings = load_api_runtime_settings_from_env()

    assert settings.environment == "production"
    assert settings.enable_admin_routes is False


def test_api_runtime_settings_apply_explicit_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SEC_COPILOT_ENV", "production")
    monkeypatch.setenv("SEC_COPILOT_ENABLE_ADMIN_ROUTES", "true")
    monkeypatch.setenv("SEC_COPILOT_LOG_LEVEL", "debug")
    monkeypatch.setenv("SEC_COPILOT_OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("SEC_COPILOT_CHROMA_DIR", str(tmp_path / "runtime-chroma"))

    settings = load_api_runtime_settings_from_env()

    assert settings.log_level == "debug"
    assert settings.enable_admin_routes is True
    assert settings.openai_model_override == "gpt-4.1-mini"
    assert settings.chroma_dir_override == tmp_path / "runtime-chroma"


def test_frontend_backend_url_loader_uses_override(monkeypatch) -> None:
    monkeypatch.setenv("SEC_COPILOT_UI_BACKEND_URL", "http://sec-copilot-api:8000")

    assert load_frontend_backend_url_from_env() == "http://sec-copilot-api:8000"
