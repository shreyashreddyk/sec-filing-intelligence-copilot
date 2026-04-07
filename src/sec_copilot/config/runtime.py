"""Runtime environment and path helpers for local and container startup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_project_root() -> Path:
    """Return the repository root used for runtime-relative paths."""

    override = os.getenv("SEC_COPILOT_PROJECT_ROOT")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT


def read_env_string(name: str, default: str | None = None) -> str | None:
    """Read one string environment variable, treating blank values as unset."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip()
    return normalized or default


def read_env_bool(name: str, default: bool) -> bool:
    """Read one boolean environment variable."""

    raw_value = read_env_string(name)
    if raw_value is None:
        return default
    lowered = raw_value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true, false, yes, no, on, off, 1, or 0.")


def read_env_positive_float(name: str, default: float) -> float:
    """Read one positive float environment variable."""

    raw_value = read_env_string(name)
    if raw_value is None:
        return default
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0 seconds.")
    return value


def resolve_runtime_path(value: str | Path, *, project_root: Path | None = None) -> Path:
    """Resolve one runtime path without depending on the current working directory."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = project_root or default_project_root()
    return (root / path).resolve()


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved runtime paths used by the application entrypoints."""

    project_root: Path
    config_dir: Path
    companies_config_path: Path
    retrieval_config_path: Path
    prompts_config_path: Path
    eval_config_path: Path
    data_dir: Path
    chroma_dir: Path


@dataclass(frozen=True)
class ApiRuntimeSettings:
    """Environment-driven API runtime settings."""

    environment: str = "development"
    log_level: str = "INFO"
    enable_admin_routes: bool = True
    openai_model_override: str | None = None
    chroma_dir_override: Path | None = None


def load_runtime_paths_from_env(project_root: Path | None = None) -> RuntimePaths:
    """Resolve repo-relative runtime paths from environment variables."""

    root = project_root or default_project_root()
    config_dir = resolve_runtime_path(read_env_string("SEC_COPILOT_CONFIG_DIR", "configs"), project_root=root)
    companies_config_path = resolve_runtime_path(
        read_env_string("SEC_COPILOT_COMPANIES_CONFIG_PATH", str(config_dir / "companies.yaml")),
        project_root=root,
    )
    retrieval_config_path = resolve_runtime_path(
        read_env_string("SEC_COPILOT_RETRIEVAL_CONFIG_PATH", str(config_dir / "retrieval.yaml")),
        project_root=root,
    )
    prompts_config_path = resolve_runtime_path(
        read_env_string("SEC_COPILOT_PROMPTS_CONFIG_PATH", str(config_dir / "prompts.yaml")),
        project_root=root,
    )
    eval_config_path = resolve_runtime_path(
        read_env_string("SEC_COPILOT_EVAL_CONFIG_PATH", str(config_dir / "eval.yaml")),
        project_root=root,
    )
    data_dir = resolve_runtime_path(read_env_string("SEC_COPILOT_DATA_DIR", "data"), project_root=root)
    chroma_dir = resolve_runtime_path(
        read_env_string("SEC_COPILOT_CHROMA_DIR", "artifacts/chroma"),
        project_root=root,
    )
    return RuntimePaths(
        project_root=root,
        config_dir=config_dir,
        companies_config_path=companies_config_path,
        retrieval_config_path=retrieval_config_path,
        prompts_config_path=prompts_config_path,
        eval_config_path=eval_config_path,
        data_dir=data_dir,
        chroma_dir=chroma_dir,
    )


def load_api_runtime_settings_from_env(project_root: Path | None = None) -> ApiRuntimeSettings:
    """Load API runtime flags and overrides from environment variables."""

    environment = read_env_string("SEC_COPILOT_ENV", "development") or "development"
    default_admin_routes = environment.lower() == "development"
    chroma_override = read_env_string("SEC_COPILOT_CHROMA_DIR")
    return ApiRuntimeSettings(
        environment=environment,
        log_level=read_env_string("SEC_COPILOT_LOG_LEVEL", "INFO") or "INFO",
        enable_admin_routes=read_env_bool("SEC_COPILOT_ENABLE_ADMIN_ROUTES", default_admin_routes),
        openai_model_override=read_env_string("SEC_COPILOT_OPENAI_MODEL"),
        chroma_dir_override=(
            resolve_runtime_path(chroma_override, project_root=project_root or default_project_root())
            if chroma_override is not None
            else None
        ),
    )


__all__ = [
    "ApiRuntimeSettings",
    "PROJECT_ROOT",
    "RuntimePaths",
    "default_project_root",
    "load_api_runtime_settings_from_env",
    "load_runtime_paths_from_env",
    "read_env_bool",
    "read_env_positive_float",
    "read_env_string",
    "resolve_runtime_path",
]
