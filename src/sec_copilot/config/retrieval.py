"""Typed loading for retrieval, embedding, and prompt configuration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


class RetrievalConfigError(ValueError):
    """Raised when retrieval config files are invalid."""


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str = Field(min_length=1)
    persist_directory: str = Field(min_length=1)
    distance_space: Literal["cosine"] = "cosine"
    default_mode: Literal["rebuild", "upsert"] = "rebuild"


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["sentence_transformers_local"] = "sentence_transformers_local"
    device: str = "auto"
    model_name: str = Field(min_length=1)
    batch_size: int = Field(default=32, ge=1)
    normalize_embeddings: bool = True
    subchunk_tokens: int = Field(default=200, ge=32)
    subchunk_overlap_tokens: int = Field(default=40, ge=0)

    @field_validator("device")
    @classmethod
    def _normalize_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"auto", "cpu", "cuda", "mps"}:
            return normalized
        if re.fullmatch(r"cuda:\d+", normalized):
            return normalized
        raise ValueError("device must be one of auto, cpu, cuda, cuda:<index>, or mps")

    @model_validator(mode="after")
    def _validate_overlap(self) -> "EmbeddingConfig":
        if self.subchunk_overlap_tokens >= self.subchunk_tokens:
            raise ValueError("subchunk_overlap_tokens must be smaller than subchunk_tokens")
        return self


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_subchunk_top_k: int = Field(default=24, ge=1)
    retrieval_top_k: int = Field(default=8, ge=1)
    weak_score_threshold: float = Field(default=0.20, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def _validate_order(self) -> "RetrievalSettings":
        if self.retrieval_subchunk_top_k < self.retrieval_top_k:
            raise ValueError("retrieval_subchunk_top_k must be >= retrieval_top_k")
        return self


class PromptingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: str = Field(min_length=1)
    prompt_top_n: int = Field(default=4, ge=1)
    prompt_chunk_max_chars: int = Field(default=1800, ge=200)
    prompt_context_max_chars: int = Field(default=7200, ge=500)

    @model_validator(mode="after")
    def _validate_budget(self) -> "PromptingConfig":
        if self.prompt_context_max_chars < self.prompt_chunk_max_chars:
            raise ValueError("prompt_context_max_chars must be >= prompt_chunk_max_chars")
        return self


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_name: Literal["openai", "mock"] = "openai"
    openai_model: str = Field(default="gpt-4.1-mini", min_length=1)


class RerankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model_name: str | None = None
    rerank_top_k: int | None = None


class GroundingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_citations: bool = True
    invalid_response_policy: Literal["invalid_grounding"] = "invalid_grounding"


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunking: dict[str, Any] = Field(default_factory=dict)
    index: IndexConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalSettings
    prompting: PromptingConfig
    provider: ProviderConfig
    reranking: RerankingConfig
    grounding: GroundingConfig


class PromptTemplateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    system: str | None = None
    user: str | None = None


class PromptCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: dict[str, PromptTemplateConfig]

    def get(self, name: str) -> PromptTemplateConfig:
        try:
            return self.prompts[name]
        except KeyError as exc:
            raise RetrievalConfigError(f"Prompt template {name!r} was not found") from exc


def load_retrieval_config(path: str | Path = "configs/retrieval.yaml") -> RetrievalConfig:
    """Load the V2 retrieval configuration."""

    return RetrievalConfig.model_validate(_load_yaml(path))


def load_prompt_catalog(path: str | Path = "configs/prompts.yaml") -> PromptCatalog:
    """Load prompt templates used by the grounded answer pipeline."""

    return PromptCatalog.model_validate(_load_yaml(path))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise RetrievalConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise RetrievalConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RetrievalConfigError(f"Config file must contain a top-level mapping: {config_path}")
    return payload


__all__ = [
    "EmbeddingConfig",
    "GroundingConfig",
    "IndexConfig",
    "PromptCatalog",
    "PromptTemplateConfig",
    "PromptingConfig",
    "ProviderConfig",
    "RetrievalConfig",
    "RetrievalConfigError",
    "RetrievalSettings",
    "RerankingConfig",
    "load_prompt_catalog",
    "load_retrieval_config",
]
