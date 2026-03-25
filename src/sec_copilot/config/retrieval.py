"""Typed loading for retrieval, embedding, reranking, and prompt configuration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


class RetrievalConfigError(ValueError):
    """Raised when retrieval config files are invalid."""


def _normalize_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda", "mps"}:
        return normalized
    if re.fullmatch(r"cuda:\d+", normalized):
        return normalized
    raise ValueError("device must be one of auto, cpu, cuda, cuda:<index>, or mps")


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
        return _normalize_device(value)

    @model_validator(mode="after")
    def _validate_overlap(self) -> "EmbeddingConfig":
        if self.subchunk_overlap_tokens >= self.subchunk_tokens:
            raise ValueError("subchunk_overlap_tokens must be smaller than subchunk_tokens")
        return self


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_subchunk_top_k: int = Field(default=24, ge=1)
    dense_top_k: int = Field(default=8, ge=1)
    bm25_top_k: int = Field(default=8, ge=1)
    fused_top_k_before_rerank: int = Field(default=12, ge=1)
    generation_context_top_k: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _validate_order(self) -> "RetrievalSettings":
        if self.dense_subchunk_top_k < self.dense_top_k:
            raise ValueError("dense_subchunk_top_k must be >= dense_top_k")
        if self.fused_top_k_before_rerank < self.dense_top_k:
            raise ValueError("fused_top_k_before_rerank must be >= dense_top_k")
        if self.fused_top_k_before_rerank < self.bm25_top_k:
            raise ValueError("fused_top_k_before_rerank must be >= bm25_top_k")
        if self.fused_top_k_before_rerank < self.generation_context_top_k:
            raise ValueError("fused_top_k_before_rerank must be >= generation_context_top_k")
        return self


class PromptingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    max_context_tokens: int = Field(default=2400, ge=128)


class BM25Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing: Literal["sec_regex_v1"] = "sec_regex_v1"


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_name: Literal["openai", "mock"] = "openai"
    openai_model: str = Field(default="gpt-4.1-mini", min_length=1)


class RerankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    required_for_generation: bool = True
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", min_length=1)
    rerank_top_k: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)
    device: str = "auto"

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        return _normalize_device(value)


class AbstentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weak_top_rerank_score_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    supporting_chunk_rerank_score_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_supporting_chunks: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "AbstentionConfig":
        if self.supporting_chunk_rerank_score_threshold > self.weak_top_rerank_score_threshold:
            raise ValueError(
                "supporting_chunk_rerank_score_threshold must be <= weak_top_rerank_score_threshold"
            )
        return self


class GroundingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citations_must_come_from_final_context: bool = True


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunking: dict[str, Any] = Field(default_factory=dict)
    index: IndexConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalSettings
    bm25: BM25Config
    fusion: FusionConfig
    prompting: PromptingConfig
    provider: ProviderConfig
    reranking: RerankingConfig
    abstention: AbstentionConfig
    grounding: GroundingConfig

    @model_validator(mode="after")
    def _validate_cross_section_settings(self) -> "RetrievalConfig":
        if self.reranking.rerank_top_k > self.retrieval.fused_top_k_before_rerank:
            raise ValueError("rerank_top_k must be <= fused_top_k_before_rerank")
        if self.retrieval.generation_context_top_k > self.reranking.rerank_top_k:
            raise ValueError("generation_context_top_k must be <= rerank_top_k")
        return self


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
    "AbstentionConfig",
    "BM25Config",
    "EmbeddingConfig",
    "FusionConfig",
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
