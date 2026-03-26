"""Cross-encoder reranking over fused parent-chunk candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from sec_copilot.config.retrieval import RerankingConfig
from sec_copilot.retrieval.embedding import resolve_embedding_device
from sec_copilot.schemas.retrieval import RetrievedChunk
from sec_copilot.utils.huggingface import resolve_huggingface_token


class Reranker(Protocol):
    """Minimal interface for reranking fused parent candidates."""

    def rerank(self, question: str, candidates: tuple[RetrievedChunk, ...]) -> tuple[RetrievedChunk, ...]:
        """Return reranked candidates in final order."""


class RerankerUnavailableError(RuntimeError):
    """Raised when the configured reranker cannot be loaded or executed."""


@dataclass(frozen=True)
class RerankResult:
    """Internal rerank result plus debug flags."""

    results: tuple[RetrievedChunk, ...]
    applied: bool
    skipped_reason: str | None = None


class CrossEncoderReranker:
    """Sentence Transformers cross-encoder reranker with sigmoid-normalized scores."""

    def __init__(self, config: RerankingConfig) -> None:
        self.config = config
        self.requested_device = config.device
        self.resolved_device = resolve_embedding_device(config.device)
        self._model = None

    def rerank(self, question: str, candidates: tuple[RetrievedChunk, ...]) -> tuple[RetrievedChunk, ...]:
        if not candidates:
            return ()
        model = self._load_model()
        sentence_pairs = [(question, candidate.text) for candidate in candidates]
        try:
            raw_scores = model.predict(
                sentence_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local model/runtime behavior
            raise RerankerUnavailableError(f"Failed to execute cross-encoder reranking: {exc}") from exc

        rescored = [
            candidate.model_copy(
                update={
                    "rerank_rank": index + 1,
                    "rerank_score": _sigmoid(float(score)),
                }
            )
            for index, (candidate, score) in enumerate(
                sorted(
                    zip(candidates, raw_scores, strict=True),
                    key=lambda item: (
                        -_sigmoid(float(item[1])),
                        -(item[0].rrf_score or 0.0),
                        item[0].dense_rank or 10_000,
                        item[0].bm25_rank or 10_000,
                        item[0].chunk_id,
                    ),
                )
            )
        ]
        return tuple(rescored[: self.config.rerank_top_k])

    def ensure_loaded(self) -> None:
        """Preflight the configured cross-encoder runtime."""

        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - import path depends on local runtime
            raise RerankerUnavailableError(f"Failed to import CrossEncoder runtime: {exc}") from exc

        try:
            self._model = CrossEncoder(
                self.config.model_name,
                device=self.resolved_device,
                token=resolve_huggingface_token(),
            )
        except Exception as exc:  # pragma: no cover - depends on model availability
            raise RerankerUnavailableError(f"Failed to load cross-encoder model {self.config.model_name!r}: {exc}") from exc
        return self._model


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


__all__ = [
    "CrossEncoderReranker",
    "RerankResult",
    "Reranker",
    "RerankerUnavailableError",
]
