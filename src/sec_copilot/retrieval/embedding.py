"""Embedding adapter and index-time subchunking for dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, Sequence

from sec_copilot.config.retrieval import EmbeddingConfig
from sec_copilot.schemas.ingestion import ChunkRecord


@dataclass(frozen=True)
class EmbeddingSubchunk:
    """Embedding-only subchunk used for index-time vector representation."""

    subchunk_id: str
    parent_chunk_id: str
    subchunk_index: int
    subchunk_count: int
    text: str
    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    document_id: str
    section_title: str
    source_url: str


class EmbeddingAdapter(Protocol):
    """Minimal adapter interface required by indexing and retrieval code."""

    model_name: str
    model_revision: str | None
    max_seq_length: int | None
    embedding_dimension: int
    requested_device: str
    resolved_device: str
    sentence_transformers_version: str | None
    torch_version: str | None
    transformers_version: str | None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text values into dense vectors."""

    def build_subchunks(self, chunk: ChunkRecord) -> tuple[EmbeddingSubchunk, ...]:
        """Split one parent chunk into embedding-only subchunks."""


class DeviceResolutionError(RuntimeError):
    """Raised when an embedding device was requested but is unavailable."""


@dataclass(frozen=True)
class TorchRuntimeCapabilities:
    """Detected torch runtime capabilities used for device resolution."""

    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool


class SentenceTransformerEmbeddingAdapter:
    """Local Sentence Transformers adapter with tokenizer-aligned subchunking."""

    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer, __version__ as sentence_transformers_version
        from torch import __version__ as torch_version
        from transformers import AutoTokenizer, __version__ as transformers_version

        self.config = config
        self.requested_device = config.device
        self.resolved_device = resolve_embedding_device(config.device)
        self.model = SentenceTransformer(config.model_name, device=self.resolved_device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        self.model_name = config.model_name
        self.model_revision = self.tokenizer.init_kwargs.get("revision")
        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        self.embedding_dimension = int(self.model.get_sentence_embedding_dimension())
        self.sentence_transformers_version = sentence_transformers_version
        self.torch_version = torch_version
        self.transformers_version = transformers_version

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def build_subchunks(self, chunk: ChunkRecord) -> tuple[EmbeddingSubchunk, ...]:
        token_ids = self.tokenizer(
            chunk.text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        if not token_ids:
            return ()

        windows = _window_ranges(
            token_count=len(token_ids),
            target_tokens=self.config.subchunk_tokens,
            overlap_tokens=self.config.subchunk_overlap_tokens,
        )

        decoded_windows: list[tuple[int, str]] = []
        for subchunk_index, (start_token, end_token) in enumerate(windows):
            text = self.tokenizer.decode(
                token_ids[start_token:end_token],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if not text:
                continue
            decoded_windows.append((subchunk_index, text))

        subchunk_count = len(decoded_windows)
        subchunks: list[EmbeddingSubchunk] = []
        for subchunk_index, text in decoded_windows:
            subchunks.append(
                EmbeddingSubchunk(
                    subchunk_id=f"{chunk.chunk_id}__emb_{subchunk_index:04d}",
                    parent_chunk_id=chunk.chunk_id,
                    subchunk_index=subchunk_index,
                    subchunk_count=subchunk_count,
                    text=text,
                    ticker=chunk.ticker,
                    form_type=chunk.form_type,
                    filing_date=chunk.filing_date,
                    accession_number=chunk.accession_number,
                    document_id=chunk.document_id,
                    section_title=chunk.section_title,
                    source_url=chunk.source_url,
                )
            )

        return tuple(subchunks)


def resolve_embedding_device(
    requested_device: str,
    capabilities: TorchRuntimeCapabilities | None = None,
) -> str:
    """Resolve the configured embedding device against the available torch runtime."""

    requested = requested_device.strip().lower()
    runtime = capabilities or _torch_runtime_capabilities()

    if requested == "auto":
        if runtime.cuda_available:
            return "cuda"
        if runtime.mps_available:
            return "mps"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "mps":
        if runtime.mps_available:
            return "mps"
        raise DeviceResolutionError("Requested device 'mps' but MPS is not available in this torch runtime.")

    if requested == "cuda":
        if runtime.cuda_available:
            return "cuda"
        raise DeviceResolutionError("Requested device 'cuda' but CUDA is not available in this torch runtime.")

    match = re.fullmatch(r"cuda:(\d+)", requested)
    if match:
        if not runtime.cuda_available:
            raise DeviceResolutionError(
                f"Requested device '{requested}' but CUDA is not available in this torch runtime."
            )
        index = int(match.group(1))
        if index >= runtime.cuda_device_count:
            raise DeviceResolutionError(
                f"Requested device '{requested}' but only {runtime.cuda_device_count} CUDA device(s) are available."
            )
        return requested

    raise DeviceResolutionError(f"Unsupported embedding device: {requested_device!r}")


def _torch_runtime_capabilities() -> TorchRuntimeCapabilities:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    return TorchRuntimeCapabilities(
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_available=mps_available,
    )


def _window_ranges(token_count: int, target_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(token_count, start + target_tokens)
        windows.append((start, end))
        if end >= token_count:
            break
        next_start = max(start + 1, end - overlap_tokens)
        start = next_start
    return windows


__all__ = [
    "DeviceResolutionError",
    "EmbeddingAdapter",
    "EmbeddingSubchunk",
    "SentenceTransformerEmbeddingAdapter",
    "TorchRuntimeCapabilities",
    "resolve_embedding_device",
]
