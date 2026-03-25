"""Chroma indexing lifecycle management for the V2 dense baseline."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Literal

import chromadb
from pydantic import BaseModel, ConfigDict, Field

from sec_copilot.config.retrieval import RetrievalConfig
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import EmbeddingAdapter
from sec_copilot.utils.normalization import filing_date_to_ordinal, normalize_form_type, normalize_ticker
from sec_copilot.utils.io import write_json


class DenseIndexError(RuntimeError):
    """Raised when the vector index cannot be built safely."""


class IndexBuildMetadata(BaseModel):
    """Persistent metadata describing one Chroma collection build."""

    model_config = ConfigDict(extra="forbid")

    collection_name: str
    collection_mode: Literal["rebuild", "upsert"]
    built_at: str
    embedding_model_name: str
    embedding_model_revision_if_available: str | None = None
    requested_embedding_device: str
    resolved_embedding_device: str
    embedding_dimension: int
    embedding_max_seq_length: int | None = None
    sentence_transformers_version: str | None = None
    torch_version: str | None = None
    transformers_version: str | None = None
    parent_chunk_count: int
    embedding_subchunk_count: int
    stale_id_count: int
    corpus_fingerprint: str
    index_policy_fingerprint: str


class IndexBuildResult(BaseModel):
    """Return shape for CLI and validation after indexing."""

    model_config = ConfigDict(extra="forbid")

    collection_name: str
    collection_mode: Literal["rebuild", "upsert"]
    requested_embedding_device: str
    resolved_embedding_device: str
    parent_chunk_count: int
    embedding_subchunk_count: int
    stale_id_count: int
    sidecar_path: str
    corpus_fingerprint: str


class ChromaIndexManager:
    """Manage Chroma collection creation, rebuilds, and metadata sidecars."""

    def __init__(self, config: RetrievalConfig, adapter: EmbeddingAdapter) -> None:
        self.config = config
        self.adapter = adapter
        self.persist_directory = Path(config.index.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection_name = config.index.collection_name

    @property
    def sidecar_path(self) -> Path:
        return self.persist_directory / f"{self.collection_name}.build.json"

    def build(
        self,
        store: ProcessedChunkStore,
        mode: Literal["rebuild", "upsert"] | None = None,
    ) -> IndexBuildResult:
        index_mode = mode or self.config.index.default_mode
        chunks = store.values()
        corpus_fingerprint = _corpus_fingerprint(chunks)
        index_policy_fingerprint = _index_policy_fingerprint(self.config, self.adapter)

        subchunks = []
        for chunk in chunks:
            subchunks.extend(self.adapter.build_subchunks(chunk))

        current_ids = {subchunk.subchunk_id for subchunk in subchunks}
        stale_id_count = 0

        if index_mode == "rebuild":
            self._delete_collection_if_exists()
            collection = self._get_or_create_collection(index_policy_fingerprint)
        else:
            self._ensure_upsert_compatibility(index_policy_fingerprint)
            collection = self._get_or_create_collection(index_policy_fingerprint)
            existing_ids = set(self._list_collection_ids(collection))
            stale_id_count = len(existing_ids - current_ids)

        batch_size = max(1, self.config.embedding.batch_size)
        for batch_start in range(0, len(subchunks), batch_size):
            batch = subchunks[batch_start : batch_start + batch_size]
            embeddings = self.adapter.embed_texts([subchunk.text for subchunk in batch])
            collection.upsert(
                ids=[subchunk.subchunk_id for subchunk in batch],
                embeddings=embeddings,
                documents=[subchunk.text for subchunk in batch],
                metadatas=[_subchunk_metadata(subchunk) for subchunk in batch],
            )

        metadata = IndexBuildMetadata(
            collection_name=self.collection_name,
            collection_mode=index_mode,
            built_at=datetime.now(UTC).isoformat(),
            embedding_model_name=self.adapter.model_name,
            embedding_model_revision_if_available=self.adapter.model_revision,
            requested_embedding_device=self.adapter.requested_device,
            resolved_embedding_device=self.adapter.resolved_device,
            embedding_dimension=self.adapter.embedding_dimension,
            embedding_max_seq_length=self.adapter.max_seq_length,
            sentence_transformers_version=self.adapter.sentence_transformers_version,
            torch_version=self.adapter.torch_version,
            transformers_version=self.adapter.transformers_version,
            parent_chunk_count=len(chunks),
            embedding_subchunk_count=len(subchunks),
            stale_id_count=stale_id_count,
            corpus_fingerprint=corpus_fingerprint,
            index_policy_fingerprint=index_policy_fingerprint,
        )
        write_json(self.sidecar_path, metadata.model_dump(mode="json"))

        return IndexBuildResult(
            collection_name=self.collection_name,
            collection_mode=index_mode,
            requested_embedding_device=self.adapter.requested_device,
            resolved_embedding_device=self.adapter.resolved_device,
            parent_chunk_count=len(chunks),
            embedding_subchunk_count=len(subchunks),
            stale_id_count=stale_id_count,
            sidecar_path=str(self.sidecar_path),
            corpus_fingerprint=corpus_fingerprint,
        )

    def load_build_metadata(self) -> IndexBuildMetadata | None:
        if not self.sidecar_path.exists():
            return None
        return IndexBuildMetadata.model_validate_json(self.sidecar_path.read_text(encoding="utf-8"))

    def get_collection(self):
        return self.client.get_collection(name=self.collection_name)

    def _delete_collection_if_exists(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            return

    def _get_or_create_collection(self, index_policy_fingerprint: str):
        metadata = {
            "hnsw:space": self.config.index.distance_space,
            "embedding_model_name": self.adapter.model_name,
            "resolved_embedding_device": self.adapter.resolved_device,
            "index_policy_fingerprint": index_policy_fingerprint,
        }
        return self.client.get_or_create_collection(name=self.collection_name, metadata=metadata)

    def _ensure_upsert_compatibility(self, index_policy_fingerprint: str) -> None:
        existing = self.load_build_metadata()
        if existing is None:
            return
        if existing.index_policy_fingerprint != index_policy_fingerprint:
            raise DenseIndexError(
                "Index policy fingerprint changed. Run rebuild instead of upsert to avoid mixing vector policies."
            )
        if existing.embedding_model_name != self.adapter.model_name:
            raise DenseIndexError("Embedding model changed. Run rebuild instead of upsert.")

    def _list_collection_ids(self, collection) -> list[str]:
        total = collection.count()
        if total == 0:
            return []
        result = collection.get(limit=total, include=[])
        return list(result.get("ids", []))


def _subchunk_metadata(subchunk) -> dict[str, str | int]:
    return {
        "parent_chunk_id": subchunk.parent_chunk_id,
        "subchunk_index": subchunk.subchunk_index,
        "subchunk_count": subchunk.subchunk_count,
        "ticker": normalize_ticker(subchunk.ticker),
        "form_type": normalize_form_type(subchunk.form_type),
        "filing_date": subchunk.filing_date,
        "filing_date_ordinal": filing_date_to_ordinal(datetime.fromisoformat(subchunk.filing_date).date()),
        "accession_number": subchunk.accession_number,
        "document_id": subchunk.document_id,
        "section_title": subchunk.section_title,
        "source_url": subchunk.source_url,
    }


def _corpus_fingerprint(chunks) -> str:
    lines = [f"{chunk.chunk_id}:{chunk.content_hash}" for chunk in chunks]
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _index_policy_fingerprint(config: RetrievalConfig, adapter: EmbeddingAdapter) -> str:
    payload = {
        "distance_space": config.index.distance_space,
        "embedding_model_name": adapter.model_name,
        "normalize_embeddings": config.embedding.normalize_embeddings,
        "subchunk_tokens": config.embedding.subchunk_tokens,
        "subchunk_overlap_tokens": config.embedding.subchunk_overlap_tokens,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ChromaIndexManager",
    "DenseIndexError",
    "IndexBuildMetadata",
    "IndexBuildResult",
]
