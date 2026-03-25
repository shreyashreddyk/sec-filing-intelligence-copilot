"""Processed chunk loading for V2 dense retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from sec_copilot.schemas.ingestion import ChunkRecord
from sec_copilot.schemas.retrieval import RetrievalFilters


class ProcessedChunkStore:
    """In-memory access layer over V1 processed chunk JSONL artifacts."""

    def __init__(self, chunks_by_id: dict[str, ChunkRecord]) -> None:
        self._chunks_by_id = chunks_by_id

    @classmethod
    def load(cls, data_dir: str | Path) -> "ProcessedChunkStore":
        base_dir = Path(data_dir)
        chunk_paths = sorted((base_dir / "processed" / "chunks").glob("**/*.jsonl"))
        chunks_by_id: dict[str, ChunkRecord] = {}
        for chunk_path in chunk_paths:
            for line in chunk_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunk = ChunkRecord(**payload)
                chunks_by_id[chunk.chunk_id] = chunk
        return cls(chunks_by_id=chunks_by_id)

    def __len__(self) -> int:
        return len(self._chunks_by_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._chunks_by_id))

    def values(self) -> tuple[ChunkRecord, ...]:
        return tuple(self._chunks_by_id[chunk_id] for chunk_id in sorted(self._chunks_by_id))

    def get(self, chunk_id: str) -> ChunkRecord | None:
        return self._chunks_by_id.get(chunk_id)

    def filtered_values(self, filters: RetrievalFilters) -> tuple[ChunkRecord, ...]:
        from sec_copilot.retrieval.filters import chunk_matches_filters

        return tuple(
            chunk
            for chunk_id, chunk in sorted(self._chunks_by_id.items())
            if chunk_matches_filters(chunk, filters)
        )

    def has_matches(self, filters: RetrievalFilters) -> bool:
        return len(self.filtered_values(filters)) > 0


__all__ = ["ProcessedChunkStore"]
