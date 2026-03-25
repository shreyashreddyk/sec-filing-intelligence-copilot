from __future__ import annotations

from pathlib import Path

from sec_copilot.eval.curated_examples import (
    CuratedEmbeddingAdapter,
    CuratedReranker,
    build_curated_config,
    build_curated_store,
)


class CuratedEmbeddingAdapterV2(CuratedEmbeddingAdapter):
    model_name = "curated-embedding-v2"


def build_store():
    return build_curated_store()


def build_config(tmp_path: Path):
    return build_curated_config(tmp_path)


def build_reranker():
    return CuratedReranker()
