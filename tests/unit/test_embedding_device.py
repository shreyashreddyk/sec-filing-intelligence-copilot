from __future__ import annotations

import sys
import types

import pytest

from sec_copilot.config.retrieval import EmbeddingConfig, RerankingConfig
from sec_copilot.rerank.cross_encoder import CrossEncoderReranker
from sec_copilot.retrieval.embedding import DeviceResolutionError, TorchRuntimeCapabilities, resolve_embedding_device


def _caps(*, cuda_available: bool, cuda_device_count: int = 0, mps_available: bool = False) -> TorchRuntimeCapabilities:
    return TorchRuntimeCapabilities(
        torch_version="test",
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_available=mps_available,
    )


def test_embedding_config_normalizes_supported_device_values() -> None:
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", device="CUDA:1")
    assert config.device == "cuda:1"

    auto_config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert auto_config.device == "auto"


def test_auto_device_prefers_cuda_then_mps_then_cpu() -> None:
    assert resolve_embedding_device("auto", capabilities=_caps(cuda_available=True, cuda_device_count=2)) == "cuda"
    assert resolve_embedding_device("auto", capabilities=_caps(cuda_available=False, mps_available=True)) == "mps"
    assert resolve_embedding_device("auto", capabilities=_caps(cuda_available=False, mps_available=False)) == "cpu"


def test_explicit_cuda_index_requires_available_device() -> None:
    assert resolve_embedding_device("cuda:1", capabilities=_caps(cuda_available=True, cuda_device_count=2)) == "cuda:1"

    with pytest.raises(DeviceResolutionError):
        resolve_embedding_device("cuda:2", capabilities=_caps(cuda_available=True, cuda_device_count=2))

    with pytest.raises(DeviceResolutionError):
        resolve_embedding_device("cuda", capabilities=_caps(cuda_available=False))


def test_explicit_mps_requires_runtime_support() -> None:
    assert resolve_embedding_device("mps", capabilities=_caps(cuda_available=False, mps_available=True)) == "mps"

    with pytest.raises(DeviceResolutionError):
        resolve_embedding_device("mps", capabilities=_caps(cuda_available=False, mps_available=False))


def test_sentence_transformer_adapter_passes_hf_token_to_model_and_tokenizer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, *, device: str, token: str | None = None) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["sentence_transformer_token"] = token
            self.max_seq_length = 256

        def get_sentence_embedding_dimension(self) -> int:
            return 384

    class FakeTokenizer:
        def __init__(self) -> None:
            self.init_kwargs = {}

        @classmethod
        def from_pretrained(cls, model_name: str, *, use_fast: bool, token: str | None = None):
            captured["tokenizer_model_name"] = model_name
            captured["use_fast"] = use_fast
            captured["tokenizer_token"] = token
            return cls()

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr("sec_copilot.retrieval.embedding.resolve_embedding_device", lambda _: "cpu")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(__version__="test-st", SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(__version__="test-tf", AutoTokenizer=FakeTokenizer),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="test-torch"))

    from sec_copilot.retrieval.embedding import SentenceTransformerEmbeddingAdapter

    adapter = SentenceTransformerEmbeddingAdapter(
        EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    )

    assert adapter.embedding_dimension == 384
    assert captured["sentence_transformer_token"] == "hf_test_token"
    assert captured["tokenizer_token"] == "hf_test_token"
    assert captured["use_fast"] is True


def test_cross_encoder_reranker_passes_hf_token_to_model_loader(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, device: str, token: str | None = None) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["token"] = token

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr("sec_copilot.rerank.cross_encoder.resolve_embedding_device", lambda _: "cpu")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    reranker = CrossEncoderReranker(
        RerankingConfig(
            enabled=True,
            required_for_generation=True,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_k=8,
            batch_size=8,
            device="cpu",
        )
    )
    reranker.ensure_loaded()

    assert captured["model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert captured["token"] == "hf_test_token"
