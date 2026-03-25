from __future__ import annotations

import pytest

from sec_copilot.config.retrieval import EmbeddingConfig
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
