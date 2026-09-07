from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from core.config import EmbedderBackend, settings

if TYPE_CHECKING:
    from core.embedder import Embedder


def make_embedder(backend: EmbedderBackend | None = None) -> Embedder:
    """Defaults to the configured backend. Imports are lazy so PyTorch-only
    installations never need the [onnx] extra.
    """
    backend = backend or settings.embedder_backend
    logger.info("make_embedder | backend={}", backend)

    if backend == "pytorch":
        from embeddings.pytorch import PytorchEmbedder

        return PytorchEmbedder(settings.embedding_model)

    if backend in ("onnx-fp32", "onnx-int8"):
        from embeddings.onnx import OnnxEmbedder

        model_dir = settings.embedder_onnx_fp32_path if backend == "onnx-fp32" else settings.embedder_onnx_int8_path
        return OnnxEmbedder(model_dir)

    msg = f"Unknown EMBEDDER_BACKEND: {backend!r}. Expected: pytorch | onnx-fp32 | onnx-int8."
    raise ValueError(msg)
