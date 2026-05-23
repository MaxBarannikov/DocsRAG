"""Embedder factory — selects backend by EMBEDDER_BACKEND env var."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from api.config import EmbedderBackend, settings
from embeddings.pytorch import PytorchEmbedder

if TYPE_CHECKING:
    from embeddings.onnx import OnnxEmbedder


def make_embedder(backend: EmbedderBackend | None = None) -> PytorchEmbedder | OnnxEmbedder:
    backend = backend or settings.embedder_backend
    logger.info(f"make_embedder | backend={backend}")

    if backend == "pytorch":
        return PytorchEmbedder(settings.embedding_model)

    if backend == "onnx-fp32":
        from embeddings.onnx import OnnxEmbedder  # noqa: PLC0415

        return OnnxEmbedder(settings.embedder_onnx_fp32_path)

    if backend == "onnx-int8":
        from embeddings.onnx import OnnxEmbedder  # noqa: PLC0415

        return OnnxEmbedder(settings.embedder_onnx_int8_path)

    msg = f"Unknown EMBEDDER_BACKEND: {backend!r}. Expected: pytorch | onnx-fp32 | onnx-int8."
    raise ValueError(msg)
