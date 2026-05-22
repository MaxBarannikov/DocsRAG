"""Embedder factory — pick a backend by `EMBEDDER_BACKEND` env var.

`OnnxEmbedder` is imported lazily so callers using only the PyTorch backend
don't need the optional `[onnx]` extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from api.config import EmbedderBackend, settings
from embeddings.pytorch import PytorchEmbedder

if TYPE_CHECKING:
    from embeddings.onnx import OnnxEmbedder


def make_embedder(backend: EmbedderBackend | None = None) -> PytorchEmbedder | OnnxEmbedder:
    """Build an embedder for the given backend (defaults to `settings.embedder_backend`).

    The returned object exposes the shared API surface used by `api/rag.py`,
    `api/retriever.py`, and `indexing/run_indexing.py`:
        - `encode(texts, batch_size=..., show_progress=..., prefix="") -> list[list[float]]`
        - `dimension: int`
        - `model_name: str`
    """
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
