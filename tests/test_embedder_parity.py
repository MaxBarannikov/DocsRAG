"""Parity gate between the PyTorch and ONNX-FP32 embedders.

Fails on any chunk below the threshold, which would mean drift in pooling,
normalization or provider numerics. Skipped when the extra, corpus or model is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from loguru import logger

pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")

from core.config import settings
from embeddings.onnx import OnnxEmbedder
from embeddings.pytorch import PytorchEmbedder
from indexing.chunker import chunk_documents
from indexing.loader import load_markdown_files

PARITY_THRESHOLD = 0.9999
# Covers the corpus's length distribution without making the test slow.
NUM_CHUNKS = 120


@pytest.fixture(scope="module")
def chunks() -> list[str]:

    docs_path = Path(settings.docs_source_path)
    if not docs_path.exists():
        pytest.skip(f"docs source not present at {docs_path} — run `make fetch-docs` first")

    docs = load_markdown_files(docs_path)
    chunked = chunk_documents(docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    texts = [c.text for c in chunked]
    if len(texts) < NUM_CHUNKS:
        pytest.skip(f"need >= {NUM_CHUNKS} chunks, corpus produced {len(texts)}")
    return texts[:NUM_CHUNKS]


@pytest.fixture(scope="module")
def pytorch_embedder() -> PytorchEmbedder:
    return PytorchEmbedder(settings.embedding_model)


@pytest.fixture(scope="module")
def onnx_embedder() -> OnnxEmbedder:
    model_dir = settings.embedder_onnx_fp32_path
    if not (model_dir / "model.onnx").exists():
        pytest.skip(f"ONNX model not exported — run `make export-onnx` to create {model_dir}/")
    return OnnxEmbedder(model_dir)


@pytest.mark.slow
def test_pytorch_onnx_parity(
    chunks: list[str],
    pytorch_embedder: PytorchEmbedder,
    onnx_embedder: OnnxEmbedder,
) -> None:
    # float64: accumulation error matters at 4-9 decimal places.
    pt_vecs = np.array(pytorch_embedder.encode(chunks, show_progress=False), dtype=np.float64)
    onnx_vecs = np.array(onnx_embedder.encode(chunks, show_progress=False), dtype=np.float64)

    assert pt_vecs.shape == onnx_vecs.shape == (NUM_CHUNKS, pytorch_embedder.dimension)

    pt_norms = np.linalg.norm(pt_vecs, axis=1)
    onnx_norms = np.linalg.norm(onnx_vecs, axis=1)
    assert np.allclose(pt_norms, 1.0, atol=1e-4), f"PyTorch norms off: min={pt_norms.min()}, max={pt_norms.max()}"
    assert np.allclose(onnx_norms, 1.0, atol=1e-4), f"ONNX norms off: min={onnx_norms.min()}, max={onnx_norms.max()}"

    cosines = (pt_vecs * onnx_vecs).sum(axis=1)  # cosine = dot product for unit vectors

    below = np.where(cosines < PARITY_THRESHOLD)[0]
    if below.size > 0:
        worst_idx = int(below[np.argmin(cosines[below])])
        worst_score = float(cosines[worst_idx])
        worst_preview = chunks[worst_idx][:200].replace("\n", " ")
        pytest.fail(
            f"PyTorch↔ONNX parity broken on {below.size}/{NUM_CHUNKS} chunks. "
            f"Worst: chunk[{worst_idx}] cosine={worst_score:.6f} < {PARITY_THRESHOLD}. "
            f"Preview: {worst_preview!r}",
        )

    logger.info(
        "parity OK across {} chunks: cosine min={:.6f}, mean={:.6f}, max={:.6f}",
        NUM_CHUNKS,
        cosines.min(),
        cosines.mean(),
        cosines.max(),
    )
