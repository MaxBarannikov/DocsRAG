"""Parity gate: PyTorch and ONNX embedders must produce nearly-identical vectors.

Task 9 step 5: validates the entire ONNX export + runtime pipeline against the
PyTorch reference. Hard fail if any chunk's cosine similarity is below the
threshold — that's the signal that something drifted (trace artifacts, graph
optimizations, provider-specific numerical paths).

Falls back to skipping (not failing) when:
    - The [onnx] extra is not installed (pytest.importorskip below).
    - The FastAPI docs corpus isn't present locally (e.g. fresh CI checkout).
    - Either embedder model file is missing (e.g. ONNX hasn't been exported yet).

If this test ever genuinely fails (cosine < threshold), see CLAUDE.md decision
#4 — fallback is to read `token_embeddings` and pool manually.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Module-level skip if [onnx] extra is missing. Both imports are pulled in by
# the OnnxEmbedder's module-load — checking here yields a clean skip message
# instead of an ImportError on collection.
pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")

from api.config import settings  # noqa: E402
from embeddings.onnx import OnnxEmbedder  # noqa: E402
from embeddings.pytorch import PytorchEmbedder  # noqa: E402
from indexing.chunker import chunk_documents  # noqa: E402
from indexing.loader import load_markdown_files  # noqa: E402

PARITY_THRESHOLD = 0.9999
NUM_CHUNKS = 120  # ~5% of the 2540-chunk corpus — meaningful sample, fast to run
ONNX_MODEL_DIR = Path("models/bge-small-en-v1.5-onnx-fp32")


@pytest.fixture(scope="module")
def chunks() -> list[str]:
    """Real chunks from the FastAPI docs corpus, sliced to NUM_CHUNKS."""
    docs_path = Path(settings.docs_source_path)
    if not docs_path.exists():
        pytest.skip(f"docs source not present at {docs_path} — run `make fetch-docs` first")

    docs = load_markdown_files(docs_path)
    chunked = chunk_documents(docs, chunk_size=1024, chunk_overlap=100)
    texts = [c.text for c in chunked]
    if len(texts) < NUM_CHUNKS:
        pytest.skip(f"need >= {NUM_CHUNKS} chunks, corpus produced {len(texts)}")
    return texts[:NUM_CHUNKS]


@pytest.fixture(scope="module")
def pytorch_embedder() -> PytorchEmbedder:
    return PytorchEmbedder(settings.embedding_model)


@pytest.fixture(scope="module")
def onnx_embedder() -> OnnxEmbedder:
    if not (ONNX_MODEL_DIR / "model.onnx").exists():
        pytest.skip(f"ONNX model not exported — run `make export-onnx` to create {ONNX_MODEL_DIR}/")
    return OnnxEmbedder(ONNX_MODEL_DIR)


def test_pytorch_onnx_parity(
    chunks: list[str],
    pytorch_embedder: PytorchEmbedder,
    onnx_embedder: OnnxEmbedder,
) -> None:
    # float64 for the cosine math — both embedders return float32, accumulating
    # error matters when comparing two unit vectors at 4-9s precision.
    pt_vecs = np.array(pytorch_embedder.encode(chunks, show_progress=False), dtype=np.float64)
    onnx_vecs = np.array(onnx_embedder.encode(chunks, show_progress=False), dtype=np.float64)

    assert pt_vecs.shape == onnx_vecs.shape == (NUM_CHUNKS, pytorch_embedder.dimension)

    # Sanity: both should already be L2-normalized (graph- or library-baked).
    pt_norms = np.linalg.norm(pt_vecs, axis=1)
    onnx_norms = np.linalg.norm(onnx_vecs, axis=1)
    assert np.allclose(pt_norms, 1.0, atol=1e-4), f"PyTorch norms off: min={pt_norms.min()}, max={pt_norms.max()}"
    assert np.allclose(onnx_norms, 1.0, atol=1e-4), f"ONNX norms off: min={onnx_norms.min()}, max={onnx_norms.max()}"

    # Cosine = dot product for unit vectors.
    cosines = (pt_vecs * onnx_vecs).sum(axis=1)

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

    print(
        f"\n  parity OK across {NUM_CHUNKS} chunks: "
        f"cosine min={cosines.min():.6f}, mean={cosines.mean():.6f}, max={cosines.max():.6f}",
    )
