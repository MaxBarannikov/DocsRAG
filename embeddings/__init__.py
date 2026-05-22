"""Embedder package — pluggable backends for sentence embeddings.

Available:
    - PytorchEmbedder (sentence-transformers, MPS/CUDA/CPU)
        from embeddings import PytorchEmbedder
    - OnnxEmbedder (ONNX Runtime, FP32 or INT8 depending on model_dir)
        from embeddings.onnx import OnnxEmbedder         # requires [onnx] extra
    - make_embedder(backend?) factory — picks by EMBEDDER_BACKEND env var
        from embeddings import make_embedder

OnnxEmbedder is deliberately NOT re-exported from this __init__: it imports
onnxruntime + transformers + tqdm at module-load time, which would break the
package import for callers who haven't installed the optional [onnx] extra.
The factory lazily imports it only when its backend is selected.
"""

from embeddings.factory import make_embedder
from embeddings.pytorch import PytorchEmbedder

__all__ = ["PytorchEmbedder", "make_embedder"]
