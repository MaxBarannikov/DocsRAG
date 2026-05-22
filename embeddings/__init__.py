"""Embedder package — pluggable backends for sentence embeddings.

Available:
    - PytorchEmbedder (sentence-transformers, MPS/CUDA/CPU)
      from embeddings import PytorchEmbedder
    - OnnxEmbedder (ONNX Runtime FP32; INT8 variant via --output in step 6)
      from embeddings.onnx import OnnxEmbedder    # requires [onnx] extra

Planned (Task 9):
    - make_embedder() factory selecting by EMBEDDER_BACKEND

OnnxEmbedder is deliberately NOT re-exported from this __init__: it imports
onnxruntime + transformers + tqdm at module-load time, which would break the
package import for callers who haven't installed the optional [onnx] extra.
PytorchEmbedder is always available.
"""

from embeddings.pytorch import PytorchEmbedder

__all__ = ["PytorchEmbedder"]
