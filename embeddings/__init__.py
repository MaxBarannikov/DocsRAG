"""Embedder package — pluggable backends for sentence embeddings.

Currently:
    - PytorchEmbedder (sentence-transformers, MPS/CUDA/CPU)

Planned (Task 9):
    - OnnxEmbedder (ONNX Runtime, FP32 + INT8)
    - make_embedder() factory selecting by EMBEDDER_BACKEND
"""

from embeddings.pytorch import PytorchEmbedder

__all__ = ["PytorchEmbedder"]
