"""Pluggable embedder backends (pytorch, onnx-fp32, onnx-int8).

OnnxEmbedder is NOT re-exported here — it requires the optional [onnx] extra and
imports onnxruntime at module load time. Use `from embeddings.onnx import OnnxEmbedder`
or let make_embedder() handle it lazily.
"""

from embeddings.factory import make_embedder
from embeddings.pytorch import PytorchEmbedder

__all__ = ["PytorchEmbedder", "make_embedder"]
