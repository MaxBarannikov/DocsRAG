"""sentence-transformers embedder with auto device selection (MPS > CUDA > CPU)."""

from collections.abc import Sequence

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


class PytorchEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> None:
        resolved_device = device if device is not None else self._select_device()
        logger.info(f"Loading PyTorch embedder '{model_name}' on device '{resolved_device}'")
        self._model = SentenceTransformer(model_name, device=resolved_device)
        self.model_name = model_name
        self.dimension = self._model.get_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    @staticmethod
    def _select_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",  # use "query: "/"passage: " for e5 models; bge doesn't need it
    ) -> list[list[float]]:
        if not texts:
            return []

        prefixed = [prefix + t for t in texts] if prefix else list(texts)
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # critical for cosine similarity
        )
        return embeddings.tolist()
