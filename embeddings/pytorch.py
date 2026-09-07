"""sentence-transformers embedder with automatic device selection (MPS > CUDA > CPU)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from core.config import settings

if TYPE_CHECKING:
    from collections.abc import Sequence


class PytorchEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        resolved_device = device if device is not None else self._select_device()
        logger.info("Loading PyTorch embedder '{}' on device '{}'", self.model_name, resolved_device)

        try:
            self._model = SentenceTransformer(self.model_name, device=resolved_device)
        except Exception as exc:
            msg = (
                f"Failed to load embedding model {self.model_name!r} on device {resolved_device!r}. "
                f"The first run downloads it from Hugging Face, so check network access."
            )
            raise RuntimeError(msg) from exc

        self.dimension = int(self._embedding_dimension() or 0)
        if self.dimension != settings.embedding_dim:
            logger.warning(
                "Embedding dimension {} differs from configured EMBEDDING_DIM={}. "
                "Indexing into an existing collection built at the other width will fail.",
                self.dimension,
                settings.embedding_dim,
            )
        logger.info("Embedding dimension: {}", self.dimension)

    def _embedding_dimension(self) -> int | None:
        """5.x renamed the getter and warns on the old name; older releases only have it."""
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        return getter()

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
        prefix: str = "",  # e5-style models need "query: "/"passage: "; bge does not
    ) -> list[list[float]]:
        if not texts:
            return []

        prefixed = [prefix + t for t in texts] if prefix else list(texts)
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # required: the collections use cosine distance
        )
        return [vector.tolist() for vector in embeddings]
