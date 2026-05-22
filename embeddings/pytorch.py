"""PyTorch sentence-transformers embedder.

Default: BAAI/bge-small-en-v1.5 (384-dim, normalized cosine). MPS on Apple
Silicon, CUDA elsewhere, CPU fallback. Used at both index- and query-time —
pooling/normalization parity between phases is mandatory (see CLAUDE.md
"Embedder reuse").
"""

from collections.abc import Sequence

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


class PytorchEmbedder:
    """sentence-transformers wrapper with batched encoding and L2 normalization."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> None:
        # `device=None` keeps the auto-detection path (MPS > CUDA > CPU).
        # Pass an explicit string to override — useful for benchmarks comparing
        # MPS vs CPU on the same machine (Task 9 step 9).
        resolved_device = device if device is not None else self._select_device()
        logger.info(f"Loading PyTorch embedder '{model_name}' on device '{resolved_device}'")
        self._model = SentenceTransformer(model_name, device=resolved_device)
        self.model_name = model_name
        self.dimension = self._model.get_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    @staticmethod
    def _select_device() -> str:
        """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",
    ) -> list[list[float]]:
        """Encode texts into dense embedding vectors.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per forward pass. 32 is a good default for
                        small models on M-series Macs.
            show_progress: Show tqdm bar (useful for long indexing jobs).
            prefix: Optional prefix prepended to each text. Use "query: " at
                    retrieval time and "passage: " at indexing time for e5 models.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
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
