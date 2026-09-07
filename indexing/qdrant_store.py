from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

if TYPE_CHECKING:
    from collections.abc import Sequence

    from indexing.chunker import Chunk

# Fixed so point ids are reproducible across machines and runs.
_POINT_ID_NAMESPACE = uuid.UUID("6f3a1d2e-7c4b-5a89-9e10-3b2c4d5e6f70")


def chunk_point_id(source_path: str, chunk_index: int) -> str:
    """Derived from the chunk's identity, so re-indexing overwrites in place instead
    of appending a second copy of the corpus.
    """
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{source_path}:{chunk_index}"))


class QdrantStore:
    def __init__(self, url: str, collection_name: str, vector_dim: int) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_dim = vector_dim

    def recreate_collection(self) -> None:
        """Destructive."""
        if self.client.collection_exists(self.collection_name):
            logger.warning("Deleting existing collection '{}'", self.collection_name)
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )
        logger.info(
            "Created collection '{}' (dim={}, distance=cosine)",
            self.collection_name,
            self.vector_dim,
        )

    def assert_dimension_matches(self) -> None:

        if not self.client.collection_exists(self.collection_name):
            return
        info = self.client.get_collection(self.collection_name)
        params = info.config.params.vectors
        existing_dim = getattr(params, "size", None)
        if existing_dim is not None and int(existing_dim) != self.vector_dim:
            msg = (
                f"Collection '{self.collection_name}' has vector dim {existing_dim}, "
                f"but the embedder produces {self.vector_dim}. Re-index with --recreate."
            )
            raise ValueError(msg)

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 100,
    ) -> None:
        if len(chunks) != len(embeddings):
            msg = f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            raise ValueError(msg)

        points = [
            PointStruct(
                id=chunk_point_id(chunk.source_path, chunk.chunk_index),
                vector=list(embedding),
                payload={
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "header_path": chunk.header_path,
                    "chunk_index": chunk.chunk_index,
                    # Lets the eval harness detect a config/index chunking mismatch.
                    "chunk_size": chunk.chunk_size,
                    "chunk_overlap": chunk.chunk_overlap,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            # wait=True, or count() below reads a stale value mid-batch.
            self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
            logger.debug("Upserted batch {}: {} points", i // batch_size + 1, len(batch))

        logger.info("Upserted {} chunks into '{}'", len(points), self.collection_name)

    def count(self) -> int:
        return int(self.client.count(self.collection_name, exact=True).count)
