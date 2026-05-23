import uuid
from collections.abc import Sequence

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from indexing.chunker import Chunk


class QdrantStore:

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_dim: int,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_dim = vector_dim

    def recreate_collection(self) -> None:
        """Drop and recreate the collection. Destructive — use for fresh indexing."""
        if self.client.collection_exists(self.collection_name):
            logger.warning(f"Deleting existing collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection '{self.collection_name}' (dim={self.vector_dim}, distance=cosine)")

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 100,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=list(emb),
                payload={
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "header_path": chunk.header_path,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

        logger.info(f"Upserted {len(points)} chunks into '{self.collection_name}'")

    def count(self) -> int:
        return self.client.count(self.collection_name).count
