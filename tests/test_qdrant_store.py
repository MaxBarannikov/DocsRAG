"""Point identity.

Regression: ids used to be random, so re-indexing appended a second copy of the corpus.
"""

from __future__ import annotations

import uuid

import pytest

from indexing.chunker import Chunk
from indexing.qdrant_store import chunk_point_id


def test_point_id_is_stable_for_the_same_chunk() -> None:
    assert chunk_point_id("tutorial/path-params.md", 3) == chunk_point_id("tutorial/path-params.md", 3)


def test_point_id_differs_across_chunk_indices_and_files() -> None:
    ids = {
        chunk_point_id("tutorial/path-params.md", 0),
        chunk_point_id("tutorial/path-params.md", 1),
        chunk_point_id("advanced/security.md", 0),
    }
    assert len(ids) == 3, "chunk_index alone is not unique across documents"


def test_point_id_is_a_valid_uuid() -> None:
    uuid.UUID(chunk_point_id("a.md", 0))


def test_upsert_rejects_mismatched_lengths() -> None:
    from indexing.qdrant_store import QdrantStore

    store = QdrantStore.__new__(QdrantStore)  # no client needed for this check
    store.collection_name = "test"
    chunks = [Chunk(text="t", source_path="a.md", header_path="", chunk_index=0, chunk_size=512, chunk_overlap=50)]

    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert_chunks(chunks, [])
