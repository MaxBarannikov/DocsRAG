from __future__ import annotations

import pytest

from indexing.chunker import MIN_CONTENT_CHARS, Chunk, chunk_documents


def test_header_path_is_the_full_header_trail(sample_markdown_docs) -> None:
    chunks = chunk_documents(sample_markdown_docs, chunk_size=512, chunk_overlap=50)
    header_paths = {chunk.header_path for chunk in chunks}
    assert "Path Parameters > Path parameters with types" in header_paths
    assert "Query Parameters" in header_paths


def test_chunk_index_restarts_for_each_document(sample_markdown_docs) -> None:
    """Pinned deliberately: retrieval code must treat chunk_index as document-local."""
    chunks = chunk_documents(sample_markdown_docs, chunk_size=512, chunk_overlap=50)
    per_document: dict[str, list[int]] = {}
    for chunk in chunks:
        per_document.setdefault(chunk.source_path, []).append(chunk.chunk_index)

    assert len(per_document) == 2
    for indices in per_document.values():
        assert indices == list(range(len(indices)))


def test_headers_are_kept_in_the_chunk_text(sample_markdown_docs) -> None:
    chunks = chunk_documents(sample_markdown_docs, chunk_size=512, chunk_overlap=50)
    assert any("# Path Parameters" in chunk.text for chunk in chunks)


def test_fragments_below_the_minimum_length_are_dropped() -> None:
    from pathlib import Path

    from indexing.loader import RawDocument

    doc = RawDocument(
        content="# T\n\nshort\n\n## U\n\n" + "long enough body text " * 10,
        source_path=Path("a.md"),
        relative_path="a.md",
    )
    for chunk in chunk_documents([doc], chunk_size=512, chunk_overlap=50):
        assert len(chunk.text.strip()) >= MIN_CONTENT_CHARS


def test_chunking_parameters_are_recorded_on_each_chunk(sample_markdown_docs) -> None:
    chunks = chunk_documents(sample_markdown_docs, chunk_size=256, chunk_overlap=25)
    assert chunks
    assert all(chunk.chunk_size == 256 and chunk.chunk_overlap == 25 for chunk in chunks)


def test_overlap_at_or_above_chunk_size_is_rejected(sample_markdown_docs) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_documents(sample_markdown_docs, chunk_size=100, chunk_overlap=100)


def test_no_documents_yields_no_chunks() -> None:
    assert chunk_documents([], chunk_size=512, chunk_overlap=50) == []


def test_chunk_is_immutable() -> None:
    chunk = Chunk(text="t", source_path="a.md", header_path="H", chunk_index=0, chunk_size=512, chunk_overlap=50)
    with pytest.raises((AttributeError, TypeError)):
        chunk.text = "other"  # type: ignore[misc]
