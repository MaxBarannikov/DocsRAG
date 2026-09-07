"""Document chunking — two passes: split on Markdown headers, then by character count."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from core.config import settings

if TYPE_CHECKING:
    from indexing.loader import RawDocument

# Below this a fragment is almost always a bare heading. Also used by the loader.
MIN_CONTENT_CHARS = 50

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    source_path: str  # path relative to the docs root
    header_path: str  # e.g. "Tutorial > Path Parameters > Data conversion"
    chunk_index: int  # 0-based index WITHIN its source document, not corpus-wide
    chunk_size: int  # chunking parameters, carried into the Qdrant payload
    chunk_overlap: int


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:

    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if chunk_overlap >= chunk_size:
        msg = f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        raise ValueError(msg)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keeping headers in the text measurably helps retrieval
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Chunk] = []
    dropped = 0

    for doc in documents:
        chunk_idx = 0
        for header_chunk in header_splitter.split_text(doc.content):
            header_path = _build_header_path(header_chunk.metadata)

            for sub in char_splitter.split_text(header_chunk.page_content):
                if len(sub.strip()) < MIN_CONTENT_CHARS:
                    dropped += 1
                    continue

                all_chunks.append(
                    Chunk(
                        text=sub,
                        source_path=doc.relative_path,
                        header_path=header_path,
                        chunk_index=chunk_idx,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                chunk_idx += 1

    logger.info(
        "Produced {} chunks from {} documents ({} dropped as too short; chunk_size={}, overlap={})",
        len(all_chunks),
        len(documents),
        dropped,
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


def _build_header_path(metadata: dict[str, object]) -> str:
    """e.g. 'Tutorial > Path Parameters > Data conversion'."""
    parts = [metadata.get(level) for level in ("h1", "h2", "h3", "h4")]
    return " > ".join(str(part) for part in parts if part)
