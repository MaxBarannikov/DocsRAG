"""Document chunking — two-pass: split by Markdown headers, then by character count."""

from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from indexing.loader import RawDocument


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single chunk ready for embedding."""

    text: str
    source_path: str  # relative path within docs root
    header_path: str  # e.g. "Tutorial > Path Parameters > Data conversion"
    chunk_index: int  # 0-based index of this chunk within the source document


_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep headers in chunk text — helps retrieval
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Chunk] = []

    for doc in documents:
        header_chunks = header_splitter.split_text(doc.content)

        chunk_idx = 0
        for hc in header_chunks:
            header_path = _build_header_path(hc.metadata)

            sub_chunks = char_splitter.split_text(hc.page_content)
            for sub in sub_chunks:
                # Filter out tiny chunks that are likely just headers with no body
                if len(sub.strip()) < 50:
                    continue

                all_chunks.append(
                    Chunk(
                        text=sub,
                        source_path=doc.relative_path,
                        header_path=header_path,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1

    logger.info(
        f"Produced {len(all_chunks)} chunks from {len(documents)} documents "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks


def _build_header_path(metadata: dict[str, str]) -> str:
    """Reconstruct hierarchical header path like 'h1 > h2 > h3'."""
    parts = [metadata.get(level) for level in ("h1", "h2", "h3", "h4")]
    return " > ".join(p for p in parts if p)
