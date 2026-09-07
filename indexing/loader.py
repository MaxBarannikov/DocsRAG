"""Markdown document loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from indexing.chunker import MIN_CONTENT_CHARS

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class RawDocument:
    content: str
    source_path: Path
    relative_path: str  # path relative to the docs root, used for citations


def load_markdown_files(docs_root: Path) -> list[RawDocument]:

    if not docs_root.exists():
        msg = f"Docs root does not exist: {docs_root}. Run `make fetch-docs` to download the documentation."
        raise FileNotFoundError(msg)

    md_files = sorted(docs_root.rglob("*.md"))
    logger.info("Found {} markdown files under {}", len(md_files), docs_root)

    documents: list[RawDocument] = []
    skipped_unreadable = 0
    skipped_empty = 0

    for md_path in md_files:
        try:
            content = md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF8 file: {}", md_path)
            skipped_unreadable += 1
            continue
        except OSError as exc:
            logger.warning("Skipping unreadable file {}: {}", md_path, exc)
            skipped_unreadable += 1
            continue

        if len(content.strip()) < MIN_CONTENT_CHARS:
            skipped_empty += 1
            continue

        try:
            relative_path = str(md_path.relative_to(docs_root))
        except ValueError:
            # rglob can escape docs_root through a symlinked directory.
            logger.warning("Skipping file outside the docs root: {}", md_path)
            skipped_unreadable += 1
            continue

        documents.append(RawDocument(content=content, source_path=md_path, relative_path=relative_path))

    logger.info(
        "Loaded {} documents ({} skipped as empty, {} as unreadable)",
        len(documents),
        skipped_empty,
        skipped_unreadable,
    )
    return documents
