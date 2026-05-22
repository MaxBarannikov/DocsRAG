"""Markdown document loader.

Recursively reads Markdown files from a source directory and produces
a list of RawDocument objects with path metadata.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True, slots=True)
class RawDocument:
    """A loaded markdown file with minimal metadata."""

    content: str
    source_path: Path
    relative_path: str  # path relative to docs root, used for display


def load_markdown_files(docs_root: Path) -> list[RawDocument]:
    """Load all .md files recursively from `docs_root`.

    Args:
        docs_root: Directory containing markdown documentation.

    Returns:
        List of RawDocument objects, one per .md file.

    Raises:
        FileNotFoundError: If `docs_root` does not exist.
    """
    if not docs_root.exists():
        raise FileNotFoundError(
            f"Docs root does not exist: {docs_root}. Run ./indexing/fetch_docs.sh to download documentation."
        )

    md_files = sorted(docs_root.rglob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files under {docs_root}")

    documents: list[RawDocument] = []
    skipped = 0

    for md_path in md_files:
        try:
            content = md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Skipping non-UTF8 file: {md_path}")
            skipped += 1
            continue

        # Skip nearly-empty files (boilerplate, redirects, etc.)
        if len(content.strip()) < 50:
            skipped += 1
            continue

        documents.append(
            RawDocument(
                content=content,
                source_path=md_path,
                relative_path=str(md_path.relative_to(docs_root)),
            )
        )

    logger.info(f"Loaded {len(documents)} documents ({skipped} skipped as empty or non-UTF8)")
    return documents
