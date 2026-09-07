"""Indexing pipeline entry point.

Usage:
    python -m indexing.run_indexing
    python -m indexing.run_indexing --chunk-size 1024 --overlap 100
    python -m indexing.run_indexing --recreate
"""

from __future__ import annotations

import argparse
import sys
import time

from core.proxy import strip_socks_proxy_env

# Must run before any httpx client is constructed.
strip_socks_proxy_env()

from loguru import logger  # noqa: E402 — after the proxy cleanup above

from core.config import settings  # noqa: E402
from core.health import DependencyUnavailableError  # noqa: E402
from embeddings import make_embedder  # noqa: E402
from indexing.chunker import Chunk, chunk_documents  # noqa: E402
from indexing.loader import load_markdown_files  # noqa: E402
from indexing.qdrant_store import QdrantStore  # noqa: E402


class IndexingError(RuntimeError):
    """Carries a message meant for a human."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index documentation into Qdrant.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Target chunk size in characters (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap in characters (default: {settings.chunk_overlap})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.active_qdrant_collection,
        help=(
            "Qdrant collection to write to "
            f"(default: routed by EMBEDDER_BACKEND, currently {settings.active_qdrant_collection!r})"
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    return parser.parse_args()


def _qdrant_reachable(url: str) -> bool:
    from qdrant_client import QdrantClient

    try:
        QdrantClient(url=url, check_compatibility=False).get_collections()
    except Exception:  # noqa: BLE001 — any transport failure means "not reachable"
        return False
    return True


def build_chunks(args: argparse.Namespace) -> list[Chunk]:

    try:
        documents = load_markdown_files(settings.docs_source_path)
    except FileNotFoundError as exc:
        raise IndexingError(str(exc)) from exc

    if not documents:
        msg = f"No documents found under {settings.docs_source_path}. Run `make fetch-docs` first."
        raise IndexingError(msg)

    try:
        chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    except ValueError as exc:
        msg = f"Invalid chunking parameters: {exc}"
        raise IndexingError(msg) from exc

    if not chunks:
        msg = "Chunking produced nothing; check the corpus and the chunk size."
        raise IndexingError(msg)

    logger.info("Loaded {} documents into {} chunks", len(documents), len(chunks))
    return chunks


def index_chunks(chunks: list[Chunk], collection: str, *, recreate: bool) -> int:
    """Returns the resulting point count."""
    if not _qdrant_reachable(settings.qdrant_url):
        msg = f"Qdrant is not reachable at {settings.qdrant_url}.\nStart the stack with `make up`, then retry."
        raise IndexingError(msg)

    embedder = make_embedder()
    logger.info("Encoding {} chunks...", len(chunks))
    embeddings = embedder.encode([c.text for c in chunks], show_progress=True)

    store = QdrantStore(url=settings.qdrant_url, collection_name=collection, vector_dim=embedder.dimension)
    if recreate or not store.client.collection_exists(collection):
        store.recreate_collection()
    else:
        store.assert_dimension_matches()

    store.upsert_chunks(chunks, embeddings)
    return store.count()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    logger.info("DocsRAG indexing pipeline")
    logger.info(
        "source={} collection={} chunk_size={} overlap={} embedder={} recreate={}",
        settings.docs_source_path,
        args.collection,
        args.chunk_size,
        args.overlap,
        settings.embedder_backend,
        args.recreate,
    )

    try:
        chunks = build_chunks(args)
        final_count = index_chunks(chunks, args.collection, recreate=args.recreate)
    except (IndexingError, DependencyUnavailableError) as exc:
        logger.error("{}", exc)
        return 1
    except Exception:  # noqa: BLE001 — CLI boundary: report and exit non-zero
        logger.exception("Indexing failed")
        return 1

    logger.info(
        "Done in {:.1f}s | chunks={} points_in_qdrant={}",
        time.perf_counter() - started,
        len(chunks),
        final_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
