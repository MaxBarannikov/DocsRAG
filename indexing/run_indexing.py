"""Indexing pipeline entry point.

Usage:
    python -m indexing.run_indexing
    python -m indexing.run_indexing --chunk-size 1024 --overlap 100
    python -m indexing.run_indexing --recreate

The pipeline:
    1. Loads markdown documents from settings.docs_source_path.
    2. Chunks them with hierarchical (header + char) splitting.
    3. Encodes chunks with the embedding model.
    4. Upserts chunks + embeddings into Qdrant.
"""

import argparse
import sys
import time

from loguru import logger

from api.config import settings
from embeddings import make_embedder
from indexing.chunker import chunk_documents
from indexing.loader import load_markdown_files
from indexing.qdrant_store import QdrantStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index documentation into Qdrant.")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Target chunk size in characters (default: {settings.chunk_size})",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap in characters (default: {settings.chunk_overlap})",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=settings.active_qdrant_collection,
        help=f"Qdrant collection (default: routed by EMBEDDER_BACKEND, currently {settings.active_qdrant_collection!r})",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    # Embedder is selected via EMBEDDER_BACKEND env var; the underlying model
    # (HF id for pytorch, dir path for onnx) lives in api/config.py settings.
    # No --embedding-model CLI flag: it would have backend-dependent semantics
    # and only the pytorch path historically used it.
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DocsRAG indexing pipeline")
    logger.info("=" * 60)
    logger.info(f"Source:           {settings.docs_source_path}")
    logger.info(f"Collection:       {args.collection}")
    logger.info(f"Chunk size:       {args.chunk_size}")
    logger.info(f"Overlap:          {args.overlap}")
    logger.info(f"Embedder backend: {settings.embedder_backend}")
    logger.info(f"Recreate:         {args.recreate}")
    logger.info("=" * 60)

    # 1. Load
    documents = load_markdown_files(settings.docs_source_path)
    if not documents:
        logger.error("No documents loaded — aborting.")
        return 1

    # 2. Chunk
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )
    if not chunks:
        logger.error("No chunks produced — aborting.")
        return 1

    # 3. Embed
    embedder = make_embedder()
    texts = [c.text for c in chunks]
    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress=True)

    # 4. Store
    store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=args.collection,
        vector_dim=embedder.dimension,
    )
    if args.recreate:
        store.recreate_collection()
    elif not store.client.collection_exists(args.collection):
        logger.info(f"Collection '{args.collection}' does not exist, creating it.")
        store.recreate_collection()

    store.upsert_chunks(chunks, embeddings)

    # Summary
    elapsed = time.perf_counter() - started
    final_count = store.count()
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Documents indexed: {len(documents)}")
    logger.info(f"Chunks produced:   {len(chunks)}")
    logger.info(f"Points in Qdrant:  {final_count}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
