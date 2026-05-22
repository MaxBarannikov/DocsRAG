"""Smoke test: query Qdrant and print top results.

Usage:
    python -m indexing.smoke_test "how to define a path parameter"
    python -m indexing.smoke_test "how to handle file uploads" --top-k 3
"""

import argparse
import sys

from loguru import logger

from api.config import settings
from embeddings import PytorchEmbedder
from indexing.qdrant_store import QdrantStore


def main() -> int:
    p = argparse.ArgumentParser(description="Run a smoke-test query against Qdrant.")
    p.add_argument("query", type=str, help="Natural language query.")
    p.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    p.add_argument(
        "--collection",
        type=str,
        default=settings.qdrant_collection,
    )
    args = p.parse_args()

    embedder = PytorchEmbedder(settings.embedding_model)
    store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=args.collection,
        vector_dim=embedder.dimension,
    )

    logger.info(f"Query: {args.query!r}")
    query_vec = embedder.encode([args.query], show_progress=False)[0]

    results = store.client.query_points(
        collection_name=args.collection,
        query=query_vec,
        limit=args.top_k,
        with_payload=True,
    ).points

    print(f"\nTop {len(results)} results:\n" + "=" * 60)
    for i, point in enumerate(results, 1):
        payload = point.payload or {}
        print(f"\n[{i}] score={point.score:.4f}")
        print(f"    source: {payload.get('source_path')}")
        print(f"    section: {payload.get('header_path')}")
        text = payload.get("text", "")
        preview = text[:300].replace("\n", " ")
        print(f"    preview: {preview}...")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
