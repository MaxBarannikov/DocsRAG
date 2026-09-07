"""Print the top retrieval results for a query.

A developer tool, not a test: nothing is asserted. Automated checks live in tests/.

    python -m indexing.query_cli "how to define a path parameter" --top-k 3
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger
from qdrant_client import QdrantClient

from core.config import settings
from core.health import DependencyUnavailableError, check_qdrant
from embeddings import make_embedder

PREVIEW_CHARS = 300


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a retrieval check against Qdrant.")
    parser.add_argument("query", type=str, help="Natural language query.")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="Number of results to return.")
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.active_qdrant_collection,
        help="Collection to query (default: routed by EMBEDDER_BACKEND).",
    )
    args = parser.parse_args()

    try:
        client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
        check_qdrant(client)
        embedder = make_embedder()
        logger.info("Query: {!r} against collection {!r}", args.query, args.collection)
        query_vector = embedder.encode([args.query], show_progress=False)[0]
        results = client.query_points(
            collection_name=args.collection,
            query=query_vector,
            limit=args.top_k,
            with_payload=True,
        ).points
    except DependencyUnavailableError as exc:
        logger.error("{}", exc)
        return 1
    except Exception:  # noqa: BLE001 — CLI boundary: report and exit non-zero
        logger.exception("Retrieval check failed")
        return 1

    if not results:
        logger.warning("No results. Is the collection indexed? Run `make reindex`.")
        return 1

    print(f"\nTop {len(results)} results:\n" + "=" * 60)  # noqa: T201 — CLI output
    for i, point in enumerate(results, start=1):
        payload = point.payload or {}
        preview = str(payload.get("text", ""))[:PREVIEW_CHARS].replace("\n", " ")
        print(  # noqa: T201 — CLI output
            f"\n[{i}] score={point.score:.4f}"
            f"\n    source:  {payload.get('source_path')}"
            f"\n    section: {payload.get('header_path')}"
            f"\n    preview: {preview}..."
        )
    print()  # noqa: T201 — CLI output
    return 0


if __name__ == "__main__":
    sys.exit(main())
