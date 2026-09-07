"""Hybrid retrieval: dense (Qdrant) + sparse (BM25), with optional cross-encoder reranking."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger
from rank_bm25 import BM25Okapi

from core.config import settings
from core.health import check_qdrant
from core.types import RetrievalHit, document_chunk_key, payload_to_document, scored_point_to_hit

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.documents import Document
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder

    from core.embedder import Embedder

# Bump to invalidate on-disk caches after a format change.
BM25_CACHE_VERSION = 1
_SCROLL_BATCH = 500
# Guards against a scroll that never advances its offset.
_MAX_SCROLL_BATCHES = 10_000
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self._bm25 = BM25Okapi([_tokenize(doc.page_content) for doc in docs])

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, top_n: int) -> list[tuple[Document, float]]:
        if not self._docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [(self._docs[i], float(scores[i])) for i in top_indices]

    @classmethod
    def build_from_qdrant(cls, client: QdrantClient, collection: str) -> BM25Index:

        logger.info("Building BM25 index from Qdrant collection '{}'...", collection)
        docs: list[Document] = []
        offset = None

        for _ in range(_MAX_SCROLL_BATCHES):
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            docs.extend(payload_to_document(point.payload) for point in points)
            if next_offset is None:
                break
            offset = next_offset
        else:
            msg = f"Scrolling '{collection}' exceeded {_MAX_SCROLL_BATCHES} batches; aborting to avoid a hang."
            raise RuntimeError(msg)

        logger.info("BM25 index built from {} documents", len(docs))
        return cls(docs)

    def save(self, path: Path, collection: str) -> None:
        """JSON rather than pickle: the cache lives in a writable, bind-mountable
        directory, and unpickling it would execute its contents at API startup.
        """
        payload = {
            "version": BM25_CACHE_VERSION,
            "collection": collection,
            "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in self._docs],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("BM25 index saved to {} ({} docs)", path, len(self._docs))

    @classmethod
    def load(cls, path: Path, collection: str, expected_count: int | None = None) -> BM25Index | None:
        """Load a cached index, or return None if it is missing, corrupt or stale."""
        from langchain_core.documents import Document as LcDocument

        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("BM25 cache at {} is unreadable ({}) — rebuilding", path, exc)
            return None

        if not isinstance(raw, dict) or raw.get("version") != BM25_CACHE_VERSION:
            logger.warning("BM25 cache at {} has an unsupported format — rebuilding", path)
            return None
        if raw.get("collection") != collection:
            logger.warning(
                "BM25 cache at {} was built for collection {!r}, not {!r} — rebuilding",
                path,
                raw.get("collection"),
                collection,
            )
            return None

        documents = [
            LcDocument(page_content=entry["page_content"], metadata=entry.get("metadata", {}))
            for entry in raw.get("documents", [])
        ]
        if expected_count is not None and len(documents) != expected_count:
            logger.warning(
                "BM25 cache at {} holds {} docs but '{}' now has {} points — rebuilding",
                path,
                len(documents),
                collection,
                expected_count,
            )
            return None

        logger.info("BM25 index loaded from {} ({} docs)", path, len(documents))
        return cls(documents)


def rrf_merge(
    dense_hits: list[RetrievalHit],
    bm25_hits: list[tuple[Document, float]],
    top_k: int,
    k: int = RRF_K,
) -> list[RetrievalHit]:
    """Fuse two ranked lists with reciprocal rank fusion.

    Keyed on `(source_path, chunk_index)`: `chunk_index` restarts per document, so
    keying on it alone collapses chunks from different files into one entry.
    """
    scores: dict[tuple[str, int], float] = {}
    doc_map: dict[tuple[str, int], Document] = {}

    for rank, hit in enumerate(dense_hits):
        key = hit.chunk_key
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = hit.document

    for rank, (doc, _score) in enumerate(bm25_hits):
        key = document_chunk_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map.setdefault(key, doc)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [RetrievalHit(document=doc_map[key], score=score) for key, score in ranked]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        qdrant_client: QdrantClient,
        bm25_index: BM25Index,
        reranker: CrossEncoder | None = None,
    ) -> None:
        self._embedder = embedder
        self._qdrant_client = qdrant_client
        self._bm25 = bm25_index
        self._reranker = reranker

    def retrieve(self, query: str, top_k: int, rerank_top_n: int | None = None) -> list[RetrievalHit]:
        rerank_top_n = rerank_top_n if rerank_top_n is not None else settings.rerank_top_n
        # Fusing two lists of length top_k leaves RRF almost nothing to merge.
        fetch_n = max(top_k, rerank_top_n, settings.hybrid_candidates)

        query_vector = self._embedder.encode([query], show_progress=False)[0]
        dense_response = self._qdrant_client.query_points(
            collection_name=settings.active_qdrant_collection,
            query=query_vector,
            limit=fetch_n,
            with_payload=True,
        )
        dense_hits = [scored_point_to_hit(point) for point in dense_response.points]
        bm25_hits = self._bm25.search(query, top_n=fetch_n)

        merged = rrf_merge(dense_hits, bm25_hits, top_k=fetch_n)

        if self._reranker is None:
            result = merged[:top_k]
        else:
            candidates = merged[:rerank_top_n]
            pairs = [[query, hit.document.page_content] for hit in candidates]
            # Upstream types `predict` over a multimodal union no list[list[str]] satisfies.
            rerank_scores = self._reranker.predict(pairs)  # type: ignore[arg-type]
            reranked = sorted(
                zip(candidates, rerank_scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            result = [RetrievalHit(document=hit.document, score=float(score)) for hit, score in reranked[:top_k]]

        logger.debug(
            "HybridRetriever | dense={} bm25={} merged={} returned={} top_score={:.3f}",
            len(dense_hits),
            len(bm25_hits),
            len(merged),
            len(result),
            result[0].score if result else 0.0,
        )
        return result


def load_or_build_bm25(client: QdrantClient) -> BM25Index:
    """Requires a populated collection — the index is built by scrolling every point out."""
    collection = settings.active_qdrant_collection
    path = settings.bm25_index_path(collection)
    expected_count = check_qdrant(client)

    if path.exists():
        cached = BM25Index.load(path, collection, expected_count)
        if cached is not None:
            return cached

    index = BM25Index.build_from_qdrant(client, collection)
    index.save(path, collection)
    return index


def load_reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder reranker '{}'", settings.reranker_model)
    return CrossEncoder(settings.reranker_model)
