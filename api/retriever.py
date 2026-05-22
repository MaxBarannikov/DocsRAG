"""Hybrid retriever: dense (Qdrant) + sparse (BM25) with optional cross-encoder reranking."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from api.config import settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from api.rag import RetrievalHit
    from indexing.embeddings import EmbeddingModel

BM25_INDEX_PATH = Path("data/bm25_index.pkl")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25 index over Qdrant collection chunks.

    Built once from all points in the collection. Serialized to disk so warm
    restarts skip the Qdrant scroll. If the collection changes (reindex), delete
    data/bm25_index.pkl to force a rebuild.
    """

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        corpus = [_tokenize(doc.page_content) for doc in docs]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_n: int) -> list[tuple[Document, float]]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [(self._docs[i], float(scores[i])) for i in top_indices]

    @classmethod
    def build_from_qdrant(cls, client: QdrantClient, collection: str) -> BM25Index:
        """Scroll all points from Qdrant and build the index."""
        logger.info("Building BM25 index from Qdrant collection '{}'...", collection)
        docs: list[Document] = []
        offset = None
        while True:
            result = client.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                payload = point.payload or {}
                docs.append(
                    Document(
                        page_content=str(payload.get("text", "")),
                        metadata={
                            "source_path": payload.get("source_path", "unknown"),
                            "header_path": payload.get("header_path", ""),
                            "chunk_index": payload.get("chunk_index", -1),
                        },
                    )
                )
            if next_offset is None:
                break
            offset = next_offset
        logger.info("BM25 index built from {} documents", len(docs))
        return cls(docs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("BM25 index saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as f:
            # Index file is written by save() in this same process; not external input.
            index = pickle.load(f)  # noqa: S301 — trusted local file written by save()
        logger.info("BM25 index loaded from {} ({} docs)", path, len(index._docs))
        return index


def _rrf_merge(
    dense_hits: list[RetrievalHit],
    bm25_hits: list[tuple[Document, float]],
    top_k: int,
    k: int = 60,
) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion over dense and sparse result lists.

    Uses chunk_index as the document identifier for deduplication.
    k=60 is the standard RRF constant that dampens high-rank advantage.
    """
    from api.rag import RetrievalHit  # noqa: PLC0415 — circular dep: api.rag imports from api.retriever

    scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for rank, hit in enumerate(dense_hits):
        chunk_idx = int(hit.document.metadata.get("chunk_index", -rank))
        scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
        doc_map[chunk_idx] = hit.document

    for rank, (doc, _) in enumerate(bm25_hits):
        chunk_idx = int(doc.metadata.get("chunk_index", -(rank + 10000)))
        scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
        if chunk_idx not in doc_map:
            doc_map[chunk_idx] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [RetrievalHit(document=doc_map[idx], score=score) for idx, score in ranked]


class HybridRetriever:
    """Dense + BM25 retrieval with optional cross-encoder reranking."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        qdrant_client: QdrantClient,
        bm25_index: BM25Index,
        reranker: CrossEncoder | None = None,
    ) -> None:
        self._embedder = embedder
        self._qdrant_client = qdrant_client
        self._bm25 = bm25_index
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int,
        rerank_top_n: int = 20,
    ) -> list[RetrievalHit]:
        from api.rag import RetrievalHit, _scored_point_to_hit  # noqa: PLC0415 — circular dep

        fetch_n = max(top_k, rerank_top_n) if self._reranker else top_k

        query_vector = self._embedder.encode([query], show_progress=False)[0]
        dense_response = self._qdrant_client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=fetch_n,
            with_payload=True,
        )
        dense_hits = [_scored_point_to_hit(p) for p in dense_response.points]

        bm25_hits = self._bm25.search(query, top_n=fetch_n)

        merged = _rrf_merge(dense_hits, bm25_hits, top_k=fetch_n)

        if self._reranker is None:
            result = merged[:top_k]
        else:
            pairs: list[tuple[str, str]] = [(query, hit.document.page_content) for hit in merged]
            rerank_scores = self._reranker.predict(pairs)
            reranked = sorted(zip(merged, rerank_scores, strict=False), key=lambda x: x[1], reverse=True)
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
    """Load BM25 index from disk; rebuild from Qdrant if missing."""
    if BM25_INDEX_PATH.exists():
        return BM25Index.load(BM25_INDEX_PATH)
    index = BM25Index.build_from_qdrant(client, settings.qdrant_collection)
    index.save(BM25_INDEX_PATH)
    return index


def load_reranker() -> CrossEncoder:
    logger.info("Loading cross-encoder reranker '{}'", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)
