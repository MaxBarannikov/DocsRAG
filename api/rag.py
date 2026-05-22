"""RAG pipeline: retrieve relevant chunks from Qdrant, generate an answer with Ollama."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from qdrant_client import QdrantClient

from api.config import settings
from api.llm import make_llm
from api.prompts import PROMPT
from api.schemas import Source
from indexing.embeddings import EmbeddingModel

if TYPE_CHECKING:
    from qdrant_client.models import ScoredPoint

    from api.retriever import HybridRetriever

RetrievalStrategy = Literal["dense", "hybrid", "hybrid_rerank"]


@dataclass(slots=True)
class RetrievalHit:
    """A single retrieval result with score."""

    document: Document
    score: float


def _scored_point_to_hit(point: ScoredPoint) -> RetrievalHit:
    payload = point.payload or {}
    doc = Document(
        page_content=str(payload.get("text", "")),
        metadata={
            "source_path": payload.get("source_path", "unknown"),
            "header_path": payload.get("header_path", ""),
            "chunk_index": payload.get("chunk_index", -1),
        },
    )
    return RetrievalHit(document=doc, score=float(point.score))


class RAGPipeline:
    """End-to-end retrieve → generate pipeline.

    Heavy resources (embedder, Qdrant client, LLM client) are constructed
    once and reused across requests. See `get_pipeline()` for the cached
    application-wide instance.
    """

    def __init__(self, retrieval_strategy: RetrievalStrategy = "dense") -> None:
        logger.info("Initializing RAGPipeline (strategy={})", retrieval_strategy)
        self._strategy = retrieval_strategy

        # Embedder: same model used during indexing — guarantees identical
        # vector space and pooling/normalization between index- and query-time.
        self._embedder = EmbeddingModel(model_name=settings.embedding_model)

        # Qdrant: direct client — bypasses langchain-qdrant metadata handling
        # which changed in 0.2.x. Our payload is flat: text, source_path,
        # header_path, chunk_index — we map it to Document ourselves.
        self._qdrant_client = QdrantClient(url=settings.qdrant_url)

        self._llm = make_llm(temperature=0.0)
        self._chain = PROMPT | self._llm | StrOutputParser()

        # Hybrid retriever is built lazily on first use to avoid loading
        # BM25 index and reranker when strategy is "dense".
        self._hybrid_retriever = None
        if retrieval_strategy in ("hybrid", "hybrid_rerank"):
            self._hybrid_retriever = self._build_hybrid_retriever(retrieval_strategy)

        active_model = settings.vllm_model if settings.inference_backend == "vllm" else settings.ollama_model
        logger.info(
            "RAGPipeline ready | collection={} | backend={} | model={} | strategy={}",
            settings.qdrant_collection,
            settings.inference_backend,
            active_model,
            retrieval_strategy,
        )

    def _build_hybrid_retriever(self, strategy: RetrievalStrategy) -> HybridRetriever:
        from api.retriever import HybridRetriever, load_or_build_bm25, load_reranker

        bm25_index = load_or_build_bm25(self._qdrant_client)
        reranker = load_reranker() if strategy == "hybrid_rerank" else None
        return HybridRetriever(
            embedder=self._embedder,
            qdrant_client=self._qdrant_client,
            bm25_index=bm25_index,
            reranker=reranker,
        )

    def retrieve(self, query: str, top_k: int, rerank_top_n: int = 20) -> list[RetrievalHit]:
        if self._hybrid_retriever is not None:
            return self._hybrid_retriever.retrieve(query, top_k=top_k, rerank_top_n=rerank_top_n)
        return self._dense_retrieve(query, top_k)

    def _dense_retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_vector = self._embedder.encode([query], show_progress=False)[0]
        response = self._qdrant_client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = [_scored_point_to_hit(r) for r in response.points]
        logger.debug(
            "Dense retrieved {} hits for query={!r} (top_score={:.3f})",
            len(hits),
            query,
            hits[0].score if hits else 0.0,
        )
        return hits

    def generate(self, question: str, hits: list[RetrievalHit], callbacks: list | None = None) -> str:
        context = self._format_context(hits)
        invoke_config = {"callbacks": callbacks} if callbacks else {}
        answer = self._chain.invoke(
            {"context": context, "question": question},
            config=invoke_config,
        )
        return answer.strip()

    def ask(
        self,
        question: str,
        top_k: int,
        include_contexts: bool,
        rerank_top_n: int = 20,
    ) -> tuple[str, list[Source], dict[str, int]]:
        """Full pipeline: optionally translate RU→EN → retrieve → generate → optionally translate EN→RU."""
        from api.tracing import get_langfuse_handler
        from api.translation import (
            contains_cyrillic,
            translate_to_english,
            translate_to_russian,
        )

        handler = get_langfuse_handler(question)
        callbacks = [handler] if handler else None

        is_russian = contains_cyrillic(question)
        translation_ms = 0
        t_start = time.perf_counter()

        if is_russian:
            t_tr0 = time.perf_counter()
            retrieval_question = translate_to_english(self._llm, question, callbacks=callbacks)
            translation_ms += int((time.perf_counter() - t_tr0) * 1000)
            logger.info("RU→EN | in={!r} | out={!r}", question, retrieval_question)
        else:
            retrieval_question = question

        t0 = time.perf_counter()
        hits = self.retrieve(retrieval_question, top_k=top_k, rerank_top_n=rerank_top_n)
        t1 = time.perf_counter()
        answer_en = self.generate(retrieval_question, hits, callbacks=callbacks)
        t2 = time.perf_counter()
        answer = answer_en

        if is_russian:
            t_tr1 = time.perf_counter()
            answer = translate_to_russian(self._llm, answer_en, callbacks=callbacks)
            translation_ms += int((time.perf_counter() - t_tr1) * 1000)
            logger.info("EN→RU | in={!r} | out={!r}", answer_en, answer)

        t_end = time.perf_counter()

        sources = [self._hit_to_source(h, include_contexts) for h in hits]
        timings = {
            "retrieval_ms": int((t1 - t0) * 1000),
            "generation_ms": int((t2 - t1) * 1000),
            "translation_ms": translation_ms,
            "total_ms": int((t_end - t_start) * 1000),
        }
        logger.info(
            "ask | strategy={} lang={} retrieval={}ms generation={}ms translation={}ms total={}ms | hits={} | q={!r}",
            self._strategy,
            "ru" if is_russian else "en",
            timings["retrieval_ms"],
            timings["generation_ms"],
            timings["translation_ms"],
            timings["total_ms"],
            len(hits),
            question[:80],
        )
        return answer, sources, timings

    def collection_points_count(self) -> int:
        """Number of points in the Qdrant collection (used by /health)."""
        info = self._qdrant_client.count(collection_name=settings.qdrant_collection, exact=True)
        return int(info.count)

    @staticmethod
    def _format_context(hits: list[RetrievalHit]) -> str:
        if not hits:
            return "(no relevant context found)"
        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            md = hit.document.metadata or {}
            source_path = md.get("source_path", "unknown")
            header_path = md.get("header_path", "")
            header_line = f" — {header_path}" if header_path else ""
            blocks.append(f"[{i}] {source_path}{header_line}\n{hit.document.page_content}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _hit_to_source(hit: RetrievalHit, include_contexts: bool) -> Source:
        md = hit.document.metadata or {}
        return Source(
            source_path=str(md.get("source_path", "unknown")),
            header_path=str(md.get("header_path", "")),
            score=hit.score,
            content=hit.document.page_content if include_contexts else None,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Application-wide singleton. Lazily constructed on first call."""
    return RAGPipeline()
