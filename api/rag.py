"""Retrieval-augmented generation pipeline: embed, retrieve, generate."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from qdrant_client import QdrantClient

from api.llm import make_llm
from api.prompts import PROMPT
from api.schemas import Source
from api.tracing import get_langfuse_handler
from api.translation import contains_cyrillic, translate_to_english, translate_to_russian
from core.config import RetrievalStrategy, settings
from core.types import RetrievalHit, scored_point_to_hit
from embeddings import make_embedder

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig

    from api.retriever import HybridRetriever

NO_CONTEXT_PLACEHOLDER = "(no relevant context found)"


class RAGPipeline:
    """Construction loads the embedding model and opens clients — use `get_pipeline()`."""

    def __init__(self, retrieval_strategy: RetrievalStrategy | None = None) -> None:
        strategy = retrieval_strategy or settings.retrieval_strategy
        if strategy not in ("dense", "hybrid", "hybrid_rerank"):
            msg = f"Unknown retrieval strategy {strategy!r}. Expected: dense | hybrid | hybrid_rerank."
            raise ValueError(msg)

        logger.info("Initializing RAGPipeline (strategy={})", strategy)
        self._strategy: RetrievalStrategy = strategy
        self._embedder = make_embedder()
        # The server version is pinned in compose; the probe only adds a startup warning.
        self._qdrant_client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
        self._llm = make_llm(temperature=0.0)
        self._chain = PROMPT | self._llm | StrOutputParser()

        self._hybrid_retriever: HybridRetriever | None = None
        if strategy in ("hybrid", "hybrid_rerank"):
            self._hybrid_retriever = self._build_hybrid_retriever(strategy)

        active_model = settings.vllm_model if settings.inference_backend == "vllm" else settings.ollama_model
        logger.info(
            "RAGPipeline ready | collection={} | backend={} | model={} | strategy={}",
            settings.active_qdrant_collection,
            settings.inference_backend,
            active_model,
            strategy,
        )

    @property
    def llm(self) -> BaseChatModel:
        """Reused by the agent graph."""
        return self._llm

    @property
    def strategy(self) -> RetrievalStrategy:
        return self._strategy

    def _build_hybrid_retriever(self, strategy: RetrievalStrategy) -> HybridRetriever:
        from api.retriever import HybridRetriever, load_or_build_bm25, load_reranker

        return HybridRetriever(
            embedder=self._embedder,
            qdrant_client=self._qdrant_client,
            bm25_index=load_or_build_bm25(self._qdrant_client),
            reranker=load_reranker() if strategy == "hybrid_rerank" else None,
        )

    def retrieve(self, query: str, top_k: int, rerank_top_n: int | None = None) -> list[RetrievalHit]:
        if self._hybrid_retriever is not None:
            return self._hybrid_retriever.retrieve(query, top_k=top_k, rerank_top_n=rerank_top_n)
        return self._dense_retrieve(query, top_k)

    def _dense_retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_vector = self._embedder.encode([query], show_progress=False)[0]
        response = self._qdrant_client.query_points(
            collection_name=settings.active_qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = [scored_point_to_hit(point) for point in response.points]
        logger.debug(
            "Dense retrieved {} hits for query={!r} (top_score={:.3f})",
            len(hits),
            query,
            hits[0].score if hits else 0.0,
        )
        return hits

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> str:
        config: RunnableConfig = {"callbacks": callbacks} if callbacks else {}
        answer = self._chain.invoke(
            {"context": self.format_context(hits), "question": question},
            config=config,
        )
        return answer.strip()

    def ask(
        self,
        question: str,
        top_k: int,
        *,
        include_contexts: bool,
        rerank_top_n: int | None = None,
    ) -> tuple[str, list[Source], dict[str, int]]:
        """Translates to and from Russian when the question is Cyrillic."""
        handler = get_langfuse_handler()
        callbacks = [handler] if handler else None

        is_russian = contains_cyrillic(question)
        translation_ms = 0
        started = time.perf_counter()

        if is_russian:
            t0 = time.perf_counter()
            retrieval_question = translate_to_english(self._llm, question, callbacks=callbacks)
            translation_ms += _elapsed_ms(t0)
            logger.info("RU->EN | in={!r} | out={!r}", question, retrieval_question)
        else:
            retrieval_question = question

        t_retrieval = time.perf_counter()
        hits = self.retrieve(retrieval_question, top_k=top_k, rerank_top_n=rerank_top_n)
        retrieval_ms = _elapsed_ms(t_retrieval)

        t_generation = time.perf_counter()
        answer = self.generate(retrieval_question, hits, callbacks=callbacks)
        generation_ms = _elapsed_ms(t_generation)

        if is_russian:
            t1 = time.perf_counter()
            answer_en = answer
            answer = translate_to_russian(self._llm, answer_en, callbacks=callbacks)
            translation_ms += _elapsed_ms(t1)
            logger.info("EN->RU | in={!r} | out={!r}", answer_en, answer)

        sources = [self.hit_to_source(hit, include_contexts=include_contexts) for hit in hits]
        timings = {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "translation_ms": translation_ms,
            # Wall-clock, so it covers the untimed glue between stages.
            "total_ms": _elapsed_ms(started),
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

        return int(self._qdrant_client.count(collection_name=settings.active_qdrant_collection, exact=True).count)

    @staticmethod
    def format_context(hits: list[RetrievalHit]) -> str:
        """Numbered, cited context block for the prompt."""
        if not hits:
            return NO_CONTEXT_PLACEHOLDER

        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            metadata = hit.document.metadata or {}
            header_path = metadata.get("header_path", "")
            header_line = f" — {header_path}" if header_path else ""
            blocks.append(f"[{i}] {metadata.get('source_path', 'unknown')}{header_line}\n{hit.document.page_content}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def hit_to_source(hit: RetrievalHit, *, include_contexts: bool) -> Source:

        metadata = hit.document.metadata or {}
        return Source(
            source_path=str(metadata.get("source_path", "unknown")),
            header_path=str(metadata.get("header_path", "")),
            score=hit.score,
            content=hit.document.page_content if include_contexts else None,
        )


def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:

    return RAGPipeline()
