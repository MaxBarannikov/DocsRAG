"""Agentic RAG graph via LangGraph.

Graph flow:
  START → query_rewriter → retriever → relevance_grader
                ↑                              |
                |     (low relevance + retries left)
                └──────────────────────────────┘
                                               |
                         (sufficient relevance or max retries)
                                               ↓
                                           generator → END
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypedDict

# langchain_ollama initialises an httpx client at import time and picks up
# SOCKS proxy env vars, which breaks it if socksio isn't installed.
# Must run before any import chain that transitively loads httpx.
for _var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_var, None)

from langchain_core.messages import HumanMessage  # noqa: E402 — imports below env-pop guard
from langgraph.graph import END, START, StateGraph  # noqa: E402
from loguru import logger  # noqa: E402

from api.llm import make_llm  # noqa: E402

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from api.rag import RAGPipeline
    from api.schemas import Source

MAX_RETRIES = 1  # one retry after initial retrieval
MIN_RELEVANT_CHUNKS = 2  # minimum chunks that must pass grading to skip retry


class GraphState(TypedDict):
    question: str  # original user question — never mutated
    query: str  # current retrieval query (rewritten on retry)
    top_k: int
    hits: list  # list[RetrievalHit] from retriever
    relevant_hits: list  # list[RetrievalHit] that passed grading
    answer: str
    sources: list  # list[Source]
    retry_count: int  # incremented by relevance_grader each pass
    timings: dict  # rewrite_ms, retrieval_ms, grading_ms, generation_ms, total_ms
    callbacks: list  # LangFuse CallbackHandler list, empty when tracing disabled


def build_agent_graph(pipeline: RAGPipeline) -> CompiledStateGraph:  # noqa: C901, PLR0915
    # LangGraph builder: 4 closure-capturing node fns inflate the statement count;
    # extracting them only moves closures up a level without simplifying the flow.
    llm = make_llm(temperature=0.0)
    # Separate grader LLM with json_mode=True so Qwen reliably outputs structured verdicts.
    grader_llm = make_llm(temperature=0.0, json_mode=True)

    def query_rewriter(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        if retry_count == 0:
            prompt = (
                "Rewrite the following question to improve document retrieval. "
                "Output only the rewritten question, nothing else.\n\n"
                f"Question: {question}"
            )
        else:
            prompt = (
                "The previous retrieval did not return sufficiently relevant documents. "
                "Rewrite the query using different keywords or phrasing to find better results. "
                "Output only the rewritten query, nothing else.\n\n"
                f"Original question: {question}\n"
                f"Previous query: {state.get('query', question)}"
            )

        callbacks = state.get("callbacks") or []
        t0 = time.perf_counter()
        response = llm.invoke([HumanMessage(content=prompt)], config={"callbacks": callbacks})
        rewrite_ms = int((time.perf_counter() - t0) * 1000)

        query = str(response.content).strip()
        logger.info("query_rewriter | retry={} rewrite={}ms | query={!r}", retry_count, rewrite_ms, query[:80])

        timings = dict(state.get("timings") or {})
        timings["rewrite_ms"] = timings.get("rewrite_ms", 0) + rewrite_ms
        return {"query": query, "timings": timings}

    def retriever(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        hits = pipeline.retrieve(state["query"], top_k=state.get("top_k", 5))
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        logger.info("retriever | hits={} retrieval={}ms | query={!r}", len(hits), retrieval_ms, state["query"][:80])

        timings = dict(state.get("timings") or {})
        timings["retrieval_ms"] = timings.get("retrieval_ms", 0) + retrieval_ms
        return {"hits": hits, "timings": timings}

    def relevance_grader(state: GraphState) -> dict[str, Any]:
        query = state["query"]
        hits = state["hits"]

        callbacks = state.get("callbacks") or []
        t0 = time.perf_counter()
        relevant_hits = []
        for hit in hits:
            prompt = (
                f"Question: {query}\n\n"
                f"Document: {hit.document.page_content[:500]}\n\n"
                "Is this document relevant to answering the question? "
                'Respond with JSON: {"relevant": true or false}'
            )
            try:
                response = grader_llm.invoke([HumanMessage(content=prompt)], config={"callbacks": callbacks})
                if json.loads(str(response.content)).get("relevant", False):
                    relevant_hits.append(hit)
            except Exception as exc:
                # On parse failure include the chunk — safer to over-retrieve than drop good context.
                logger.warning("relevance_grader parse error (including chunk): {}", exc)
                relevant_hits.append(hit)

        grading_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "relevance_grader | relevant={}/{} grading={}ms",
            len(relevant_hits),
            len(hits),
            grading_ms,
        )

        timings = dict(state.get("timings") or {})
        timings["grading_ms"] = timings.get("grading_ms", 0) + grading_ms
        return {
            "relevant_hits": relevant_hits,
            "retry_count": state.get("retry_count", 0) + 1,
            "timings": timings,
        }

    def generator(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        # Fall back to all hits if grading produced nothing relevant.
        hits_for_gen = state.get("relevant_hits") or state["hits"]

        callbacks = state.get("callbacks") or []
        t0 = time.perf_counter()
        answer = pipeline.generate(question, hits_for_gen, callbacks=callbacks or None)
        generation_ms = int((time.perf_counter() - t0) * 1000)

        sources = [pipeline._hit_to_source(h, include_contexts=False) for h in hits_for_gen]

        timings = dict(state.get("timings") or {})
        timings["generation_ms"] = generation_ms
        timings["total_ms"] = sum(timings.values())
        logger.info(
            "generator | generation={}ms total={}ms answer_len={}",
            generation_ms,
            timings["total_ms"],
            len(answer),
        )
        return {"answer": answer, "sources": sources, "timings": timings}

    def should_retry(state: GraphState) -> str:
        relevant = state.get("relevant_hits", [])
        retry_count = state.get("retry_count", 0)
        # retry_count was already incremented by relevance_grader, so compare with <=.
        if len(relevant) < MIN_RELEVANT_CHUNKS and retry_count <= MAX_RETRIES:
            logger.info(
                "should_retry → retry (relevant={} < {}, retry_count={} ≤ {})",
                len(relevant),
                MIN_RELEVANT_CHUNKS,
                retry_count,
                MAX_RETRIES,
            )
            return "retry"
        return "generate"

    graph: StateGraph = StateGraph(GraphState)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("retriever", retriever)
    graph.add_node("relevance_grader", relevance_grader)
    graph.add_node("generator", generator)

    graph.add_edge(START, "query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "relevance_grader")
    graph.add_conditional_edges(
        "relevance_grader",
        should_retry,
        {"retry": "query_rewriter", "generate": "generator"},
    )
    graph.add_edge("generator", END)

    return graph.compile()


class AgentPipeline:
    """Wraps the compiled LangGraph agent with the same ask() interface as RAGPipeline.

    Allows the eval harness and the /agent/ask endpoint to treat it identically
    to RAGPipeline without special-casing.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline
        self._graph = build_agent_graph(pipeline)

    def ask(
        self,
        question: str,
        top_k: int,
        *,
        include_contexts: bool,
        rerank_top_n: int = 20,  # noqa: ARG002 — interface parity with RAGPipeline.ask
    ) -> tuple[str, list[Source], dict[str, int]]:
        # Lazy imports: translation pulls api.llm (circular), tracing is optional.
        from api.tracing import get_langfuse_handler  # noqa: PLC0415
        from api.translation import (  # noqa: PLC0415
            contains_cyrillic,
            translate_to_english,
            translate_to_russian,
        )

        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []

        is_russian = contains_cyrillic(question)
        translation_ms = 0
        t_start = time.perf_counter()

        if is_russian:
            t_tr0 = time.perf_counter()
            graph_question = translate_to_english(self._pipeline._llm, question, callbacks=callbacks or None)
            translation_ms += int((time.perf_counter() - t_tr0) * 1000)
            logger.info("RU→EN (agent) | in={!r} | out={!r}", question, graph_question)
        else:
            graph_question = question

        initial: GraphState = {
            "question": graph_question,
            "query": graph_question,
            "top_k": top_k,
            "hits": [],
            "relevant_hits": [],
            "answer": "",
            "sources": [],
            "retry_count": 0,
            "timings": {},
            "callbacks": callbacks,
        }

        result = self._graph.invoke(initial)
        answer_en = result["answer"]
        answer = answer_en

        if is_russian:
            t_tr1 = time.perf_counter()
            answer = translate_to_russian(self._pipeline._llm, answer_en, callbacks=callbacks or None)
            translation_ms += int((time.perf_counter() - t_tr1) * 1000)
            logger.info("EN→RU (agent) | in={!r} | out={!r}", answer_en, answer)

        final_hits = result.get("relevant_hits") or result.get("hits", [])
        sources = [self._pipeline._hit_to_source(h, include_contexts=include_contexts) for h in final_hits]

        # Graph's own total_ms only covers in-graph stages; recompute end-to-end
        # so translation is included when the question was Russian.
        total_ms = int((time.perf_counter() - t_start) * 1000)
        timings: dict[str, int] = {
            "retrieval_ms": result["timings"].get("retrieval_ms", 0),
            "generation_ms": result["timings"].get("generation_ms", 0),
            "translation_ms": translation_ms,
            "total_ms": total_ms,
            "rewrite_ms": result["timings"].get("rewrite_ms", 0),
            "grading_ms": result["timings"].get("grading_ms", 0),
        }

        logger.info(
            "AgentPipeline.ask | lang={} retries={} rewrite={}ms retrieval={}ms grading={}ms generation={}ms translation={}ms total={}ms",
            "ru" if is_russian else "en",
            result.get("retry_count", 0) - 1,  # subtract last increment
            timings["rewrite_ms"],
            timings["retrieval_ms"],
            timings["grading_ms"],
            timings["generation_ms"],
            timings["translation_ms"],
            timings["total_ms"],
        )

        return answer, sources, timings


@lru_cache(maxsize=1)
def get_agent_pipeline() -> AgentPipeline:
    """Application-wide singleton for the agentic pipeline."""
    from api.rag import get_pipeline  # noqa: PLC0415 — circular dep: api.rag imports types from api.graph context

    return AgentPipeline(get_pipeline())
