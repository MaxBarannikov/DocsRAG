"""Agentic RAG via LangGraph.

    query_rewriter -> retriever -> relevance_grader -> generator

The grader loops back to rewriting when too few chunks pass, at most MAX_RETRIES times.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.callbacks import BaseCallbackHandler  # noqa: TC002 — see GraphState note below
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, ValidationError

from api.llm import make_llm
from api.metrics import rag_grader_parse_failures_total
from api.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_RETRY_PROMPT, RELEVANCE_GRADER_PROMPT
from api.tracing import get_langfuse_handler
from api.translation import contains_cyrillic, translate_to_english, translate_to_russian

# Imported at runtime, not under TYPE_CHECKING: LangGraph resolves the GraphState
# annotations with get_type_hints() when the graph is built, and `from __future__ import
# annotations` leaves them as strings that must be evaluable against module globals.
from core.types import RetrievalHit  # noqa: TC001 — must be resolvable at runtime

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from api.rag import RAGPipeline
    from api.schemas import Source

MAX_RETRIES = 1  # one retry after the initial retrieval
MIN_RELEVANT_CHUNKS = 2  # chunks that must pass grading for the retry to be skipped
# Relevance is decidable from the opening lines; full chunks double the prompt cost.
GRADER_PREVIEW_CHARS = 500


class RelevanceVerdict(BaseModel):
    relevant: bool


class GraphState(TypedDict):
    question: str
    query: str
    top_k: int
    hits: list[RetrievalHit]
    relevant_hits: list[RetrievalHit]
    answer: str
    retry_count: int
    timings: dict[str, int]
    callbacks: list[BaseCallbackHandler]


def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _accumulate(state: GraphState, key: str, value: int) -> dict[str, int]:

    timings = dict(state.get("timings") or {})
    timings[key] = timings.get(key, 0) + value
    return timings


class AgentGraphNodes:
    """Methods rather than closures so each node can be tested on its own."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline
        self._llm = pipeline.llm
        # Without json_mode, Qwen answers the grading prompt in prose at temperature 0.
        self._grader_llm = make_llm(temperature=0.0, json_mode=True)

    def query_rewriter(self, state: GraphState) -> dict[str, Any]:
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        if retry_count == 0:
            messages = QUERY_REWRITE_PROMPT.format_messages(question=question)
        else:
            messages = QUERY_REWRITE_RETRY_PROMPT.format_messages(
                question=question,
                previous_query=state.get("query", question),
            )

        started = time.perf_counter()
        response = self._llm.invoke(messages, config={"callbacks": state.get("callbacks") or []})
        rewrite_ms = _elapsed_ms(started)

        query = str(response.content).strip()
        if not query:
            logger.warning("query_rewriter returned an empty query — falling back to the original question")
            query = question

        logger.info("query_rewriter | retry={} rewrite={}ms | query={!r}", retry_count, rewrite_ms, query[:80])
        return {"query": query, "timings": _accumulate(state, "rewrite_ms", rewrite_ms)}

    def retriever(self, state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        hits = self._pipeline.retrieve(state["query"], top_k=state["top_k"])
        retrieval_ms = _elapsed_ms(started)

        logger.info("retriever | hits={} retrieval={}ms | query={!r}", len(hits), retrieval_ms, state["query"][:80])
        return {"hits": hits, "timings": _accumulate(state, "retrieval_ms", retrieval_ms)}

    def _grade_one(self, query: str, hit: RetrievalHit, callbacks: list[BaseCallbackHandler]) -> bool:
        """Fails open on an unparseable verdict, but records it."""
        messages = RELEVANCE_GRADER_PROMPT.format_messages(
            question=query,
            document=hit.document.page_content[:GRADER_PREVIEW_CHARS],
        )
        try:
            response = self._grader_llm.invoke(messages, config={"callbacks": callbacks})
            return RelevanceVerdict.model_validate_json(str(response.content)).relevant
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            # Counted, not just logged: sustained failures silently degrade the agent
            # back into plain dense RAG.
            rag_grader_parse_failures_total.inc()
            logger.warning("relevance_grader could not parse a verdict (keeping the chunk): {}", exc)
            return True

    def relevance_grader(self, state: GraphState) -> dict[str, Any]:
        callbacks = state.get("callbacks") or []
        started = time.perf_counter()
        relevant_hits = [hit for hit in state["hits"] if self._grade_one(state["query"], hit, callbacks)]
        grading_ms = _elapsed_ms(started)

        logger.info(
            "relevance_grader | relevant={}/{} grading={}ms",
            len(relevant_hits),
            len(state["hits"]),
            grading_ms,
        )
        return {
            "relevant_hits": relevant_hits,
            "retry_count": state.get("retry_count", 0) + 1,
            "timings": _accumulate(state, "grading_ms", grading_ms),
        }

    def generator(self, state: GraphState) -> dict[str, Any]:
        hits = state.get("relevant_hits") or state["hits"]
        callbacks = state.get("callbacks") or []

        started = time.perf_counter()
        answer = self._pipeline.generate(state["question"], hits, callbacks=callbacks or None)
        generation_ms = _elapsed_ms(started)

        logger.info("generator | generation={}ms answer_len={}", generation_ms, len(answer))
        return {"answer": answer, "timings": _accumulate(state, "generation_ms", generation_ms)}


def should_retry(state: GraphState) -> str:
    relevant = state.get("relevant_hits", [])
    # relevance_grader has already incremented retry_count by the time this runs.
    retry_count = state.get("retry_count", 0)
    if len(relevant) < MIN_RELEVANT_CHUNKS and retry_count <= MAX_RETRIES:
        logger.info(
            "should_retry -> retry (relevant={} < {}, attempt {} of {})",
            len(relevant),
            MIN_RELEVANT_CHUNKS,
            retry_count,
            MAX_RETRIES + 1,
        )
        return "retry"
    return "generate"


def build_agent_graph(pipeline: RAGPipeline) -> CompiledStateGraph:
    nodes = AgentGraphNodes(pipeline)

    graph: StateGraph = StateGraph(GraphState)
    graph.add_node("query_rewriter", nodes.query_rewriter)
    graph.add_node("retriever", nodes.retriever)
    graph.add_node("relevance_grader", nodes.relevance_grader)
    graph.add_node("generator", nodes.generator)

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
    """The agent graph behind the same RU-EN wrapper as `RAGPipeline`."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline
        self._graph = build_agent_graph(pipeline)

    def ask(
        self,
        question: str,
        top_k: int,
        *,
        include_contexts: bool,
        rerank_top_n: int | None = None,  # noqa: ARG002 — interface parity with RAGPipeline.ask
    ) -> tuple[str, list[Source], dict[str, int]]:
        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []
        llm = self._pipeline.llm

        is_russian = contains_cyrillic(question)
        translation_ms = 0
        started = time.perf_counter()

        if is_russian:
            t0 = time.perf_counter()
            graph_question = translate_to_english(llm, question, callbacks=callbacks or None)
            translation_ms += _elapsed_ms(t0)
            logger.info("RU->EN (agent) | in={!r} | out={!r}", question, graph_question)
        else:
            graph_question = question

        initial: GraphState = {
            "question": graph_question,
            "query": graph_question,
            "top_k": top_k,
            "hits": [],
            "relevant_hits": [],
            "answer": "",
            "retry_count": 0,
            "timings": {},
            "callbacks": callbacks,
        }
        result = self._graph.invoke(initial)
        answer = result.get("answer", "")

        if is_russian:
            t1 = time.perf_counter()
            answer_en = answer
            answer = translate_to_russian(llm, answer_en, callbacks=callbacks or None)
            translation_ms += _elapsed_ms(t1)
            logger.info("EN->RU (agent) | in={!r} | out={!r}", answer_en, answer)

        final_hits = result.get("relevant_hits") or result.get("hits", [])
        sources = [self._pipeline.hit_to_source(hit, include_contexts=include_contexts) for hit in final_hits]

        graph_timings = result.get("timings", {})
        # The grader counts attempts, not retries.
        retries = max(result.get("retry_count", 1) - 1, 0)
        timings: dict[str, int] = {
            "retrieval_ms": graph_timings.get("retrieval_ms", 0),
            "generation_ms": graph_timings.get("generation_ms", 0),
            "rewrite_ms": graph_timings.get("rewrite_ms", 0),
            "grading_ms": graph_timings.get("grading_ms", 0),
            "translation_ms": translation_ms,
            "total_ms": _elapsed_ms(started),
            "retry_count": retries,
        }

        logger.info(
            "AgentPipeline.ask | lang={} retries={} rewrite={}ms retrieval={}ms "
            "grading={}ms generation={}ms translation={}ms total={}ms",
            "ru" if is_russian else "en",
            retries,
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
    from api.rag import get_pipeline  # module-scope would make api.rag <-> api.graph a cycle

    return AgentPipeline(get_pipeline())
