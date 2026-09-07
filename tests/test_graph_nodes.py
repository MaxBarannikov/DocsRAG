"""Graph nodes, exercised without an LLM or a vector store."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from api.graph import (
    MAX_RETRIES,
    MIN_RELEVANT_CHUNKS,
    AgentGraphNodes,
    GraphState,
    RelevanceVerdict,
    build_agent_graph,
    should_retry,
)
from tests.conftest import make_hit


class StubLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[object] = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        reply = self._replies.pop(0) if self._replies else ""
        return SimpleNamespace(content=reply)


@pytest.fixture
def nodes(monkeypatch) -> AgentGraphNodes:

    def _retrieve(*_args: object, **_kwargs: object) -> list:
        return []

    def _generate(*_args: object, **_kwargs: object) -> str:
        return "answer"

    def _make_llm(**_kwargs: object) -> StubLLM:
        return StubLLM([])

    pipeline = SimpleNamespace(llm=StubLLM([]), retrieve=_retrieve, generate=_generate)
    monkeypatch.setattr("api.graph.make_llm", _make_llm)
    return AgentGraphNodes(pipeline)  # type: ignore[arg-type]


def _state(**overrides: object) -> GraphState:
    base = {
        "question": "How do I define a path parameter?",
        "query": "How do I define a path parameter?",
        "top_k": 5,
        "hits": [],
        "relevant_hits": [],
        "answer": "",
        "retry_count": 0,
        "timings": {},
        "callbacks": [],
    }
    base.update(overrides)
    return cast("GraphState", base)


def test_relevance_verdict_parses_the_expected_json() -> None:
    assert RelevanceVerdict.model_validate_json('{"relevant": true}').relevant is True
    assert RelevanceVerdict.model_validate_json('{"relevant": false}').relevant is False


def test_grader_keeps_a_chunk_when_the_verdict_cannot_be_parsed(nodes) -> None:
    """Dropping context we cannot judge is worse than keeping it."""
    from api.metrics import rag_grader_parse_failures_total

    nodes._grader_llm = StubLLM(["Sure! This document looks relevant to me."])
    before = rag_grader_parse_failures_total._value.get()

    assert nodes._grade_one("q", make_hit("a.md", 0), []) is True
    # Counted, so a grader that never parses shows up in metrics.
    assert rag_grader_parse_failures_total._value.get() == before + 1


def test_grader_drops_a_chunk_judged_irrelevant(nodes) -> None:
    nodes._grader_llm = StubLLM(['{"relevant": false}'])
    assert nodes._grade_one("q", make_hit("a.md", 0), []) is False


def test_grader_increments_the_attempt_counter(nodes) -> None:
    nodes._grader_llm = StubLLM(['{"relevant": true}', '{"relevant": false}'])
    result = nodes.relevance_grader(_state(hits=[make_hit("a.md", 0), make_hit("b.md", 0)]))

    assert len(result["relevant_hits"]) == 1
    assert result["retry_count"] == 1
    assert result["timings"]["grading_ms"] >= 0


def test_query_rewriter_falls_back_to_the_original_question_when_empty(nodes) -> None:
    nodes._llm = StubLLM(["   "])
    result = nodes.query_rewriter(_state())
    assert result["query"] == "How do I define a path parameter?"


def test_should_retry_asks_for_another_pass_when_too_few_chunks_pass() -> None:
    too_few = [make_hit("a.md", 0)] * (MIN_RELEVANT_CHUNKS - 1)
    assert should_retry(_state(relevant_hits=too_few, retry_count=1)) == "retry"


def test_should_retry_stops_once_the_retry_budget_is_spent() -> None:
    too_few = [make_hit("a.md", 0)] * (MIN_RELEVANT_CHUNKS - 1)
    assert should_retry(_state(relevant_hits=too_few, retry_count=MAX_RETRIES + 1)) == "generate"


def test_should_retry_proceeds_when_enough_chunks_pass() -> None:
    enough = [make_hit("a.md", i) for i in range(MIN_RELEVANT_CHUNKS)]
    assert should_retry(_state(relevant_hits=enough, retry_count=1)) == "generate"


def test_graph_compiles(monkeypatch) -> None:
    """LangGraph resolves the GraphState annotations at build time, so every type in
    them has to exist at runtime, not only under TYPE_CHECKING.
    """

    def _make_llm(**_kwargs: object) -> StubLLM:
        return StubLLM([])

    def _retrieve(*_args: object, **_kwargs: object) -> list:
        return []

    def _generate(*_args: object, **_kwargs: object) -> str:
        return ""

    monkeypatch.setattr("api.graph.make_llm", _make_llm)
    pipeline = SimpleNamespace(llm=StubLLM([]), retrieve=_retrieve, generate=_generate)

    graph = build_agent_graph(pipeline)  # type: ignore[arg-type]

    assert {"query_rewriter", "retriever", "relevance_grader", "generator"} <= set(graph.nodes)
