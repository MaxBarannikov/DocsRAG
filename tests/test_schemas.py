"""Request and response model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import MAX_QUESTION_CHARS, MAX_TOP_K, AgentAskResponse, AskRequest, AskResponse, Source


def test_defaults() -> None:
    request = AskRequest(question="What is FastAPI?")
    assert request.top_k == 5
    assert request.include_contexts is False


@pytest.mark.parametrize("question", ["", " " * 0])
def test_empty_question_is_rejected(question: str) -> None:
    with pytest.raises(ValidationError):
        AskRequest(question=question)


def test_overlong_question_is_rejected() -> None:
    AskRequest(question="a" * MAX_QUESTION_CHARS)
    with pytest.raises(ValidationError):
        AskRequest(question="a" * (MAX_QUESTION_CHARS + 1))


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
def test_top_k_outside_the_allowed_range_is_rejected(top_k: int) -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="q", top_k=top_k)


@pytest.mark.parametrize("top_k", [1, 5, MAX_TOP_K])
def test_top_k_inside_the_allowed_range_is_accepted(top_k: int) -> None:
    assert AskRequest(question="q", top_k=top_k).top_k == top_k


def test_source_content_defaults_to_none() -> None:
    assert Source(source_path="a.md", score=0.5).content is None


def test_agent_response_extends_the_shared_answer_fields() -> None:
    shared = {
        "question": "q",
        "answer": "a",
        "sources": [],
        "retrieval_ms": 1,
        "generation_ms": 2,
        "total_ms": 3,
    }
    assert AskResponse.model_validate(shared).translation_ms == 0
    agent = AgentAskResponse.model_validate({**shared, "retry_count": 1})
    assert agent.retry_count == 1
    assert agent.rewrite_ms == 0
