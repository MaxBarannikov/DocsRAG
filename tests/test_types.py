from __future__ import annotations

from typing import Any, cast

from core.types import UNKNOWN_SOURCE, payload_to_document, scored_point_to_hit


class StubPoint:
    """Only the attributes `scored_point_to_hit` reads."""

    def __init__(self, score: float | None, payload: dict[str, Any] | None) -> None:
        self.score = score
        self.payload = payload


def test_payload_fields_survive_the_round_trip() -> None:
    """Guards the hand-written mapping that replaced langchain-qdrant."""
    document = payload_to_document(
        {
            "text": "Path parameters are declared with curly braces.",
            "source_path": "tutorial/path-params.md",
            "header_path": "Path Parameters > Types",
            "chunk_index": 7,
        }
    )
    assert document.page_content == "Path parameters are declared with curly braces."
    assert document.metadata["source_path"] == "tutorial/path-params.md"
    assert document.metadata["header_path"] == "Path Parameters > Types"
    assert document.metadata["chunk_index"] == 7


def test_missing_payload_fields_fall_back_without_raising() -> None:
    document = payload_to_document(None)
    assert document.page_content == ""
    assert document.metadata["source_path"] == UNKNOWN_SOURCE
    assert document.metadata["chunk_index"] == -1


def test_scored_point_conversion_keeps_the_score() -> None:
    point = StubPoint(score=0.87, payload={"text": "t", "source_path": "a.md", "chunk_index": 1})
    hit = scored_point_to_hit(cast("Any", point))
    assert hit.score == 0.87
    assert hit.chunk_key == ("a.md", 1)


def test_a_missing_score_becomes_zero_instead_of_raising() -> None:
    point = StubPoint(score=None, payload={"text": "t"})
    assert scored_point_to_hit(cast("Any", point)).score == 0.0
