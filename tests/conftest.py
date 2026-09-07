"""Shared fixtures. Everything here is offline; tests needing real infrastructure skip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from core.types import RetrievalHit
from indexing.loader import RawDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

    from api.schemas import Source


class FakeEmbedder:
    """Unit vectors derived from a hash of the text, stable across runs."""

    def __init__(self, dimension: int = 8) -> None:
        self.model_name = "fake-embedder"
        self.dimension = dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",
    ) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = abs(hash(prefix + text))
            raw = [((seed >> (i * 3)) % 97) + 1 for i in range(self.dimension)]
            norm = sum(v * v for v in raw) ** 0.5
            vectors.append([v / norm for v in raw])
        return vectors


class FakePipeline:
    def __init__(
        self,
        *,
        answer: str = "FastAPI declares path parameters in the path string. [tutorial/path-params.md]",
        sources: list[Source] | None = None,
        timings: dict[str, int] | None = None,
        error: Exception | None = None,
        points: int = 2540,
        strategy: str = "dense",
    ) -> None:
        from api.schemas import Source as SourceModel

        self.answer = answer
        self.sources = (
            sources
            if sources is not None
            else [
                SourceModel(
                    source_path="tutorial/path-params.md",
                    header_path="Path Parameters",
                    score=0.87,
                    content="Path parameters are declared with curly braces.",
                )
            ]
        )
        self.timings = timings or {
            "retrieval_ms": 12,
            "generation_ms": 900,
            "translation_ms": 0,
            "total_ms": 915,
        }
        self.error = error
        self.points = points
        self.strategy = strategy
        self.calls: list[dict[str, object]] = []

    def ask(
        self,
        question: str,
        top_k: int,
        *,
        include_contexts: bool,
        rerank_top_n: int | None = None,
    ) -> tuple[str, list[Source], dict[str, int]]:
        self.calls.append({"question": question, "top_k": top_k, "include_contexts": include_contexts})
        if self.error is not None:
            raise self.error
        sources = self.sources if include_contexts else [s.model_copy(update={"content": None}) for s in self.sources]
        return self.answer, sources, self.timings

    def collection_points_count(self) -> int:
        if self.error is not None:
            raise self.error
        return self.points


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


def make_hit(source_path: str, chunk_index: int, score: float = 0.5, text: str = "chunk text") -> RetrievalHit:

    from langchain_core.documents import Document

    return RetrievalHit(
        document=Document(
            page_content=text,
            metadata={"source_path": source_path, "chunk_index": chunk_index, "header_path": "Section"},
        ),
        score=score,
    )


def make_document(source_path: str, chunk_index: int, text: str = "chunk text"):
    return make_hit(source_path, chunk_index, text=text).document


@pytest.fixture
def sample_markdown_docs() -> list[RawDocument]:

    from pathlib import Path

    first = (
        "# Path Parameters\n\n"
        "You can declare path parameters with the same syntax used by Python format strings, "
        "which makes them readable and explicit in the route definition.\n\n"
        "## Path parameters with types\n\n"
        "Declaring the type gives you parsing and validation for free, and the editor knows "
        "the type as well, so completion and error checks work inside the function body.\n"
    )
    second = (
        "# Query Parameters\n\n"
        "Function parameters that are not part of the path are interpreted as query parameters, "
        "and FastAPI converts and validates them exactly as it does for path parameters.\n"
    )
    return [
        RawDocument(
            content=first,
            source_path=Path("tutorial/path-params.md"),
            relative_path="tutorial/path-params.md",
        ),
        RawDocument(content=second, source_path=Path("tutorial/query.md"), relative_path="tutorial/query.md"),
    ]
