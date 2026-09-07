"""Retrieval types shared by the retrieval, generation and evaluation layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from langchain_core.documents import Document

if TYPE_CHECKING:
    from qdrant_client.models import ScoredPoint

    from api.schemas import Source

UNKNOWN_SOURCE = "unknown"


@dataclass(slots=True)
class RetrievalHit:
    """A retrieved chunk and its score.

    The score's scale depends on how the hit was produced: cosine similarity for dense
    search, a reciprocal-rank sum after fusion, an unbounded logit after reranking.
    """

    document: Document
    score: float

    @property
    def chunk_key(self) -> tuple[str, int]:
        return document_chunk_key(self.document)


def document_chunk_key(document: Document) -> tuple[str, int]:
    """Corpus-wide identity of a chunk.

    `chunk_index` alone is not unique — the chunker restarts it per source document —
    so anything that deduplicates or merges hits must key on the pair.
    """
    metadata = document.metadata or {}
    return (
        str(metadata.get("source_path", UNKNOWN_SOURCE)),
        int(metadata.get("chunk_index", -1)),
    )


def payload_to_document(payload: dict | None) -> Document:
    """Map a Qdrant payload onto a Document.

    Hand-written because langchain-qdrant 0.2.x stopped propagating flat payload
    fields into `Document.metadata`.
    """
    payload = payload or {}
    return Document(
        page_content=str(payload.get("text", "")),
        metadata={
            "source_path": payload.get("source_path", UNKNOWN_SOURCE),
            "header_path": payload.get("header_path", ""),
            "chunk_index": payload.get("chunk_index", -1),
        },
    )


def scored_point_to_hit(point: ScoredPoint) -> RetrievalHit:
    score = getattr(point, "score", None)  # optional in the client's schema
    return RetrievalHit(
        document=payload_to_document(point.payload),
        score=float(score) if score is not None else 0.0,
    )


class AskablePipeline(Protocol):
    """What the eval harness needs from a pipeline; RAGPipeline and AgentPipeline both fit."""

    def ask(
        self,
        question: str,
        top_k: int,
        *,
        include_contexts: bool,
        rerank_top_n: int = ...,
    ) -> tuple[str, list[Source], dict[str, int]]: ...
