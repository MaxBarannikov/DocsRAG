"""LangChain embeddings adapter over the project's own encoder.

Ragas needs embeddings for `answer_relevancy`; handing it the chat model would be
wrong. Reusing the retrieval encoder also keeps the metric in the same vector space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from core.embedder import Embedder


class ProjectEmbeddings(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts, show_progress=False)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text], show_progress=False)[0]
