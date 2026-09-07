"""Schema for the experiment configs in configs/*.yaml."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvalStrategy = Literal["dense", "hybrid", "hybrid_rerank", "agentic"]


class EvalConfig(BaseModel):
    """`chunk_size` and `chunk_overlap` describe the index a run expects to query; they
    do not trigger re-indexing. The harness warns when they disagree with the collection.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(..., ge=1, description="Chunk size the target index was built with")
    chunk_overlap: int = Field(..., ge=0, description="Chunk overlap the target index was built with")
    top_k: int = Field(..., ge=1, le=50)
    embedding_model: str
    llm_model: str
    retrieval_strategy: EvalStrategy = "dense"
    rerank_top_n: int = Field(default=20, ge=1)
    embedder_backend: str | None = None
