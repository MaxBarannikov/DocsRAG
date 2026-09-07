"""Request and response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 20


class Source(BaseModel):
    source_path: str = Field(
        ...,
        description="Path to the source markdown file, relative to the docs root",
        examples=["tutorial/path-params.md"],
    )
    header_path: str = Field(
        default="",
        description="Hierarchical markdown headers leading to the chunk",
        examples=["Path Parameters > Path parameters with types"],
    )
    score: float = Field(
        ...,
        description=(
            "Relevance score, higher is better. Its scale depends on the retrieval "
            "strategy: cosine similarity in [0, 1] for dense search, a small "
            "reciprocal-rank sum for hybrid fusion, an unbounded logit after reranking."
        ),
    )
    content: str | None = Field(
        default=None,
        description="Raw chunk text. Populated only when include_contexts is true.",
    )


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="Natural-language question about the indexed documentation",
        examples=["How do I define a path parameter in FastAPI?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_TOP_K,
        description="Number of chunks to retrieve from the vector store",
    )
    include_contexts: bool = Field(
        default=False,
        description="Return raw chunk texts in sources[].content, for debugging",
    )


class _AnswerBase(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    retrieval_ms: int = Field(..., description="Retrieval latency in milliseconds")
    generation_ms: int = Field(..., description="LLM generation latency in milliseconds")
    translation_ms: int = Field(
        default=0,
        description="Combined RU-EN and EN-RU translation latency; 0 for English questions",
    )
    total_ms: int = Field(..., description="End-to-end wall-clock latency in milliseconds")


class AskResponse(_AnswerBase):
    """Answer from the retrieve-then-generate pipeline."""


class AgentAskResponse(_AnswerBase):
    """Answer from the agentic pipeline, with its extra stages broken out."""

    rewrite_ms: int = Field(default=0, description="Query-rewriting latency across all attempts")
    grading_ms: int = Field(default=0, description="Relevance-grading latency across all attempts")
    retry_count: int = Field(default=0, description="Retrieval retries performed (0 = no retry was needed)")


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    qdrant_collection: str
    qdrant_points: int
    embedding_model: str
    embedder_backend: str = Field(..., description="Active embedder backend: pytorch | onnx-fp32 | onnx-int8")
    inference_backend: str = Field(..., description="Active inference backend: ollama | vllm")
    llm_model: str = Field(..., description="Model actually serving requests on the active backend")
    retrieval_strategy: str = Field(..., description="Active strategy: dense | hybrid | hybrid_rerank")
