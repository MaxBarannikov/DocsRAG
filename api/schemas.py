from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    source_path: str = Field(
        ...,
        description="Relative path to the source markdown file",
        examples=["tutorial/path-params.md"],
    )
    header_path: str = Field(
        default="",
        description="Hierarchical markdown headers leading to the chunk",
        examples=["Path Parameters > Path parameters with types"],
    )
    score: float = Field(
        ...,
        description="Relevance score (higher is better). Cosine similarity [0, 1] for dense retrieval; unbounded logit for cross-encoder reranking.",
    )
    content: str | None = Field(
        default=None,
        description="Raw chunk text. Populated only when include_contexts=True.",
    )


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question about the indexed documentation",
        examples=["How do I define a path parameter in FastAPI?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from the vector store",
    )
    include_contexts: bool = Field(
        default=False,
        description="If true, returns raw chunk texts in `sources[].content` for debugging",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    retrieval_ms: int = Field(..., description="Retrieval latency in milliseconds")
    generation_ms: int = Field(..., description="LLM generation latency in milliseconds")
    translation_ms: int = Field(
        default=0,
        description="Combined RU→EN + EN→RU translation latency; 0 for English questions",
    )
    total_ms: int = Field(..., description="End-to-end latency in milliseconds")


class AgentAskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    retrieval_ms: int = Field(..., description="Total retrieval latency across all iterations")
    generation_ms: int = Field(..., description="LLM generation latency")
    total_ms: int = Field(..., description="End-to-end latency including rewriting and grading")
    rewrite_ms: int = Field(default=0, description="Query rewriting latency")
    grading_ms: int = Field(default=0, description="Relevance grading latency")
    translation_ms: int = Field(
        default=0,
        description="Combined RU→EN + EN→RU translation latency; 0 for English questions",
    )
    retry_count: int = Field(default=0, description="Number of retrieval retries performed (0 = no retry needed)")


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    qdrant_collection: str
    qdrant_points: int
    ollama_model: str
    vllm_model: str
    embedding_model: str
    embedder_backend: str = Field(..., description="Active embedder backend: pytorch | onnx-fp32 | onnx-int8")
    inference_backend: str
