"""FastAPI entrypoint for the DocsRAG API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from api.config import settings
from api.graph import AgentPipeline, get_agent_pipeline
from api.metrics import (
    rag_answer_length_chars,
    rag_generation_duration_seconds,
    rag_requests_total,
    rag_retrieval_duration_seconds,
    rag_top_k,
)
from api.rag import RAGPipeline, get_pipeline
from api.schemas import AgentAskResponse, AskRequest, AskResponse, HealthResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up the RAG pipeline at startup so the first request isn't slow."""
    logger.info("Starting DocsRAG API")
    get_pipeline()  # constructs and caches the singleton; agent pipeline shares it
    logger.info("DocsRAG API ready")
    yield
    logger.info("Shutting down DocsRAG API")


app = FastAPI(
    title="DocsRAG API",
    description="Q&A over FastAPI documentation via retrieval-augmented generation.",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]
AgentPipelineDep = Annotated[AgentPipeline, Depends(get_agent_pipeline)]


@app.get("/health", response_model=HealthResponse)
def health(pipeline: PipelineDep) -> HealthResponse:
    """Liveness + readiness check.

    Touches Qdrant to confirm the collection is reachable and reports
    configured model names. If Qdrant is down, returns 503.
    """
    try:
        points = pipeline.collection_points_count()
    except Exception as exc:
        logger.error("Health check failed: {}", exc)
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {exc}") from exc

    return HealthResponse(
        status="ok",
        qdrant_collection=settings.qdrant_collection,
        qdrant_points=points,
        ollama_model=settings.ollama_model,
        vllm_model=settings.vllm_model,
        embedding_model=settings.embedding_model,
        inference_backend=settings.inference_backend,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: PipelineDep) -> AskResponse:
    try:
        answer, sources, timings = pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            include_contexts=request.include_contexts,
        )
    except Exception as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    rag_requests_total.labels(endpoint="ask").inc()
    rag_retrieval_duration_seconds.labels(endpoint="ask").observe(timings["retrieval_ms"] / 1000)
    rag_generation_duration_seconds.labels(endpoint="ask").observe(timings["generation_ms"] / 1000)
    rag_top_k.observe(request.top_k)
    rag_answer_length_chars.observe(len(answer))

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_ms=timings["retrieval_ms"],
        generation_ms=timings["generation_ms"],
        translation_ms=timings.get("translation_ms", 0),
        total_ms=timings["total_ms"],
    )


@app.post("/agent/ask", response_model=AgentAskResponse)
def agent_ask(request: AskRequest, agent: AgentPipelineDep) -> AgentAskResponse:
    """Answer a question using the agentic RAG graph (query rewriting + relevance grading)."""
    try:
        answer, sources, timings = agent.ask(
            question=request.question,
            top_k=request.top_k,
            include_contexts=request.include_contexts,
        )
    except Exception as exc:
        logger.exception("Agent pipeline failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    rag_requests_total.labels(endpoint="agent_ask").inc()
    rag_retrieval_duration_seconds.labels(endpoint="agent_ask").observe(timings["retrieval_ms"] / 1000)
    rag_generation_duration_seconds.labels(endpoint="agent_ask").observe(timings["generation_ms"] / 1000)
    rag_top_k.observe(request.top_k)
    rag_answer_length_chars.observe(len(answer))

    return AgentAskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_ms=timings["retrieval_ms"],
        generation_ms=timings["generation_ms"],
        total_ms=timings["total_ms"],
        rewrite_ms=timings.get("rewrite_ms", 0),
        grading_ms=timings.get("grading_ms", 0),
        translation_ms=timings.get("translation_ms", 0),
        retry_count=timings.get("retry_count", 0),
    )
