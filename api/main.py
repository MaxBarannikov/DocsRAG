"""FastAPI application exposing the RAG endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

from core.proxy import strip_socks_proxy_env

# Must run before any httpx client is constructed; the imports below build them.
strip_socks_proxy_env()

from fastapi import Depends, FastAPI, HTTPException, status  # noqa: E402 — after the proxy cleanup above
from loguru import logger  # noqa: E402
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402

from api.graph import AgentPipeline, get_agent_pipeline  # noqa: E402
from api.metrics import rag_errors_total, rag_requests_total, record_rag_result  # noqa: E402
from api.rag import RAGPipeline, get_pipeline  # noqa: E402
from api.schemas import (  # noqa: E402
    AgentAskResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    Source,
)
from api.security import require_api_key  # noqa: E402
from core.config import settings  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from core.types import AskablePipeline

GENERIC_ERROR_DETAIL = "Internal error while answering the question. See server logs for details."


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm the pipeline so the first request does not pay for loading it."""
    logger.info("Starting DocsRAG API")
    try:
        get_pipeline()
    except Exception:  # noqa: BLE001 — a 503 from /health beats a crash-looping container
        logger.exception("Pipeline warmup failed; the API will start but /health will report unhealthy")
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
ApiKeyDep = Annotated[None, Depends(require_api_key)]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """503 when Qdrant cannot be reached."""
    try:
        pipeline = get_pipeline()
        points = pipeline.collection_points_count()
    except Exception as exc:
        logger.error("Health check failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant is unreachable.",
        ) from exc

    active_model = settings.vllm_model if settings.inference_backend == "vllm" else settings.ollama_model
    return HealthResponse(
        status="ok",
        qdrant_collection=settings.active_qdrant_collection,
        qdrant_points=points,
        embedding_model=settings.embedding_model,
        embedder_backend=settings.embedder_backend,
        inference_backend=settings.inference_backend,
        llm_model=active_model,
        retrieval_strategy=pipeline.strategy,
    )


def _run(
    pipeline: AskablePipeline,
    request: AskRequest,
    endpoint: str,
) -> tuple[str, list[Source], dict[str, int]]:
    """Run a pipeline, recording request and error counters around it."""
    rag_requests_total.labels(endpoint=endpoint).inc()
    try:
        answer, sources, timings = pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            include_contexts=request.include_contexts,
        )
    except Exception as exc:
        rag_errors_total.labels(endpoint=endpoint).inc()
        # Exception text can carry internal hostnames and paths; keep it out of the response.
        logger.exception("{} pipeline failed", endpoint)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=GENERIC_ERROR_DETAIL,
        ) from exc

    record_rag_result(endpoint, timings, request.top_k, answer)
    return answer, sources, timings


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: PipelineDep, _: ApiKeyDep) -> AskResponse:
    answer, sources, timings = _run(pipeline, request, endpoint="ask")
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
def agent_ask(request: AskRequest, agent: AgentPipelineDep, _: ApiKeyDep) -> AgentAskResponse:
    answer, sources, timings = _run(agent, request, endpoint="agent_ask")
    return AgentAskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_ms=timings["retrieval_ms"],
        generation_ms=timings["generation_ms"],
        translation_ms=timings.get("translation_ms", 0),
        total_ms=timings["total_ms"],
        rewrite_ms=timings.get("rewrite_ms", 0),
        grading_ms=timings.get("grading_ms", 0),
        retry_count=timings.get("retry_count", 0),
    )
