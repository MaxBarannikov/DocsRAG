"""Routing, validation, error handling and response shaping, against a fake pipeline."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.main import app
from api.rag import get_pipeline
from tests.conftest import FakePipeline


@pytest.fixture
def stub_app(monkeypatch):
    """`/health` and the lifespan handler both call `get_pipeline()` directly rather
    than through Depends, so both need patching or the tests hit real infrastructure.
    """

    def _install(pipeline: FakePipeline, agent: FakePipeline | None = None) -> TestClient:
        monkeypatch.setattr(api_main, "get_pipeline", lambda: pipeline)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        app.dependency_overrides[api_main.get_agent_pipeline] = lambda: agent or pipeline
        return TestClient(app)

    yield _install
    app.dependency_overrides.clear()


@pytest.fixture
def client(stub_app, fake_pipeline: FakePipeline):
    with stub_app(fake_pipeline) as test_client:
        yield test_client


def test_health_reports_the_active_configuration(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant_points"] == 2540
    assert body["retrieval_strategy"] == "dense"
    assert {"embedder_backend", "inference_backend", "llm_model"} <= set(body)


def test_health_returns_503_when_qdrant_is_unreachable(stub_app) -> None:
    broken = FakePipeline(error=ConnectionError("connection refused to qdrant:6333"))

    with stub_app(broken) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
    # The internal host and port must not reach the client.
    assert "6333" not in response.text


def test_ask_returns_the_answer_and_timings(client) -> None:
    response = client.post("/ask", json={"question": "How do I define a path parameter?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("FastAPI declares path parameters")
    assert body["retrieval_ms"] == 12
    assert body["total_ms"] == 915
    assert body["sources"][0]["source_path"] == "tutorial/path-params.md"


def test_ask_hides_chunk_text_unless_contexts_are_requested(client) -> None:
    without = client.post("/ask", json={"question": "q"}).json()
    assert without["sources"][0]["content"] is None

    with_contexts = client.post("/ask", json={"question": "q", "include_contexts": True}).json()
    assert with_contexts["sources"][0]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 999},
        {"question": "a" * 2001},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, payload: dict) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_pipeline_failures_do_not_leak_internal_details(stub_app) -> None:
    secret = "postgresql://user:hunter2@internal-db:5432/prod"  # noqa: S105 — fake DSN, asserted absent from the response
    broken = FakePipeline(error=RuntimeError(secret))

    with stub_app(broken) as test_client:
        response = test_client.post("/ask", json={"question": "q"})

    assert response.status_code == 500
    assert secret not in response.text
    assert "hunter2" not in response.text


def test_agent_ask_surfaces_the_retry_count(stub_app, fake_pipeline) -> None:
    """retry_count used to be absent from the timings dict, so it was always 0."""
    agent = FakePipeline(
        timings={
            "retrieval_ms": 20,
            "generation_ms": 800,
            "translation_ms": 0,
            "total_ms": 2100,
            "rewrite_ms": 300,
            "grading_ms": 980,
            "retry_count": 1,
        }
    )
    with stub_app(fake_pipeline, agent) as test_client:
        body = test_client.post("/agent/ask", json={"question": "q"}).json()

    assert body["retry_count"] == 1
    assert body["rewrite_ms"] == 300
    assert body["grading_ms"] == 980


def test_metrics_endpoint_is_exposed(client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_requests_total" in response.text
