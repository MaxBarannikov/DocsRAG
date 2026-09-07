"""Prometheus metrics. Buckets follow the latencies actually observed on this stack."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

rag_requests_total = Counter(
    "rag_requests_total",
    "RAG requests received, by endpoint",
    ["endpoint"],
)

rag_errors_total = Counter(
    "rag_errors_total",
    "RAG requests that failed, by endpoint",
    ["endpoint"],
)

rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval stage duration",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

rag_generation_duration_seconds = Histogram(
    "rag_generation_duration_seconds",
    "Generation stage duration",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

rag_translation_duration_seconds = Histogram(
    "rag_translation_duration_seconds",
    "Combined RU-EN translation duration; 0 for English questions",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

rag_top_k = Histogram(
    "rag_top_k",
    "Number of chunks requested (top_k)",
    ["endpoint"],
    buckets=[1, 2, 3, 5, 10, 20],
)

rag_answer_length_chars = Histogram(
    "rag_answer_length_chars",
    "Answer length in characters",
    ["endpoint"],
    buckets=[100, 250, 500, 1000, 2000, 5000],
)

rag_grader_parse_failures_total = Counter(
    "rag_grader_parse_failures_total",
    "Relevance-grader replies that could not be parsed as JSON",
)


def record_rag_result(endpoint: str, timings: dict[str, int], top_k: int, answer: str) -> None:

    rag_retrieval_duration_seconds.labels(endpoint=endpoint).observe(timings.get("retrieval_ms", 0) / 1000)
    rag_generation_duration_seconds.labels(endpoint=endpoint).observe(timings.get("generation_ms", 0) / 1000)
    rag_translation_duration_seconds.labels(endpoint=endpoint).observe(timings.get("translation_ms", 0) / 1000)
    rag_top_k.labels(endpoint=endpoint).observe(top_k)
    rag_answer_length_chars.labels(endpoint=endpoint).observe(len(answer))
