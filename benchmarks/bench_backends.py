"""Benchmark: Ollama vs vllm-metal generation latency.

Runs a fixed set of questions through the RAG pipeline for each backend,
measures generation_ms and total_ms, and prints a comparison table.

Usage:
    # Make sure Ollama AND vllm-metal are both running, then:
    uv run python benchmarks/bench_backends.py
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    "What is FastAPI?",
    "How do I define a path parameter in FastAPI?",
    "How does dependency injection work in FastAPI?",
    "How do I handle file uploads in FastAPI?",
    "What is the difference between async and sync endpoints in FastAPI?",
]

BACKENDS: list[tuple[str, str, str]] = [
    ("Ollama (Qwen2.5-7B q4_K_M)", "ollama", "qwen2.5:7b-instruct-q4_K_M"),
    ("vllm-metal (Qwen2.5-7B 4bit MLX)", "vllm", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
]


def run_backend(backend_env: str, questions: list[str]) -> list[dict]:
    os.environ["INFERENCE_BACKEND"] = backend_env

    import importlib
    import api.config
    import api.llm
    import api.rag

    importlib.reload(api.config)
    importlib.reload(api.llm)
    importlib.reload(api.rag)
    api.rag.get_pipeline.cache_clear()

    from api.rag import RAGPipeline

    pipeline = RAGPipeline()
    results = []
    for q in questions:
        try:
            _, _, timings = pipeline.ask(q, top_k=3, include_contexts=False)
            results.append(timings)
        except Exception as exc:
            print(f"  ERROR on '{q}': {exc}")
            results.append({"retrieval_ms": 0, "generation_ms": 0, "total_ms": 0})
    return results


def stats(values: list[int]) -> str:
    if not values:
        return "n/a"
    return f"avg={statistics.mean(values):.0f}ms  p50={statistics.median(values):.0f}ms  min={min(values)}ms  max={max(values)}ms"


def main() -> None:
    print("=" * 70)
    print("DocsRAG backend benchmark — generation latency")
    print(f"Questions: {len(QUESTIONS)}")
    print("=" * 70)

    all_results: dict[str, list[dict]] = {}

    for label, backend_env, _ in BACKENDS:
        print(f"\n▶ {label}")
        results = run_backend(backend_env, QUESTIONS)
        all_results[label] = results
        gen_times = [r["generation_ms"] for r in results]
        total_times = [r["total_ms"] for r in results]
        for i, (q, r) in enumerate(zip(QUESTIONS, results)):
            print(f"  [{i + 1}] gen={r['generation_ms']}ms  total={r['total_ms']}ms  | {q[:55]}")
        print(f"  generation: {stats(gen_times)}")
        print(f"  total:      {stats(total_times)}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Backend':<40} {'avg gen':>10} {'avg total':>10}")
    print("-" * 60)
    for label, results in all_results.items():
        gen_avg = statistics.mean(r["generation_ms"] for r in results)
        total_avg = statistics.mean(r["total_ms"] for r in results)
        print(f"{label:<40} {gen_avg:>9.0f}ms {total_avg:>9.0f}ms")

    labels = list(all_results.keys())
    if len(labels) == 2:
        gen_a = statistics.mean(r["generation_ms"] for r in all_results[labels[0]])
        gen_b = statistics.mean(r["generation_ms"] for r in all_results[labels[1]])
        if gen_b > 0:
            ratio = gen_a / gen_b
            faster = labels[1] if ratio > 1 else labels[0]
            print(f"\n→ {faster} is {max(ratio, 1 / ratio):.2f}x faster on generation")

    print("=" * 70)


if __name__ == "__main__":
    main()
