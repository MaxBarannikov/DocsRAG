"""Ollama vs vllm-metal generation latency. Both servers must be running.

Failed questions are counted and excluded from the statistics: averaging a crashed
backend's zeros would make it look arbitrarily fast.

    python -m benchmarks.bench_backends --top-k 5 --warmup
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

QUESTIONS = [
    "What is FastAPI?",
    "How do I define a path parameter in FastAPI?",
    "How does dependency injection work in FastAPI?",
    "How do I handle file uploads in FastAPI?",
    "What is the difference between async and sync endpoints in FastAPI?",
]

COMPARABLE_BACKENDS = 2  # a head-to-head ratio only makes sense with exactly two

BACKENDS: list[tuple[str, str]] = [
    ("Ollama", "ollama"),
    ("vllm-metal", "vllm"),
]


class BackendResult:
    def __init__(self, label: str, model: str) -> None:
        self.label = label
        self.model = model
        self.timings: list[dict[str, int]] = []
        self.failures = 0

    @property
    def succeeded(self) -> int:
        return len(self.timings)

    def stats(self, key: str) -> dict[str, float] | None:
        values = [t[key] for t in self.timings if key in t]
        if not values:
            return None
        return {
            "avg": statistics.mean(values),
            "p50": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }


def run_backend(label: str, backend_env: str, questions: list[str], top_k: int, *, warmup: bool) -> BackendResult:
    """Reloading the config and pipeline modules is how the backend is swapped: the
    pipeline reads it once at construction. Benchmark-only; the service never does this.
    """
    os.environ["INFERENCE_BACKEND"] = backend_env

    import importlib

    import core.config

    importlib.reload(core.config)
    import api.llm
    import api.rag

    importlib.reload(api.llm)
    importlib.reload(api.rag)
    api.rag.get_pipeline.cache_clear()

    pipeline = api.rag.RAGPipeline()
    settings = core.config.settings
    model = settings.vllm_model if backend_env == "vllm" else settings.ollama_model
    result = BackendResult(label, model)

    if warmup:
        print(f"  warming up {label}...")
        try:
            pipeline.ask(questions[0], top_k=top_k, include_contexts=False)
        except Exception as exc:  # noqa: BLE001 — a failed warmup is not fatal
            print(f"  warmup failed: {exc}")

    for question in questions:
        try:
            _, _, timings = pipeline.ask(question, top_k=top_k, include_contexts=False)
        except Exception as exc:  # noqa: BLE001 — record and continue to the next question
            print(f"  ERROR on {question!r}: {exc}")
            result.failures += 1
            continue
        result.timings.append(timings)
    return result


def print_report(results: list[BackendResult]) -> None:
    print("\n" + "=" * 78)
    print(f"{'Backend':<34}{'runs':>6}{'avg gen':>10}{'p50 gen':>10}{'min':>9}{'max':>9}")
    print("-" * 78)

    for result in results:
        gen = result.stats("generation_ms")
        if gen is None:
            print(f"{result.label:<34}{'0/' + str(result.failures):>6}   all questions failed")
            continue
        runs = f"{result.succeeded}/{result.succeeded + result.failures}"
        print(
            f"{result.label:<34}{runs:>6}"
            f"{gen['avg']:>9.0f}ms{gen['p50']:>9.0f}ms{gen['min']:>8.0f}ms{gen['max']:>8.0f}ms"
        )
    print("=" * 78)

    for result in results:
        print(f"  {result.label}: model={result.model}, failures={result.failures}")

    comparable = [r for r in results if r.stats("generation_ms") is not None]
    if len(comparable) == COMPARABLE_BACKENDS:
        a, b = comparable
        avg_a = a.stats("generation_ms")["avg"]  # type: ignore[index]
        avg_b = b.stats("generation_ms")["avg"]  # type: ignore[index]
        if avg_a > 0 and avg_b > 0:
            faster, slower = (b, a) if avg_b < avg_a else (a, b)
            ratio = max(avg_a, avg_b) / min(avg_a, avg_b)
            print(f"\n  {faster.label} is {ratio:.1f}x faster than {slower.label} on generation.")
    else:
        print("\n  Not enough successful backends to compare.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "Backend benchmark")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--warmup", action="store_true", help="Discard one generation per backend first")
    args = parser.parse_args()

    results = []
    for label, backend_env in BACKENDS:
        print(f"\n=== {label} ===")
        try:
            results.append(run_backend(label, backend_env, QUESTIONS, args.top_k, warmup=args.warmup))
        except Exception as exc:  # noqa: BLE001 — a backend that will not start is reported, not fatal
            print(f"  could not start backend: {exc}")
            failed = BackendResult(label, "unavailable")
            failed.failures = len(QUESTIONS)
            results.append(failed)

    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
