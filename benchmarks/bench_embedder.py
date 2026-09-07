"""Embedder latency (p50/p95/p99) and throughput, over real documentation chunks.

The comparison only means something if every backend computes the *same* embedding
function, so `--verify-parity` checks each one against PyTorch before timing it.

    python -m benchmarks.bench_embedder --single-runs 100 --verify-parity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.config import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    EncodeFn = Callable[[list[str]], list[list[float]]]

SINGLE_QUERY = "How do I define a path parameter in FastAPI?"
BATCH_SIZES = (1, 8, 32, 128)
WARMUP_RUNS = 10
DEFAULT_SINGLE_RUNS = 50
THROUGHPUT_RUNS_PER_BATCH = 5
TORCHSCRIPT_PATH = Path("models/bge-small-en-v1.5.pt")
PARITY_MIN_COSINE = 0.999


def load_batch_texts(n: int) -> list[str]:
    from indexing.chunker import chunk_documents
    from indexing.loader import load_markdown_files

    docs = load_markdown_files(Path(settings.docs_source_path))
    chunks = chunk_documents(docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    texts = [c.text for c in chunks[:n]]
    if len(texts) < n:
        msg = f"Need {n} chunks but the corpus yielded {len(texts)}. Run `make fetch-docs` first."
        raise SystemExit(msg)
    return texts


def measure_latency(encode_fn: EncodeFn, runs: int) -> dict[str, float]:
    for _ in range(WARMUP_RUNS):
        encode_fn([SINGLE_QUERY])
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        encode_fn([SINGLE_QUERY])
        times.append((time.perf_counter() - started) * 1000)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "min": float(min(times)),
        "max": float(max(times)),
    }


def measure_throughput(encode_fn: EncodeFn, batch: list[str], runs: int) -> float:
    for _ in range(2):
        encode_fn(batch)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        encode_fn(batch)
        times.append(time.perf_counter() - started)
    return len(batch) / float(np.mean(times))


def make_torchscript_encode_fn(model_path: Path, tokenizer_name: str, device: str = "cpu") -> EncodeFn:
    """The traced graph returns `last_hidden_state`, so pooling happens here. bge-small
    pools on CLS; mean-pooling instead would silently benchmark a different function.
    """
    import torch
    from transformers import AutoTokenizer

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    torch_device = torch.device(device)
    batch_size = 32

    def encode(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(torch_device)
            with torch.no_grad():
                last_hidden = model(inputs["input_ids"], inputs["attention_mask"])
            pooled = last_hidden[:, 0]  # CLS token, matching bge-small's pooling config
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.extend(normalized.cpu().tolist())
        return all_embeddings

    return encode


def cosine_against_reference(encode_fn: EncodeFn, reference: EncodeFn, texts: list[str]) -> float:
    """Lowest per-text cosine against the reference encoder."""
    a = np.asarray(reference(texts), dtype=np.float64)
    b = np.asarray(encode_fn(texts), dtype=np.float64)
    if a.shape != b.shape:
        return 0.0
    return float((a * b).sum(axis=1).min())


def bench_backend(
    name: str,
    encode_fn: EncodeFn,
    single_runs: int,
    batch_texts: list[str],
) -> dict[str, object]:
    print(f"\n> {name}")
    latency = measure_latency(encode_fn, single_runs)
    print(
        f"  latency (single query, ms): p50={latency['p50']:5.1f}  p95={latency['p95']:5.1f}  "
        f"p99={latency['p99']:5.1f}  (min={latency['min']:.1f}, max={latency['max']:.1f})"
    )
    throughput = {bs: measure_throughput(encode_fn, batch_texts[:bs], THROUGHPUT_RUNS_PER_BATCH) for bs in BATCH_SIZES}
    print("  throughput (vec/s): " + "  ".join(f"bs={bs}: {throughput[bs]:7.1f}" for bs in BATCH_SIZES))
    return {"latency": latency, "throughput": throughput}


def collect_backends(*, skip_onnx: bool) -> list[tuple[str, EncodeFn]]:

    import torch

    from embeddings.pytorch import PytorchEmbedder

    backends: list[tuple[str, EncodeFn]] = []

    if torch.backends.mps.is_available():
        mps = PytorchEmbedder(device="mps")
        backends.append(("PyTorch-MPS", lambda t: mps.encode(t, show_progress=False)))
    elif torch.cuda.is_available():
        cuda = PytorchEmbedder(device="cuda")
        backends.append(("PyTorch-CUDA", lambda t: cuda.encode(t, show_progress=False)))
    else:
        print("  (no GPU backend available — benchmarking CPU only)")

    cpu = PytorchEmbedder(device="cpu")
    backends.append(("PyTorch-CPU", lambda t: cpu.encode(t, show_progress=False)))

    if not skip_onnx:
        for label, path in (
            ("ONNX-CPU-FP32", settings.embedder_onnx_fp32_path),
            ("ONNX-CPU-INT8", settings.embedder_onnx_int8_path),
        ):
            try:
                from embeddings.onnx import OnnxEmbedder

                embedder = OnnxEmbedder(path)
            except Exception as exc:  # noqa: BLE001 — a missing artifact skips one row, not the run
                print(f"  skipping {label}: {exc}")
                continue
            backends.append((label, lambda t, e=embedder: e.encode(t, show_progress=False)))

    if TORCHSCRIPT_PATH.exists():
        backends.append(
            ("TorchScript-CPU", make_torchscript_encode_fn(TORCHSCRIPT_PATH, settings.embedding_model, device="cpu"))
        )

    return backends


def print_summary(results: dict[str, dict[str, object]]) -> None:
    print("\n" + "=" * 78)
    print("Summary: single-query latency (ms)")
    print("=" * 78)
    print(f"| {'Backend':<16} | {'p50':>6} | {'p95':>6} | {'p99':>6} |")
    print(f"|{'-' * 18}|{'-' * 8}|{'-' * 8}|{'-' * 8}|")
    for name, result in results.items():
        latency = result["latency"]
        print(f"| {name:<16} | {latency['p50']:>6.1f} | {latency['p95']:>6.1f} | {latency['p99']:>6.1f} |")

    print("\n" + "=" * 78)
    print("Summary: throughput (vectors/sec)")
    print("=" * 78)
    print(f"| {'Backend':<16} | " + " | ".join(f"bs={bs:<3}" for bs in BATCH_SIZES) + " |")
    print(f"|{'-' * 18}|" + "|".join("-" * 8 for _ in BATCH_SIZES) + "|")
    for name, result in results.items():
        throughput = result["throughput"]
        print(f"| {name:<16} | " + " | ".join(f"{throughput[bs]:>5.1f}" for bs in BATCH_SIZES) + " |")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedder backend benchmark")
    parser.add_argument("--single-runs", type=int, default=DEFAULT_SINGLE_RUNS)
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX backends")
    parser.add_argument("--verify-parity", action="store_true", help="Check each backend against PyTorch first")
    parser.add_argument("--json", type=Path, default=None, help="Also write raw results to this file")
    args = parser.parse_args()

    print("=" * 78)
    print("DocsRAG embedder backend benchmark")
    print(f"single-query runs: {args.single_runs}  |  warmup: {WARMUP_RUNS}  |  batch sizes: {list(BATCH_SIZES)}")
    print("=" * 78)

    batch_texts = load_batch_texts(max(BATCH_SIZES))
    backends = collect_backends(skip_onnx=args.skip_onnx)
    if not backends:
        print("No embedder backend could be constructed.")
        return 1

    if args.verify_parity:
        reference_name, reference_fn = backends[0]
        parity_texts = batch_texts[:16]
        print(f"\nParity check against {reference_name} (min cosine over {len(parity_texts)} chunks):")
        for name, encode_fn in backends[1:]:
            cosine = cosine_against_reference(encode_fn, reference_fn, parity_texts)
            verdict = "ok" if cosine >= PARITY_MIN_COSINE else "DIFFERENT EMBEDDING FUNCTION"
            print(f"  {name:<16} {cosine:.6f}  {verdict}")

    results = {name: bench_backend(name, fn, args.single_runs, batch_texts) for name, fn in backends}
    print_summary(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nRaw results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
