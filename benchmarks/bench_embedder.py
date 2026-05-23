"""Embedder backend benchmark — single-query latency (p50/p95/p99) and throughput (vec/s).

Backends: PyTorch-MPS, PyTorch-CPU, ONNX-CPU-FP32, ONNX-CPU-INT8, TorchScript-CPU (if exported).
Uses real FastAPI docs chunks for production-like text length distribution.

Usage:
    uv run python benchmarks/bench_embedder.py
    uv run python benchmarks/bench_embedder.py --single-runs 100
    uv run python benchmarks/bench_embedder.py --skip-onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SINGLE_QUERY = "How do I define a path parameter in FastAPI?"
BATCH_SIZES = (1, 8, 32, 128)
WARMUP_RUNS = 10
DEFAULT_SINGLE_RUNS = 50
THROUGHPUT_RUNS_PER_BATCH = 5


def load_batch_texts(n: int) -> list[str]:
    from api.config import settings
    from indexing.chunker import chunk_documents
    from indexing.loader import load_markdown_files

    docs = load_markdown_files(Path(settings.docs_source_path))
    chunks = chunk_documents(docs, chunk_size=1024, chunk_overlap=100)
    texts = [c.text for c in chunks[:n]]
    if len(texts) < n:
        msg = f"need {n} chunks, corpus has {len(texts)}"
        raise SystemExit(msg)
    return texts


def measure_latency(encode_fn, runs: int) -> dict[str, float]:
    """Single-query encode latency in ms, with warmup."""
    for _ in range(WARMUP_RUNS):
        encode_fn([SINGLE_QUERY])
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        encode_fn([SINGLE_QUERY])
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "min": min(times),
        "max": max(times),
    }


def measure_throughput(encode_fn, batch: list[str], runs: int) -> float:
    """Vectors per second for the given batch (avg over `runs` repetitions)."""
    for _ in range(2):
        encode_fn(batch)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        encode_fn(batch)
        times.append(time.perf_counter() - t0)
    return len(batch) / float(np.mean(times))


def _make_torchscript_encode_fn(model_path: Path, tokenizer_name: str, device: str = "cpu"):
    """Build an encode_fn around a traced TorchScript backbone.

    Traced backbone returns last_hidden_state; we mean-pool + L2-normalize
    to match OnnxEmbedder / PytorchEmbedder output semantics.
    """
    import torch
    from transformers import AutoTokenizer

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    torch_device = torch.device(device)

    def encode(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(torch_device)
            with torch.no_grad():
                last_hidden = model(inputs["input_ids"], inputs["attention_mask"])
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.extend(normalized.cpu().tolist())
        return all_embeddings

    return encode


def bench_backend(name: str, encode_fn, single_runs: int, batch_texts: list[str]) -> dict:
    print(f"\n▶ {name}")
    lat = measure_latency(encode_fn, single_runs)
    print(
        f"  latency (single query, ms): p50={lat['p50']:5.1f}  p95={lat['p95']:5.1f}  p99={lat['p99']:5.1f}  "
        f"(min={lat['min']:.1f}, max={lat['max']:.1f})"
    )
    tputs: dict[int, float] = {}
    for bs in BATCH_SIZES:
        tputs[bs] = measure_throughput(encode_fn, batch_texts[:bs], THROUGHPUT_RUNS_PER_BATCH)
    tput_str = "  ".join(f"bs={bs}: {tputs[bs]:7.1f}" for bs in BATCH_SIZES)
    print(f"  throughput (vec/s): {tput_str}")
    return {"latency": lat, "throughput": tputs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--single-runs", type=int, default=DEFAULT_SINGLE_RUNS, help="runs for latency measurement")
    p.add_argument("--skip-onnx", action="store_true", help="skip ONNX backends (when [onnx] extra not installed)")
    args = p.parse_args()

    print("=" * 78)
    print("DocsRAG embedder backend benchmark")
    print(f"single-query runs: {args.single_runs}  |  warmup: {WARMUP_RUNS}  |  batch sizes: {list(BATCH_SIZES)}")
    print("=" * 78)

    batch_texts = load_batch_texts(max(BATCH_SIZES))
    results: dict[str, dict] = {}

    from embeddings.pytorch import PytorchEmbedder

    pt_mps = PytorchEmbedder(device="mps")
    results["PyTorch-MPS"] = bench_backend(
        "PyTorch-MPS",
        lambda t: pt_mps.encode(t, show_progress=False),
        args.single_runs,
        batch_texts,
    )

    pt_cpu = PytorchEmbedder(device="cpu")
    results["PyTorch-CPU"] = bench_backend(
        "PyTorch-CPU",
        lambda t: pt_cpu.encode(t, show_progress=False),
        args.single_runs,
        batch_texts,
    )

    if not args.skip_onnx:
        from api.config import settings
        from embeddings.onnx import OnnxEmbedder

        onnx_fp32 = OnnxEmbedder(settings.embedder_onnx_fp32_path)
        results["ONNX-CPU-FP32"] = bench_backend(
            "ONNX-CPU-FP32",
            lambda t: onnx_fp32.encode(t, show_progress=False),
            args.single_runs,
            batch_texts,
        )

        onnx_int8 = OnnxEmbedder(settings.embedder_onnx_int8_path)
        results["ONNX-CPU-INT8"] = bench_backend(
            "ONNX-CPU-INT8",
            lambda t: onnx_int8.encode(t, show_progress=False),
            args.single_runs,
            batch_texts,
        )

    torchscript_path = Path("models/bge-small-en-v1.5.pt")
    if torchscript_path.exists():
        encode_fn = _make_torchscript_encode_fn(torchscript_path, "BAAI/bge-small-en-v1.5", device="cpu")
        results["TorchScript-CPU"] = bench_backend(
            "TorchScript-CPU",
            encode_fn,
            args.single_runs,
            batch_texts,
        )

    print("\n" + "=" * 78)
    print("Summary: single-query latency (ms)")
    print("=" * 78)
    print(f"| {'Backend':<16} | {'p50':>6} | {'p95':>6} | {'p99':>6} |")
    print(f"|{'-' * 18}|{'-' * 8}|{'-' * 8}|{'-' * 8}|")
    for name, r in results.items():
        lat = r["latency"]
        print(f"| {name:<16} | {lat['p50']:>6.1f} | {lat['p95']:>6.1f} | {lat['p99']:>6.1f} |")

    print("\n" + "=" * 78)
    print("Summary: throughput (vectors/sec)")
    print("=" * 78)
    bs_headers = " | ".join(f"bs={bs:<3}" for bs in BATCH_SIZES)
    print(f"| {'Backend':<16} | {bs_headers} |")
    print(f"|{'-' * 18}|" + "|".join(f"{'-' * 8}" for _ in BATCH_SIZES) + "|")
    for name, r in results.items():
        tputs = r["throughput"]
        cells = " | ".join(f"{tputs[bs]:>5.1f}" for bs in BATCH_SIZES)
        print(f"| {name:<16} | {cells} |")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
