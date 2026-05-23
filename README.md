# DocsRAG

Self-hosted RAG (Retrieval-Augmented Generation) system for technical documentation Q&A.

## TL;DR

End-to-end production-grade RAG system over FastAPI documentation (153 markdown files, 2540 chunks). Built from scratch as a learning project to demonstrate modern MLOps practices for LLM systems. Highlights:

- **Evaluation-driven design.** Every retrieval/generation decision is backed by Ragas metrics on a 25-question golden dataset, tracked in MLflow.
- **Honest negative results.** Tested hybrid search and agentic RAG; both lost to plain dense retrieval on this corpus and the README explains why.
- **Switchable inference.** Same API code runs against Ollama (dev) or vLLM (prod) via a single env variable. Benchmark on M4 Max: vllm-metal **3.8× faster** than Ollama.
- **Switchable embedder.** Same factory pattern for `sentence-transformers` (PyTorch on MPS/CUDA/CPU) or ONNX Runtime (FP32 / INT8) via `EMBEDDER_BACKEND`. ONNX-CPU-FP32 single-query latency is **3.2× faster than PyTorch-MPS** on bge-small (1.7 ms p50 vs 5.5 ms).
- **Cross-language Q&A.** Ask in Russian, get Russian answers — implemented as a thin RU↔EN translation wrapper over the English-only pipeline. No reindex required.
- **Full observability stack.** Prometheus + Grafana for system metrics, LangFuse for LLM tracing — both fully optional and additive.

## Key Findings

These are the non-obvious results from running the full eval pipeline. Each is the kind of thing you only learn by actually building and measuring:

1. **Bigger chunks beat more chunks.** Going from `chunk_size=512, top_k=10` to `chunk_size=1024, top_k=5` gave better metrics on every dimension (faithfulness +0.064, context_precision +0.072) while sending fewer tokens to the LLM. More semantic context per chunk > more chunks.

2. **BM25 hurts on semantically rich corpora.** Hybrid retrieval (dense + BM25 via RRF) underperformed pure dense by **−0.093 faithfulness**. Technical documentation about FastAPI is full of natural-language explanation; keyword overlap from BM25 added more noise than signal. A cross-encoder reranker recovered some quality but still didn't beat dense.

3. **Agentic RAG is a precision/recall trade, not a free win.** A LangGraph agent with relevance grading improved `context_precision` by +0.055 but cut `context_recall` by −0.107 — the binary grader discards borderline-relevant chunks that actually contained answers. Pick agentic when precision matters more than coverage; pick simple RAG otherwise.

4. **vLLM on Apple Silicon is real and fast.** The MLX-based `vllm-metal` server delivers OpenAI-compatible API with **3.8× faster generation** than llama.cpp-based Ollama (891ms vs 3375ms avg) on M4 Max. Same code path works for production CUDA vLLM — swap the image, set `VLLM_BASE_URL`.

5. **Cosine + normalized embeddings is non-negotiable.** Forgetting `normalize_embeddings=True` in `sentence-transformers` silently breaks retrieval quality without obvious errors. The bug doesn't surface until you measure with Ragas.

6. **ONNX on CPU beat PyTorch on MPS — and INT8 didn't fit.** The bge-small embedder is small enough (30M params) that ORT's per-call overhead + graph optimizations dominate over MPS's GPU dispatch overhead: **ONNX-CPU-FP32 single-query latency is 3.2× faster than PyTorch-MPS** (1.7 ms vs 5.5 ms p50), with byte-identical retrieval (cosine parity = 1.0, Ragas Δ ≤ 0.01). The intuition "GPU should always win" is wrong for sub-100M models. Dynamic INT8 went the other way — `context_recall` dropped 0.070 (12.6% relative), well past the 0.05 acceptance budget, and the model got flagged as unusable for retrieval. INT8 noise is invisible on cosine-of-same-text (0.997) but compounds across top-k ranking. Both results are documented honestly in the [Embedder ONNX optimization section](#embedder-onnx-optimization) — the negative INT8 result is as informative as the FP32 win.

## Goals

A production-grade RAG system demonstrating modern MLOps practices:
- End-to-end RAG pipeline with hybrid retrieval and reranking
- Agentic workflow via LangGraph (query rewriting, relevance grading)
- Quality evaluation with Ragas, experiment tracking with MLflow
- Full observability: LLM tracing (LangFuse) + system metrics (Prometheus/Grafana)
- Multi-backend inference: Ollama for development, vLLM for production

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic |
| LLM | Qwen 2.5 7B Instruct via Ollama (dev) / vllm-metal MLX (prod) |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, English, normalized cosine) |
| Embedder backend | `pytorch` (MPS/CUDA/CPU) · `onnx-fp32` (3.4× faster, byte-identical retrieval) · `onnx-int8` (benchmark artifact, fails Ragas budget) |
| Vector DB | Qdrant (cosine similarity) |
| Orchestration | LangChain + LangGraph |
| Retrieval | Dense (Qdrant) + Sparse (BM25) + Cross-encoder reranker |
| Evaluation | Ragas + MLflow |
| Observability | LangFuse, Prometheus, Grafana |
| Prod inference | vLLM (vllm-metal on Apple Silicon, vllm+CUDA in cloud) |
| Packaging | Docker Compose, uv |

## System Architecture

```mermaid
graph LR
    User[Client] -->|POST /ask| API[FastAPI Service]

    subgraph Inference
        API -->|embed query| Embed[bge-small-en-v1.5<br/>sentence-transformers]
        API -->|vector search| Qdrant[(Qdrant<br/>2540 chunks)]
        API -->|chat completion| LLM{LLM Backend}
        LLM -->|dev| Ollama[Ollama<br/>Qwen 2.5 7B]
        LLM -->|prod| vLLM[vllm-metal / vLLM<br/>Qwen 2.5 7B 4bit]
    end

    subgraph Observability
        API -.metrics.-> Prom[Prometheus]
        API -.traces.-> LF[LangFuse]
        Prom --> Graf[Grafana]
    end

    subgraph Eval
        RunEval[evaluation/run_eval.py] -->|Ragas metrics| MLflow[(MLflow)]
        RunEval -.uses.-> API
    end

    classDef storage fill:#e8d5ff,stroke:#5a3e8a
    classDef service fill:#d4e8ff,stroke:#3e5a8a
    classDef obs fill:#d5ffe8,stroke:#3e8a5a
    class Qdrant,MLflow storage
    class API,Embed,Ollama,vLLM,LLM service
    class Prom,LF,Graf,RunEval obs
```

The API is the only stateful service. Qdrant holds chunk embeddings; MLflow holds eval runs. Ollama / vLLM are stateless inference servers swapped via `INFERENCE_BACKEND` env var. Observability is fully additive — the system runs unchanged without LangFuse keys or with Prometheus disabled.

## Quick Start

> **Platform note:** this guide was developed and tested on **Apple Silicon (MacBook M4 Max)**. Core services (Qdrant, API, MLflow) run in Docker and should work on any platform. Ollama, embeddings (MPS), and vllm-metal are macOS ARM64-specific — behaviour on other systems is not guaranteed.

### Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Docker Desktop | Qdrant, API, MLflow, Prometheus, Grafana | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Python 3.12 | Local tooling (eval, indexing, benchmarks) | `brew install python@3.12` |
| uv | Fast Python package manager | `brew install uv` |
| Ollama (macOS app) | LLM inference — runs natively for Metal GPU | [ollama.com](https://ollama.com) |

### Step 1 — Clone and install

```bash
git clone <repo-url>
cd DocsRAG

# Create virtualenv and install all dependencies
make install
```

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Required fields in `.env`:

```
HF_TOKEN=hf_...   # Hugging Face token (required to download the embedding model)
```

All other values can be left as-is for local development.

### Step 3 — Pull the LLM model into Ollama

```bash
# Make sure the Ollama app is running (icon in the menu bar)
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### Step 4 — Start infrastructure

```bash
make up        # starts Qdrant + API + MLflow + Prometheus + Grafana
make health    # checks that everything is up
```

The first API start takes ~30–60 s — the embedding model (~130 MB) is being downloaded.

### Step 5 — Index documents

```bash
make fetch-docs   # downloads 153 FastAPI docs markdown files into data/raw/
make reindex      # indexes into Qdrant (chunk_size=1024, overlap=100)
```

### Step 6 — Ask a question

```bash
make warmup                                             # loads the LLM into Ollama RAM
make ask Q='How do I define a path parameter in FastAPI?'
```

Expected response in ~3–5 s with source citations (`tutorial/path-params.md`).

### Step 7 — Observability

```bash
make grafana-ui    # http://localhost:3000 — login admin/admin, DocsRAG dashboard
make prometheus-ui # http://localhost:9090 — raw metrics
make mlflow-ui     # http://localhost:5000 — eval experiment results
```

LangFuse tracing is enabled by adding keys to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```
Get keys at: [cloud.langfuse.com](https://cloud.langfuse.com) → project → Settings → API Keys.

### Step 8 — vllm-metal backend (Apple Silicon only)

Faster inference via MLX (3.8× faster than Ollama on M4 Max). `vllm-metal` 0.2.0 is a plugin to upstream `vllm`, so both packages must be installed into the project venv via `make install-vllm`. They're not in `pyproject.toml` / `uv.lock`: vllm's own pyproject hard-pins CUDA-only deps (`nvidia-cudnn-frontend`, `cuda-python`, `flashinfer-python`...) that have no macOS wheels, and the manual two-phase install (CPU requirements → main build) can't be expressed in uv's universal resolver — verified experimentally.

```bash
# 1. Install vllm core + vllm-metal plugin into .venv (builds vllm from source)
source .venv/bin/activate
make install-vllm

# 2. Start the server (default model: Qwen2.5-7B-Instruct-4bit; override via VLLM_MODEL in .env)
make vllm-start

# 3. Check status
make vllm-status

# 4. Switch the API to vllm
echo "INFERENCE_BACKEND=vllm" >> .env
make restart

# 5. Verify with a request
make ask Q='What is FastAPI?'
```

Versions are pinned via `VLLM_VERSION` / `VLLM_METAL_WHEEL` variables at the top of the `Makefile` — bump them explicitly when upgrading.

**Caveat:** installing `vllm` pulls a large dependency tree (torch, transformers, kernels) and overrides versions of packages also used by the RAG pipeline. After install run `make health` to confirm the API still starts. If you ever run `uv pip sync uv.lock`, vllm + metal will be removed — just re-run `make install-vllm`.

### Step 9 — Run evaluation

```bash
# Ollama (baseline)
make eval CONFIG=configs/chunk_1024.yaml

# vllm-metal (requires a running vllm-metal server)
INFERENCE_BACKEND=vllm uv run python evaluation/run_eval.py --config configs/chunk_1024.yaml
```

### Step 10 — ONNX embedder backend

The embedder can run on three swappable backends: `pytorch` (default), `onnx-fp32`, `onnx-int8`. Switch via `EMBEDDER_BACKEND` in `.env`. ONNX is an opt-in extra (~200 MB of deps) — installed separately to keep CI lean.

```bash
# 1. Install [onnx] extra into the venv
source .venv/bin/activate
make install-onnx

# 2. Export bge-small to ONNX FP32 (produces models/bge-small-en-v1.5-onnx-fp32/, 127 MB)
make export-onnx

# 3. Quantize FP32 → INT8 dynamic per-channel (produces models/bge-small-en-v1.5-onnx-int8/, 32 MB)
make quantize-onnx

# 4. Validate parity vs PyTorch baseline (cosine > 0.9999 expected; actual: 1.000000 across 120 chunks)
pytest tests/test_embedder_parity.py -v -s

# 5. Reindex into the two collections (docsrag → ONNX FP32 vectors, docsrag_int8 → ONNX INT8 vectors).
#    Each ~30-60 s. Skip these if you only want to bench latency without changing the index.
make reindex-onnx
make reindex-int8

# 6. (Optional bonus) Trace bge backbone to TorchScript .pt for an extra bench row
make export-torchscript

# 7. Bench 4 backends (or 5 if TorchScript .pt exists) — ~2-3 min
make bench-embedder

# 8. Run Ragas eval for each ONNX backend
make eval CONFIG=configs/onnx_fp32.yaml
make eval CONFIG=configs/onnx_int8.yaml

# 9. Switch the API to ONNX backend (optional — default stays pytorch)
echo "EMBEDDER_BACKEND=onnx-fp32" >> .env
make restart
make health    # should show "embedder_backend": "onnx-fp32"
```

See [Embedder ONNX optimization](#embedder-onnx-optimization) below for the numeric results and architectural decisions.

## Troubleshooting

Real failure modes hit while building / running this project:

- **First `/ask` after `make up` is slow (10–20 s), subsequent calls fast.** Ollama lazy-loads the model into RAM on the first request. Not a bug. Run `make warmup` after `make up` to pay this cost once outside the user's request.

- **API container starts but `/ask` returns `ConnectionError`.** The Ollama menu-bar app isn't running on the host. From inside the container: `curl -s http://host.docker.internal:11434/api/tags` should return JSON. If not, start the Ollama app or `ollama serve` natively. (Variant A — Ollama on the host — is the default; Variant B with Ollama in a sibling container is commented out in `docker-compose.yml`.)

- **`docker compose build` fails on `torch`.** Docker Desktop memory limit. Bump to 8+ GB in Settings → Resources → Memory.

- **Embedding model re-downloads on every `up --build`.** The `hf_cache` Docker volume in `docker-compose.yml` mounts `/app/.cache/huggingface`. If you removed it, the ~470 MB download repeats on every container rebuild.

- **`Failed to export span batch code: 401, reason: Unauthorized` floods logs during eval.** LangFuse keys in `.env` are wrong / expired / for the wrong region. Either fix them (`cloud.langfuse.com` → Settings → API Keys; check whether your project is in US or EU region — the keypairs are different) or clear them (`LANGFUSE_PUBLIC_KEY=` and `LANGFUSE_SECRET_KEY=`) to disable tracing entirely. The 401s don't break Ragas — numbers in MLflow are unaffected — they're just noise.

- **`make vllm-start` fails with `vllm: command not found`.** You're not in the project venv (`source .venv/bin/activate`) or upstream `vllm` isn't installed yet (`make install-vllm`).

- **`uv pip sync uv.lock` silently removes vllm-metal and/or ONNX deps.** They're not pinned in `pyproject.toml` / `uv.lock` (vllm: CUDA-only deps without macOS wheels; ONNX: kept as opt-in extra). After any `uv pip sync`, recovery is `make install-vllm` and/or `make install-onnx`.

- **Russian answers come back with garbled Cyrillic.** On `INFERENCE_BACKEND=vllm` with the default 7B-4bit MLX, the translator's EN→RU pass produces latin-with-acute artefacts mid-word (e.g. `разdéлвние`). Switch `VLLM_MODEL` in `.env` to `mlx-community/Qwen2.5-14B-Instruct-4bit` and `make vllm-start` again. Ollama and 14B-MLX both handle Russian cleanly.

- **`pytest tests/test_embedder_parity.py` errors on `ModuleNotFoundError: No module named 'embeddings'`** but `python -c "from embeddings ..."` works. The editable install's package list (`__editable__.docsrag-0.1.0.pth`) was registered before a new top-level package was added. Recovery: `uv pip install -e . --no-deps`.

## API

The RAG API runs on `http://localhost:8000`.

### `GET /health`

```bash
make health
```

Returns Qdrant collection status, point count, and configured model names.

### `POST /ask`

```bash
make ask Q='How does dependency injection work in FastAPI?'
```

Parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | — | Natural language question |
| `top_k` | int | 5 | Number of chunks to retrieve |
| `include_contexts` | bool | false | Include raw chunk text in response |

Response includes `answer`, `sources` (with `source_path`, `header_path`, `score`), and timing breakdown (`retrieval_ms`, `generation_ms`, `translation_ms`, `total_ms`).

### `POST /agent/ask`

Routes the question through the LangGraph agent (query rewriting + relevance grading + conditional retry). Same request/response shape as `/ask`. Use when precision matters more than recall.

### Cross-language support

The same endpoints (`/ask` and `/agent/ask`) accept questions in **Russian** without any flag or extra parameter. The pipeline detects Cyrillic in the question, translates it to English for retrieval and generation, then translates the answer back to Russian before returning it.

```bash
make ask Q='Как определить path-параметр в FastAPI?'
```

Response shape is unchanged; `translation_ms` reports the combined RU→EN + EN→RU latency (`0` for English questions). File-path citations like `[tutorial/path-params.md]` and code blocks are preserved verbatim. Translation steps log at INFO level (`make api-logs` shows `RU→EN | in=... | out=...`).

**Backend choice matters for Russian.** On `INFERENCE_BACKEND=vllm` with the default `Qwen2.5-7B-Instruct-4bit` (MLX), the EN→RU step produces garbled Cyrillic — Latin-with-acute artefacts mid-word (e.g. `разdéлвние` instead of `разработки`). Use `Qwen2.5-14B-Instruct-4bit` for clean Russian output (set `VLLM_MODEL` in `.env`). On `INFERENCE_BACKEND=ollama` (default), the GGUF-quantized 7B handles Russian cleanly — no model swap needed. The 4bit MLX quantization of Qwen 2.5 7B has a vocabulary/sampling artefact that the larger 14B model avoids.

**Why translation, not multilingual embeddings?** The index uses `BAAI/bge-small-en-v1.5` (English-only) and was tuned on an English golden dataset (faithfulness 0.882, context_recall 0.557). Swapping to a multilingual embedder (`bge-m3`, `multilingual-e5`) requires a full reindex on a ~2 GB model and would degrade the validated English baseline. Translation is reversible, leaves the index untouched, and reuses the existing multilingual LLM (Qwen 2.5) — at a cost of two extra LLM calls per Russian query (~+1.5 s on Ollama 7B, ~+8 s on vllm-metal 7B, ~+13 s on vllm-metal 14B).

## Evaluation

Evaluation uses [Ragas](https://docs.ragas.io) metrics over a 25-question golden dataset derived from FastAPI documentation. Results are tracked in MLflow (`http://localhost:5000`).

```bash
make eval CONFIG=configs/chunk_1024.yaml   # run evaluation (dense baseline)
make mlflow-ui                             # open MLflow UI
```

**Metric reference:**
- **faithfulness** — does the answer follow from the retrieved context (no hallucination)?
- **answer_relevancy** — does the answer actually address the question?
- **context_precision** — of the retrieved chunks, how many are actually relevant?
- **context_recall** — of the chunks needed to answer, how many were retrieved?

`faithfulness` and `answer_relevancy` measure generation quality; `context_precision` and `context_recall` measure retrieval quality.

### sweep — chunk size and top-k (dense retrieval)

| Config | chunk\_size | overlap | top\_k | faithfulness | answer\_relevancy | context\_precision | context\_recall |
|---|---|---|---|---|---|---|---|
| chunk\_256 | 256 | 25 | 5 | 0.646 | 0.767 | 0.417 | 0.353 |
| baseline | 512 | 50 | 5 | 0.757 | 0.849 | 0.506 | 0.431 |
| topk\_3 | 512 | 50 | 3 | 0.719 | 0.775 | 0.517 | 0.403 |
| topk\_10 | 512 | 50 | 10 | 0.818 | **0.892** | 0.526 | 0.517 |
| **chunk\_1024** ✓ | **1024** | **100** | **5** | **0.882** | 0.886 | **0.598** | **0.557** |

**Takeaway:** chunk size dominates top-k. Doubling chunk size (512→1024) gave a bigger metric jump than doubling top-k (5→10), and used half the chunks. Frozen baseline for all subsequent experiments: `chunk_1024`.

### hybrid search and reranking (chunk\_size=1024, top\_k=5)

| Strategy | faithfulness | answer\_relevancy | context\_precision | context\_recall |
|---|---|---|---|---|
| **dense** ✓ | **0.882** | 0.886 | **0.598** | **0.557** |
| hybrid (dense + BM25 → RRF) | 0.789 | 0.818 | 0.556 | 0.523 |
| hybrid\_rerank (+ cross-encoder) | 0.825 | **0.890** | 0.566 | 0.510 |

**Finding:** dense retrieval outperforms both hybrid variants on this dataset. BM25 adds keyword-match noise to semantically rich technical documentation where the dense embeddings already perform well. The cross-encoder partially recovers `answer_relevancy` and `context_precision` but cannot fully offset the RRF noise. Dense remains the production strategy.

**When hybrid would likely help instead:** corpora with many exact-match terms that embeddings struggle with — error codes, API tokens, version numbers, product SKUs, function names without surrounding prose. The FastAPI docs corpus is the opposite: prose-heavy explanations where dense semantics shine.

### agentic RAG (chunk\_size=1024, top\_k=5, dense retrieval)

| Strategy | faithfulness | answer\_relevancy | context\_precision | context\_recall |
|---|---|---|---|---|
| dense (baseline) | **0.882** | 0.886 | 0.598 | **0.557** |
| agentic | 0.817 | 0.813 | **0.653** | 0.450 |

**Finding:** agentic grading improves `context_precision` (+0.055) by filtering irrelevant chunks before generation, but at the cost of `context_recall` (−0.107): the binary relevance grader discards borderline-relevant chunks. `faithfulness` and `answer_relevancy` drop slightly because graded-out context sometimes contained answers. Dense remains the better end-to-end strategy; the agentic pipeline is useful when precision matters more than recall.

## Agentic RAG Graph

The `/agent/ask` endpoint runs questions through a LangGraph agent that rewrites the query, grades retrieved chunks for relevance, and retries retrieval if necessary.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([START]):::first
    query_rewriter(query_rewriter)
    retriever(retriever)
    relevance_grader(relevance_grader)
    generator(generator)
    __end__([END]):::last
    __start__ --> query_rewriter;
    query_rewriter --> retriever;
    retriever --> relevance_grader;
    relevance_grader -. generate .-> generator;
    relevance_grader -. retry .-> query_rewriter;
    generator --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

**Nodes:**
- `query_rewriter` — LLM rewrites the question to improve retrieval; on retry uses different phrasing
- `retriever` — dense vector search via Qdrant
- `relevance_grader` — LLM scores each chunk as relevant/not relevant (JSON verdict)
- `generator` — generates the final answer from relevant chunks only

**Retry logic:** if fewer than 2 chunks pass grading and no retry has been attempted, the graph loops back to `query_rewriter`. Maximum 1 retry.

## vLLM backend + benchmark (Apple Silicon, M4 Max)

Inference backend is switchable via `INFERENCE_BACKEND=ollama|vllm` in `.env`.  
With `vllm`, the API uses `ChatOpenAI` pointing at a [vllm-metal](https://github.com/vllm-project/vllm-metal) endpoint (OpenAI-compatible, same API as production vLLM on CUDA).

**Benchmark — generation latency, 5 warm questions, top\_k=3:**

| Backend | avg gen | p50 gen | min | max |
|---|---|---|---|---|
| Ollama (Qwen2.5-7B q4\_K\_M, llama.cpp) | 3375ms | 3439ms | 2064ms | 4327ms |
| **vllm-metal (Qwen2.5-7B 4bit, MLX)** | **891ms** | **911ms** | **705ms** | **1075ms** |

**Finding:** vllm-metal is **3.8× faster** on generation latency vs Ollama on M4 Max. MLX uses Apple Silicon unified memory more efficiently than llama.cpp. On a CUDA GPU, the same code (with `vllm/vllm-openai` image) would provide similar or greater speedup.

**Quality comparison (Ragas eval, 25 samples, chunk\_size=1024, top\_k=5):**

| Metric | Ollama q4\_K\_M | vllm-metal 4bit | Δ |
|---|---|---|---|
| faithfulness | **0.882** | 0.827 | −0.055 |
| answer\_relevancy | 0.886 | **0.907** | +0.021 |
| context\_precision | 0.598 | 0.598 | ≈0 |
| context\_recall | 0.557 | 0.557 | ≈0 |

Context metrics are identical (same retrieval). Minor faithfulness/relevancy gap reflects quantization format differences (GGUF q4\_K\_M vs MLX 4bit), not a meaningful quality difference at this sample size.

To reproduce:
```bash
# 1. Install vllm-metal (macOS ARM64 only — see Step 8 in Quick Start for the
#    full install (vllm core 0.20.1 from source + plugin wheel))

# 2. Start the vllm server (plugin auto-registers as a vllm platform backend)
vllm serve mlx-community/Qwen2.5-7B-Instruct-4bit --host 127.0.0.1 --port 8001

# 3. Verify the server is up
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "messages": [{"role": "user", "content": "Hi!"}], "max_tokens": 20}'

# 4. Switch the API to vllm backend
echo "INFERENCE_BACKEND=vllm" >> .env

# 5. Run benchmark (Ollama must also be running)
uv run python benchmarks/bench_backends.py

# 6. Run Ragas eval on vllm
INFERENCE_BACKEND=vllm uv run python evaluation/run_eval.py --config configs/chunk_1024.yaml
```

## Embedder ONNX optimization

Swappable embedder backend mirroring `make_llm()` pattern: same factory, three runtimes. Demonstrates ONNX Runtime + dynamic INT8 quantization on a classical-ML serving stack.

### Backends

| Backend | Model file | Size | Where it reads / writes |
|---|---|---|---|
| `pytorch` | HF `BAAI/bge-small-en-v1.5` | ~130 MB | `docsrag` collection |
| `onnx-fp32` | `models/bge-small-en-v1.5-onnx-fp32/model.onnx` | 127 MB | `docsrag` collection (parity-equivalent to PyTorch) |
| `onnx-int8` | `models/bge-small-en-v1.5-onnx-int8/model.onnx` | 32 MB | `docsrag_int8` collection (separate — vectors differ) |

Selected via `EMBEDDER_BACKEND` env var. `settings.active_qdrant_collection` routes Qdrant queries to the right collection automatically (see `api/config.py`).

### Single-query latency benchmark (M4 Max, `make bench-embedder`, 50 runs, 10 warmup)

| Backend | p50 | p95 | p99 |
|---|---|---|---|
| PyTorch-MPS | 5.7 ms | 9.0 ms | 9.2 ms |
| PyTorch-CPU | 6.7 ms | 7.1 ms | 7.4 ms |
| TorchScript-CPU | 4.7 ms | 4.9 ms | 4.9 ms |
| **ONNX-CPU-FP32** | **1.7 ms** | **1.8 ms** | **1.8 ms** |
| ONNX-CPU-INT8 | 1.5 ms | 1.6 ms | 1.7 ms |

**Headline:** ONNX-CPU-FP32 is **3.4× faster than PyTorch-MPS** on single-query latency. The intuition "MPS GPU should win for embeddings" doesn't hold for small models — bge-small is 30M params, and the per-call MPS dispatch overhead + Python ↔ Metal boundary cost dominates over the actual matrix math. ONNX Runtime minimizes that overhead and applies graph-level optimizations (operator fusion, constant folding) that pay no GPU-roundtrip cost. TorchScript-CPU gives a modest ~30% win over plain PyTorch-CPU (graph freezing + IR optimizations) but doesn't approach ONNX — TorchScript's IR is shallower than ORT's optimization pipeline. INT8 trims another ~12% on p50 with no longer the tail-variance issue we saw on the earlier run.

### Throughput benchmark (vectors/sec)

| Backend | bs=1 | bs=8 | bs=32 | bs=128 |
|---|---|---|---|---|
| PyTorch-MPS | 156 | 319 | 206 | **351** |
| PyTorch-CPU | 57 | 106 | 80 | 135 |
| TorchScript-CPU | 62 | 115 | 83 | 97 |
| ONNX-CPU-FP32 | 106 | 64 | 32 | 38 |
| ONNX-CPU-INT8 | 78 | 76 | 35 | 41 |

**Workload mapping:** ONNX-CPU-FP32 for `/ask` (latency wins). PyTorch-MPS for `make reindex` (throughput wins at bs≥8 — **9× faster** than ONNX at bs=128). The ONNX throughput drop at larger batches is most likely caused by per-batch padding to the longest sequence — ORT runs the full attention pattern regardless of attention mask, while `sentence-transformers` and TorchScript have more efficient masked-attention paths (TorchScript scales like PyTorch-CPU, no drop, confirming the bottleneck is ORT-specific). Not blocking for production (single-query path is what `/ask` uses), investigation deferred.

### Quality validation

**PyTorch ↔ ONNX FP32 parity** (`tests/test_embedder_parity.py`, 120 real chunks):

```
parity OK across 120 chunks: cosine min=1.000000, mean=1.000000, max=1.000000
```

Byte-perfect numerical equivalence — `optimum-cli export onnx` traces the existing `sentence-transformers` Python code into the graph, including the pooling and L2 normalization. We read the graph-baked `sentence_embedding` output directly (no manual pooling in our wrapper), so there's no implementation gap to drift through.

**ONNX FP32 ↔ INT8 quantization noise** (120 chunks):

| | per-tensor (rejected) | per-channel (production) |
|---|---|---|
| mean cosine | 0.9759 | **0.9973** |
| min cosine | 0.9456 | 0.9816 |
| chunks below 0.99 | 100% | 1.7% |
| size | 32.25 MB | 32.45 MB |

Per-channel quantization (own zero-point / scale per channel) recovers almost all the parity loss at +0.2 MB cost. Per-tensor would have likely blown the Ragas budget; per-channel keeps it safe. The `--per-tensor` flag in `scripts/quantize_onnx.py` preserves the alt variant for reproducibility.

### Ragas evaluation — end-to-end gate

Each ONNX backend is graded by Ragas against the same 25-question golden dataset, with strict budgets vs the cached PyTorch `chunk_1024` baseline. Acceptance: FP32 must stay within ±0.02 on all four metrics; INT8 may drop faithfulness / context_recall by at most 0.05 before being flagged unusable.

| Metric | PyTorch baseline | ONNX FP32 | Δ FP32 | ONNX INT8 | Δ INT8 |
|---|---|---|---|---|---|
| faithfulness | 0.882 | 0.889 | +0.007 | 0.873 | −0.009 |
| answer_relevancy | 0.886 | 0.886 | −0.001 | 0.867 | −0.019 |
| context_precision | 0.598 | 0.598 | ≈0 | 0.589 | −0.009 |
| context_recall | 0.557 | 0.557 | ≈0 | **0.487** | **−0.070** |

**ONNX FP32 — PASS.** Every delta within ±0.01; retrieval-metrics byte-identical to PyTorch (direct consequence of cosine 1.0 parity); generation-metrics within LLM-as-judge noise. Combined with the 3.2× single-query latency win above, ONNX FP32 is a strict improvement over PyTorch-MPS on `/ask`. Switching the API to `EMBEDDER_BACKEND=onnx-fp32` is a no-quality-cost upgrade.

**ONNX INT8 — FAIL.** `context_recall` dropped 0.070, exceeding the 0.05 budget. The 0.997 mean cosine vs FP32 turns into top-5 reshuffling that drops borderline-relevant chunks: precision stays fine (graded chunks are still relevant), but coverage is not. **INT8 is documented as unusable for production embedding on this model** and kept only as a benchmark artifact in `docsrag_int8`. This is a small-model-specific failure mode — bge-small is ~30M params, where dynamic INT8 noise has nowhere to hide. Larger encoders (`bge-base` 110M, `bge-large` 335M) typically tolerate INT8 much better; if switching the embedder model upwards in the future, re-run the quantization + Ragas gate before drawing a verdict for that model.

Static quantization with a calibration set is the documented escalation path for trying to recover INT8 quality, but on a personal-project budget the negative result is itself the deliverable — measuring honestly that INT8 doesn't work here is more informative than bashing on it until it does. Numbers above live in MLflow under experiment `docsrag-rag-eval` (`make mlflow-ui`).

## Lessons Learned

Things this project taught me that aren't in any RAG tutorial:

**1. Eval-driven beats intuition-driven, every time.**  
Three different decisions in this project (chunk size, hybrid vs dense, agentic vs simple) felt obvious going in and turned out the opposite when measured. Without Ragas + MLflow I would have shipped worse versions of all three with full confidence. The evaluation harness was the single highest-leverage thing in the project.

**2. "Production-ready" means swappable.**  
The single `INFERENCE_BACKEND` env var is the difference between a one-off demo and a system you can actually deploy. The exact same RAG code runs against Ollama on a laptop or vLLM on H100s — only the URL changes. Designing for that boundary from the start (via OpenAI-compatible API) was free; retrofitting it would have been painful.

**3. Determinism is a feature.**  
`temperature=0.0` everywhere isn't paranoia — it's what makes Ragas evaluation actually reproducible across runs. Once that's broken, every "the metric improved" claim becomes "the metric improved or maybe noise."

**4. Observability has to be additive.**  
LangFuse and Prometheus were added with zero changes to the RAG pipeline logic. The pipeline doesn't know whether tracing is on. If observability is invasive (callbacks threading through business logic, conditional code paths for "metrics enabled"), it gets ripped out the first time it breaks something. Decouple it.

**5. Embed once, embed everywhere — but make sure it's literally the same embedder.**  
Both `indexing/run_indexing.py` and `api/rag.py` go through the same `embeddings.factory.make_embedder()` factory. Using a different LangChain wrapper at query time (even one that "should" be equivalent) silently degrades retrieval because of subtle differences in pooling/normalization. The bug doesn't crash, it just makes things slightly worse. Eval would catch it; trust wouldn't. This is also why ONNX backend was validated with a hard cosine-parity gate (`tests/test_embedder_parity.py`, threshold 0.9999) before being trusted to read the existing index — the parity test would have caught any pooling drift between the PyTorch wrapper and the ONNX graph.

**6. Honest negative results > impressive demos.**  
Showing that hybrid+rerank lost to dense, and that agentic RAG sacrificed recall for precision, is more interesting than claiming everything got better. Anyone can build a stack of trendy components; understanding the trade-offs is the actual MLOps skill.

## Production Considerations

What I'd do differently if this were a real production system:

**Inference scaling**  
- Move vLLM to a CUDA host with `vllm/vllm-openai` Docker image; same `INFERENCE_BACKEND=vllm` codepath works without changes.
- Front it with a load balancer; vLLM supports continuous batching and multiple replicas.
- Add request queueing with backpressure — `/ask` should fail fast under overload, not pile up.

**Vector DB**  
- Qdrant in HA mode with replicas; the current single-node setup loses data on disk failure.
- Add a payload index on `source_path` if filtering by document section becomes a feature.
- Periodic re-indexing pipeline (Airflow / Prefect) instead of a manual `make reindex`.

**Evaluation**  
- Expand the golden dataset from 25 to 200+ questions, ideally human-curated from real user logs.
- Run eval in CI on every PR that touches `api/` or `indexing/` — fail builds on metric regression.
- Add LLM-as-judge eval alongside Ragas to cover dimensions Ragas doesn't measure (style, completeness).

**Observability**  
- Self-hosted LangFuse instead of cloud free tier, integrated with org SSO.
- Alert rules in Prometheus on `rag_generation_duration_seconds` p99, on Qdrant unavailability, on `rag_requests_total{status="error"}` rate.
- Cost tracking — log token counts and compute per-request inference cost into Grafana.

**Security**  
- API auth (the current `/ask` is open). At minimum API keys, ideally OIDC.
- Prompt injection mitigation — the current system prompt is firm but not exhaustively tested adversarially.
- PII filtering on retrieved chunks if the corpus ever contains user data.

**Quality**  
- Streaming responses (`/ask/stream`) — current p95 is 5–8s, perceived latency would drop dramatically with token streaming.
- Caching: identical-question cache keyed on question hash + retrieval config. ~30% of FAQ-style traffic is duplicates.
- Re-ranking with a domain-tuned cross-encoder once the corpus stabilizes — generic `bge-reranker-v2-m3` is a starting point, not a finish line.

## Design Notes

Non-obvious architectural decisions to know before extending the code:

- **Pipeline is a single cached instance.** `api/rag.py::get_pipeline()` is decorated with `@lru_cache(maxsize=1)` and FastAPI's `lifespan` calls it once at startup. The embedder (~130 MB), Qdrant client, and LLM wrapper are heavy — never construct `RAGPipeline()` directly inside a request handler. Inject via `Annotated[RAGPipeline, Depends(get_pipeline)]`.

- **Endpoints are `def`, not `async def`.** Both `qdrant_client.query_points()` and `chain.invoke()` are blocking. FastAPI runs sync endpoints in a thread pool; using `async def` would block the event loop instead. When we add streaming (`/ask/stream`), we'll switch to `async def` with `chain.astream()` — until then, sync is correct.

- **Same embedder factory at indexing and query time.** `embeddings.factory.make_embedder()` is used by both `indexing/run_indexing.py` and `api/rag.py`. **Do not** substitute a different LangChain wrapper at query time — even ones that "should" be equivalent differ in pooling / normalization, silently degrading retrieval. The cosine-parity test (`tests/test_embedder_parity.py`) exists precisely because this is hard to spot without measurement.

- **Direct `qdrant_client.query_points()`, not `langchain-qdrant`.** `api/rag.py` maps Qdrant `ScoredPoint.payload` to `langchain_core.documents.Document` manually in `_scored_point_to_hit()`. The bypass was necessary because `langchain-qdrant 0.2.x` changed metadata handling and stopped propagating flat payload fields into `Document.metadata`. If you swap retrieval back to `langchain-qdrant` later, verify chunk metadata survives the round-trip.

- **`temperature=0.0` everywhere, plus explicit sampling params.** `api/llm.py` sets `top_p=1.0`, `max_tokens=1024` (Ollama: `num_predict=1024`), and `frequency_penalty=0.3` for the vllm path. Two reasons: backend parity (Ollama and vllm-metal have different defaults — explicit params make benchmarks fair), and bounded cost (a runaway generation is capped at ~1024 tokens). The `frequency_penalty` is an anti-loop brake for Qwen 2.5 at temp=0 on vllm (Ollama's `repeat_penalty=1.1` is the equivalent on its side).

- **`load_dotenv()` at module level in `api/config.py`.** This happens *before* `Settings()` is instantiated, intentionally. Pydantic Settings can read `.env` on its own, but third-party libraries that read `os.environ` directly (e.g. `huggingface_hub` for `HF_TOKEN`) need the values pushed into the process environment first. Removing the `load_dotenv()` line breaks HF downloads on first run.

- **Observability is additive.** LangFuse and the custom Prometheus metrics in `api/metrics.py` were bolted on without touching the RAG pipeline logic — `api/tracing.py::get_langfuse_handler()` returns `None` when keys are unset, the pipeline doesn't know whether tracing is on. If you ever conditionalize core code on whether tracing is enabled, you've broken the invariant.

## Project Structure

```
docsrag/
├── api/              # FastAPI service
│   ├── main.py       # /health, /ask, /agent/ask endpoints + Prometheus instrumentation
│   ├── rag.py        # RAGPipeline: embed → retrieve → generate
│   ├── retriever.py  # HybridRetriever: BM25Index + RRF + CrossEncoder
│   ├── graph.py      # Agentic RAG graph via LangGraph
│   ├── llm.py        # LLM factory: ChatOllama or ChatOpenAI→vLLM
│   ├── translation.py # RU↔EN wrapper — routes Russian questions through translation
│   ├── metrics.py    # Prometheus custom metrics
│   ├── tracing.py    # LangFuse callback helper
│   ├── prompts.py    # System + user + translation prompts
│   ├── schemas.py    # Pydantic request/response models
│   └── config.py     # Pydantic Settings
├── embeddings/       # Embedder backends
│   ├── pytorch.py    # PytorchEmbedder (sentence-transformers, MPS/CUDA/CPU)
│   ├── onnx.py       # OnnxEmbedder (raw onnxruntime, sentence_embedding output)
│   └── factory.py    # make_embedder(backend) — picks by EMBEDDER_BACKEND
├── indexing/         # Indexing pipeline
│   ├── loader.py     # Markdown loader
│   ├── chunker.py    # Hierarchical chunker (header + recursive)
│   ├── qdrant_store.py
│   ├── run_indexing.py
│   └── smoke_test.py
├── evaluation/       # Evaluation framework
│   ├── golden_dataset.json  # 25 hand-verified Q&A pairs
│   └── run_eval.py          # Ragas + MLflow eval harness (honours embedder_backend in YAML)
├── configs/          # Experiment configs (YAML)
│   ├── baseline.yaml
│   ├── chunk_256.yaml
│   ├── chunk_1024.yaml      # dense baseline (frozen)
│   ├── hybrid.yaml          # dense + BM25 → RRF
│   ├── hybrid_rerank.yaml   # dense + BM25 → RRF + cross-encoder
│   ├── agentic.yaml         # LangGraph agentic RAG
│   ├── onnx_fp32.yaml       # chunk_1024 + EMBEDDER_BACKEND=onnx-fp32
│   ├── onnx_int8.yaml       # chunk_1024 + EMBEDDER_BACKEND=onnx-int8
│   ├── topk_3.yaml
│   └── topk_10.yaml
├── scripts/          # One-off operational scripts
│   ├── export_onnx.py        # bge → ONNX FP32 via optimum-cli
│   ├── quantize_onnx.py      # FP32 → INT8 (dynamic, per-channel)
│   └── export_torchscript.py # bge backbone → TorchScript .pt (bench-only artifact)
├── models/           # ONNX-exported models (gitignored, ~160 MB total)
├── observability/    # Prometheus, Grafana, LangFuse
├── benchmarks/
│   ├── bench_backends.py  # Ollama vs vllm-metal
│   └── bench_embedder.py  # PyTorch-MPS/CPU vs ONNX-CPU FP32/INT8
├── tests/
│   └── test_embedder_parity.py  # PyTorch ↔ ONNX FP32 cosine parity gate
├── docker-compose.yml
└── Makefile
```

## Current State

- **Qdrant collections:** `docsrag` (2540 chunks, shared by `pytorch` and `onnx-fp32` backends — parity-equivalent), `docsrag_int8` (2540 chunks, `onnx-int8` backend). Both at chunk\_size=1024, overlap=100.
- **Embeddings:** `BAAI/bge-small-en-v1.5` — 384-dim, English, normalized cosine similarity. Three swappable runtimes via `EMBEDDER_BACKEND` env var: `pytorch` (default), `onnx-fp32`, `onnx-int8`.
- **Retrieval strategy:** dense vector search (best by eval); hybrid and hybrid\_rerank available via config
- **Generation:** `temperature=0.0` for determinism; answers cite sources as `[file.md]`
- **Inference backend:** `INFERENCE_BACKEND=ollama` (default) or `vllm` — switchable via `.env`
- **Observability:** LangFuse tracing, Prometheus `/metrics`, Grafana dashboard at `:3000`
- **Languages:** English (native), Russian (via RU↔EN translation wrapper; English path untouched)

## Makefile Reference

```bash
make up            # Start Qdrant + API + MLflow (Ollama must be running natively)
make down          # Stop services
make build         # Build API Docker image
make health        # GET /health
make ask Q="..."   # POST /ask
make warmup        # Load LLM into Ollama RAM (run after make up)
make reindex                                      # Recreate the active collection (CHUNK_SIZE=1024 OVERLAP=100 defaults; honours EMBEDDER_BACKEND)
make reindex CHUNK_SIZE=512 CHUNK_OVERLAP=50      # Override chunk params
make reindex-onnx                                 # Recreate docsrag with EMBEDDER_BACKEND=onnx-fp32
make reindex-int8                                 # Recreate docsrag_int8 with EMBEDDER_BACKEND=onnx-int8
make smoke         # Retrieval sanity check
make eval          # Run evaluation (CONFIG=configs/baseline.yaml by default; pass CONFIG=configs/onnx_*.yaml for ONNX)
make mlflow-ui     # Open MLflow UI in browser
make prometheus-ui # Open Prometheus UI (http://localhost:9090)
make grafana-ui    # Open Grafana dashboard (http://localhost:3000, admin/admin)
# ONNX
make install-onnx     # uv pip install -e ".[onnx]" — adds optimum + onnxruntime
make export-onnx      # bge-small → models/bge-small-en-v1.5-onnx-fp32/ (FORCE=1 to re-export)
make quantize-onnx    # FP32 → INT8 dynamic per-channel → models/bge-small-en-v1.5-onnx-int8/
make bench-embedder   # Latency + throughput across PyTorch-MPS/CPU + ONNX-CPU FP32/INT8 (+ TorchScript-CPU if exported)
make export-torchscript # Trace bge backbone to TorchScript .pt — bench-only artifact
make lint          # ruff
make format        # ruff --fix + black
make type-check    # mypy
make test          # pytest
```
