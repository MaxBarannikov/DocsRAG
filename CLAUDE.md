# CLAUDE.md

This file gives Claude Code the context it needs to work productively in this repository. Read it before making changes.

---

## Project: DocsRAG

Self-hosted RAG system for Q&A over technical documentation. Pet project explicitly designed to demonstrate MLOps competencies (LangChain, Qdrant, Ragas, MLflow, LangGraph, vLLM, observability) for a CV.

**Documentation source:** FastAPI docs (153 markdown files) cloned into `data/raw/fastapi/docs/en/docs/`.

**Hardware:** MacBook M4 Max, 64 GB RAM. Apple Silicon — embeddings use MPS automatically.

**Language conventions:** the developer writes in Russian, code and technical terms in English. When responding in chat, prefer Russian for prose and English for code/technical terms. CLAUDE.md itself is in English so tooling stays language-agnostic.

---

## Setup mode: Variant A (current)

**Important architectural decision — do not silently change.**

- **Ollama runs natively on macOS** (installed via `brew install --cask ollama-app`, GUI app in the menu bar).
- **Qdrant runs in Docker** via `docker-compose.yml`.
- **API runs in Docker** and reaches Ollama via `host.docker.internal:11434`.
- The compose file contains `extra_hosts: host.docker.internal:host-gateway` for portability to Linux.

**Variant B (Ollama in Docker)** is intentionally kept as commented-out blocks in `docker-compose.yml` and noted in code comments. If we ever switch:
1. Uncomment the `ollama` service and `ollama_data` volume in `docker-compose.yml`.
2. Change `OLLAMA_BASE_URL` in the `api` service to `http://ollama:11434`.
3. Remove the `extra_hosts` block from `api`.
4. Add `ollama` to `api.depends_on`.
5. Pull the model: `docker exec -it docsrag-ollama ollama pull qwen2.5:7b-instruct-q4_K_M`.

When adding new code that touches Ollama, leave a `# Variant B: ...` comment explaining the change needed.

---

## Stack and key dependencies

- **LLM:** Qwen 2.5 7B Instruct (`qwen2.5:7b-instruct-q4_K_M`) via Ollama.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, English, normalized for cosine similarity). No task prefixes required — `encode()` is called without `prefix` at both index- and query-time.
- **Vector DB:** Qdrant **v1.17** (floating tag in `docker-compose.yml` — latest 1.17.x at pull time). Distance: cosine. Collection: `docsrag`.
- **Chunking:** hierarchical — `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter`. Current production config: `chunk_size=1024, overlap=100` (best by eval; default CLI values are 512/50).
- **Python 3.12**, `uv` package manager.
- **Quality:** `ruff-check` (lint) + `ruff-format` (format) + `mypy` (types) + `pytest`. All wired via `pre-commit` (`make format`).
- **Logging:** `loguru`.
- **Config:** Pydantic Settings + `python-dotenv`.

### Critical config detail

`api/config.py` calls `load_dotenv()` at module level **before** `Settings` is instantiated. This is intentional: it pushes `.env` into `os.environ` so third-party libraries (e.g. `huggingface_hub` reading `HF_TOKEN`) see the values. **Do not remove this `load_dotenv()` call**, even though Pydantic Settings can read `.env` on its own — it's there for the third-party libs.

---

## Repository layout

```
docsrag/
├── api/                       # FastAPI service (Tasks 3, 5)
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings + load_dotenv()
│   ├── main.py                # FastAPI app, /health, /ask, /agent/ask, lifespan
│   ├── rag.py                 # RAGPipeline class, get_pipeline() singleton
│   ├── retriever.py           # BM25Index, HybridRetriever, RRF, CrossEncoder (Task 5)
│   ├── graph.py               # AgentPipeline — LangGraph agentic RAG (Task 6)
│   ├── llm.py                 # make_llm() factory — ChatOllama or ChatOpenAI→vLLM (Task 8)
│   ├── metrics.py             # Prometheus custom metrics (Task 7)
│   ├── tracing.py             # LangFuse callback helper (Task 7)
│   ├── prompts.py             # System + user prompt templates (incl. RU↔EN translation prompts)
│   ├── translation.py         # RU↔EN routing — wraps the pipeline when the question is Russian
│   └── schemas.py             # Pydantic request/response models
├── embeddings/                # Embedder backends (Task 9)
│   ├── __init__.py            # re-exports PytorchEmbedder + make_embedder
│   ├── pytorch.py             # PytorchEmbedder (sentence-transformers)
│   ├── onnx.py                # OnnxEmbedder (FP32 or INT8 dir, raw ORT session)
│   └── factory.py             # make_embedder(backend) — picks by EMBEDDER_BACKEND
├── indexing/                  # Indexing pipeline (Task 2)
│   ├── __init__.py
│   ├── loader.py              # Markdown loader, RawDocument dataclass
│   ├── chunker.py             # Hierarchical chunker, Chunk dataclass
│   ├── qdrant_store.py        # QdrantStore class with recreate/upsert
│   ├── run_indexing.py        # CLI entry point (uses make_embedder)
│   ├── smoke_test.py          # CLI for retrieval sanity-checks
│   └── fetch_docs.sh          # Downloads FastAPI docs
├── evaluation/                # Tasks 4, 5
│   ├── golden_dataset.json    # 25 hand-verified Q&A pairs
│   └── run_eval.py            # Ragas + MLflow eval harness (honours embedder_backend in YAML)
├── observability/             # Task 7 — Prometheus, Grafana, dashboard
├── scripts/                   # One-off operational scripts (Task 9)
│   ├── export_onnx.py         # bge → ONNX FP32 via optimum-cli
│   ├── quantize_onnx.py       # ONNX FP32 → INT8 (dynamic, per-channel)
│   └── export_torchscript.py  # bge backbone → TorchScript .pt (Task 9 step 11, bench-only)
├── benchmarks/                # Task 8 + Task 9 benchmarks
│   ├── bench_backends.py      # Ollama vs vllm-metal (Task 8)
│   └── bench_embedder.py      # PyTorch-MPS/CPU vs ONNX-CPU FP32/INT8 (Task 9)
├── tests/
│   └── test_embedder_parity.py # PyTorch ↔ ONNX FP32 cosine parity (Task 9 step 5)
├── configs/                   # Experiment configs (YAML)
│   ├── baseline.yaml          # chunk_size=512, overlap=50, top_k=5, dense
│   ├── chunk_256.yaml
│   ├── chunk_1024.yaml        # ← frozen baseline (chunk_size=1024, overlap=100, top_k=5)
│   ├── hybrid.yaml            # dense + BM25 → RRF
│   ├── hybrid_rerank.yaml     # dense + BM25 → RRF + cross-encoder
│   ├── agentic.yaml           # LangGraph agentic RAG (Task 6)
│   ├── onnx_fp32.yaml         # chunk_1024 + EMBEDDER_BACKEND=onnx-fp32 (Task 9)
│   ├── onnx_int8.yaml         # chunk_1024 + EMBEDDER_BACKEND=onnx-int8 (Task 9, reads docsrag_int8)
│   ├── topk_3.yaml
│   └── topk_10.yaml
├── models/                    # ONNX + TorchScript artifacts (gitignored, ≈289 MB total)
│   ├── bge-small-en-v1.5-onnx-fp32/   # 128 MB — produced by scripts/export_onnx.py
│   ├── bge-small-en-v1.5-onnx-int8/   # 33 MB — produced by scripts/quantize_onnx.py
│   └── bge-small-en-v1.5.pt           # 128 MB — produced by scripts/export_torchscript.py (bench-only)
├── data/
│   ├── raw/fastapi/docs/en/docs/   # 153 markdown files
│   └── bm25_index.pkl              # BM25 index (gitignored, rebuilt on demand)
├── docker-compose.yml         # Qdrant + API + Prometheus + Grafana + MLflow. api/, indexing/, embeddings/, models/ bind-mounted read-only into the API container.
├── Dockerfile                 # Multi-stage uv build for the API
├── pyproject.toml
├── uv.lock                    # Committed — required for Docker build
├── Makefile
├── .env.example
├── .env                       # Real HF_TOKEN, gitignored
├── .github/workflows/ci.yml   # Lint + format + type-check (no tests yet)
├── README.md
└── CLAUDE.md                  # This file
```

---

## Project plan and current status

The project is structured as 8 sequential tasks. Each task ends with a working artifact and a commit. Don't start a new task until the current one is committed and CI is green.

- [x] **Task 1** — Infrastructure and repository setup
- [x] **Task 2** — Indexing pipeline (smoke tests passing; current collection: 2540 chunks at chunk_size=1024)
- [x] **Task 3** — Basic RAG API (FastAPI + LangChain + Ollama) — verified, committed
- [x] **Task 4** — Evaluation framework (Ragas + MLflow) — 5 configs swept, baseline frozen
- [x] **Task 5** — Hybrid search (BM25 + dense) + cross-encoder reranker — done, dense remains best strategy
- [x] **Task 6** — Agentic RAG via LangGraph — done; precision↑ (0.653) recall↓ (0.450) vs dense baseline
- [x] **Task 7** — Observability — LangFuse tracing + Prometheus metrics + Grafana dashboard
- [x] **Task 8** — vLLM backend (vllm-metal/MLX on Apple Silicon) + benchmark — vllm-metal 3.8× faster than Ollama
- [x] **Task 9** — ONNX optimization of embedder — done; ONNX-CPU-FP32 is 3.4× faster than PyTorch-MPS on `/ask` with byte-identical Ragas metrics; INT8 documented as unusable on this model (context_recall −0.070 vs budget −0.05)

Tasks 5–7 can be reordered if needed. Task 8 was originally the finale; Task 9 was added later to round out the MLOps story (classical ML serving formats — ONNX/TorchScript — alongside the LLM serving from Task 8). See the `## Task 9 results` section below.

---

## Current state of data and services

### Evaluation baseline (frozen — Tasks 5+ compare against this)

Best config: **chunk_size=1024, overlap=100, top_k=5**

| Config | chunk\_size | overlap | top\_k | faithfulness | answer\_relevancy | context\_precision | context\_recall |
|---|---|---|---|---|---|---|---|
| chunk\_256 | 256 | 25 | 5 | 0.646 | 0.767 | 0.417 | 0.353 |
| baseline | 512 | 50 | 5 | 0.757 | 0.849 | 0.506 | 0.431 |
| topk\_3 | 512 | 50 | 3 | 0.719 | 0.775 | 0.517 | 0.403 |
| topk\_10 | 512 | 50 | 10 | 0.818 | **0.892** | 0.526 | 0.517 |
| **chunk\_1024** ✓ | **1024** | **100** | **5** | **0.882** | 0.886 | **0.598** | **0.557** |

All subsequent tasks will run `make eval CONFIG=configs/chunk_1024.yaml` (after reindexing) to compare against these numbers.

### Task 5 results — hybrid search (chunk\_size=1024, top\_k=5)

| Strategy | faithfulness | answer\_relevancy | context\_precision | context\_recall |
|---|---|---|---|---|
| **dense** ✓ | **0.882** | 0.886 | **0.598** | **0.557** |
| hybrid (dense + BM25 → RRF) | 0.789 | 0.818 | 0.556 | 0.523 |
| hybrid\_rerank (+ cross-encoder) | 0.825 | **0.890** | 0.566 | 0.510 |

Dense wins. BM25 adds noise on this semantically rich corpus; cross-encoder partially compensates but not enough to beat dense. **Active strategy: dense.**

### Qdrant collection `docsrag`

- **Points:** 2540 chunks (chunk_size=1024, overlap=100)
- **Vector dim:** 384
- **Distance:** cosine
- **Payload keys** (verified by scrolling a point):
  - `text` — chunk content
  - `source_path` — relative MD path, e.g. `tutorial/path-params.md`
  - `header_path` — markdown header trail, e.g. `Path Parameters > Path parameters with types`
  - `chunk_index` — running chunk number

**Important:** retrieval in `api/rag.py` uses `qdrant_client.query_points()` directly (not `langchain-qdrant`). The payload is mapped to `Document` manually in `_scored_point_to_hit()`: `text` → `page_content`, `source_path`/`header_path`/`chunk_index` → `metadata`. This bypass was necessary because `langchain-qdrant 0.2.x` changed metadata handling and stopped propagating flat payload fields into `Document.metadata`.

Hybrid retrieval is implemented in `api/retriever.py`: `BM25Index` (scrolls all Qdrant points, serialized to `data/bm25_index.pkl`) + `HybridRetriever` (dense + BM25 → RRF → optional cross-encoder). Active strategy is controlled by `retrieval_strategy` in the YAML config. Delete `data/bm25_index.pkl` after reindexing to force a BM25 rebuild.

### Smoke-test queries that should work

- "how to define a path parameter in FastAPI" → top-1 score ~0.85, sources from `tutorial/path-params.md`, `tutorial/body.md`
- "how to handle file uploads" → top-1 ~0.79, mostly `tutorial/request-files.md`
- "dependency injection examples" → all top-3 from `tutorial/dependencies/`

Known retrieval noise: `_llm-test.md` (FastAPI repo internal helper file) sometimes appears in top-k. Not critical, can be filtered later if it hurts metrics.

---

## Commands (Makefile-driven)

```bash
# Setup
make install              # uv venv + editable install with [dev] extras

# Lifecycle
make up                   # Bring up Qdrant + API; check Ollama on host
make down
make build                # Build the API image
make rebuild              # --no-cache rebuild + restart API (only after pyproject/Dockerfile changes)
make restart              # Recreate API container — picks up code edits (bind mount) and .env/compose changes
make logs                 # Tail all services
make api-logs             # Tail API only
make api-shell            # Bash inside the API container
make ollama-status        # Check Ollama on host

# RAG API
make health               # GET /health (pretty-printed)
make ask Q="..."          # POST /ask with a question (defaults to a path-params example)
make warmup               # Send one request to load the LLM into Ollama RAM

# Indexing
make fetch-docs           # Download FastAPI docs
make index                # Run indexing (incremental)
make reindex              # Recreate the collection — uses EMBEDDER_BACKEND from env (defaults to pytorch / docsrag)
make reindex-onnx         # Recreate docsrag with EMBEDDER_BACKEND=onnx-fp32 (Task 9)
make reindex-int8         # Recreate docsrag_int8 with EMBEDDER_BACKEND=onnx-int8 (Task 9)
make smoke                # Retrieval sanity-check

# ONNX (Task 9, opt-in via `make install-onnx`)
make install-onnx         # uv pip install -e ".[onnx]" — adds optimum + onnxruntime
make export-onnx          # bge-small → models/bge-small-en-v1.5-onnx-fp32/  (idempotent; FORCE=1 to re-export)
make quantize-onnx        # FP32 → INT8 dynamic per-channel → models/bge-small-en-v1.5-onnx-int8/
make bench-embedder       # Latency + throughput across PyTorch-MPS/CPU + ONNX-CPU FP32/INT8

# Evaluation
make eval                 # Run Ragas eval (CONFIG=configs/baseline.yaml by default)
make eval CONFIG=configs/onnx_fp32.yaml   # ONNX FP32 backend, reads docsrag
make eval CONFIG=configs/onnx_int8.yaml   # ONNX INT8 backend, reads docsrag_int8
make mlflow-ui            # Open MLflow UI in browser (http://localhost:5000)

# Quality
make format               # pre-commit run -a (ruff-format + ruff-check --fix + mypy)
make test                 # pytest -v
make clean                # Remove local caches
```

**`make ask` quoting:** the Makefile JSON-escapes `Q`, so apostrophes and double quotes in the question are safe. Prefer single quotes in shell: `make ask Q='What is "Depends"?'`.

**Makefile `.env` loading:** the top of the `Makefile` does `include .env` + `export` so Make targets see the same variables as docker-compose and the Python app. Without this, `VLLM_MODEL=...` set in `.env` would be ignored by `make vllm-start` (it would fall back to the `?=` default in the Makefile). Side effect: `vllm serve` warns "Unknown vLLM environment variable detected: VLLM_MODEL" — harmless, it's our project var, not theirs.

### Dev loop: `make restart` vs `make rebuild`

The API container bind-mounts `./api`, `./indexing`, `./embeddings`, and `./models` as read-only volumes (see `docker-compose.yml`), so source edits + freshly-exported ONNX models are visible inside the container without rebuilding the image.

- **`make restart`** — default dev loop. `docker compose up -d --force-recreate api`. Picks up any code edit in `api/` or `indexing/` AND any change to `.env` / `docker-compose.yml`. ~5–10 s.
- **`make rebuild`** — only when `pyproject.toml`, `uv.lock`, or `Dockerfile` change. Full no-cache image rebuild + restart. Minutes.

We deliberately don't expose a plain `docker compose restart api` target — it would save ~3 s on pure code edits but doesn't re-read env vars, which makes the "did my `.env` change apply?" question depend on which restart you used. One target, one mental model.

Bind-mount caveat: if you remove `./api:/app/api:ro` from compose to test the image as-built (e.g. for CI parity), code edits will require `make rebuild` again.

---

## Architectural conventions

### Pipeline singleton

`api/rag.py` exports `get_pipeline()` decorated with `@lru_cache(maxsize=1)`. This is the single application-wide `RAGPipeline` instance. The embedding model (~130 MB) and Qdrant client are heavy — never construct `RAGPipeline()` directly inside a request handler. Inject via `Annotated[RAGPipeline, Depends(get_pipeline)]`.

The FastAPI `lifespan` handler in `api/main.py` calls `get_pipeline()` once at startup so the first `/ask` doesn't pay initialization cost.

### Sync vs async endpoints

Endpoints are intentionally `def`, **not** `async def`. Both `qdrant_client.query_points()` and `chain.invoke()` are blocking. FastAPI runs sync endpoints in a thread pool; using `async def` would block the event loop. When we move to streaming in a later task, we'll switch to `async def` with `chain.astream()`.

### Embedder reuse between indexing and querying

Embedders live in the top-level `embeddings/` package (Task 9). `api/rag.py` and `indexing/run_indexing.py` both go through `embeddings.factory.make_embedder()`, which picks one of three backends by `EMBEDDER_BACKEND` env var:
- `pytorch` — `PytorchEmbedder` wrapping `sentence-transformers` (MPS/CUDA/CPU)
- `onnx-fp32` — `OnnxEmbedder` reading the graph-baked `sentence_embedding` output (parity-equivalent to PyTorch, cosine 1.0 on the parity test)
- `onnx-int8` — same `OnnxEmbedder` pointed at the INT8 model dir; reads from a separate `docsrag_int8` collection via `settings.active_qdrant_collection`

**Do not** swap in `HuggingFaceEmbeddings` or any other LangChain wrapper — this would risk subtle differences in pooling/normalization between index- and query-time, silently degrading retrieval.

Both `PytorchEmbedder` and `OnnxEmbedder` produce L2-normalized vectors (PyTorch via `normalize_embeddings=True`; ONNX via the graph-baked normalization step) — required for cosine similarity to behave as expected.

### Prompts live in their own module

`api/prompts.py` keeps the system + user prompt templates. Don't inline prompts in `rag.py`. Reasons:
1. Prompt iteration without touching pipeline code.
2. Eval harness logs `prompt_version` as an MLflow param — this makes it easy to A/B test prompt variants across runs.

The system prompt currently mandates:
- Answer only from CONTEXT.
- If insufficient: reply exactly `I don't know based on the provided documentation.`
- Cite sources inline as `[file.md]`.
- Use code blocks for code.

### `temperature=0.0`

The LLM is called with `temperature=0.0` for determinism — required for reproducible Ragas evaluation (same question must produce the same answer across runs). If we want to study creativity vs faithfulness later, expose temperature as a param, don't hardcode a different default.

### Explicit sampling: `top_p=1.0`, `max_tokens=1024`, `frequency_penalty=0.3` (vllm)

`api/llm.py` sets `top_p=1.0` and `max_tokens=1024` (Ollama: `num_predict=1024`) for **both** backends, plus `frequency_penalty=0.3` for the vllm path. Reasons:
- **Backend parity.** Ollama and vllm-metal have different defaults. Without an explicit cap, vllm-metal can truncate long answers while Ollama runs unbounded (`num_predict=-1`), which makes side-by-side comparisons unfair.
- **Predictable cost.** A bug that triggers a runaway generation is bounded to ~1024 tokens.
- **1024 fits.** Longest production answers we've seen are ~600 tokens. Bump `MAX_TOKENS` in `api/llm.py` if real answers get clipped.
- **Anti-loop on vllm.** Qwen 2.5 at `temperature=0` can fall into degenerate citation/phrase loops. `frequency_penalty=0.3` is a mild brake. Ollama applies `repeat_penalty=1.1` by default — same intent, different name.

**History note for `frequency_penalty`:** vllm-metal **0.1.0** silently ignored OpenAI penalty params (the MLX sampler didn't implement them — verified with `curl frequency_penalty=2.0` returning 200 identical tokens). **0.2.0 honors them** (same curl test shows token mutation). The value was kept across the 0.1.0→0.2.0 transition; on 0.1.0 it was dead code, on 0.2.0 it's live again. Production CUDA vllm has always honored it.

### Logging contract

Every `/ask` logs:
- `retrieval_ms`, `generation_ms`, `total_ms`
- Number of hits
- First 80 chars of the question

These same numbers go into the response so we can later instrument them as Prometheus metrics in Task 7 with no extra plumbing.

### Source attribution

`AskResponse.sources` always returns `source_path`, `header_path`, `score`. Raw chunk text in `sources[].content` is **only** populated when the request includes `include_contexts: true` (default `false`). This was a deliberate choice — debuggability without bloating production responses.

### Cross-language support (RU questions)

The corpus and embeddings are English-only (`BAAI/bge-small-en-v1.5`), so Russian queries embed into a useless vector space. Instead of swapping to a multilingual embedder (full reindex, larger model), we wrap the pipeline with translation:

1. `api/translation.py::contains_cyrillic()` — regex-based language detection. Anything with `[А-Яа-яЁё]` is routed through translation.
2. `api/translation.py::translate_to_english()` — RU question → EN via the same `make_llm()` (Qwen 2.5 is multilingual).
3. RAG runs on the English question against the English index.
4. `api/translation.py::translate_to_russian()` — EN answer → RU before returning to the user.

English questions short-circuit the wrappers entirely (`is_russian = False`) — zero overhead on the production path.

**Files involved:**
- `api/translation.py` — language detection + two helper functions.
- `api/prompts.py` — `TRANSLATE_RU_TO_EN_PROMPT` and `TRANSLATE_EN_TO_RU_PROMPT`. The EN→RU prompt explicitly forbids translating code blocks, file-path citations (`[file.md]`), and well-known English technical terms (path operation, dependency injection, etc.).
- `api/rag.py::RAGPipeline.ask()` and `api/graph.py::AgentPipeline.ask()` — the wrapping logic.
- `api/schemas.py` — `AskResponse.translation_ms` / `AgentAskResponse.translation_ms` reports the combined latency; `0` for English questions.

**Cost:** +2 LLM calls per Russian query → ≈+1.5 s on Ollama 7B, ≈+8 s on vllm-metal 7B, ≈+13 s on vllm-metal 14B. Model-dependent on vllm: 14B-4bit is slower but produces clean Cyrillic; 7B-4bit garbles it.

**Translation debug logs (INFO level):**
```
RU→EN | in='Главные преимущества FastAPI?' | out='What are the main advantages of FastAPI?'
EN→RU | in='The main advantages of FastAPI are...' | out='Основные преимущества FastAPI...'
```
Both `api/rag.py::ask()` and `api/graph.py::AgentPipeline.ask()` emit these. Use `make api-logs` to inspect translator quality directly when debugging cross-language regressions.

**Known weaknesses:**
- Translation quality bottlenecks retrieval quality. A bad RU→EN translation will retrieve the wrong chunks.
- Model choice matters more than quantization. **Qwen2.5-7B-4bit (MLX)** garbles Cyrillic in the EN→RU step (latin-with-acute characters mid-word like `разdéлвние`). **Qwen2.5-14B-4bit (MLX)** does not. Ollama `qwen2.5:7b-instruct-q4_K_M` (llama.cpp, GGUF) handles it cleanly too. The difference is MLX 4bit specifically — likely a tokenizer/vocab quantization artefact.
- Citations like `[tutorial/path-params.md]` survive EN→RU only because the prompt explicitly forbids touching them. If you swap the translator model, retest.
- The fixed canonical response `"I don't know based on the provided documentation."` also gets translated, so Ragas exact-string checks on Russian golden datasets won't match. The current `golden_dataset.json` is English-only, so this doesn't bite eval yet.

**Why not multilingual embeddings (e.g. `BAAI/bge-m3`)?** Considered. Requires a full reindex on a ~2 GB model, and would degrade English-on-English retrieval quality (which is what the eval baseline is built on). Cheap translation is reversible and didn't touch the index; multilingual embeddings would be a one-way decision.

---

## Known pitfalls (read before debugging)

1. **`/ask` returns empty/garbage answers** — check `_scored_point_to_hit()` in `api/rag.py`: it maps `payload["text"]` to `page_content`. If the key name changed in the index, `page_content` will be empty. Confirm with: `logger.debug("first chunk len={}", len(hits[0].document.page_content))` in `retrieve()`.

2. **First `/ask` is slow (10–20s), subsequent ones fast** — Ollama lazy-loads the model into RAM on first request. Use `make warmup` after `make up`. Not a bug.

3. **API container starts but `/ask` fails with `ConnectionError`** — the Ollama menu-bar app isn't running on the host. From inside the container: `curl -s http://host.docker.internal:11434/api/tags` should return JSON. If it doesn't, start the Ollama app or run `ollama serve` natively.

4. **`docker compose build` fails on `torch`** — Docker Desktop memory limit. Bump it to 8+ GB in Settings → Resources.

5. **Container `import torch` fails with `libgomp.so.1` not found** — `libgomp1` install must be present in the runtime stage of the Dockerfile. It is. Don't remove it when refactoring.

6. **Embedding model redownloads on every `docker compose up --build`** — the `hf_cache` volume in `docker-compose.yml` mounts `/app/.cache/huggingface`. If you remove that volume, the ~470 MB download repeats.

7. **Qdrant healthcheck looks weird (raw `/dev/tcp` bash)** — yes, intentional. The Qdrant slim image has neither `curl` nor `wget`. The healthcheck uses `CMD` (not `CMD-SHELL`) with an explicit `bash -c` to ensure `/dev/tcp` works — it's a bash-only feature and `CMD-SHELL` runs `sh` by default. Don't "fix" it.

8. **`uv.lock` missing** — required for the Docker build (`uv sync --frozen`). If it's not in the repo, run `uv lock` before `make build`.

9. **`_llm-test.md` chunks pollute retrieval** — known noise from upstream FastAPI repo. Tolerable now; consider a payload-filter at retrieval time if it hurts Task 4 metrics.

10. **Embedder logs progress bars on every query** — `retrieve()` calls `self._embedder.encode([query], show_progress=False)`. If you see tqdm spam in API logs, the `show_progress=False` flag was lost.

11. **Ragas hangs or times out** — Ragas defaults to parallel workers that overwhelm single-threaded Ollama. The harness sets `RunConfig(timeout=180, max_retries=3, max_workers=1)`. If you see repeated `TimeoutError` in Ragas eval, verify this `RunConfig` is still in place.

12. **`ImportError` for `langchain_ollama` when running eval** — SOCKS proxy env vars (`ALL_PROXY`, `HTTPS_PROXY`, etc.) break httpx inside Ragas. `evaluation/run_eval.py` unsets them at module level before any imports. Don't move or remove that block.

13. **Ragas `OutputParserException` from Qwen** — Qwen at `temperature=0` deterministically returns plain text instead of JSON for certain samples when Ragas expects a structured evaluation response. Fixed by setting `format="json"` in `_SanitizedChatOllama` (the Ragas-only LLM wrapper in `evaluation/run_eval.py`) — Ollama constrains token sampling to valid JSON at the model level. Do not use `format="json"` for the main RAG pipeline LLM (`ChatOllama` in `api/rag.py`) — it would corrupt natural-language answers.

14. **`make vllm-start` fails with `vllm: command not found`** — you're either not in the project venv (`source .venv/bin/activate`) or upstream `vllm` isn't installed. The 0.2.0 plugin requires upstream `vllm` to be present in the same venv — see "Running vllm-metal locally" above.

15. **`uv pip sync uv.lock` silently removes vllm + vllm-metal** — they're not in `pyproject.toml` (vllm's pyproject hard-pins CUDA deps with no macOS wheels — confirmed experimentally on `try-uv`), so the lockfile-driven sync removes them as "extra" packages. Recovery: `make install-vllm`.

16. **Russian answers have garbled Cyrillic on vllm-metal** (like `разdéлвние` instead of `разработки`) — the 4bit MLX quantization of **Qwen2.5-7B** produces this. Switch `VLLM_MODEL` in `.env` to `mlx-community/Qwen2.5-14B-Instruct-4bit` and restart vllm. Ollama and 14B-4bit are both immune.

17. **vllm-metal 0.2.0 server warns `Unknown vLLM environment variable detected: VLLM_MODEL`** — harmless. The variable belongs to our project (consumed via `$(VLLM_MODEL)` in the Makefile), not vllm itself. vllm scans env for its own `VLLM_*` vars and complains about strangers.

---

## Style and review preferences

- **Concise, direct prose.** Russian for chat replies, English for code/comments/docs.
- **Explain "why", not just "how".** Especially for non-obvious decisions (`content_payload_key`, sync `def`, embedder reuse, etc.).
- **Flag pitfalls proactively.** If a change has a likely failure mode, point it out before the developer hits it.
- **Type hints everywhere.** mypy must pass.
- **Prefer paraphrasing over copying** when summarizing other docs.
- **One commit per task.** Don't bundle Task 4 work into the Task 3 commit.
- **Update README + this CLAUDE.md** as the project state changes — they should never go stale.
- **Do not run linters or formatters unless the user explicitly asks.** No `ruff format`, `ruff check`, `mypy`, `make format`, `pre-commit run`, or `make test` on your own initiative — even after writing or editing code. The developer runs these themselves (locally or in CI). If a change you made would obviously fail a check (e.g. unused import you just added), fix it inline; otherwise leave verification to the developer.

---

## Task 6 results — Agentic RAG via LangGraph

**Implementation:** `api/graph.py` — LangGraph StateGraph with 4 nodes: `query_rewriter → retriever → relevance_grader → generator`. Conditional retry if fewer than 2 chunks pass grading (max 1 retry). Exposed as `/agent/ask` endpoint in `api/main.py`. Config: `configs/agentic.yaml`.

### Eval results (chunk_size=1024, top_k=5)

| Strategy | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| dense (baseline) | **0.882** | **0.886** | 0.598 | **0.557** |
| agentic | 0.817 | 0.813 | **0.653** | 0.450 |

**Finding:** binary relevance grading shifts the precision/recall tradeoff — `context_precision` improves (+0.055) but `context_recall` drops sharply (−0.107). The grader discards borderline-relevant chunks that contained actual answers. Dense remains the better end-to-end strategy. The `/agent/ask` endpoint is available but `/ask` (dense) is the production default.

---

## Task 7 results — Observability

**Implementation:** additive instrumentation, no pipeline changes.

- **`api/metrics.py`** — 5 Prometheus metrics: `rag_requests_total` (counter by endpoint), `rag_retrieval_duration_seconds`, `rag_generation_duration_seconds` (histograms by endpoint), `rag_top_k`, `rag_answer_length_chars`.
- **`api/main.py`** — `prometheus-fastapi-instrumentator` exposes `/metrics` with HTTP latency + request counts; custom metrics recorded after each `/ask` and `/agent/ask`.
- **`api/tracing.py`** — `get_langfuse_handler()`: returns `CallbackHandler` if `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set, `None` otherwise. Fully optional — pipeline works unchanged without keys.
- **`api/rag.py`** + **`api/graph.py`** — callbacks threaded through `generate()` and all LLM invoke calls; all LLM steps in one request group under one LangFuse trace.
- **`docker-compose.yml`** — `prometheus` (port 9090) scrapes `/metrics` every 15s; `grafana` (port 3000) auto-provisions Prometheus datasource + DocsRAG dashboard (8 panels).
- **`observability/`** — prometheus config, grafana provisioning, pre-built dashboard JSON.

**LangFuse:** cloud (free tier). Keys in `.env`: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.

**Grafana:** `http://localhost:3000`, login `admin`/`admin`.

---

## Task 8 results — vLLM backend + benchmark

**Implementation:** switchable inference backend via `INFERENCE_BACKEND=ollama|vllm` env var.

- **`api/llm.py`** — `make_llm()` factory: returns `ChatOllama` or `ChatOpenAI(base_url=vllm_base_url)` depending on `INFERENCE_BACKEND`. `json_mode=True` maps to `format="json"` for Ollama and `response_format={"type":"json_object"}` for vLLM.
- **`api/rag.py`**, **`api/graph.py`** — replaced direct `ChatOllama` instantiation with `make_llm()`.
- **`benchmarks/bench_backends.py`** — runs 5 warm questions through each backend, prints latency table.
- **vllm-metal** — MLX-based plugin for upstream vLLM on Apple Silicon, OpenAI-compatible API. **Currently on 0.2.0** (upgraded from 0.1.0 — see "Running vllm-metal locally" below).

### Benchmark results (M4 Max, warm, 5 questions, top_k=3, vllm-metal 0.1.0 + 7B-4bit)

| Backend | avg gen | p50 gen | min | max |
|---|---|---|---|---|
| Ollama (Qwen2.5-7B q4\_K\_M, llama.cpp) | 3375ms | 3439ms | 2064ms | 4327ms |
| **vllm-metal (Qwen2.5-7B 4bit, MLX)** | **891ms** | **911ms** | **705ms** | **1075ms** |

**Finding:** vllm-metal is **3.8× faster** on generation. MLX leverages Apple Silicon unified memory more efficiently than llama.cpp. On production CUDA hardware the same API code works without changes — swap `vllm-metal` for `vllm/vllm-openai` Docker image and set `VLLM_BASE_URL`.

**Quality (Ragas, 25 samples):** context_precision/recall identical (same retrieval). faithfulness −0.055, answer_relevancy +0.021 vs Ollama — minor quantization format difference, not meaningful at this sample size.

### 7B vs 14B on vllm-metal 0.2.0 (after upgrade)

The 0.1.0→0.2.0 upgrade and the addition of Russian Q&A support changed the picture. Observed on the same RU question ("Главные преимущества FastAPI?"):

| Model on vllm-metal 0.2.0 | retrieval | generation | translation | total | RU quality |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-4bit | 550ms | 3938ms | 7724ms | 12.2s | ❌ corrupted Cyrillic ("разdéлвние" instead of "разработки") |
| **Qwen2.5-14B-Instruct-4bit** (current default) | 475ms | 6264ms | 12622ms | **19.4s** | ✅ clean Russian, proper citations preserved |

7B-4bit MLX produces garbled Cyrillic in EN→RU translation (latin-with-acute artefacts mid-word). 14B-4bit handles multilingual generation cleanly. **For vllm-backed Russian queries we default to 14B-4bit**, accepting ~2× slower generation in exchange for usable output. English-only queries are unaffected — 7B is fine if you don't need RU.

Set the model via `VLLM_MODEL` in `.env`; both default and override paths read from there (Makefile auto-loads `.env` since the change in that file).

### Running vllm-metal locally

**Important — install procedure changed in `vllm-metal` 0.2.0.** It's no longer a standalone server; it's a plugin for upstream `vllm`. Both packages must be installed into the project venv via `make install-vllm`. Neither is in `pyproject.toml`/`uv.lock`.

**Why not pyproject?** Tried it experimentally on branch `try-uv`: added vllm + vllm-metal under `[project.optional-dependencies]` with macOS markers and `[tool.uv.sources]` URLs, ran `CXXFLAGS="-Wno-parentheses" UV_INDEX_STRATEGY=unsafe-best-match uv lock` — lock succeeded (3.9 s) but pulled in 84 CUDA packages. `uv sync --dry-run --extra vllm-mac` failed on `nvidia-cudnn-frontend==1.18.0`: no macOS wheel, no sdist. Behind it queued `cuda-python`, `flashinfer-python`, `nvidia-cutlass-dsl`, `torchvision`, `torchaudio` with the same diagnosis. Root cause: vllm's own pyproject hard-pins CUDA-only deps without environment markers, and the manual two-phase workaround (CPU requirements → main build) can't be encoded in uv's universal resolver. Don't re-attempt without solving both.

Install:

```bash
source .venv/bin/activate
make install-vllm   # ~5-15 min: downloads vllm 0.20.1 tarball, builds from source, installs metal plugin
vllm --version      # must print "Platform plugin metal is activated"
```

Then set `INFERENCE_BACKEND=vllm` in `.env`, run `make vllm-start` + `make restart`.

Versions are pinned via `VLLM_VERSION` / `VLLM_METAL_WHEEL` variables at the top of the `Makefile` — bump them explicitly when upgrading.

**Gotchas:**
- Installing vllm pulls torch / transformers / kernels and bumps versions of packages also used by the RAG pipeline. `make health` after install to confirm the API still starts.
- `uv pip sync uv.lock` will remove vllm + the plugin. Recovery: `make install-vllm`.
- The `vllm-start` target uses `vllm serve $(VLLM_MODEL) --host 127.0.0.1 --port $(VLLM_PORT)`. The old `vllm-metal --model ...` syntax is dead.

---

## Task 9 results — ONNX optimization of embedder

**Status:** complete. All 12 steps done. Verdict: ONNX-CPU-FP32 is a strict improvement over PyTorch-MPS for `/ask` (3.4× lower latency, byte-identical retrieval); INT8 fails the Ragas budget on this small model and is retained as a benchmark artifact.

**Goal:** convert `BAAI/bge-small-en-v1.5` to ONNX Runtime (FP32 + INT8), expose a swappable embedder backend (mirror of Task 8's `make_llm()`), demonstrate latency/quality tradeoff. Reranker, cross-encoder, LLM itself — out of scope.

### Architectural decisions (sealed)

1. **Two Qdrant collections, not three.** `docsrag` for pytorch + onnx-fp32 (parity-equivalent); `docsrag_int8` for INT8 (vectors differ). `settings.active_qdrant_collection` routes by `EMBEDDER_BACKEND`. Cached MLflow numbers serve as the PyTorch reference.
2. **Top-level `embeddings/` package** owns embedders. Repays the old `api → indexing` import debt. Factory `make_embedder()` reads `settings.embedder_backend`; `OnnxEmbedder` is lazy-imported so PyTorch-only callers don't need the `[onnx]` extra.
3. **TorchScript stays minimal.** One export script + one row in the bench table (step 11). No factory integration. TorchScript in 2026 is legacy.

4. **ONNX export reads `sentence_embedding` (graph-baked pooling).** Empirically `optimum-cli export onnx --task feature-extraction` on sentence-transformers produces both `token_embeddings` (raw per-token) and `sentence_embedding` (already pooled + L2-normalized — byte-identical to PyTorch since the same code was traced). We use the second: less code, zero pooling-bug risk, parity test came back at cosine 1.000000 across 120 chunks.
5. **Dynamic INT8 with `per_channel=True`.** A/B during step 6: per-tensor gave mean cosine 0.9759 vs FP32 (75% under 0.98 — Ragas-risky); per-channel gave 0.9973 (only 1.7% under 0.99) at +0.2 MB cost. `--per-tensor` flag preserves the alt variant.
6. **`[onnx]` is opt-in.** `pyproject.toml` extra, Docker image stays lean. Local dev gets ONNX backends; production API container remains on PyTorch.

### Implementation status

| # | Step | Status |
|---|---|---|
| 1 | `[onnx]` extra + `make install-onnx` | done |
| 2 | Migrate `EmbeddingModel` → `embeddings/pytorch.py` (renamed `PytorchEmbedder`) | done |
| 3 | `scripts/export_onnx.py` → `models/bge-small-en-v1.5-onnx-fp32/` (128 MB dir) | done |
| 4 | `embeddings/onnx.py::OnnxEmbedder` reading `sentence_embedding` | done |
| 5 | Parity test `tests/test_embedder_parity.py` | green: cosine 1.000000 over 120 chunks |
| 6 | `scripts/quantize_onnx.py` → INT8 (33 MB dir, per-channel) | done |
| 7 | `embeddings/factory.py::make_embedder()` + `settings.active_qdrant_collection` + `embedder_backend` in `/health` | done |
| 8 | Reindex `docsrag` (FP32) + `docsrag_int8` (both 2540 points at chunk_size=1024) | done |
| 9 | `benchmarks/bench_embedder.py` — 4 backends, p50/p95/p99 + throughput | done (results below) |
| 10 | Ragas eval on `configs/onnx_fp32.yaml` + `configs/onnx_int8.yaml` | done — FP32 PASS, INT8 FAIL on context_recall (see below) |
| 11 | TorchScript bonus — `scripts/export_torchscript.py` + TorchScript-CPU row in bench | done |
| 12 | Final docs: this section is now `## Task 9 results`; README Task 9 section + reproducibility steps in place | done |

### Bench-9 — single-query latency (M4 Max, bge-small, 50 runs, 10 warmup)

| Backend | p50 | p95 | p99 |
|---|---|---|---|
| PyTorch-MPS | 5.7 ms | 9.0 ms | 9.2 ms |
| PyTorch-CPU | 6.7 ms | 7.1 ms | 7.4 ms |
| TorchScript-CPU | 4.7 ms | 4.9 ms | 4.9 ms |
| **ONNX-CPU-FP32** | **1.7 ms** | **1.8 ms** | **1.8 ms** |
| ONNX-CPU-INT8 | 1.5 ms | 1.6 ms | 1.7 ms |

**Headline:** ONNX-CPU-FP32 is **3.4× faster than PyTorch-MPS** on single-query and **2.8× faster than TorchScript-CPU**. The risk "MPS may be hard to beat" did not materialize — bge-small is small enough that ORT's per-call overhead + graph-level optimizations (operator fusion, constant folding) dominate vs MPS GPU-dispatch overhead. TorchScript gives a modest 30% win over plain PyTorch-CPU (graph IR + freezing) but doesn't match ORT's optimization depth. INT8 trims another ~12% on p50 and now has very tight tail too.

### Bench-9 — throughput (vectors/sec, bs = input size)

| Backend | bs=1 | bs=8 | bs=32 | bs=128 |
|---|---|---|---|---|
| PyTorch-MPS | 156 | 319 | 206 | **351** |
| PyTorch-CPU | 57 | 106 | 80 | 135 |
| TorchScript-CPU | 62 | 115 | 83 | 97 |
| ONNX-CPU-FP32 | 106 | 64 | 32 | 38 |
| ONNX-CPU-INT8 | 78 | 76 | 35 | 41 |

**Anomaly (deferred investigation):** ONNX throughput decreases with batch size — likely per-batch padding to longest sequence (ORT runs the full attention pattern regardless of mask, while PyTorch / TorchScript paths use more efficient masked attention). TorchScript scales like PyTorch-CPU (no anomaly), confirming the bottleneck is ORT-specific.

**Workload mapping:** ONNX-CPU-FP32 for `/ask` (latency); PyTorch-MPS for `make reindex` (throughput at bs≥8 — **9× faster** than ONNX at bs=128).

### Step 10 — Ragas eval results

Baseline = cached PyTorch `chunk_1024` (faithfulness 0.882, answer_relevancy 0.886, context_precision 0.598, context_recall 0.557). Acceptance budgets: FP32 ±0.02 on all metrics; INT8 ≤ −0.05 on faithfulness / context_recall.

| Metric | PyTorch baseline | ONNX FP32 | Δ FP32 | ONNX INT8 | Δ INT8 |
|---|---|---|---|---|---|
| faithfulness | 0.882 | 0.889 | +0.007 | 0.873 | −0.009 |
| answer_relevancy | 0.886 | 0.886 | −0.001 | 0.867 | −0.019 |
| context_precision | 0.598 | 0.598 | ≈0 | 0.589 | −0.009 |
| context_recall | 0.557 | 0.557 | ≈0 | **0.487** | **−0.070** |

**FP32 verdict — PASS.** All deltas within ±0.01. Retrieval-metrics identical to PyTorch (cosine 1.0 parity translates to byte-identical top-k). Generation-metrics within LLM-as-judge noise. Production-ready for `/ask` — combined with 3.4× latency win on bench-9, ONNX FP32 is a strict improvement over PyTorch-MPS on the hot path.

**INT8 verdict — FAIL.** `context_recall` dropped −0.070, exceeds the −0.05 budget. The 0.997 mean cosine vs FP32 turns into top-5 reshuffling that drops borderline-relevant chunks. Precision is fine (graded chunks are still relevant), but coverage is not. **INT8 documented as unusable for production on this model**, kept as a benchmark artifact (`docsrag_int8` collection + bench-9 row). Larger encoder models (`bge-base` 110M, `bge-large` 335M) typically tolerate INT8 much better; this is a small-model-specific failure mode. Static quantization with calibration would likely close some of the gap but doesn't have a guaranteed margin and adds complexity — escalation rejected as not worth the engineering cost on a personal project.

### Acceptance gates

- done: parity cosine > 0.9999 (actual 1.000000).
- ✓ done: Ragas FP32 within ±0.02 of cached `chunk_1024` baseline (actual max Δ 0.007).
- ✗ done with documented failure: Ragas INT8 context_recall dropped −0.070, exceeds budget; verdict above.
- done: bench table populated for 4 backends; TorchScript row in step 11.
- done: `make health` + `make smoke` + existing `chunk_1024.yaml` eval path still green.
- done: CI green without `[onnx]` extra (opt-in).

### Residual risks

- **INT8 + small BERT fragility.** Materialized as predicted — see Step 10 results. Documented, not blocking; INT8 path retained as a benchmark artifact (`/ask` defaults remain pytorch/onnx-fp32). If switching to a larger encoder (bge-base / bge-large), re-run quantization + Ragas; the failure here is model-size-specific, not a general INT8 verdict.
- **`uv pip install -e .` doesn't refresh editable MAPPING when `packages.find` changes.** Hit once at step 2 (new `embeddings/` package not picked up). Symptom: `pytest` fails imports while `python -c` works (CWD vs sys.path). Fix: `uv pip install -e . --no-deps` after `pyproject.toml` `packages.find` edits.

---

## When in doubt

- Read the corresponding section of this file before improvising.
- If something contradicts what you find here, the file is likely stale — surface the discrepancy and update the file rather than silently doing the new thing.
- If a task starts ballooning in scope, stop, ship the minimum, mark `TODO` notes, and move on. The plan exists to prevent rabbit-holes.
