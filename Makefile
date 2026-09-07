# Read .env so recipes can interpolate values like $(VLLM_MODEL) and $(VLLM_PORT).
#
# Deliberately NOT `export`ed. Make's parser is not a dotenv parser: it keeps quotes
# as part of the value, so `KEY="secret"` would be exported as `"secret"` with the
# quotes attached. Since python-dotenv does not override variables already present in
# the environment, that corrupted value would win over the correct one inside every
# child process — which showed up as LangFuse rejecting valid keys with a 401.
#
# Without `export`, each consumer parses .env itself: Python through python-dotenv,
# Docker Compose through its own loader. Both handle quoting correctly.
ifneq (,$(wildcard .env))
    include .env
endif

.PHONY: help install up down logs ollama-status \
        lint type-check format test ci \
        fetch-docs index reindex smoke \
        build rebuild restart api-logs api-shell health ask warmup \
        eval mlflow-ui prometheus-ui grafana-ui \
        install-vllm vllm-start vllm-status install-onnx export-onnx quantize-onnx \
        reindex-onnx reindex-int8 bench-embedder bench-backends export-torchscript clean

# Default question for `make ask` if Q is not provided
Q ?= How do I define a path parameter in FastAPI?

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install       - Install Python dependencies (uv venv + editable install)"
	@echo ""
	@echo "  Lifecycle:"
	@echo "    make up            - Start all Docker services (Qdrant, API, Prometheus, Grafana, MLflow)"
	@echo "    make down          - Stop Docker services"
	@echo "    make build         - Build the API Docker image"
	@echo "    make rebuild       - Rebuild the API image without cache and restart (use after pyproject/Dockerfile changes)"
	@echo "    make restart       - Recreate the API container (picks up code edits via bind mount AND .env/compose changes)"
	@echo "    make logs          - Tail logs of all services"
	@echo "    make api-logs      - Tail logs of the API service only"
	@echo "    make api-shell     - Open a shell inside the running API container"
	@echo "    make ollama-status - Check Ollama status (native install)"
	@echo "    make install-vllm  - Install vllm + vllm-metal plugin into .venv (macOS arm64 only)"
	@echo "    make vllm-start    - Start vllm-metal server (VLLM_MODEL / VLLM_PORT overridable)"
	@echo "    make vllm-status   - Check vllm-metal status"
	@echo "    make install-onnx  - Install ONNX Runtime + optimum into .venv"
	@echo "    make export-onnx   - Export embedder to ONNX FP32 (ONNX_MODEL=... overridable, FORCE=1 to re-export)"
	@echo "    make quantize-onnx - Quantize ONNX FP32 → INT8 dynamic (FORCE=1 to re-quantize)"
	@echo "    make bench-embedder - Latency + throughput benchmark across embedder backends (VERIFY=1 to check parity)"
	@echo "    make bench-backends - Ollama vs vllm-metal generation latency (WARMUP=1 to discard a first run)"
	@echo "    make export-torchscript - Trace bge backbone to TorchScript .pt (bench-only artifact)"
	@echo ""
	@echo "  RAG API:"
	@echo "    make health        - GET /health"
	@echo "    make ask Q=\"...\"   - POST /ask with a question (default: path-params example)"
	@echo "    make warmup        - Send a warmup question to load the LLM into Ollama RAM"
	@echo ""
	@echo "  Indexing:"
	@echo "    make fetch-docs    - Download FastAPI documentation"
	@echo "    make index         - Index new/changed docs into the existing collection (idempotent)"
	@echo "    make reindex       - Recreate the collection (CHUNK_SIZE=1024 CHUNK_OVERLAP=100 by default)"
	@echo "    make reindex-onnx  - Reindex docsrag with ONNX FP32 backend (parity-equivalent to pytorch)"
	@echo "    make reindex-int8  - Reindex docsrag_int8 with ONNX INT8 backend (separate collection)"
	@echo "    make smoke         - Print the top retrieval results for a sample query"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make eval          - Run Ragas evaluation (CONFIG=configs/chunk_1024.yaml by default)"
	@echo "    make mlflow-ui     - Open the MLflow UI"
	@echo "    make prometheus-ui - Open the Prometheus UI"
	@echo "    make grafana-ui    - Open the Grafana dashboard"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint          - Run ruff check"
	@echo "    make format        - Run all pre-commit hooks (ruff-check --fix, ruff-format, mypy)"
	@echo "    make type-check    - Run mypy"
	@echo "    make test          - Run pytest"
	@echo "    make ci            - Run the full CI gate locally"
	@echo ""
	@echo "  Misc:"
	@echo "    make clean         - Remove caches and build artifacts"

# Setup

install:
	uv venv --python 3.12
	uv pip install -e ".[dev,eval]"
	@echo ""
	@echo "✓ Installed. Activate the environment before running make targets that call python:"
	@echo "    source .venv/bin/activate"

# Lifecycle

up:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 3
	@curl -sf http://localhost:6333/collections > /dev/null && echo "✓ Qdrant is up" || echo "✗ Qdrant not ready"
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama is up (native)" || echo "✗ Ollama not ready — start the Ollama app or run 'ollama serve'"
	@curl -sf http://localhost:8000/health > /dev/null && echo "✓ API is up" || echo "… API still starting (model loading takes ~10-20s on first run); check 'make api-logs'"

down:
	docker compose down

build:
	docker compose build api

rebuild:
	docker compose build --no-cache api
	docker compose up -d api

# Apply code edits and/or .env changes without rebuilding the image.
# Source is bind-mounted in docker-compose.yml, so the new container picks up
# the latest files. `--force-recreate` also re-reads environment variables.
# Use `make rebuild` only when pyproject.toml/uv.lock or the Dockerfile changed.
restart:
	docker compose up -d --force-recreate api

logs:
	docker compose logs -f

api-logs:
	docker compose logs -f api

api-shell:
	docker compose exec api /bin/bash

ollama-status:
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama API responding" || echo "✗ Ollama API not responding — start the Ollama app or run 'ollama serve'"

VLLM_MODEL       ?= mlx-community/Qwen2.5-7B-Instruct-4bit
VLLM_PORT        ?= 8001
# Pinned for `make install-vllm`. Bump explicitly when upgrading.
VLLM_VERSION     ?= 0.20.1
VLLM_METAL_WHEEL ?= https://github.com/vllm-project/vllm-metal/releases/download/v0.2.0-20260509-055449/vllm_metal-0.2.0-cp312-cp312-macosx_11_0_arm64.whl

# vllm + vllm-metal can't live in pyproject.toml / uv.lock: vllm's own pyproject
# hard-pins CUDA-only deps (nvidia-cudnn-frontend, cuda-python, flashinfer...)
# with no macOS wheels, so the install is two-phase — CPU requirements first
# (substituting compatible versions), then vllm itself. uv's universal resolver
# has no way to express that. CXXFLAGS works around a clang/parentheses warning
# that macOS upgrades to an error. Verified experimentally before settling on this
# two-phase install.
install-vllm:
	@uname -sm | grep -q "Darwin arm64" || { echo "✗ macOS arm64 only"; exit 1; }
	@test -n "$$VIRTUAL_ENV" || { echo "✗ Activate project venv first: source .venv/bin/activate"; exit 1; }
	@echo "→ Downloading vllm $(VLLM_VERSION) source..."
	curl -sL -o /tmp/vllm-$(VLLM_VERSION).tar.gz https://github.com/vllm-project/vllm/releases/download/v$(VLLM_VERSION)/vllm-$(VLLM_VERSION).tar.gz
	tar xf /tmp/vllm-$(VLLM_VERSION).tar.gz -C /tmp
	@echo "→ Installing vllm CPU requirements..."
	uv pip install -r /tmp/vllm-$(VLLM_VERSION)/requirements/cpu.txt --index-strategy unsafe-best-match
	@echo "→ Building vllm core..."
	cd /tmp/vllm-$(VLLM_VERSION) && CXXFLAGS="-Wno-parentheses" uv pip install .
	rm -rf /tmp/vllm-$(VLLM_VERSION) /tmp/vllm-$(VLLM_VERSION).tar.gz
	@echo "→ Installing vllm-metal plugin..."
	uv pip install "$(VLLM_METAL_WHEEL)"
	@echo "→ Verifying — should print 'Platform plugin metal is activated':"
	vllm --version

# vllm-metal 0.2.0+ is a plugin to upstream vllm — the CLI is `vllm serve`,
# not `vllm-metal --model ...`. The plugin registers itself via entry_points
# and prints "Platform plugin metal is activated" on startup.
vllm-start:
	vllm serve $(VLLM_MODEL) --host 127.0.0.1 --port $(VLLM_PORT)

vllm-status:
	@curl -sf http://127.0.0.1:$(VLLM_PORT)/v1/models > /dev/null && echo "✓ vllm-metal responding on port $(VLLM_PORT)" || echo "✗ vllm-metal not running — run 'make vllm-start'"

# ONNX Runtime + optimum add ~200MB to the venv (ORT + onnx + protobuf + huggingface
# exporters), unnecessary for the default PyTorch-MPS path. Kept as an opt-in extra
# so CI stays lean and `make install` doesn't drag it in.
install-onnx:
	@test -n "$$VIRTUAL_ENV" || { echo "✗ Activate project venv first: source .venv/bin/activate"; exit 1; }
	uv pip install -e ".[onnx]"
	@echo "→ Verifying ONNX Runtime install:"
	@python -c "import onnxruntime as ort; print('ONNX Runtime', ort.__version__, '| providers:', ort.get_available_providers())"

# Export the embedder to ONNX FP32. Default: bge-small-en-v1.5 → models/bge-small-en-v1.5-onnx-fp32/.
# Idempotent — pass FORCE=1 to re-export.
ONNX_MODEL ?= BAAI/bge-small-en-v1.5
export-onnx:
	@python scripts/export_onnx.py --model $(ONNX_MODEL) $(if $(FORCE),--force,)

# Quantize the FP32 ONNX model to INT8 (dynamic, per-channel by default;
# pass --per-tensor to the script for the alternative).
# Default input: models/bge-small-en-v1.5-onnx-fp32/ → models/bge-small-en-v1.5-onnx-int8/.
# Idempotent — pass FORCE=1 to re-quantize. Custom paths: use the script directly.
quantize-onnx:
	@python scripts/quantize_onnx.py $(if $(FORCE),--force,)

# Embedder backend benchmark: PyTorch-MPS, PyTorch-CPU, ONNX-CPU-FP32, ONNX-CPU-INT8.
# Reports single-query p50/p95/p99 latency + throughput at batch sizes 1/8/32/128.
# Adds a TorchScript-CPU row automatically if models/bge-small-en-v1.5.pt exists
# (produced by `make export-torchscript`).
# VERIFY=1 cross-checks each backend against PyTorch before timing it, so a row can
# never silently measure a different embedding function.
bench-embedder:
	python -m benchmarks.bench_embedder $(if $(VERIFY),--verify-parity,)

# Ollama vs vllm-metal generation latency. Both servers must be running.
bench-backends:
	python -m benchmarks.bench_backends $(if $(WARMUP),--warmup,)

# Trace the bge backbone to TorchScript .pt — bench-only artifact.
# Output: models/bge-small-en-v1.5.pt. Idempotent — pass FORCE=1 to re-trace.
export-torchscript:
	@python scripts/export_torchscript.py $(if $(FORCE),--force,)

# RAG API

health:
	@curl -sf http://localhost:8000/health | python -m json.tool || echo "✗ API not reachable on http://localhost:8000"

ask:
	@curl -s -X POST http://localhost:8000/ask \
		-H 'Content-Type: application/json' \
		-d '{"question": $(call quote,$(Q)), "top_k": 5, "include_contexts": false}' \
		| python -m json.tool --no-ensure-ascii

warmup:
	@echo "Sending warmup request — this loads the LLM into Ollama's RAM..."
	@curl -s -X POST http://localhost:8000/ask \
		-H 'Content-Type: application/json' \
		-d '{"question": "What is FastAPI?", "top_k": 3, "include_contexts": false}' \
		> /dev/null && echo "✓ Warmup complete" || echo "✗ Warmup failed"

# Indexing

fetch-docs:
	./indexing/fetch_docs.sh

index:
	python -m indexing.run_indexing

# BM25 caches are keyed by collection; reindexing invalidates all of them.
BM25_CACHE = data/bm25_index_*.json

CHUNK_SIZE ?= 1024
CHUNK_OVERLAP ?= 100

reindex:
	python -m indexing.run_indexing --recreate --chunk-size $(CHUNK_SIZE) --overlap $(CHUNK_OVERLAP)
	rm -f $(BM25_CACHE)

# Reindex with ONNX FP32 backend → docsrag collection (parity-equivalent to pytorch).
# Drops + recreates docsrag. Safe to run while API serves at EMBEDDER_BACKEND=pytorch
# since FP32 ONNX and PyTorch vectors are numerically identical.
reindex-onnx:
	EMBEDDER_BACKEND=onnx-fp32 python -m indexing.run_indexing --recreate --chunk-size $(CHUNK_SIZE) --overlap $(CHUNK_OVERLAP)
	rm -f $(BM25_CACHE)

# Reindex with ONNX INT8 backend → docsrag_int8 collection (separate index — vectors differ).
# Creates docsrag_int8 fresh; doesn't touch docsrag.
reindex-int8:
	EMBEDDER_BACKEND=onnx-int8 python -m indexing.run_indexing --recreate --chunk-size $(CHUNK_SIZE) --overlap $(CHUNK_OVERLAP)
	rm -f $(BM25_CACHE)

smoke:
	python -m indexing.query_cli "how to define a path parameter in FastAPI"

# Quality

lint:
	ruff check .

type-check:
	mypy .

format:
	pre-commit run -a

test:
	pytest

# The same gate CI runs, minus the Docker build.
ci: lint type-check test
	ruff format --check .
	@echo "✓ All quality gates passed"

# Evaluation

# The frozen baseline; the 512-token configs are kept for reference only.
MLFLOW_PORT ?= 5555
CONFIG ?= configs/chunk_1024.yaml

eval:
	python evaluation/run_eval.py --config $(CONFIG)

mlflow-ui:
	@open http://localhost:$(MLFLOW_PORT) || xdg-open http://localhost:$(MLFLOW_PORT)

prometheus-ui:
	@open http://localhost:9090 || xdg-open http://localhost:9090

grafana-ui:
	@open http://localhost:3000 || xdg-open http://localhost:3000

# Misc

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	@echo "✓ Cleaned local caches"

# Helpers

# Safely JSON-quote a make variable: escape backslashes and double-quotes,
# then wrap in double-quotes. Lets `make ask Q='...'` survive apostrophes,
# quotes, and shell metacharacters in the question.
quote = "$(subst ",\",$(subst \,\\,$(1)))"
