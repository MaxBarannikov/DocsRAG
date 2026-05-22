# Load .env if present so `make` targets honour the same config as
# docker-compose and the Python app. Make doesn't read .env on its own —
# without this block, $(VLLM_MODEL) etc. fall back to the `?=` defaults
# below even when .env sets them. `export` propagates the vars to recipes.
ifneq (,$(wildcard .env))
    include .env
    export
endif

.PHONY: help install up down logs ollama-status \
        format test \
        fetch-docs index reindex smoke \
        build rebuild restart api-logs api-shell health ask warmup \
        eval mlflow-ui prometheus-ui grafana-ui \
        install-vllm vllm-start vllm-status install-onnx clean

# Default question for `make ask` if Q is not provided
Q ?= How do I define a path parameter in FastAPI?

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install       - Install Python dependencies (uv venv + editable install)"
	@echo ""
	@echo "  Lifecycle:"
	@echo "    make up            - Start all Docker services (Qdrant + API) and check Ollama"
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
	@echo "    make install-onnx  - Install ONNX Runtime + optimum into .venv (optional, for Task 9 work)"
	@echo ""
	@echo "  RAG API:"
	@echo "    make health        - GET /health"
	@echo "    make ask Q=\"...\"   - POST /ask with a question (default: path-params example)"
	@echo "    make warmup        - Send a warmup question to load the LLM into Ollama RAM"
	@echo ""
	@echo "  Indexing:"
	@echo "    make fetch-docs    - Download FastAPI documentation"
	@echo "    make index         - Run the indexing pipeline (incremental)"
	@echo "    make reindex       - Recreate the collection (CHUNK_SIZE=1024 CHUNK_OVERLAP=100 by default)"
	@echo "    make smoke         - Run a smoke retrieval test"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint          - Run ruff linter"
	@echo "    make format        - Format code with ruff + black"
	@echo "    make type-check    - Run mypy"
	@echo "    make test          - Run pytest"
	@echo ""
	@echo "  Misc:"
	@echo "    make clean         - Remove caches and build artifacts"

# Setup

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

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
# upgraded to error on macOS. Confirmed experimentally on branch `try-uv`.
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

CHUNK_SIZE ?= 1024
CHUNK_OVERLAP ?= 100

reindex:
	python -m indexing.run_indexing --recreate --chunk-size $(CHUNK_SIZE) --overlap $(CHUNK_OVERLAP)
	rm -f data/bm25_index.pkl

smoke:
	python -m indexing.smoke_test "how to define a path parameter in FastAPI"

# Quality 

format:
	pre-commit run -a

test:
	pytest -v

# Evaluation

CONFIG ?= configs/baseline.yaml

eval:
	python evaluation/run_eval.py --config $(CONFIG)

mlflow-ui:
	@open http://localhost:5000 || xdg-open http://localhost:5000

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
