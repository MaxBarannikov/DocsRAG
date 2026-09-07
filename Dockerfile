# syntax=docker/dockerfile:1

# --- Stage 1: resolve and build dependencies with uv -------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so editing source does not invalidate this layer.
# --frozen fails on lockfile drift instead of silently resolving something else.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# The runtime needs only the packages the service imports. Evaluation tooling
# (mlflow, ragas) and the ONNX extra stay out: they add gigabytes and are used
# from the host, not from this container.
COPY core/ ./core/
COPY embeddings/ ./embeddings/
COPY api/ ./api/
COPY indexing/ ./indexing/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 is required by torch/sentence-transformers on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user. The HF cache lives under the home directory so a
# read-only application tree does not prevent model downloads.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # The venv is byte-compiled at build time; this only stops the container from
    # writing .pyc files back into the read-only source bind mounts used in dev.
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

USER appuser

EXPOSE 8000

# The slim image ships no curl or wget, so the check is a plain Python request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
