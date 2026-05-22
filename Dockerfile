# Stage 1: build dependencies with uv
FROM python:3.12-slim AS builder

# Install uv (fast Rust-based package manager)
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first (better Docker layer caching).
# Only copy lock + project metadata so changes to source code don't bust this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy the project and install it
COPY api/ ./api/
COPY indexing/ ./indexing/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: runtime
FROM python:3.12-slim AS runtime

# libgomp1 is required by torch/sentence-transformers on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv and the app code from the builder stage
COPY --from=builder /app /app

# Use the venv's Python by default
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cache HuggingFace downloads in a stable location (mounted as volume in compose)
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

# uvicorn run — single worker is fine for now; we'll tune workers in Task 8.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]