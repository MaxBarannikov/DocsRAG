"""Project-wide configuration via Pydantic Settings.

All settings can be overridden via environment variables or a .env file.
Centralizing here means we don't sprinkle hardcoded URLs/paths across modules.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

EmbedderBackend = Literal["pytorch", "onnx-fp32", "onnx-int8"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Field(default_factory=lambda: Path("data"))

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docsrag"
    # Separate collection for the INT8 backend: INT8 embeddings differ
    # numerically from FP32, so they live in their own index (Task 9 plan,
    # architectural decision #1).
    qdrant_collection_int8: str = "docsrag_int8"

    # Inference backend: "ollama" (default, local dev) or "vllm" (production)
    inference_backend: str = "ollama"

    # Ollama
    # Variant A (native Ollama on host): http://localhost:11434
    # Variant B (Ollama in Docker, called from another container): http://ollama:11434
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"

    # vLLM — OpenAI-compatible endpoint (vllm-metal locally, vllm on CUDA in prod)
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # bge-small-en-v1.5 produces 384-dim vectors

    # Embedder backend — selects the runtime via embeddings.factory.make_embedder().
    # "pytorch":   sentence-transformers, MPS/CUDA/CPU. Default — preserves prior behavior.
    # "onnx-fp32": ONNX Runtime, byte-equivalent to PyTorch (Task 9 parity gate).
    # "onnx-int8": ONNX Runtime + dynamic per-channel INT8. ~4× smaller, ~0.997 cosine vs FP32.
    embedder_backend: EmbedderBackend = "pytorch"
    embedder_onnx_fp32_path: Path = Field(default_factory=lambda: Path("models/bge-small-en-v1.5-onnx-fp32"))
    embedder_onnx_int8_path: Path = Field(default_factory=lambda: Path("models/bge-small-en-v1.5-onnx-int8"))

    # Indexing defaults (best config from Task 4 sweep)
    chunk_size: int = 1024
    chunk_overlap: int = 100

    # Document source
    docs_source_path: Path = Field(default_factory=lambda: Path("data/raw/fastapi/docs/en/docs"))

    # LangFuse tracing (optional — leave empty to disable)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def active_qdrant_collection(self) -> str:
        """Collection name routed by `embedder_backend`.

        INT8 embeddings differ from FP32 numerically, so they get their own
        collection. PyTorch and ONNX FP32 share `qdrant_collection` — they're
        parity-gated equivalent (Task 9 step 5: cosine 1.000000 on 120 chunks).
        """
        if self.embedder_backend == "onnx-int8":
            return self.qdrant_collection_int8
        return self.qdrant_collection


settings = Settings()
