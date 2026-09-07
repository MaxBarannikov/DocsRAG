"""Application configuration, shared by the API, indexing, evaluation and benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Redundant for Settings itself, but libraries that read os.environ directly
# (huggingface_hub, the langfuse SDK) need .env present in the process environment.
load_dotenv()

EmbedderBackend = Literal["pytorch", "onnx-fp32", "onnx-int8"]
InferenceBackend = Literal["ollama", "vllm"]
RetrievalStrategy = Literal["dense", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,  # the eval harness overrides embedder_backend at runtime
    )

    data_dir: Path = Field(default=Path("data"), description="Root for generated local artifacts")
    docs_source_path: Path = Path("data/raw/fastapi/docs/en/docs")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docsrag"
    # INT8 vectors differ numerically from FP32, hence a separate collection.
    qdrant_collection_int8: str = "docsrag_int8"

    retrieval_strategy: RetrievalStrategy = "dense"
    top_k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=20, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Per-arm candidates before fusion. Fusing only top_k leaves RRF nothing to merge.
    hybrid_candidates: int = Field(default=50, ge=1)

    inference_backend: InferenceBackend = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    vllm_api_key: str = "EMPTY"  # unchecked by vLLM, required by the OpenAI client

    # Set explicitly so Ollama and vLLM stay comparable; their defaults differ.
    llm_max_tokens: int = Field(default=1024, ge=1)
    llm_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    # Anti-loop brake for Qwen at temperature=0; Ollama's repeat_penalty does the same.
    llm_frequency_penalty: float = Field(default=0.3, ge=-2.0, le=2.0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, description="Expected width; checked against the loaded model")
    embedder_backend: EmbedderBackend = "pytorch"
    embedder_onnx_fp32_path: Path = Path("models/bge-small-en-v1.5-onnx-fp32")
    embedder_onnx_int8_path: Path = Path("models/bge-small-en-v1.5-onnx-int8")

    # In characters, not tokens.
    chunk_size: int = Field(default=1024, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)

    hf_token: str = ""  # optional; bge-small is public
    api_key: str = ""  # empty disables authentication on /ask and /agent/ask

    mlflow_tracking_uri: str = "http://localhost:5555"
    mlflow_experiment_name: str = "docsrag-rag-eval"

    # Empty keys disable tracing.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def active_qdrant_collection(self) -> str:
        if self.embedder_backend == "onnx-int8":
            return self.qdrant_collection_int8
        return self.qdrant_collection

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def bm25_index_path(self, collection: str) -> Path:
        """Keyed by collection so switching backend cannot reuse an index from another corpus."""
        return self.data_dir / f"bm25_index_{collection}.json"


settings = Settings()
