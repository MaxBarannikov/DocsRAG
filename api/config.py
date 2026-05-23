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
    # INT8 embeddings differ numerically from FP32 → separate collection.
    qdrant_collection_int8: str = "docsrag_int8"

    inference_backend: str = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"

    # vLLM — OpenAI-compatible endpoint
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # bge-small-en-v1.5 produces 384-dim vectors

    embedder_backend: EmbedderBackend = "pytorch"
    embedder_onnx_fp32_path: Path = Field(default_factory=lambda: Path("models/bge-small-en-v1.5-onnx-fp32"))
    embedder_onnx_int8_path: Path = Field(default_factory=lambda: Path("models/bge-small-en-v1.5-onnx-int8"))

    chunk_size: int = 1024
    chunk_overlap: int = 100

    docs_source_path: Path = Field(default_factory=lambda: Path("data/raw/fastapi/docs/en/docs"))

    # LangFuse tracing (optional — leave empty to disable)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def active_qdrant_collection(self) -> str:
        """Route by embedder backend: INT8 gets its own collection, FP32/PyTorch share one."""
        if self.embedder_backend == "onnx-int8":
            return self.qdrant_collection_int8
        return self.qdrant_collection


settings = Settings()
