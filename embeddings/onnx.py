"""ONNX Runtime embedder for models exported by scripts/export_onnx.py.

Reads the graph-baked `sentence_embedding` output, already pooled and normalized, so
no pooling is reimplemented here — that is what keeps it numerically identical to the
PyTorch backend. Requires the [onnx] extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import onnxruntime as ort
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from core.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SENTENCE_EMBEDDING_OUTPUT = "sentence_embedding"


class OnnxEmbedder:
    def __init__(
        self,
        model_dir: Path | str | None = None,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else settings.embedder_onnx_fp32_path
        self.model_name = self.model_dir.name
        logger.info("Loading ONNX embedder from '{}' on provider '{}'", self.model_dir, provider)

        available = ort.get_available_providers()
        if provider not in available:
            msg = f"ONNX Runtime provider {provider!r} is not available. Installed providers: {available}."
            raise ValueError(msg)

        model_path = self.model_dir / "model.onnx"
        if not model_path.exists():
            msg = f"model.onnx not found at {model_path}. Run `make export-onnx` first to create it."
            raise FileNotFoundError(msg)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        except Exception as exc:
            msg = (
                f"No usable tokenizer in {self.model_dir}. The export step copies the tokenizer "
                f"files alongside model.onnx — re-run `make export-onnx FORCE=1`."
            )
            raise RuntimeError(msg) from exc

        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=[provider])

        output_names = {o.name for o in self._session.get_outputs()}
        if SENTENCE_EMBEDDING_OUTPUT not in output_names:
            msg = (
                f"Expected a {SENTENCE_EMBEDDING_OUTPUT!r} output in {model_path}; got {sorted(output_names)}. "
                f"Re-export with FORCE=1."
            )
            raise RuntimeError(msg)

        self._input_names = {i.name for i in self._session.get_inputs()}
        self.dimension = self._probe_dimension()
        if self.dimension != settings.embedding_dim:
            logger.warning(
                "Embedding dimension {} differs from configured EMBEDDING_DIM={}.",
                self.dimension,
                settings.embedding_dim,
            )
        logger.info("Embedding dimension: {}", self.dimension)

    def _run(self, batch: list[str]) -> list[list[float]]:
        encoded = self._tokenizer(batch, padding=True, truncation=True, return_tensors="np")
        # Tokenizers may emit token_type_ids; feed only what this graph declares.
        feed = {name: encoded[name] for name in self._input_names if name in encoded}
        outputs = self._session.run([SENTENCE_EMBEDDING_OUTPUT], feed)
        return [vector.tolist() for vector in outputs[0]]

    def _probe_dimension(self) -> int:
        """One-shot inference; the graph's output shape is symbolic."""
        return len(self._run(["dimension probe"])[0])

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",  # kept for parity with PytorchEmbedder; bge needs no prefix
    ) -> list[list[float]]:
        if not texts:
            return []

        prefixed = [prefix + t for t in texts] if prefix else list(texts)

        batch_starts: Iterable[int] = range(0, len(prefixed), batch_size)
        if show_progress:
            batch_starts = tqdm(batch_starts, desc="ONNX encode", unit="batch")

        all_embeddings: list[list[float]] = []
        for start in batch_starts:
            all_embeddings.extend(self._run(prefixed[start : start + batch_size]))
        return all_embeddings
