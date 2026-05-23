"""ONNX Runtime embedder for sentence-transformers models exported via scripts/export_onnx.py.

Reads the graph-baked `sentence_embedding` output (already pooled + L2-normalized),
so no manual pooling is needed. Requires the [onnx] extra: `make install-onnx`.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import onnxruntime as ort
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoTokenizer

DEFAULT_MODEL_DIR = Path("models/bge-small-en-v1.5-onnx-fp32")
SENTENCE_EMBEDDING_OUTPUT = "sentence_embedding"


class OnnxEmbedder:
    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = self.model_dir.name
        logger.info(f"Loading ONNX embedder from '{self.model_dir}' on provider '{provider}'")

        model_path = self.model_dir / "model.onnx"
        if not model_path.exists():
            msg = f"model.onnx not found at {model_path}. Run `make export-onnx` first to create it."
            raise FileNotFoundError(msg)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=[provider],
        )

        output_names = {o.name for o in self._session.get_outputs()}
        if SENTENCE_EMBEDDING_OUTPUT not in output_names:
            msg = (
                f"Expected '{SENTENCE_EMBEDDING_OUTPUT}' output in {model_path}; "
                f"got {sorted(output_names)}. Re-export with FORCE=1."
            )
            raise RuntimeError(msg)

        self.dimension = self._probe_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def _probe_dimension(self) -> int:
        """One-shot inference to read the output shape — the graph shape is symbolic so static inspection fails."""
        encoded = self._tokenizer(
            ["dimension probe"],
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        outputs = self._session.run(
            [SENTENCE_EMBEDDING_OUTPUT],
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )
        return int(outputs[0].shape[-1])

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",  # kept for API parity with PytorchEmbedder; unused for bge
    ) -> list[list[float]]:
        if not texts:
            return []

        prefixed = [prefix + t for t in texts] if prefix else list(texts)

        batch_starts: Iterable[int] = range(0, len(prefixed), batch_size)
        if show_progress:
            batch_starts = tqdm(batch_starts, desc="ONNX encode", unit="batch")

        all_embeddings: list[list[float]] = []
        for start in batch_starts:
            batch = prefixed[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            outputs = self._session.run(
                [SENTENCE_EMBEDDING_OUTPUT],
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                },
            )
            all_embeddings.extend(outputs[0].tolist())

        return all_embeddings
