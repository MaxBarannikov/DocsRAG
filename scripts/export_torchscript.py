"""Export a sentence-transformers backbone to TorchScript (traced).

Task 9 step 11 — bonus row in the embedder benchmark. NOT integrated into the
production embedder factory (TorchScript in 2026 is a legacy format, kept here
only for the bench-table comparison; ONNX is the production-relevant alt).

What gets traced:
    The underlying HuggingFace AutoModel (BERT-family transformer backbone),
    wrapped so it returns `last_hidden_state` directly. Pooling and L2 norm
    are done in the bench wrapper (analogous to the manual-pooling path we
    considered for ONNX before settling on the graph-baked `sentence_embedding`
    output — see CLAUDE.md decision #4).

Output: a single .pt file at models/<basename>.pt + tokenizer files copied
from the HF cache via a separate AutoTokenizer.from_pretrained() in the bench.
Idempotent: skips if model.pt already exists unless --force is passed.

Usage:
    python scripts/export_torchscript.py
    python scripts/export_torchscript.py --model BAAI/bge-base-en-v1.5
    python scripts/export_torchscript.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _default_output(model_name: str) -> Path:
    basename = model_name.rsplit("/", 1)[-1]
    return Path("models") / f"{basename}.pt"


def main() -> int:
    p = argparse.ArgumentParser(description="Trace a sentence-transformers backbone to TorchScript.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default: {DEFAULT_MODEL})")
    p.add_argument("--output", default=None, help="Output .pt path (default: models/<basename>.pt)")
    p.add_argument("--force", action="store_true", help="Re-export even if output exists")
    args = p.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    output_path: Path = Path(args.output) if args.output else _default_output(args.model)
    if output_path.exists() and not args.force:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ {output_path} already exists ({size_mb:.1f} MB) — skipping. Pass --force to re-export.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"→ Loading {args.model}")
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Wrap so the traced forward returns last_hidden_state directly (a tensor),
    # avoiding the HF ModelOutput dataclass that TorchScript can't trace cleanly.
    class _BackboneWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    wrapped = _BackboneWrapper(model)
    wrapped.eval()

    sample = tokenizer(["sample text for tracing"], return_tensors="pt", padding=True, truncation=True)
    print("→ Tracing forward pass (expect TracerWarnings about attention-mask boolean casts — benign)")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (sample["input_ids"], sample["attention_mask"]), strict=False)

    torch.jit.save(traced, str(output_path))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Saved TorchScript model → {output_path} ({size_mb:.1f} MB)")
    print(f"  Tokenizer (loaded fresh in the bench): {args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
