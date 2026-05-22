"""Export a sentence-transformers model to ONNX FP32.

Wraps `optimum-cli export onnx --task feature-extraction`. For a
sentence-transformers model the exported graph has TWO outputs:
    - `token_embeddings`     : [batch, seq_len, hidden_dim] raw transformer output
    - `sentence_embedding`   : [batch, hidden_dim]          already pooled + L2-normalized

Our runtime wrapper (embeddings/onnx.py, Task 9 step 4) reads
`sentence_embedding` directly — pooling and normalization are baked into the
graph and are byte-identical to PyTorch sentence-transformers. See CLAUDE.md
"Task 9 plan" architectural decision #4 for why.

Idempotent: if `{output}/model.onnx` already exists, exits 0 without
re-exporting unless --force is passed.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model BAAI/bge-base-en-v1.5
    python scripts/export_onnx.py --force          # re-export even if model.onnx exists
    python scripts/export_onnx.py --output models/custom/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _check_onnx_extra_installed() -> None:
    """Fail fast with a helpful hint if the [onnx] extra is not installed."""
    try:
        import onnx  # noqa: F401
        import optimum  # noqa: F401
    except ImportError as e:
        missing = e.name or "<unknown>"
        msg = f"Missing ONNX dependency ({missing!r}). Install the optional extra:\n    make install-onnx"
        raise SystemExit(msg) from e


def _default_output_dir(model_name: str) -> Path:
    """BAAI/bge-small-en-v1.5 → models/bge-small-en-v1.5-onnx-fp32/."""
    basename = model_name.rsplit("/", 1)[-1]
    return Path("models") / f"{basename}-onnx-fp32"


def main() -> int:
    p = argparse.ArgumentParser(description="Export a sentence-transformers model to ONNX FP32.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: models/<model-basename>-onnx-fp32/)",
    )
    p.add_argument("--force", action="store_true", help="Re-export even if model.onnx exists")
    args = p.parse_args()

    _check_onnx_extra_installed()

    output_dir: Path = Path(args.output) if args.output else _default_output_dir(args.model)
    model_file = output_dir / "model.onnx"

    if model_file.exists() and not args.force:
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"✓ {model_file} already exists ({size_mb:.1f} MB) — skipping. Pass --force to re-export.")
        return 0

    if output_dir.exists() and args.force:
        print(f"→ Removing existing {output_dir}/ (--force)")
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--task",
        "feature-extraction",
        "--model",
        args.model,
        str(output_dir),
    ]
    print(f"→ Exporting {args.model} → {output_dir}/")
    print(f"   {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"✗ optimum-cli failed with exit code {result.returncode}")
        return result.returncode

    if not model_file.exists():
        print(f"✗ Export completed but {model_file} not found in {output_dir}/")
        return 1

    import onnx

    print(f"→ Validating ONNX graph: {model_file}")
    onnx.checker.check_model(str(model_file))

    size_mb = model_file.stat().st_size / 1024 / 1024
    artifacts = sorted(p.name for p in output_dir.iterdir())
    print(f"✓ Exported {args.model} → {model_file} ({size_mb:.1f} MB)")
    print(f"  Output dir contents: {', '.join(artifacts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
