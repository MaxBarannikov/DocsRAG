"""Quantize an ONNX FP32 model to INT8 (dynamic).

Wraps `optimum.onnxruntime.ORTQuantizer` with the standard preset:
    - Dynamic: weights quantized to INT8 offline, activations on-the-fly per inference.
      No calibration set needed — simplest path. If retrieval quality drops too
      much, the fallback is static quantization with a calibration corpus
      (CLAUDE.md Task 9 step 6).
    - avx512_vnni preset: optimal on modern Intel CPUs; on ARM / Apple Silicon
      it falls back to portable INT8 kernels without VNNI instructions.
    - per_channel=True: own zero-point + scale per channel. Default after A/B
      on 120 real chunks (Task 9 step 6): per_tensor gave mean cosine 0.9759
      vs FP32 (75% below 0.98), per_channel gave 0.9973 (1.7% below 0.99).
      Size cost: +0.2 MB on 32 MB total — overwhelming win. The --per-tensor
      flag is kept for reproducibility of the alt variant.

Input:  models/<name>-onnx-fp32/   (from scripts/export_onnx.py)
Output: models/<name>-onnx-int8/   (tokenizer files copied from input)

Idempotent: skips if {output}/model.onnx already exists, unless --force.

Usage:
    python scripts/quantize_onnx.py
    python scripts/quantize_onnx.py --input models/bge-base-en-v1.5-onnx-fp32/
    python scripts/quantize_onnx.py --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_INPUT = Path("models/bge-small-en-v1.5-onnx-fp32")

# Tokenizer / config files that ORTQuantizer doesn't carry over by default.
# We copy them so OnnxEmbedder can load the quantized dir with AutoTokenizer.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
)


def _check_onnx_extra_installed() -> None:
    try:
        import onnx  # noqa: F401
        import optimum  # noqa: F401
    except ImportError as e:
        missing = e.name or "<unknown>"
        msg = f"Missing ONNX dependency ({missing!r}). Install: make install-onnx"
        raise SystemExit(msg) from e


def _default_output(input_dir: Path) -> Path:
    """<name>-onnx-fp32/ → <name>-onnx-int8/ (sibling dir)."""
    name = input_dir.name
    if name.endswith("-onnx-fp32"):
        new_name = name[: -len("-onnx-fp32")] + "-onnx-int8"
    else:
        new_name = name + "-int8"
    return input_dir.parent / new_name


def _normalize_output_filename(output_dir: Path) -> Path | None:
    """ORTQuantizer's output filename varies across optimum versions
    (model.onnx vs model_quantized.onnx). Normalize to model.onnx so
    OnnxEmbedder works without special-casing.

    Returns the final path to model.onnx, or None if nothing was produced.
    """
    target = output_dir / "model.onnx"
    if target.exists():
        return target
    candidates = sorted(output_dir.glob("*.onnx"))
    if len(candidates) == 1:
        candidates[0].rename(target)
        print(f"→ Renamed {candidates[0].name} → model.onnx")
        return target
    if len(candidates) == 0:
        return None
    raise RuntimeError(f"Multiple .onnx files produced, expected one: {[c.name for c in candidates]}")


def main() -> int:
    p = argparse.ArgumentParser(description="Quantize an ONNX FP32 model to INT8 (dynamic).")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"FP32 model dir (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=None, help="Output INT8 dir (default: sibling with -int8 suffix)")
    p.add_argument("--force", action="store_true", help="Re-quantize even if output model.onnx exists")
    p.add_argument(
        "--per-tensor",
        action="store_true",
        help="Use per-tensor quantization instead of the default per-channel (worse accuracy, ~same size).",
    )
    args = p.parse_args()

    _check_onnx_extra_installed()

    input_dir: Path = args.input
    input_model = input_dir / "model.onnx"
    if not input_model.exists():
        msg = f"Input FP32 model not found at {input_model}. Run `make export-onnx` first."
        raise SystemExit(msg)

    output_dir: Path = args.output if args.output else _default_output(input_dir)
    output_model = output_dir / "model.onnx"

    if output_model.exists() and not args.force:
        size_mb = output_model.stat().st_size / 1024 / 1024
        print(f"✓ {output_model} already exists ({size_mb:.1f} MB) — skipping. Pass --force to re-quantize.")
        return 0

    if output_dir.exists() and args.force:
        print(f"→ Removing existing {output_dir}/ (--force)")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    print(f"→ Loading quantizer from {input_dir}/")
    quantizer = ORTQuantizer.from_pretrained(input_dir, file_name="model.onnx")

    per_channel = not args.per_tensor
    print(f"→ Configuring dynamic INT8 (avx512_vnni preset, per_channel={per_channel})")
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=per_channel)

    print(f"→ Quantizing → {output_dir}/")
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

    final_model = _normalize_output_filename(output_dir)
    if final_model is None:
        print(f"✗ Quantization completed but no .onnx file found in {output_dir}/")
        return 1

    print("→ Copying tokenizer + config files from input")
    copied = []
    for fname in TOKENIZER_FILES:
        src = input_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            copied.append(fname)

    input_size = input_model.stat().st_size / 1024 / 1024
    output_size = final_model.stat().st_size / 1024 / 1024
    print(f"✓ Quantized {input_model} ({input_size:.1f} MB) → {final_model} ({output_size:.1f} MB)")
    print(f"  Compression: {input_size / output_size:.2f}x ({100 * (1 - output_size / input_size):.0f}% smaller)")
    print(f"  Tokenizer files copied: {', '.join(copied)}")
    print(f"  Output dir contents: {', '.join(sorted(p.name for p in output_dir.iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
