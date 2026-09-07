"""Dynamic INT8 quantization of an ONNX FP32 model, from export_onnx.py.

Per-channel by default: it holds mean cosine 0.9973 against FP32, where per-tensor
manages only 0.9759. Idempotent; pass --force to re-quantize.

    python scripts/quantize_onnx.py --input models/bge-base-en-v1.5-onnx-fp32/ --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from scripts._paths import resolve_artifact_dir

DEFAULT_INPUT = Path("models/bge-small-en-v1.5-onnx-fp32")

MIN_EXPECTED_COMPRESSION = 2.0  # dynamic INT8 on a BERT encoder lands near 4x

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
    new_name = name[: -len("-onnx-fp32")] + "-onnx-int8" if name.endswith("-onnx-fp32") else name + "-int8"
    return input_dir.parent / new_name


def _normalize_output_filename(output_dir: Path) -> Path | None:
    """Normalize to model.onnx; ORTQuantizer's filename varies across optimum versions
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
    msg = f"Multiple .onnx files produced, expected one: {[c.name for c in candidates]}"
    raise RuntimeError(msg)


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

    try:
        output_dir = resolve_artifact_dir(args.output or _default_output(input_dir))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    if not copied:
        # OnnxEmbedder loads its tokenizer from this directory.
        print(f"✗ No tokenizer files found in {input_dir}/ — the output directory is not loadable.")
        return 1

    input_size = input_model.stat().st_size / 1024 / 1024
    output_size = final_model.stat().st_size / 1024 / 1024
    if output_size <= 0:
        print(f"✗ Quantized model at {final_model} is empty.")
        return 1
    compression = input_size / output_size
    print(f"✓ Quantized {input_model} ({input_size:.1f} MB) → {final_model} ({output_size:.1f} MB)")
    print(f"  Compression: {compression:.2f}x ({100 * (1 - output_size / input_size):.0f}% smaller)")
    if compression < MIN_EXPECTED_COMPRESSION:
        # Near 1x means the quantizer emitted an unquantized copy.
        print(
            f"✗ Expected at least {MIN_EXPECTED_COMPRESSION:.1f}x compression from INT8 quantization. "
            f"The output is probably still FP32 — inspect {output_dir}/ before using it."
        )
        return 1
    print(f"  Tokenizer files copied: {', '.join(copied)}")
    print(f"  Output dir contents: {', '.join(sorted(p.name for p in output_dir.iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
