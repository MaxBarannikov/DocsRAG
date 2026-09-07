from __future__ import annotations

from pathlib import Path

MODELS_ROOT = Path("models")


def resolve_artifact_dir(path: Path | str) -> Path:
    """Refuse anything outside models/.

    `--force` deletes the target before writing, so an unchecked `--output` would let
    a typo remove an arbitrary directory. Symlinks could escape the check.
    """
    root = MODELS_ROOT.resolve()
    candidate = Path(path)

    if candidate.is_symlink():
        msg = f"Refusing to use a symlinked output path: {candidate}"
        raise ValueError(msg)

    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        msg = f"Output directory must live under {MODELS_ROOT}/ (got {resolved}). Refusing to continue."
        raise ValueError(msg)
    return resolved
