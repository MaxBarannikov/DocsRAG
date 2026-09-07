from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.config import EvalConfig
from evaluation.run_eval import load_config

CONFIG_DIR = Path("configs")
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))


def test_config_directory_is_not_empty() -> None:
    assert CONFIG_FILES, f"no YAML configs found in {CONFIG_DIR}/"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_loads_and_validates(path: Path) -> None:
    config = load_config(path)
    assert config.chunk_overlap < config.chunk_size
    assert config.top_k >= 1


def test_unknown_keys_are_rejected() -> None:
    """extra='forbid' turns a typo into a startup error, not a silent no-op."""
    with pytest.raises(ValueError, match="Extra inputs"):
        EvalConfig.model_validate(
            {
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "top_k": 5,
                "embedding_model": "m",
                "llm_model": "l",
                "chunk_sizee": 512,
            }
        )


def test_missing_required_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Field required"):
        EvalConfig.model_validate({"chunk_size": 1024})
