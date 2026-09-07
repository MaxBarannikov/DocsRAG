"""Eval results must survive a tracking-server outage.

Regression: a 12-minute run once crashed inside `log_to_mlflow` before printing
anything, discarding every score it had just computed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.config import EvalConfig
from evaluation.run_eval import save_results

SCORES = {"faithfulness": 0.882, "context_recall": 0.557, "eval_time_sec": 720.0}
COUNTS = {"faithfulness": 24, "context_recall": 25}


@pytest.fixture
def config() -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "top_k": 5,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "qwen2.5:7b-instruct-q4_K_M",
            "retrieval_strategy": "hybrid",
        }
    )


def test_scores_are_written_before_mlflow_is_contacted(config, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("evaluation.run_eval.settings.data_dir", tmp_path)

    path = save_results(config, SCORES, COUNTS, Path("configs/hybrid.yaml"), n_samples=25)

    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["scores"] == SCORES
    assert payload["n_samples"] == 25
    assert payload["config"]["retrieval_strategy"] == "hybrid"
    assert payload["config_file"] == "hybrid.yaml"
    assert payload["prompt_version"]
    # A mean over 24 of 25 samples must not be recorded as if it covered all of them.
    assert payload["scored_samples"] == COUNTS
    # The judging stack is what made older runs incomparable; record it with the run.
    assert set(payload["judge_stack"]) == {"ragas", "langchain-core", "langchain-ollama"}


def test_each_run_gets_its_own_file(config, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("evaluation.run_eval.settings.data_dir", tmp_path)

    save_results(config, SCORES, COUNTS, Path("configs/hybrid.yaml"), n_samples=25)
    save_results(config, SCORES, COUNTS, Path("configs/dense.yaml"), n_samples=25)

    assert len(list((tmp_path / "eval_runs").glob("*.json"))) == 2
