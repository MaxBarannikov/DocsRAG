"""Evaluation harness: golden dataset → RAGPipeline → Ragas metrics → MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import yaml
from loguru import logger
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

# Ragas uses httpx internally; unset SOCKS proxy if set to avoid import errors
for _var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_var, None)

from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import settings  # noqa: E402
from api.rag import RAGPipeline  # noqa: E402

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "docsrag-rag-eval"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_dataset(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        data = json.load(f)
    logger.info("Loaded {} golden samples from {}", len(data), path)
    return data


def build_pipeline(config: dict[str, Any]):
    strategy = config.get("retrieval_strategy", "dense")
    if strategy == "agentic":
        from api.graph import AgentPipeline

        base = RAGPipeline(retrieval_strategy="dense")
        return AgentPipeline(base)
    return RAGPipeline(retrieval_strategy=strategy)


def run_pipeline(
    pipeline: RAGPipeline,
    samples: list[dict[str, str]],
    top_k: int,
    rerank_top_n: int = 20,
) -> list[SingleTurnSample]:
    """Run every question through the pipeline, collect answers and contexts."""
    results: list[SingleTurnSample] = []
    for i, sample in enumerate(samples, start=1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        logger.info("[{}/{}] {}", i, len(samples), question[:80])

        answer, sources, timings = pipeline.ask(
            question=question,
            top_k=top_k,
            include_contexts=True,
            rerank_top_n=rerank_top_n,
        )
        contexts = [s.content for s in sources if s.content]

        results.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
        logger.debug(
            "  answer_len={} contexts={} retrieval={}ms generation={}ms",
            len(answer),
            len(contexts),
            timings["retrieval_ms"],
            timings["generation_ms"],
        )

    return results


def build_ragas_llm(config: dict[str, Any]) -> LangchainLLMWrapper:
    llm = ChatOllama(
        model=config["llm_model"],
        base_url=settings.ollama_base_url,
        temperature=0.0,
        format="json",
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings(config: dict[str, Any]) -> LangchainEmbeddingsWrapper:
    embeddings = OllamaEmbeddings(
        model=config["llm_model"],
        base_url=settings.ollama_base_url,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def compute_metrics(
    ragas_samples: list[SingleTurnSample],
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: LangchainEmbeddingsWrapper,
) -> dict[str, float]:
    dataset = EvaluationDataset(samples=ragas_samples)

    # ragas 0.2.x: metrics are module-level singletons, configure in place.
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # max_workers=1: Ollama is single-threaded; parallelism only causes timeouts.
    run_config = RunConfig(timeout=180, max_retries=3, max_workers=1)

    logger.info("Running Ragas evaluation on {} samples...", len(ragas_samples))
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    scores: dict[str, float] = {}
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            scores[col] = float(df[col].mean())

    return scores


def log_to_mlflow(
    config: dict[str, Any],
    scores: dict[str, float],
    config_path: str,
    n_samples: int,
) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "chunk_size": config["chunk_size"],
                "chunk_overlap": config["chunk_overlap"],
                "top_k": config["top_k"],
                "embedding_model": config["embedding_model"],
                "embedder_backend": config.get("embedder_backend", settings.embedder_backend),
                "llm_model": config["llm_model"],
                "prompt_version": config["prompt_version"],
                "retrieval_strategy": config.get("retrieval_strategy", "dense"),
                "rerank_top_n": config.get("rerank_top_n", 20),
                "n_samples": n_samples,
                "config_file": Path(config_path).name,
            }
        )
        mlflow.log_metrics(scores)
        run_url = f"{MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    return run_url


def main() -> None:
    import time

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info("Config: {}", config)

    if "embedder_backend" in config:
        settings.embedder_backend = config["embedder_backend"]
        logger.info(
            "Override embedder_backend={} | active_qdrant_collection={}",
            settings.embedder_backend,
            settings.active_qdrant_collection,
        )

    t_start = time.perf_counter()

    samples = load_dataset(GOLDEN_DATASET_PATH)
    pipeline = build_pipeline(config)
    ragas_llm = build_ragas_llm(config)
    ragas_embeddings = build_ragas_embeddings(config)

    rerank_top_n = config.get("rerank_top_n", 20)
    ragas_samples = run_pipeline(pipeline, samples, top_k=config["top_k"], rerank_top_n=rerank_top_n)
    scores = compute_metrics(ragas_samples, ragas_llm, ragas_embeddings)

    scores["eval_time_sec"] = round(time.perf_counter() - t_start, 1)

    run_url = log_to_mlflow(config, scores, str(config_path), len(samples))

    print("\n" + "=" * 60)
    print(f"Config:  {config_path.name}")
    print(f"Samples: {len(samples)}")
    print("-" * 60)
    for metric, value in scores.items():
        print(f"  {metric:<25} {value:.4f}")
    print("=" * 60)
    print(f"MLflow run: {run_url}")


if __name__ == "__main__":
    main()
