"""Evaluation harness: golden dataset -> pipeline -> Ragas metrics -> MLflow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.proxy import strip_socks_proxy_env

# Ragas and langchain-ollama build httpx clients at import time; see core.proxy.
strip_socks_proxy_env()

import yaml  # noqa: E402 — after the proxy cleanup above
from loguru import logger  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from api.prompts import prompt_version  # noqa: E402
from api.rag import RAGPipeline  # noqa: E402
from core.config import settings  # noqa: E402
from core.health import (  # noqa: E402
    DependencyUnavailableError,
    check_mlflow,
    check_ollama,
    check_qdrant,
)
from evaluation.config import EvalConfig  # noqa: E402
from evaluation.ragas_embeddings import ProjectEmbeddings  # noqa: E402

if TYPE_CHECKING:
    from ragas.dataset_schema import SingleTurnSample

    from core.types import AskablePipeline

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
# Recorded with every run: faithfulness, answer_relevancy and context_recall are
# LLM-judged, so a version bump in this stack shifts them even when retrieval is
# byte-identical. Without it, two runs cannot be told apart from a real regression.
JUDGE_STACK_PACKAGES = ("ragas", "langchain-core", "langchain-ollama")
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
RAGAS_TIMEOUT_SECONDS = 180
RAGAS_MAX_RETRIES = 3


def require_eval_extra() -> None:
    """ragas and mlflow live in the [eval] extra, so the CLI can be missing them."""
    from importlib.util import find_spec

    missing = [name for name in ("ragas", "mlflow") if find_spec(name) is None]
    if missing:
        msg = f"Missing evaluation dependencies: {', '.join(missing)}.\nInstall them with `make install`."
        raise DependencyUnavailableError(msg)


def judge_stack_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in JUDGE_STACK_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def load_config(path: Path) -> EvalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path} does not contain a YAML mapping."
        raise TypeError(msg)
    return EvalConfig.model_validate(raw)


def load_dataset(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        msg = f"{path} must contain a non-empty list of samples."
        raise ValueError(msg)
    for i, sample in enumerate(data):
        missing = {"question", "ground_truth"} - set(sample)
        if missing:
            msg = f"Sample {i} in {path} is missing required keys: {sorted(missing)}."
            raise ValueError(msg)
    logger.info("Loaded {} golden samples from {}", len(data), path)
    return data


def warn_on_chunking_mismatch(pipeline: RAGPipeline, config: EvalConfig) -> None:
    """Without this, a run against a differently-chunked collection would still log the
    config's values to MLflow, corrupting the experiment record.
    """
    collection = settings.active_qdrant_collection
    try:
        points, _ = pipeline._qdrant_client.scroll(  # noqa: SLF001 — diagnostic read, not part of the pipeline API
            collection_name=collection, limit=1, with_payload=True, with_vectors=False
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic check, never fatal
        logger.warning("Could not verify chunking parameters of '{}': {}", collection, exc)
        return

    if not points:
        logger.warning("Collection '{}' is empty — run `make reindex` first.", collection)
        return

    payload = points[0].payload or {}
    indexed_size, indexed_overlap = payload.get("chunk_size"), payload.get("chunk_overlap")
    if indexed_size is None:
        logger.warning(
            "Collection '{}' predates chunking metadata; cannot verify that it matches "
            "chunk_size={} / chunk_overlap={}. Re-index to enable this check.",
            collection,
            config.chunk_size,
            config.chunk_overlap,
        )
        return

    if (indexed_size, indexed_overlap) != (config.chunk_size, config.chunk_overlap):
        logger.error(
            "CHUNKING MISMATCH: '{}' was built with chunk_size={} overlap={}, but the config "
            "declares {}/{}. The metrics below would be logged under the wrong parameters. "
            "Re-index with `make reindex CHUNK_SIZE={} CHUNK_OVERLAP={}` before trusting this run.",
            collection,
            indexed_size,
            indexed_overlap,
            config.chunk_size,
            config.chunk_overlap,
            config.chunk_size,
            config.chunk_overlap,
        )


def build_pipeline(config: EvalConfig) -> tuple[AskablePipeline, RAGPipeline]:
    """Returns the pipeline under test alongside the underlying RAGPipeline."""
    if config.retrieval_strategy == "agentic":
        from api.graph import AgentPipeline

        base = RAGPipeline(retrieval_strategy="dense")
        return AgentPipeline(base), base

    base = RAGPipeline(retrieval_strategy=config.retrieval_strategy)
    return base, base


def run_pipeline(
    pipeline: AskablePipeline,
    samples: list[dict[str, str]],
    config: EvalConfig,
) -> list[SingleTurnSample]:

    from ragas.dataset_schema import SingleTurnSample

    results: list[SingleTurnSample] = []
    failures = 0

    for i, sample in enumerate(samples, start=1):
        question = sample["question"]
        logger.info("[{}/{}] {}", i, len(samples), question[:80])
        try:
            answer, sources, timings = pipeline.ask(
                question=question,
                top_k=config.top_k,
                include_contexts=True,
                rerank_top_n=config.rerank_top_n,
            )
        except Exception:  # noqa: BLE001 — one bad question must not discard the whole run
            logger.exception("Sample {} failed; continuing with the remaining questions", i)
            failures += 1
            continue

        contexts = [s.content for s in sources if s.content]
        if not contexts:
            logger.warning("Sample {} retrieved no contexts; its context metrics will be undefined", i)

        results.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=sample["ground_truth"],
            )
        )
        logger.debug(
            "  answer_len={} contexts={} retrieval={}ms generation={}ms",
            len(answer),
            len(contexts),
            timings.get("retrieval_ms", 0),
            timings.get("generation_ms", 0),
        )

    if failures:
        logger.error("{} of {} samples failed and are excluded from the metrics", failures, len(samples))
    return results


def compute_metrics(
    ragas_samples: list[SingleTurnSample],
    config: EvalConfig,
    embedder_pipeline: RAGPipeline,
) -> tuple[dict[str, float], dict[str, int]]:
    """Returns the mean per metric, and how many samples each mean was taken over.

    A metric can fail on individual samples when the judge returns unparseable
    output, so the counts are what makes a mean interpretable.
    """
    from langchain_ollama import ChatOllama  # judge model, independent of the pipeline backend
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    # Without format="json", Qwen answers Ragas's structured prompts in prose.
    judge = LangchainLLMWrapper(
        ChatOllama(
            model=config.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            format="json",
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        ProjectEmbeddings(embedder_pipeline._embedder)  # noqa: SLF001 — reuse the already-loaded encoder
    )

    # ragas 0.2.x exposes metrics as module-level singletons.
    faithfulness.llm = judge
    answer_relevancy.llm = judge
    answer_relevancy.embeddings = embeddings
    context_precision.llm = judge
    context_recall.llm = judge

    logger.info("Running Ragas evaluation on {} samples...", len(ragas_samples))
    result = evaluate(
        dataset=EvaluationDataset(samples=ragas_samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        # Ollama serves one request at a time; parallel workers only produce timeouts.
        run_config=RunConfig(timeout=RAGAS_TIMEOUT_SECONDS, max_retries=RAGAS_MAX_RETRIES, max_workers=1),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in METRIC_NAMES:
        if name not in df.columns:
            logger.error("Metric {!r} produced no results at all and is omitted", name)
            continue
        column = df[name]
        n_scored = int(column.notna().sum())
        if n_scored < len(column):
            logger.warning("Metric {!r}: scored {}/{} samples", name, n_scored, len(column))
        if n_scored == 0:
            logger.error("Metric {!r} failed on every sample and is omitted", name)
            continue
        scores[name] = float(column.mean())
        counts[name] = n_scored
    return scores, counts


def save_results(
    config: EvalConfig,
    scores: dict[str, float],
    counts: dict[str, int],
    config_path: Path,
    n_samples: int,
) -> Path:
    """Write the run to disk before MLflow is touched, so a tracking outage cannot
    discard a run that took minutes to produce.
    """
    payload = {
        "config_file": config_path.name,
        "finished_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "n_samples": n_samples,
        "prompt_version": prompt_version(),
        "embedder_backend": config.embedder_backend or settings.embedder_backend,
        "judge_stack": judge_stack_versions(),
        "config": config.model_dump(),
        "scores": scores,
        "scored_samples": counts,
    }
    out_dir = settings.data_dir / "eval_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{config_path.stem}-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_report(
    config_path: Path,
    scores: dict[str, float],
    counts: dict[str, int],
    n_used: int,
    n_total: int,
) -> None:
    print("\n" + "=" * 60)  # noqa: T201 — CLI report
    print(f"Config:   {config_path.name}")  # noqa: T201
    print(f"Samples:  {n_used} of {n_total}")  # noqa: T201
    print("-" * 60)  # noqa: T201
    for metric, value in scores.items():
        scored = counts.get(metric)
        suffix = f"   ({scored}/{n_used} scored)" if scored is not None and scored < n_used else ""
        print(f"  {metric:<25} {value:.4f}{suffix}")  # noqa: T201
    print("=" * 60)  # noqa: T201


def log_to_mlflow(
    config: EvalConfig,
    scores: dict[str, float],
    counts: dict[str, int],
    config_path: Path,
    n_samples: int,
) -> str:
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "top_k": config.top_k,
                "embedding_model": config.embedding_model,
                "embedder_backend": config.embedder_backend or settings.embedder_backend,
                "llm_model": config.llm_model,
                "prompt_version": prompt_version(),
                "retrieval_strategy": config.retrieval_strategy,
                "rerank_top_n": config.rerank_top_n,
                "n_samples": n_samples,
                "config_file": config_path.name,
                **judge_stack_versions(),
            }
        )
        mlflow.log_metrics(scores)
        mlflow.log_metrics({f"{name}_scored_samples": n for name, n in counts.items()})
        return f"{settings.mlflow_tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config file")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 — surface any config problem as a clean message
        logger.error("Invalid config {}: {}", args.config, exc)
        return 1

    logger.info("Config: {}", config.model_dump())

    if config.embedder_backend:
        settings.embedder_backend = config.embedder_backend  # type: ignore[assignment]
        logger.info(
            "Override embedder_backend={} | active collection={}",
            settings.embedder_backend,
            settings.active_qdrant_collection,
        )

    started = time.perf_counter()
    samples = load_dataset(GOLDEN_DATASET_PATH)

    try:
        # Before the pipeline, which loads the embedding model. Ollama is always
        # needed: it is the Ragas judge even when the pipeline runs on vLLM.
        require_eval_extra()
        check_ollama()
        check_mlflow()
        check_qdrant(QdrantClient(url=settings.qdrant_url, check_compatibility=False))
        pipeline, base_pipeline = build_pipeline(config)
    except DependencyUnavailableError as exc:
        logger.error("{}", exc)
        return 1

    warn_on_chunking_mismatch(base_pipeline, config)

    ragas_samples = run_pipeline(pipeline, samples, config)
    if not ragas_samples:
        logger.error("Every sample failed — nothing to evaluate.")
        return 1

    scores, counts = compute_metrics(ragas_samples, config, base_pipeline)
    if not scores:
        logger.error("No metric produced a usable score.")
        return 1

    scores["eval_time_sec"] = round(time.perf_counter() - started, 1)

    results_path = save_results(config, scores, counts, args.config, len(ragas_samples))
    print_report(args.config, scores, counts, len(ragas_samples), len(samples))
    print(f"Results:    {results_path}")  # noqa: T201

    try:
        run_url = log_to_mlflow(config, scores, counts, args.config, len(ragas_samples))
        print(f"MLflow run: {run_url}")  # noqa: T201
    except Exception as exc:  # noqa: BLE001 — the run is already saved; tracking is not worth failing over
        logger.error("Could not log to MLflow at {}: {}", settings.mlflow_tracking_uri, exc)
        logger.error("The scores above are saved at {} and can be re-logged later.", results_path)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
