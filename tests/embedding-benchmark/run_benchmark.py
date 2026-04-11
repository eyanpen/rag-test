#!/usr/bin/env python3
"""Dual-tower retrieval benchmark runner.

Entry point that orchestrates dataset loading, model filtering, embedding,
metric computation, and report generation.

Usage:
    python tests/embedding-benchmark/run_benchmark.py
    python tests/embedding-benchmark/run_benchmark.py --models "BAAI/bge-m3,intfloat/e5-mistral-7b-instruct"
    python tests/embedding-benchmark/run_benchmark.py --output-dir /tmp/benchmark-results
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime

# Ensure project root is on sys.path so `tests.models` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.models import EMBEDDING_MODELS, EmbeddingModelConfig  # noqa: E402
from embedding_client import EmbeddingClient, EmbeddingAPIError  # noqa: E402
from retrieval_evaluator import RetrievalEvaluator  # noqa: E402
from report_generator import ReportGenerator  # noqa: E402


logger = logging.getLogger("embedding_benchmark")

API_BASE = "http://10.210.156.69:8633"


def _setup_logging(output_dir: str) -> None:
    """Configure logging to stdout and benchmark.log in *output_dir*."""
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File handler — benchmark.log inside output dir
    log_path = os.path.join(output_dir, "benchmark.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run dual-tower retrieval benchmark on embedding models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to evaluate (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("tests", "embedding-benchmark"),
        help="Directory for output files (default: tests/embedding-benchmark/).",
    )
    return parser.parse_args()


def _load_dataset(output_dir: str) -> list:
    """Load dataset.json from the script directory or *output_dir*.

    Looks in the same directory as this script first, then falls back to
    *output_dir*.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "dataset.json"),
        os.path.join(output_dir, "dataset.json"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info("Loaded dataset from %s (%d items)", path, len(dataset))
            return dataset

    logger.error("dataset.json not found in %s", " or ".join(candidates))
    sys.exit(1)


def _filter_models(
    models: list, model_filter: str | None
) -> list:
    """Return models whose name appears in the comma-separated *model_filter*.

    If *model_filter* is None, return all models unchanged.
    """
    if model_filter is None:
        return list(models)

    requested = {name.strip() for name in model_filter.split(",")}
    filtered = [m for m in models if m.name in requested]

    if not filtered:
        logger.error(
            "No matching models found for --models=%s. Available: %s",
            model_filter,
            ", ".join(m.name for m in models),
        )
        sys.exit(1)

    logger.info(
        "Filtered to %d model(s): %s",
        len(filtered),
        ", ".join(m.display_name for m in filtered),
    )
    return filtered


def _evaluate_model_mode(
    client: EmbeddingClient,
    model: EmbeddingModelConfig,
    dataset: list,
    mode: str,
    query_prefix: str,
    doc_prefix: str,
) -> dict:
    """Run retrieval evaluation for a single model in a single mode.

    Returns a ModelModeResult dict.
    """
    ranks: list[int] = []
    difficulties: list[str] = []
    total_candidates_per_query: list[int] = []

    start_time = time.time()

    for idx, item in enumerate(dataset):
        try:
            query = item["query"]
            relevant_doc = item["relevant_doc"]
            distractor_docs = item["distractor_docs"]

            # Build candidates: relevant_doc first, then distractors
            candidates = [relevant_doc] + distractor_docs
            relevant_index = 0

            # Embed query
            query_emb = client.get_embeddings(model.name, [query], prefix=query_prefix)[0]

            # Embed candidates
            candidate_embs = client.get_embeddings(model.name, candidates, prefix=doc_prefix)

            # Rank and record
            rank = RetrievalEvaluator.rank_candidates(query_emb, candidate_embs, relevant_index)
            ranks.append(rank)
            difficulties.append(item["difficulty"])
            total_candidates_per_query.append(len(candidates))

        except Exception as exc:
            logger.warning(
                "Error evaluating item %d for %s (%s mode): %s — skipping item",
                idx, model.display_name, mode, exc,
            )
            continue

    embedding_time = time.time() - start_time

    # Compute metrics
    if ranks:
        metrics_by_difficulty = RetrievalEvaluator.compute_metrics_by_difficulty(
            ranks, total_candidates_per_query, difficulties
        )
        overall_metrics = metrics_by_difficulty["overall"]
        status = "completed"
        error_message = None
    else:
        overall_metrics = {"mrr": 0.0, "recall_at_1": 0.0, "recall_at_5": 0.0, "ndcg_at_10": 0.0}
        metrics_by_difficulty = {
            "easy": dict(overall_metrics), "medium": dict(overall_metrics),
            "hard": dict(overall_metrics), "overall": dict(overall_metrics),
        }
        status = "error"
        error_message = "All items failed during evaluation"

    return {
        "model_display_name": model.display_name,
        "model_name": model.name,
        "mode": mode,
        "metrics": overall_metrics,
        "metrics_by_difficulty": metrics_by_difficulty,
        "embedding_time_seconds": round(embedding_time, 2),
        "status": status,
        "error_message": error_message,
    }


def _compute_delta(
    model: EmbeddingModelConfig, sym_result: dict, dt_result: dict
) -> dict:
    """Compute delta row: dual-tower metrics minus symmetric metrics."""
    sym_m = sym_result["metrics"]
    dt_m = dt_result["metrics"]
    return {
        "model_display_name": model.display_name,
        "model_name": model.name,
        "delta_mrr": dt_m["mrr"] - sym_m["mrr"],
        "delta_recall_at_1": dt_m["recall_at_1"] - sym_m["recall_at_1"],
        "delta_recall_at_5": dt_m["recall_at_5"] - sym_m["recall_at_5"],
        "delta_ndcg_at_10": dt_m["ndcg_at_10"] - sym_m["ndcg_at_10"],
    }


def _log_mode_summary(model: EmbeddingModelConfig, mode: str, result: dict) -> None:
    """Print progress summary after a model-mode evaluation completes."""
    m = result["metrics"]
    logger.info(
        "  %s [%s] — MRR=%.4f  R@1=%.4f  R@5=%.4f  NDCG@10=%.4f  (%.1fs)",
        model.display_name, mode,
        m["mrr"], m["recall_at_1"], m["recall_at_5"], m["ndcg_at_10"],
        result["embedding_time_seconds"],
    )


def main() -> None:
    """CLI entry point for the dual-tower retrieval benchmark."""
    args = _parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Logging — stdout + benchmark.log
    _setup_logging(args.output_dir)

    logger.info("=== Dual-Tower Retrieval Benchmark ===")

    # Load evaluation dataset
    dataset = _load_dataset(args.output_dir)

    # Load and filter models
    models = _filter_models(EMBEDDING_MODELS, args.models)
    logger.info("Models to evaluate: %d", len(models))

    # --- Per-model evaluation loop (Task 7.2) ---
    client = EmbeddingClient(api_base=API_BASE)
    all_results: list[dict] = []
    all_deltas: list[dict] = []
    skipped_models: list[dict] = []

    for model in models:
        # --- Health check ---
        try:
            logger.info("Health-checking model: %s", model.display_name)
            client.get_embeddings(model.name, ["hello"])
            logger.info("Health check passed: %s", model.display_name)
        except Exception as exc:
            reason = str(exc)
            logger.warning("Health check FAILED for %s: %s", model.display_name, reason)
            skipped_models.append({"model": model.display_name, "reason": reason})
            continue

        # --- Symmetric mode evaluation ---
        sym_result = _evaluate_model_mode(
            client, model, dataset, mode="symmetric", query_prefix="", doc_prefix=""
        )
        all_results.append(sym_result)
        _log_mode_summary(model, "symmetric", sym_result)

        # --- Dual-tower mode evaluation ---
        dt_result = None
        if model.supports_dual_tower:
            dt_result = _evaluate_model_mode(
                client, model, dataset,
                mode="dual-tower",
                query_prefix=model.query_prefix,
                doc_prefix=model.document_prefix,
            )
            all_results.append(dt_result)
            _log_mode_summary(model, "dual-tower", dt_result)

            # Compute delta row
            delta = _compute_delta(model, sym_result, dt_result)
            all_deltas.append(delta)

        logger.info("Completed evaluation for model: %s", model.display_name)

    # --- Results assembly and output ---

    # Exit non-zero if all models were skipped (no completed results)
    completed_results = [r for r in all_results if r["status"] == "completed"]
    if not completed_results and skipped_models:
        logger.error("All models failed. No results to report.")
        sys.exit(1)

    # Rank all model-mode results by MRR descending
    sorted_results = sorted(
        all_results,
        key=lambda r: r["metrics"]["mrr"],
        reverse=True,
    )

    # Build dataset_stats
    difficulty_distribution = dict(Counter(item["difficulty"] for item in dataset))
    dataset_stats = {
        "total_items": len(dataset),
        "domains": sorted(set(item["domain"] for item in dataset)),
        "difficulty_distribution": difficulty_distribution,
    }

    # Build BenchmarkResults
    benchmark_results = {
        "results": sorted_results,
        "deltas": all_deltas,
        "skipped_models": skipped_models,
        "timestamp": datetime.now().isoformat(),
        "dataset_stats": dataset_stats,
    }

    # Save results.json
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Generate report.md via ReportGenerator
    rg = ReportGenerator(
        api_base=API_BASE,
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    )
    report = rg.generate_report(benchmark_results)
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Print output file paths
    print(f"\nBenchmark complete. Output files:")
    print(f"  Results: {results_path}")
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()
