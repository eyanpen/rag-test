#!/usr/bin/env python3
"""Re-run evaluation only: read existing predictions, recompute metrics, regenerate reports."""
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "GraphRAG-Benchmark"))

import litellm
litellm.drop_params = True

from models import (
    BenchmarkConfig, BenchmarkSummary, DatasetPhaseResult,
    EMBEDDING_MODELS, EmbeddingModelConfig, EvaluationResult,
    ModelResult, PredictionItem, sanitize_name,
)
from report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluations")

MODEL_MAP = {sanitize_name(m.name): m for m in EMBEDDING_MODELS}


async def evaluate_predictions(
    config: BenchmarkConfig,
    predictions: List[PredictionItem],
    model_name: str,
    dataset_name: str,
) -> EvaluationResult:
    from Evaluation.metrics.rouge import compute_rouge_score
    from Evaluation.metrics.answer_accuracy import compute_answer_correctness
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    t0 = time.time()
    rouge_scores, acc_scores = [], []

    eval_llm = ChatOpenAI(model="openai/gpt-oss-120b", base_url=config.api_base_url, api_key="no-key")
    eval_emb = OpenAIEmbeddings(
        model="BAAI/bge-m3", base_url=config.api_base_url,
        api_key="no-key", check_embedding_ctx_length=False,
    )

    for p in predictions:
        if p.error or not p.generated_answer.strip():
            continue
        try:
            rouge_scores.append(await compute_rouge_score(p.generated_answer, p.ground_truth))
        except Exception as e:
            log.debug(f"ROUGE failed for {p.id}: {e}")
            rouge_scores.append(float("nan"))
        try:
            score = await compute_answer_correctness(p.question, p.generated_answer, p.ground_truth, eval_llm, eval_emb)
            acc_scores.append(score)
            log.info(f"  {p.id}: answer_correctness={score:.4f}")
        except Exception as e:
            log.warning(f"  {p.id}: answer_correctness FAILED: {e}")
            acc_scores.append(float("nan"))

    def safe_mean(vals):
        clean = [v for v in vals if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    return EvaluationResult(
        model_name=model_name, dataset_name=dataset_name,
        rouge_l=safe_mean(rouge_scores), answer_correctness=safe_mean(acc_scores),
        eval_time_seconds=time.time() - t0,
    )


async def main():
    config = BenchmarkConfig(datasets=["medical", "novel"])
    summary = BenchmarkSummary(config=config, start_time=datetime.now().isoformat())

    # Restore dataset phases from old summary
    old_summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if os.path.exists(old_summary_path):
        with open(old_summary_path) as f:
            old = json.load(f)
        for dp in old.get("dataset_phases", []):
            summary.dataset_phases.append(DatasetPhaseResult(
                dataset_name=dp["dataset"], phase_1_9_time_seconds=dp["phase_1_9_time"],
                output_dir=dp["output_dir"], error=dp["error"],
            ))
        # Restore timing from old results
        old_results = {(r["model_name"], r["dataset"]): r for r in old.get("results", [])}
    else:
        old_results = {}

    pred_files = sorted(Path(PRED_DIR).glob("*.json"))
    log.info(f"Found {len(pred_files)} prediction files")

    for pf in pred_files:
        stem = pf.stem  # e.g. "baai-bge-m3__medical"
        parts = stem.split("__", 1)
        if len(parts) != 2:
            continue
        model_key, ds_name = parts
        model_cfg = MODEL_MAP.get(model_key)
        if not model_cfg:
            log.warning(f"Unknown model key: {model_key}, skipping")
            continue

        with open(pf) as f:
            raw = json.load(f)
        predictions = [PredictionItem(**item) for item in raw]

        mr = ModelResult(model=model_cfg, dataset_name=ds_name, predictions=predictions)

        # Restore old timing
        old_r = old_results.get((model_cfg.name, ds_name), {})
        mr.embedding_time_seconds = old_r.get("embedding_time", 0.0)
        mr.query_time_seconds = old_r.get("query_time", 0.0)

        log.info(f"=== Evaluating {model_cfg.display_name} × {ds_name} ({len(predictions)} predictions) ===")
        try:
            mr.evaluation = await evaluate_predictions(config, predictions, model_cfg.display_name, ds_name)
            eval_path = os.path.join(EVAL_DIR, f"{model_key}__{ds_name}.json")
            with open(eval_path, "w") as f:
                json.dump(asdict(mr.evaluation), f, indent=2, ensure_ascii=False)
            ac = mr.evaluation.answer_correctness
            ac_str = f"{ac:.4f}" if not math.isnan(ac) else "NaN"
            log.info(f"  → rouge_l={mr.evaluation.rouge_l:.4f}, answer_correctness={ac_str}")
        except Exception as e:
            log.error(f"  Evaluation failed: {e}")
            mr.error = str(e)

        summary.model_results.append(mr)

    summary.end_time = datetime.now().isoformat()
    summary.total_time_seconds = time.time() - time.time()  # not meaningful for re-eval

    report = ReportGenerator.generate_markdown(summary)
    with open(os.path.join(OUTPUT_DIR, "benchmark_report.md"), "w") as f:
        f.write(report)

    summary_data = ReportGenerator.generate_summary_json(summary)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    log.info("Done! Reports regenerated.")


if __name__ == "__main__":
    asyncio.run(main())
