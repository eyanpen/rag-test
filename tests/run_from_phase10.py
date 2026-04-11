#!/usr/bin/env python3
"""从 Phase 10 开始执行 Embedding 基准测试（跳过 Phase 1-9）。

前提：Phase 1-9 的 parquet 产物已存在于 workspace/{dataset}/output/ 下。
流程：Phase 10 × N 模型 → graph sync → query → evaluate → report
"""
import asyncio
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import litellm
litellm.drop_params = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "GraphRAG-Benchmark"))

from models import (
    BenchmarkConfig, BenchmarkSummary, DatasetPhaseResult,
    EMBEDDING_MODELS, ModelResult, make_graph_name, sanitize_name,
)
from concurrency import AdaptiveConcurrencyController
from dataset_loader import DatasetLoader
from graph_sync import sync_graph_to_falkordb
from report_generator import ReportGenerator
from run_embedding_benchmark import (
    run_phase_10, check_model_availability, run_queries,
    evaluate_predictions,
)

log = logging.getLogger(__name__)


async def async_main(config: BenchmarkConfig):
    summary = BenchmarkSummary(config=config, start_time=datetime.now().isoformat())
    t_start = time.time()

    os.makedirs(os.path.join(config.output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "evaluations"), exist_ok=True)

    data_root = config.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_DIR, data_root)

    # ── Step 1: Load questions & verify Phase 1-9 parquet exists ──
    dataset_questions: Dict[str, List[Dict[str, str]]] = {}
    for ds_name in config.datasets:
        workspace_dir = os.path.join(config.output_dir, "workspace", ds_name)
        output_dir = os.path.join(workspace_dir, "output")

        required = ["entities.parquet", "relationships.parquet", "text_units.parquet",
                     "community_reports.parquet", "communities.parquet"]
        missing = [f for f in required if not os.path.isfile(os.path.join(output_dir, f))]
        if missing:
            log.error(f"Dataset {ds_name}: missing Phase 1-9 parquet: {missing}")
            summary.dataset_phases.append(DatasetPhaseResult(
                dataset_name=ds_name, phase_1_9_time_seconds=0,
                output_dir=output_dir, error=f"Missing parquet: {missing}",
            ))
            continue

        summary.dataset_phases.append(DatasetPhaseResult(
            dataset_name=ds_name, phase_1_9_time_seconds=0,
            output_dir=output_dir,
        ))

        input_dir = os.path.join(workspace_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        questions = None
        if ds_name == "medical":
            questions = DatasetLoader.prepare_medical(data_root, input_dir)
        elif ds_name == "novel":
            questions = DatasetLoader.prepare_novel(data_root, input_dir)

        if questions is None:
            log.warning(f"Dataset {ds_name}: questions load failed, skipping")
            continue

        if config.sample_size and config.sample_size < len(questions):
            questions = questions[:config.sample_size]
        dataset_questions[ds_name] = questions
        log.info(f"Dataset {ds_name}: {len(questions)} questions ready")

    # ── Step 2: Model availability check ──
    available_models = []
    for model in config.models:
        if check_model_availability(config.api_base_url, model):
            available_models.append(model)
        else:
            log.warning(f"Model {model.display_name} unavailable, skipping")
            for ds_name in dataset_questions:
                summary.model_results.append(ModelResult(
                    model=model, dataset_name=ds_name, error="Model unavailable",
                ))

    # ── Step 3: Phase 10 + Query + Evaluate ──
    for ds_name, questions in dataset_questions.items():
        workspace_dir = os.path.join(config.output_dir, "workspace", ds_name)

        for model in available_models:
            mr = ModelResult(model=model, dataset_name=ds_name)
            log.info(f"=== {model.display_name} × {ds_name} ===")

            try:
                log.info(f"Phase 10: embedding with {model.display_name}")
                mr.embedding_time_seconds = await run_phase_10(config, ds_name, workspace_dir, model)
            except Exception as e:
                log.error(f"Phase 10 failed for {model.display_name}/{ds_name}: {e}")
                mr.error = str(e)
                summary.model_results.append(mr)
                continue

            try:
                sync_graph_to_falkordb(
                    output_dir=os.path.join(workspace_dir, "output"),
                    host=config.falkordb_host, port=config.falkordb_port,
                    graph_name=make_graph_name(model.name, ds_name),
                )
            except Exception as e:
                log.warning(f"Graph sync failed: {e}")

            try:
                log.info(f"Querying {len(questions)} questions")
                t0 = time.time()
                mr.predictions = await run_queries(config, ds_name, workspace_dir, model, questions)
                mr.query_time_seconds = time.time() - t0
            except Exception as e:
                log.error(f"Query failed: {e}")
                mr.error = str(e)
                summary.model_results.append(mr)
                continue

            safe = sanitize_name(model.name)
            pred_path = os.path.join(config.output_dir, "predictions", f"{safe}__{ds_name}.json")
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump([asdict(p) for p in mr.predictions], f, indent=2, ensure_ascii=False)

            if mr.predictions:
                try:
                    mr.evaluation, question_details = await evaluate_predictions(
                        config, mr.predictions, model.display_name, ds_name)
                    eval_path = os.path.join(config.output_dir, "evaluations", f"{safe}__{ds_name}.json")
                    with open(eval_path, "w", encoding="utf-8") as f:
                        json.dump(asdict(mr.evaluation), f, indent=2, ensure_ascii=False)
                    details_path = os.path.join(config.output_dir, "evaluations", f"{safe}__{ds_name}_details.json")
                    with open(details_path, "w", encoding="utf-8") as f:
                        json.dump(question_details, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    log.error(f"Evaluation failed: {e}")

            summary.model_results.append(mr)

    # ── Step 4: Report ──
    summary.end_time = datetime.now().isoformat()
    summary.total_time_seconds = time.time() - t_start

    report = ReportGenerator.generate_markdown(summary)
    report_path = os.path.join(config.output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report: {report_path}")

    summary_data = ReportGenerator.generate_summary_json(summary)
    summary_path = os.path.join(config.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    log.info(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="从 Phase 10 开始执行 Embedding 基准测试")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--dataset", default="all", choices=["medical", "novel", "all"])
    parser.add_argument("--models", default="", help="逗号分隔模型名，默认全部")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "benchmark_results"))
    parser.add_argument("--data-root", default=os.path.join(PROJECT_DIR, "GraphRAG-Benchmark", "Datasets"))
    parser.add_argument("--graphrag-root", default="/home/eyanpen/sourceCode/rnd-ai-engine-features/graphrag")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "benchmark.log"), mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logging.getLogger("LiteLLM").setLevel(logging.INFO)
    logging.getLogger("litellm").setLevel(logging.INFO)

    models = list(EMBEDDING_MODELS)
    if args.models:
        names = [n.strip() for n in args.models.split(",")]
        models = [m for m in EMBEDDING_MODELS if m.name in names]

    datasets = ["medical", "novel"] if args.dataset == "all" else [args.dataset]

    config = BenchmarkConfig(
        sample_size=args.sample,
        datasets=datasets,
        models=models,
        output_dir=args.output_dir,
        data_root=args.data_root,
        graphrag_root=args.graphrag_root,
    )

    ac = AdaptiveConcurrencyController()
    ac.install_hooks()

    asyncio.run(async_main(config))


if __name__ == "__main__":
    main()
