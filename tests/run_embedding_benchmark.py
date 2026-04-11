#!/usr/bin/env python3
"""Embedding 模型基准测试执行器

基于 Microsoft GraphRAG pipeline 分割策略：
- Phase 1-9（文档加载→社区报告）每个数据集只执行一次
- Phase 10（向量嵌入）按 Embedding 模型数量执行 N 次
- 向量存入 FalkorDB，图命名为 <model>_<dataset>
"""
import asyncio
import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pandas as pd
import litellm
litellm.drop_params = True

# Add GraphRAG-Benchmark to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "GraphRAG-Benchmark"))

from models import (
    BenchmarkConfig,
    BenchmarkSummary,
    DatasetPhaseResult,
    EMBEDDING_MODELS,
    EmbeddingModelConfig,
    EvaluationResult,
    ModelResult,
    PredictionItem,
    make_graph_name,
    sanitize_name,
)
from concurrency import AdaptiveConcurrencyController
from dataset_loader import DatasetLoader
from graph_sync import sync_graph_to_falkordb
from report_generator import ReportGenerator

log = logging.getLogger(__name__)


# ── GraphRAG Pipeline Integration ──


def _build_graphrag_config(
    config: BenchmarkConfig,
    dataset_name: str,
    input_dir: str,
    output_dir: str,
    cache_dir: str,
    embedding_model_name: str = "BAAI/bge-m3",
    embedding_dim: int = 1024,
    graph_name: str = "default",
    prompts_dir: str | None = None,
) -> Any:
    """Build a GraphRagConfig programmatically for a dataset."""
    from graphrag_llm.config import ModelConfig
    from graphrag_storage import StorageConfig
    from graphrag_cache import CacheConfig
    from graphrag_storage.tables.table_provider_config import TableProviderConfig
    from graphrag_vectors import VectorStoreConfig, IndexSchema
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.config.models.extract_graph_config import ExtractGraphConfig
    from graphrag.config.models.summarize_descriptions_config import SummarizeDescriptionsConfig
    from graphrag.config.models.community_reports_config import CommunityReportsConfig
    from graphrag.config.embeddings import all_embeddings

    completion_models = {
        "default_completion_model": ModelConfig(
            model_provider="openai",
            model=config.llm_model,
            api_base=config.api_base_url,
            api_key="no-key",
        ),
    }
    embedding_models = {
        "default_embedding_model": ModelConfig(
            model_provider="openai",
            model=embedding_model_name,
            api_base=config.api_base_url,
            api_key="no-key",
        ),
    }

    # Build index_schema for all embedding types
    index_schema = {}
    for emb_name in all_embeddings:
        index_schema[emb_name] = IndexSchema(
            index_name=emb_name,
            vector_size=embedding_dim,
        )

    vector_store = VectorStoreConfig(
        type="falkordb",
        host=config.falkordb_host,
        port=config.falkordb_port,
        graph_name=graph_name,
        vector_size=embedding_dim,
        index_schema=index_schema,
    )

    # Resolve prompt file paths from prompts_dir if available
    def _prompt_path(filename: str) -> str | None:
        if prompts_dir:
            p = os.path.join(prompts_dir, filename)
            if os.path.isfile(p):
                return str(Path(p).resolve())
        return None

    grc = GraphRagConfig(
        completion_models=completion_models,
        embedding_models=embedding_models,
        input_storage=StorageConfig(base_dir=str(Path(input_dir).resolve())),
        output_storage=StorageConfig(base_dir=str(Path(output_dir).resolve())),
        update_output_storage=StorageConfig(base_dir=str(Path(output_dir).resolve()) + "/update"),
        cache=CacheConfig(storage=StorageConfig(base_dir=str(Path(cache_dir).resolve()))),
        table_provider=TableProviderConfig(),
        vector_store=vector_store,
        extract_graph=ExtractGraphConfig(
            prompt=_prompt_path("extract_graph.txt"),
        ),
        summarize_descriptions=SummarizeDescriptionsConfig(
            prompt=_prompt_path("summarize_descriptions.txt"),
        ),
        community_reports=CommunityReportsConfig(
            graph_prompt=_prompt_path("community_report_graph.txt"),
        ),
    )
    return grc


async def run_prompt_tune(config: BenchmarkConfig, workspace_dir: str) -> str:
    """Run prompt-tune for a dataset, save prompts to workspace_dir/prompts/.

    Returns the prompts directory path.
    """
    import graphrag.api as api

    prompts_dir = os.path.join(workspace_dir, "prompts")
    if os.path.isfile(os.path.join(prompts_dir, "extract_graph.txt")):
        log.info(f"Prompt-tune: reusing existing prompts in {prompts_dir}")
        return prompts_dir

    input_dir = os.path.join(workspace_dir, "input")
    output_dir = os.path.join(workspace_dir, "output")
    cache_dir = os.path.join(workspace_dir, "cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    grc = _build_graphrag_config(config, "", input_dir, output_dir, cache_dir)

    log.info("Prompt-tune: generating indexing prompts...")
    extract_prompt, summarize_prompt, community_prompt = await api.generate_indexing_prompts(
        config=grc,
        limit=15,
        selection_method=api.DocSelectionType.RANDOM,
        discover_entity_types=True,
    )

    os.makedirs(prompts_dir, exist_ok=True)
    for filename, content in [
        ("extract_graph.txt", extract_prompt),
        ("summarize_descriptions.txt", summarize_prompt),
        ("community_report_graph.txt", community_prompt),
    ]:
        with open(os.path.join(prompts_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    log.info(f"Prompt-tune: prompts saved to {prompts_dir}")
    return prompts_dir


async def run_phase_1_9(config: BenchmarkConfig, dataset_name: str, workspace_dir: str, prompts_dir: str | None = None) -> DatasetPhaseResult:
    """Execute Phase 1-9 (shared) for a dataset using GraphRAG pipeline."""
    from graphrag.index.workflows.factory import PipelineFactory
    from graphrag.index.run.run_pipeline import run_pipeline
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

    input_dir = os.path.join(workspace_dir, "input")
    output_dir = os.path.join(workspace_dir, "output")
    cache_dir = os.path.join(workspace_dir, "cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    _pre_embedding_workflows = [
        "create_base_text_units",
        "create_final_documents",
        "extract_graph",
        "finalize_graph",
        "extract_covariates",
        "create_communities",
        "create_final_text_units",
        "create_community_reports",
    ]
    PipelineFactory.register_pipeline(
        "pre_embedding",
        ["load_input_documents", *_pre_embedding_workflows],
    )

    grc = _build_graphrag_config(config, dataset_name, input_dir, output_dir, cache_dir, prompts_dir=prompts_dir)

    t0 = time.time()
    try:
        pipeline = PipelineFactory.create_pipeline(grc, method="pre_embedding")
        callbacks = NoopWorkflowCallbacks()
        async for result in run_pipeline(pipeline, grc, callbacks):
            if result.error:
                log.error(f"Phase 1-9 workflow '{result.workflow}' failed: {result.error}")
                return DatasetPhaseResult(
                    dataset_name=dataset_name,
                    phase_1_9_time_seconds=time.time() - t0,
                    output_dir=output_dir,
                    error=str(result.error),
                )
            log.info(f"Phase 1-9 workflow '{result.workflow}' completed")

        elapsed = time.time() - t0
        log.info(f"Phase 1-9 for {dataset_name} completed in {elapsed:.1f}s")
        return DatasetPhaseResult(
            dataset_name=dataset_name,
            phase_1_9_time_seconds=elapsed,
            output_dir=output_dir,
        )
    except Exception as e:
        log.error(f"Phase 1-9 for {dataset_name} failed: {e}")
        return DatasetPhaseResult(
            dataset_name=dataset_name,
            phase_1_9_time_seconds=time.time() - t0,
            output_dir=output_dir,
            error=str(e),
        )


async def run_phase_10(
    config: BenchmarkConfig,
    dataset_name: str,
    workspace_dir: str,
    model_config: EmbeddingModelConfig,
) -> float:
    """Execute Phase 10 (generate_text_embeddings) for a specific embedding model.

    Returns elapsed time in seconds.
    """
    from graphrag.index.workflows.generate_text_embeddings import run_workflow
    from graphrag.index.run.utils import create_run_context
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider
    from graphrag_cache import create_cache

    input_dir = os.path.join(workspace_dir, "input")
    output_dir = os.path.join(workspace_dir, "output")
    cache_dir = os.path.join(workspace_dir, "cache")

    emb_cache = os.path.join(cache_dir, "text_embedding")
    if os.path.isdir(emb_cache):
        shutil.rmtree(emb_cache)

    graph_name = make_graph_name(model_config.name, dataset_name)

    grc = _build_graphrag_config(
        config, dataset_name, input_dir, output_dir, cache_dir,
        embedding_model_name=model_config.name,
        embedding_dim=model_config.dim,
        graph_name=graph_name,
    )

    output_storage = create_storage(grc.output_storage)
    output_table_provider = create_table_provider(grc.table_provider, output_storage)
    cache = create_cache(grc.cache)

    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    context = create_run_context(
        output_storage=output_storage,
        output_table_provider=output_table_provider,
        cache=cache,
        callbacks=NoopWorkflowCallbacks(),
    )

    t0 = time.time()
    sys.path.insert(0, SCRIPT_DIR)
    import falkordb_vector_store  # noqa: F401 — registers "falkordb" type

    await run_workflow(grc, context)
    elapsed = time.time() - t0
    log.info(f"Phase 10 for {model_config.display_name}/{dataset_name} → graph '{graph_name}' in {elapsed:.1f}s")
    return elapsed


# ── Query & Evaluation ──


def check_model_availability(api_base_url: str, model: EmbeddingModelConfig) -> bool:
    """Test if an embedding model is available on the API server."""
    try:
        r = httpx.post(
            f"{api_base_url}/embeddings",
            json={"model": model.name, "input": "test"},
            timeout=30,
        )
        return r.status_code == 200
    except Exception as e:
        log.warning(f"Model {model.display_name} unavailable: {e}")
        return False


async def run_queries(
    config: BenchmarkConfig,
    dataset_name: str,
    workspace_dir: str,
    model_config: EmbeddingModelConfig,
    questions: List[Dict[str, str]],
) -> List[PredictionItem]:
    """Run GraphRAG local search queries for a model+dataset combination."""
    from graphrag.query.factory import get_local_search_engine
    from graphrag.query.indexer_adapters import (
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_text_units,
        read_indexer_reports,
    )

    output_dir = os.path.join(workspace_dir, "output")
    input_dir = os.path.join(workspace_dir, "input")
    cache_dir = os.path.join(workspace_dir, "cache")
    graph_name = make_graph_name(model_config.name, dataset_name)

    grc = _build_graphrag_config(
        config, dataset_name, input_dir, output_dir, cache_dir,
        embedding_model_name=model_config.name,
        embedding_dim=model_config.dim,
        graph_name=graph_name,
    )

    entities_df = pd.read_parquet(os.path.join(output_dir, "entities.parquet"))
    relationships_df = pd.read_parquet(os.path.join(output_dir, "relationships.parquet"))
    text_units_df = pd.read_parquet(os.path.join(output_dir, "text_units.parquet"))
    community_reports_df = pd.read_parquet(os.path.join(output_dir, "community_reports.parquet"))
    communities_df = pd.read_parquet(os.path.join(output_dir, "communities.parquet"))

    entities = read_indexer_entities(entities_df, communities_df, community_level=None)
    relationships = read_indexer_relationships(relationships_df)
    text_units = read_indexer_text_units(text_units_df)
    reports = read_indexer_reports(community_reports_df, communities_df, community_level=None)

    sys.path.insert(0, SCRIPT_DIR)
    from falkordb_vector_store import FalkorDBVectorStore
    from graphrag.config.embeddings import entity_description_embedding

    description_store = FalkorDBVectorStore(
        host=config.falkordb_host,
        port=config.falkordb_port,
        graph_name=graph_name,
        index_name=entity_description_embedding,
        vector_size=model_config.dim,
    )
    description_store.connect()

    search_engine = get_local_search_engine(
        config=grc,
        reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates={},
        response_type="Multiple Paragraphs",
        description_embedding_store=description_store,
    )

    predictions = []
    for q in questions:
        qid = q["question_id"]
        qtext = q["question_text"]
        ground_truth = q.get("ground_truth", "")
        try:
            result = await search_engine.search(qtext)
            context_texts = []
            if hasattr(result, "context_data"):
                for key, val in result.context_data.items():
                    if isinstance(val, pd.DataFrame) and not val.empty:
                        context_texts.append(val.to_string(index=False, max_rows=5))
                    elif isinstance(val, str):
                        context_texts.append(val)
            predictions.append(PredictionItem(
                id=qid, question=qtext, source=dataset_name,
                context=context_texts,
                generated_answer=result.response,
                ground_truth=ground_truth,
            ))
        except Exception as e:
            log.error(f"Query failed for {qid}: {e}")
            predictions.append(PredictionItem(
                id=qid, question=qtext, source=dataset_name,
                context=[], generated_answer="", ground_truth=ground_truth,
                error=str(e),
            ))
    return predictions


async def evaluate_predictions(
    config: BenchmarkConfig,
    predictions: List[PredictionItem],
    model_name: str,
    dataset_name: str,
) -> EvaluationResult:
    """Compute ROUGE-L and Answer Correctness for predictions."""
    from Evaluation.metrics.rouge import compute_rouge_score

    t0 = time.time()
    rouge_scores: List[float] = []
    acc_scores: List[float] = []

    eval_llm = None
    eval_emb = None

    for p in predictions:
        if p.error or not p.generated_answer.strip():
            continue

        try:
            score = await compute_rouge_score(p.generated_answer, p.ground_truth)
            rouge_scores.append(score)
        except Exception as e:
            log.debug(f"ROUGE failed for {p.id}: {e}")
            rouge_scores.append(float("nan"))

        try:
            from Evaluation.metrics.answer_accuracy import compute_answer_correctness
            if eval_llm is None:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                eval_llm = ChatOpenAI(
                    model=config.llm_model,
                    base_url=config.api_base_url,
                    api_key="no-key",
                )
                eval_emb = OpenAIEmbeddings(
                    model="BAAI/bge-m3",
                    base_url=config.api_base_url,
                    api_key="no-key",
                    check_embedding_ctx_length=False,
                )
            score = await compute_answer_correctness(p.question, p.generated_answer, p.ground_truth, eval_llm, eval_emb)
            acc_scores.append(score)
        except Exception as e:
            log.debug(f"Answer correctness failed for {p.id}: {e}")
            acc_scores.append(float("nan"))

    def safe_mean(vals: List[float]) -> float:
        clean = [v for v in vals if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    return EvaluationResult(
        model_name=model_name,
        dataset_name=dataset_name,
        rouge_l=safe_mean(rouge_scores),
        answer_correctness=safe_mean(acc_scores),
        eval_time_seconds=time.time() - t0,
    )


# ── Main ──


async def async_main(config: BenchmarkConfig):
    """Async main orchestrator: datasets → Phase 1-9 → Phase 10 × N → query → evaluate → report."""
    summary = BenchmarkSummary(config=config, start_time=datetime.now().isoformat())
    t_start = time.time()

    os.makedirs(os.path.join(config.output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "evaluations"), exist_ok=True)

    data_root = config.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_DIR, data_root)

    # ── Step 1: Prepare datasets, prompt-tune, and run Phase 1-9 ──
    dataset_questions: Dict[str, List[Dict[str, str]]] = {}
    dataset_prompts: Dict[str, str] = {}  # ds_name -> prompts_dir

    for ds_name in config.datasets:
        workspace_dir = os.path.join(config.output_dir, "workspace", ds_name)
        input_dir = os.path.join(workspace_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        questions = None
        if ds_name == "medical":
            questions = DatasetLoader.prepare_medical(data_root, input_dir)
        elif ds_name == "novel":
            questions = DatasetLoader.prepare_novel(data_root, input_dir)

        if questions is None:
            log.warning(f"Dataset {ds_name} failed to load, skipping")
            summary.dataset_phases.append(DatasetPhaseResult(
                dataset_name=ds_name, phase_1_9_time_seconds=0,
                output_dir="", error="Dataset load failed",
            ))
            continue

        if config.sample_size and config.sample_size < len(questions):
            questions = questions[:config.sample_size]
        dataset_questions[ds_name] = questions
        log.info(f"Dataset {ds_name}: {len(questions)} questions prepared")

        # Prompt-tune for this dataset
        log.info(f"=== Prompt-tune for {ds_name} ===")
        try:
            prompts_dir = await run_prompt_tune(config, workspace_dir)
            dataset_prompts[ds_name] = prompts_dir
        except Exception as e:
            log.warning(f"Prompt-tune failed for {ds_name}, using default prompts: {e}")
            dataset_prompts[ds_name] = None

        log.info(f"=== Phase 1-9 for {ds_name} ===")
        phase_result = await run_phase_1_9(config, ds_name, workspace_dir, prompts_dir=dataset_prompts.get(ds_name))
        summary.dataset_phases.append(phase_result)

        if phase_result.error:
            log.error(f"Phase 1-9 failed for {ds_name}, skipping all models")
            continue

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

    # ── Step 3: Phase 10 + Query + Evaluate per model × dataset ──
    for ds_name, questions in dataset_questions.items():
        phase = next((p for p in summary.dataset_phases if p.dataset_name == ds_name), None)
        if phase is None or phase.error:
            continue

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

            # Sync graph structure (entities + relationships) into FalkorDB
            try:
                sync_graph_to_falkordb(
                    output_dir=os.path.join(workspace_dir, "output"),
                    host=config.falkordb_host,
                    port=config.falkordb_port,
                    graph_name=make_graph_name(model.name, ds_name),
                )
            except Exception as e:
                log.warning(f"Graph sync failed for {model.display_name}/{ds_name}: {e}")

            try:
                log.info(f"Querying {len(questions)} questions with {model.display_name}")
                t0 = time.time()
                mr.predictions = await run_queries(config, ds_name, workspace_dir, model, questions)
                mr.query_time_seconds = time.time() - t0
            except Exception as e:
                log.error(f"Query failed for {model.display_name}/{ds_name}: {e}")
                mr.error = str(e)
                summary.model_results.append(mr)
                continue

            safe = sanitize_name(model.name)
            pred_path = os.path.join(config.output_dir, "predictions", f"{safe}__{ds_name}.json")
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump([asdict(p) for p in mr.predictions], f, indent=2, ensure_ascii=False)

            if mr.predictions:
                log.info(f"Evaluating {model.display_name}/{ds_name}")
                try:
                    mr.evaluation = await evaluate_predictions(config, mr.predictions, model.display_name, ds_name)
                    eval_path = os.path.join(config.output_dir, "evaluations", f"{safe}__{ds_name}.json")
                    with open(eval_path, "w", encoding="utf-8") as f:
                        json.dump(asdict(mr.evaluation), f, indent=2, ensure_ascii=False)
                except Exception as e:
                    log.error(f"Evaluation failed: {e}")

            summary.model_results.append(mr)

    # ── Step 4: Generate reports ──
    summary.end_time = datetime.now().isoformat()
    summary.total_time_seconds = time.time() - t_start

    report = ReportGenerator.generate_markdown(summary)
    report_path = os.path.join(config.output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report saved to {report_path}")

    summary_data = ReportGenerator.generate_summary_json(summary)
    summary_path = os.path.join(config.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    log.info(f"Summary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Embedding 模型基准测试")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--dataset", default="all",
                        choices=["medical", "novel", "all"])
    parser.add_argument("--models", default="", help="Comma-separated model names, default=all")
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

    if args.models:
        names = [n.strip() for n in args.models.split(",")]
        models = [m for m in EMBEDDING_MODELS if m.name in names]
    else:
        models = list(EMBEDDING_MODELS)

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
