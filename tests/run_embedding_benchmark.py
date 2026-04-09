#!/usr/bin/env python3
"""Embedding 模型基准测试执行器

基于 Microsoft GraphRAG pipeline 分割策略：
- Phase 1-9（文档加载→社区报告）每个数据集只执行一次
- Phase 10（向量嵌入）按 Embedding 模型数量执行 N 次
- 向量存入 FalkorDB，图命名为 <model>_<dataset>
"""
import asyncio
import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

# Add GraphRAG-Benchmark to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "GraphRAG-Benchmark"))

log = logging.getLogger(__name__)

# ── Data Models ──


@dataclass
class EmbeddingModelConfig:
    name: str
    dim: int
    max_tokens: int
    display_name: str


@dataclass
class BenchmarkConfig:
    api_base_url: str = "http://10.210.156.69:8633"
    llm_model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    falkordb_host: str = "10.210.156.69"
    falkordb_port: int = 6379
    sample_size: int = 5
    datasets: List[str] = field(default_factory=lambda: ["kevin_scott"])
    models: List["EmbeddingModelConfig"] = field(default_factory=lambda: EMBEDDING_MODELS)
    output_dir: str = "tests/benchmark_results"
    data_root: str = "graphrag-benchmarking-datasets/data"
    graphrag_root: str = "/home/eyanpen/sourceCode/rnd-ai-engine-features/graphrag"


@dataclass
class DatasetPhaseResult:
    """数据集 Phase 1-9 共享阶段结果"""
    dataset_name: str
    phase_1_9_time_seconds: float
    output_dir: str
    error: Optional[str] = None


@dataclass
class PredictionItem:
    id: str
    question: str
    source: str
    context: List[str]
    generated_answer: str
    ground_truth: str
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    model_name: str
    dataset_name: str
    rouge_l: float
    answer_correctness: float
    eval_time_seconds: float


@dataclass
class ModelResult:
    model: EmbeddingModelConfig
    dataset_name: str
    predictions: List[PredictionItem] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    embedding_time_seconds: float = 0.0
    query_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    config: BenchmarkConfig
    dataset_phases: List[DatasetPhaseResult] = field(default_factory=list)
    model_results: List[ModelResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_time_seconds: float = 0.0


# ── Constants ──

EMBEDDING_MODELS = [
    EmbeddingModelConfig("BAAI/bge-m3", 1024, 8192, "BGE-M3 (default)"),
    EmbeddingModelConfig("BAAI/bge-m3/heavy", 1024, 8192, "BGE-M3 (heavy)"),
    EmbeddingModelConfig("BAAI/bge-m3/interactive", 1024, 8192, "BGE-M3 (interactive)"),
    EmbeddingModelConfig("intfloat/e5-mistral-7b-instruct", 4096, 4096, "E5-Mistral-7B"),
    EmbeddingModelConfig("intfloat/multilingual-e5-large-instruct", 1024, 512, "mE5-Large"),
    EmbeddingModelConfig("nomic-ai/nomic-embed-text-v1.5", 768, 8192, "Nomic-v1.5"),
    EmbeddingModelConfig("Qwen/Qwen3-Embedding-8B", 4096, 8192, "Qwen3-Emb-8B"),
    EmbeddingModelConfig("Qwen/Qwen3-Embedding-8B-Alt", 4096, 32768, "Qwen3-Emb-8B-Alt"),
]


def sanitize_name(model_name: str) -> str:
    """Replace '/' with '-' and convert to lowercase. Used for FalkorDB graph naming."""
    return model_name.replace("/", "-").lower()


def make_graph_name(model_name: str, dataset_name: str) -> str:
    """Generate FalkorDB graph name: sanitize(model) + '_' + dataset."""
    return sanitize_name(model_name) + "_" + dataset_name


# ── Adaptive Concurrency Controller ──


class AdaptiveConcurrencyController:
    def __init__(self, init=10, min_val=2, max_val=50):
        self.current = init
        self.min_val = min_val
        self.max_val = max_val
        self._success_streak = 0
        self._lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(init)
        self._stats: List[float] = []
        self._stats_lock = threading.Lock()
        self._req_start_times: Dict[int, float] = {}

    def adjust(self, is_error: bool):
        with self._lock:
            old = self.current
            if is_error:
                self._success_streak = 0
                self.current = max(self.min_val, self.current - 1)
            else:
                self._success_streak += 1
                if self._success_streak >= 5:
                    self._success_streak = 0
                    self.current = min(self.max_val, self.current + 1)
            if self.current != old:
                log.info(f"[ADAPTIVE] concurrency {old} → {self.current}")
                self._semaphore = asyncio.Semaphore(self.current)

    def install_hooks(self):
        ctrl = self
        _orig_init = httpx.AsyncClient.__init__
        _orig_send = httpx.AsyncClient.send

        async def _on_request(request):
            ctrl._req_start_times[id(request)] = time.time()

        async def _on_response(response):
            start = ctrl._req_start_times.pop(id(response.request), None)
            if start:
                dur = time.time() - start
                log.debug(
                    f"[TIMING] {response.request.method} {response.request.url} "
                    f"→ {response.status_code} in {dur:.2f}s"
                )
                with ctrl._stats_lock:
                    ctrl._stats.append(dur)
                ctrl.adjust(response.status_code >= 500)

        def patched_init(self_client, *args, **kwargs):
            hooks = kwargs.get("event_hooks", {})
            hooks.setdefault("request", []).append(_on_request)
            hooks.setdefault("response", []).append(_on_response)
            kwargs["event_hooks"] = hooks
            _orig_init(self_client, *args, **kwargs)

        async def throttled_send(self_client, request, *args, **kwargs):
            async with ctrl._semaphore:
                return await _orig_send(self_client, request, *args, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        httpx.AsyncClient.send = throttled_send
        self._start_stats_printer()

    def _start_stats_printer(self):
        def _run():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._print_stats())
        threading.Thread(target=_run, daemon=True).start()

    async def _print_stats(self):
        while True:
            await asyncio.sleep(60)
            with self._stats_lock:
                snap = self._stats.copy()
                self._stats.clear()
            if snap:
                log.info(
                    f"[STATS] Last 60s: count={len(snap)}, "
                    f"min={min(snap):.2f}s, max={max(snap):.2f}s, "
                    f"avg={sum(snap)/len(snap):.2f}s"
                )


# ── Dataset Loader ──


class DatasetLoader:
    @staticmethod
    def _load_csv(path: str, q_col: str, id_col: str) -> Optional[List[Dict[str, str]]]:
        """Load questions from a CSV file. Returns None on error."""
        try:
            questions = []
            with open(path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if q_col not in (reader.fieldnames or []) or id_col not in (reader.fieldnames or []):
                    log.warning(f"CSV {path} missing columns: need '{q_col}' and '{id_col}', got {reader.fieldnames}")
                    return None
                for row in reader:
                    questions.append({"question_id": row.get(id_col, ""), "question_text": row.get(q_col, "")})
            return questions
        except FileNotFoundError:
            log.warning(f"CSV file not found: {path}")
            return None
        except Exception as e:
            log.warning(f"Failed to load CSV {path}: {e}")
            return None

    @staticmethod
    def prepare_kevin_scott(data_root: str, output_dir: str) -> Optional[List[Dict[str, str]]]:
        """Merge Episode part files and write to output_dir. Return questions or None."""
        try:
            input_dir = os.path.join(data_root, "kevinScott", "input")
            if not os.path.isdir(input_dir):
                log.warning(f"Kevin Scott input dir not found: {input_dir}")
                return None
            os.makedirs(output_dir, exist_ok=True)
            files = glob.glob(os.path.join(input_dir, "*.txt"))
            if not files:
                log.warning(f"No txt files found in {input_dir}")
                return None
            episodes: Dict[str, Dict[int, str]] = {}
            for f in files:
                fname = os.path.basename(f)
                m = re.match(r"(.+)-part(\d+)\.txt$", fname)
                if m:
                    ep_name, part_num = m.group(1), int(m.group(2))
                else:
                    ep_name, part_num = fname.replace(".txt", ""), 0
                with open(f, encoding="utf-8") as fh:
                    episodes.setdefault(ep_name, {})[part_num] = fh.read()
            for ep_name, parts in episodes.items():
                merged = "\n".join(parts[k] for k in sorted(parts))
                with open(os.path.join(output_dir, f"{ep_name}.txt"), "w", encoding="utf-8") as fout:
                    fout.write(merged)
            csv_path = os.path.join(data_root, "kevinScott", "Kevin Scott Questions.csv")
            return DatasetLoader._load_csv(csv_path, "question_text", "question_id")
        except Exception as e:
            log.warning(f"Failed to prepare kevin_scott: {e}")
            return None

    @staticmethod
    def prepare_msft(data_root: str, question_type: str, output_dir: str) -> Optional[List[Dict[str, str]]]:
        """Copy MSFT txt files to output_dir. Return questions or None."""
        try:
            txt_dir = os.path.join(data_root, "MSFT", "txt")
            if not os.path.isdir(txt_dir):
                log.warning(f"MSFT txt dir not found: {txt_dir}")
                return None
            os.makedirs(output_dir, exist_ok=True)
            for f in glob.glob(os.path.join(txt_dir, "*.txt")):
                shutil.copy2(f, output_dir)
            csv_name = f"MSFT {'Multi' if question_type == 'multi' else 'Single'} Transcript Questions.csv"
            csv_path = os.path.join(data_root, "MSFT", csv_name)
            return DatasetLoader._load_csv(csv_path, "question_text", "question_id")
        except Exception as e:
            log.warning(f"Failed to prepare msft_{question_type}: {e}")
            return None

    @staticmethod
    def prepare_hotpotqa(data_root: str, output_dir: str) -> Optional[List[Dict[str, str]]]:
        """Copy HotPotQA test_*.txt files to output_dir. Return questions or None."""
        try:
            input_dir = os.path.join(data_root, "HotPotQA", "input")
            if not os.path.isdir(input_dir):
                log.warning(f"HotPotQA input dir not found: {input_dir}")
                return None
            os.makedirs(output_dir, exist_ok=True)
            for f in glob.glob(os.path.join(input_dir, "test_*.txt")):
                shutil.copy2(f, output_dir)
            csv_path = os.path.join(data_root, "HotPotQA", "HotPotQA Filtered Questions.csv")
            return DatasetLoader._load_csv(csv_path, "question_text", "question_id")
        except Exception as e:
            log.warning(f"Failed to prepare hotpotqa: {e}")
            return None


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
) -> Any:
    """Build a GraphRagConfig programmatically for a dataset."""
    from graphrag_llm.config import ModelConfig
    from graphrag_storage import StorageConfig
    from graphrag_cache import CacheConfig
    from graphrag_storage.tables.table_provider_config import TableProviderConfig
    from graphrag_vectors import VectorStoreConfig, IndexSchema
    from graphrag.config.models.graph_rag_config import GraphRagConfig
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

    grc = GraphRagConfig(
        completion_models=completion_models,
        embedding_models=embedding_models,
        input_storage=StorageConfig(base_dir=str(Path(input_dir).resolve())),
        output_storage=StorageConfig(base_dir=str(Path(output_dir).resolve())),
        update_output_storage=StorageConfig(base_dir=str(Path(output_dir).resolve()) + "/update"),
        cache=CacheConfig(storage=StorageConfig(base_dir=str(Path(cache_dir).resolve()))),
        table_provider=TableProviderConfig(),
        vector_store=vector_store,
    )
    return grc


async def run_phase_1_9(config: BenchmarkConfig, dataset_name: str, workspace_dir: str) -> DatasetPhaseResult:
    """Execute Phase 1-9 (shared) for a dataset using GraphRAG pipeline."""
    from graphrag.index.workflows.factory import PipelineFactory
    from graphrag.index.run.run_pipeline import run_pipeline
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

    input_dir = os.path.join(workspace_dir, "input")
    output_dir = os.path.join(workspace_dir, "output")
    cache_dir = os.path.join(workspace_dir, "cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Register pre-embedding pipeline (Phase 1-9 only)
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

    grc = _build_graphrag_config(config, dataset_name, input_dir, output_dir, cache_dir)

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
    # Import and register FalkorDB vector store
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

    # Read parquet tables
    entities_df = pd.read_parquet(os.path.join(output_dir, "entities.parquet"))
    relationships_df = pd.read_parquet(os.path.join(output_dir, "relationships.parquet"))
    text_units_df = pd.read_parquet(os.path.join(output_dir, "text_units.parquet"))
    community_reports_df = pd.read_parquet(os.path.join(output_dir, "community_reports.parquet"))
    communities_df = pd.read_parquet(os.path.join(output_dir, "communities.parquet"))

    entities = read_indexer_entities(entities_df, communities_df, community_level=None)
    relationships = read_indexer_relationships(relationships_df)
    text_units = read_indexer_text_units(text_units_df)
    reports = read_indexer_reports(community_reports_df, communities_df, community_level=None)

    # Create FalkorDB vector store for entity description embeddings
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
        try:
            result = await search_engine.asearch(qtext)
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
                ground_truth="",
            ))
        except Exception as e:
            log.error(f"Query failed for {qid}: {e}")
            predictions.append(PredictionItem(
                id=qid, question=qtext, source=dataset_name,
                context=[], generated_answer="", ground_truth="",
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

    # Lazy-init LLM and embeddings for answer correctness
    eval_llm = None
    eval_emb = None

    for p in predictions:
        if p.error or not p.generated_answer.strip():
            continue

        # ROUGE-L
        try:
            score = await compute_rouge_score(p.generated_answer, p.ground_truth if p.ground_truth else p.question)
            rouge_scores.append(score)
        except Exception as e:
            log.debug(f"ROUGE failed for {p.id}: {e}")
            rouge_scores.append(float("nan"))

        # Answer Correctness
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
                )
            gt = p.ground_truth if p.ground_truth else p.question
            score = await compute_answer_correctness(p.question, p.generated_answer, gt, eval_llm, eval_emb)
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


# ── Report Generator ──


class ReportGenerator:
    @staticmethod
    def generate_markdown(summary: BenchmarkSummary) -> str:
        lines = ["# Embedding 模型基准测试报告\n"]
        lines.append(f"> 生成时间: {summary.end_time}\n")

        # 1. 测试环境信息
        lines.append("## 1. 测试环境信息\n")
        lines.append("| 项目 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| API 地址 | {summary.config.api_base_url} |")
        lines.append(f"| LLM 模型 | {summary.config.llm_model} |")
        lines.append(f"| FalkorDB | {summary.config.falkordb_host}:{summary.config.falkordb_port} |")
        lines.append(f"| 数据集 | {', '.join(summary.config.datasets)} |")
        lines.append(f"| 采样数量 | {summary.config.sample_size} |")
        lines.append(f"| 总耗时 | {summary.total_time_seconds:.1f}s |")
        lines.append("")

        # 2. 模型清单
        lines.append("## 2. 模型清单\n")
        lines.append("| 模型 | 维度 | 最大Token | 状态 |")
        lines.append("|------|------|----------|------|")
        for m in summary.config.models:
            status = "✅"
            for r in summary.model_results:
                if r.model.name == m.name and r.error and "unavailable" in r.error.lower():
                    status = "❌ 不可用"
                    break
            lines.append(f"| {m.display_name} | {m.dim} | {m.max_tokens} | {status} |")
        lines.append("")

        # 3. Phase 1-9 共享耗时
        if summary.dataset_phases:
            lines.append("## 3. Phase 1-9 共享耗时\n")
            lines.append("| 数据集 | 耗时 | 状态 |")
            lines.append("|--------|------|------|")
            for dp in summary.dataset_phases:
                status = "✅" if not dp.error else f"❌ {dp.error}"
                lines.append(f"| {dp.dataset_name} | {dp.phase_1_9_time_seconds:.1f}s | {status} |")
            lines.append("")

        # 4. Per-dataset metrics
        ds_names = sorted(set(r.dataset_name for r in summary.model_results))
        for ds in ds_names:
            ds_results = [r for r in summary.model_results if r.dataset_name == ds and r.evaluation]
            if not ds_results:
                continue
            lines.append(f"## 4. 指标对比表 — {ds}\n")
            lines.append("| 模型 | ROUGE-L | Answer Correctness | Phase 10 耗时 | 查询耗时 |")
            lines.append("|------|---------|-------------------|-------------|---------|")
            rouges = [r.evaluation.rouge_l for r in ds_results if r.evaluation and not math.isnan(r.evaluation.rouge_l)]
            accs = [r.evaluation.answer_correctness for r in ds_results if r.evaluation and not math.isnan(r.evaluation.answer_correctness)]
            best_rouge = max(rouges) if rouges else None
            best_acc = max(accs) if accs else None
            for r in ds_results:
                ev = r.evaluation
                rv = f"{ev.rouge_l:.4f}" if not math.isnan(ev.rouge_l) else "NaN"
                av = f"{ev.answer_correctness:.4f}" if not math.isnan(ev.answer_correctness) else "NaN"
                if best_rouge is not None and not math.isnan(ev.rouge_l) and ev.rouge_l == best_rouge:
                    rv = f"**{rv}**"
                if best_acc is not None and not math.isnan(ev.answer_correctness) and ev.answer_correctness == best_acc:
                    av = f"**{av}**"
                lines.append(f"| {r.model.display_name} | {rv} | {av} | {r.embedding_time_seconds:.1f}s | {r.query_time_seconds:.1f}s |")
            lines.append("")

        # 5. BGE-M3 模式对比
        bge_results = [r for r in summary.model_results if r.model.name.startswith("BAAI/bge-m3") and r.evaluation]
        if bge_results:
            lines.append("## 5. BGE-M3 模式对比\n")
            lines.append("| 模式 | 数据集 | ROUGE-L | Answer Correctness |")
            lines.append("|------|--------|---------|-------------------|")
            rouges = [r.evaluation.rouge_l for r in bge_results if not math.isnan(r.evaluation.rouge_l)]
            accs = [r.evaluation.answer_correctness for r in bge_results if not math.isnan(r.evaluation.answer_correctness)]
            br = max(rouges) if rouges else None
            ba = max(accs) if accs else None
            for r in bge_results:
                ev = r.evaluation
                rv = f"{ev.rouge_l:.4f}" if not math.isnan(ev.rouge_l) else "NaN"
                av = f"{ev.answer_correctness:.4f}" if not math.isnan(ev.answer_correctness) else "NaN"
                if br and not math.isnan(ev.rouge_l) and ev.rouge_l == br:
                    rv = f"**{rv}**"
                if ba and not math.isnan(ev.answer_correctness) and ev.answer_correctness == ba:
                    av = f"**{av}**"
                lines.append(f"| {r.model.display_name} | {r.dataset_name} | {rv} | {av} |")
            lines.append("")

        # 6. 总体排名
        lines.append("## 6. 总体排名\n")
        model_scores: Dict[str, List[float]] = {}
        for r in summary.model_results:
            if r.evaluation:
                vals = []
                if not math.isnan(r.evaluation.rouge_l):
                    vals.append(r.evaluation.rouge_l)
                if not math.isnan(r.evaluation.answer_correctness):
                    vals.append(r.evaluation.answer_correctness)
                if vals:
                    model_scores.setdefault(r.model.display_name, []).extend(vals)
        ranked = sorted(model_scores.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0, reverse=True)
        lines.append("| 排名 | 模型 | 加权平均分 |")
        lines.append("|------|------|----------|")
        for i, (name, scores) in enumerate(ranked, 1):
            avg = sum(scores) / len(scores) if scores else 0
            lines.append(f"| {i} | {name} | {avg:.4f} |")
        lines.append("")

        # 7. 耗时统计
        lines.append("## 7. 耗时统计\n")
        lines.append("| 模型 | 数据集 | Phase 10 | 查询时间 | 评估时间 | 总时间 |")
        lines.append("|------|--------|---------|---------|---------|--------|")
        for r in summary.model_results:
            et = r.evaluation.eval_time_seconds if r.evaluation else 0
            total = r.embedding_time_seconds + r.query_time_seconds + et
            lines.append(
                f"| {r.model.display_name} | {r.dataset_name} "
                f"| {r.embedding_time_seconds:.1f}s | {r.query_time_seconds:.1f}s "
                f"| {et:.1f}s | {total:.1f}s |"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def generate_summary_json(summary: BenchmarkSummary) -> dict:
        results = []
        for r in summary.model_results:
            entry: Dict[str, Any] = {
                "model": r.model.display_name,
                "model_name": r.model.name,
                "dataset": r.dataset_name,
                "error": r.error,
                "embedding_time": r.embedding_time_seconds,
                "query_time": r.query_time_seconds,
                "num_predictions": len(r.predictions),
            }
            if r.evaluation:
                entry["rouge_l"] = r.evaluation.rouge_l
                entry["answer_correctness"] = r.evaluation.answer_correctness
                entry["eval_time"] = r.evaluation.eval_time_seconds
            results.append(entry)

        dataset_phases = []
        for dp in summary.dataset_phases:
            dataset_phases.append({
                "dataset": dp.dataset_name,
                "phase_1_9_time": dp.phase_1_9_time_seconds,
                "output_dir": dp.output_dir,
                "error": dp.error,
            })

        return {
            "config": {
                "api_base_url": summary.config.api_base_url,
                "llm_model": summary.config.llm_model,
                "falkordb": f"{summary.config.falkordb_host}:{summary.config.falkordb_port}",
                "sample_size": summary.config.sample_size,
                "datasets": summary.config.datasets,
                "models": [m.name for m in summary.config.models],
            },
            "start_time": summary.start_time,
            "end_time": summary.end_time,
            "total_time_seconds": summary.total_time_seconds,
            "dataset_phases": dataset_phases,
            "results": results,
        }


# ── Main ──


async def async_main(config: BenchmarkConfig):
    """Async main orchestrator: datasets → Phase 1-9 → Phase 10 × N → query → evaluate → report."""
    summary = BenchmarkSummary(config=config, start_time=datetime.now().isoformat())
    t_start = time.time()

    os.makedirs(os.path.join(config.output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "evaluations"), exist_ok=True)

    # Resolve data_root relative to project dir
    data_root = config.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_DIR, data_root)

    # ── Step 1: Prepare datasets and run Phase 1-9 ──
    dataset_questions: Dict[str, List[Dict[str, str]]] = {}

    for ds_name in config.datasets:
        workspace_dir = os.path.join(config.output_dir, "workspace", ds_name)
        input_dir = os.path.join(workspace_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        # Load dataset
        questions = None
        if ds_name == "kevin_scott":
            questions = DatasetLoader.prepare_kevin_scott(data_root, input_dir)
        elif ds_name == "msft_multi":
            questions = DatasetLoader.prepare_msft(data_root, "multi", input_dir)
        elif ds_name == "msft_single":
            questions = DatasetLoader.prepare_msft(data_root, "single", input_dir)
        elif ds_name == "hotpotqa":
            questions = DatasetLoader.prepare_hotpotqa(data_root, input_dir)

        if questions is None:
            log.warning(f"Dataset {ds_name} failed to load, skipping")
            summary.dataset_phases.append(DatasetPhaseResult(
                dataset_name=ds_name, phase_1_9_time_seconds=0,
                output_dir="", error="Dataset load failed",
            ))
            continue

        # Sample questions
        if config.sample_size and config.sample_size < len(questions):
            questions = questions[:config.sample_size]
        dataset_questions[ds_name] = questions
        log.info(f"Dataset {ds_name}: {len(questions)} questions prepared")

        # Run Phase 1-9
        log.info(f"=== Phase 1-9 for {ds_name} ===")
        phase_result = await run_phase_1_9(config, ds_name, workspace_dir)
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
        # Check if Phase 1-9 succeeded for this dataset
        phase = next((p for p in summary.dataset_phases if p.dataset_name == ds_name), None)
        if phase is None or phase.error:
            continue

        workspace_dir = os.path.join(config.output_dir, "workspace", ds_name)

        for model in available_models:
            mr = ModelResult(model=model, dataset_name=ds_name)
            log.info(f"=== {model.display_name} × {ds_name} ===")

            # Phase 10
            try:
                log.info(f"Phase 10: embedding with {model.display_name}")
                mr.embedding_time_seconds = await run_phase_10(config, ds_name, workspace_dir, model)
            except Exception as e:
                log.error(f"Phase 10 failed for {model.display_name}/{ds_name}: {e}")
                mr.error = str(e)
                summary.model_results.append(mr)
                continue

            # Query
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

            # Save predictions
            safe = sanitize_name(model.name)
            pred_path = os.path.join(config.output_dir, "predictions", f"{safe}__{ds_name}.json")
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump([asdict(p) for p in mr.predictions], f, indent=2, ensure_ascii=False)

            # Evaluate
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
    parser.add_argument("--dataset", default="kevin_scott",
                        choices=["kevin_scott", "msft_multi", "msft_single", "hotpotqa", "all"])
    parser.add_argument("--models", default="", help="Comma-separated model names, default=all")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "benchmark_results"))
    parser.add_argument("--data-root", default=os.path.join(PROJECT_DIR, "graphrag-benchmarking-datasets", "data"))
    parser.add_argument("--graphrag-root", default="/home/eyanpen/sourceCode/rnd-ai-engine-features/graphrag")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dual logging: console INFO + file DEBUG
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "benchmark.log"), mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger().handlers[0].setLevel(logging.INFO)

    # Select models
    if args.models:
        names = [n.strip() for n in args.models.split(",")]
        models = [m for m in EMBEDDING_MODELS if m.name in names]
    else:
        models = list(EMBEDDING_MODELS)

    datasets = ["kevin_scott", "msft_multi", "msft_single", "hotpotqa"] if args.dataset == "all" else [args.dataset]

    config = BenchmarkConfig(
        sample_size=args.sample,
        datasets=datasets,
        models=models,
        output_dir=args.output_dir,
        data_root=args.data_root,
        graphrag_root=args.graphrag_root,
    )

    # Install adaptive concurrency hooks
    ac = AdaptiveConcurrencyController()
    ac.install_hooks()

    asyncio.run(async_main(config))


if __name__ == "__main__":
    main()
