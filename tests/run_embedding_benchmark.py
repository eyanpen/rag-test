#!/usr/bin/env python3
"""Embedding 模型基准测试执行器"""
import asyncio
import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

import httpx

# Add GraphRAG-Benchmark to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "GraphRAG-Benchmark"))

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, OpenAIEmbeddingService
from tqdm import tqdm

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
    sample_size: int = 5
    datasets: List[str] = field(default_factory=lambda: ["kevin_scott"])
    models: List[EmbeddingModelConfig] = field(default_factory=list)
    output_dir: str = "tests/benchmark_results"
    data_root: str = "graphrag-benchmarking-datasets/data"

@dataclass
class DatasetResult:
    name: str
    documents: Dict[str, str] = field(default_factory=dict)
    questions: List[Dict[str, str]] = field(default_factory=list)

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
    index_time_seconds: float = 0.0
    query_time_seconds: float = 0.0
    error: Optional[str] = None

@dataclass
class BenchmarkSummary:
    config: BenchmarkConfig
    results: List[ModelResult] = field(default_factory=list)
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

DOMAIN = (
    "Analyze this content and identify the key entities. "
    "Focus on how they interact with each other, the locations mentioned, "
    "and their relationships."
)
EXAMPLE_QUERIES = [
    "What are the main topics discussed?",
    "How do the key entities relate to each other?",
    "What is the significance of the events described?",
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event",
                "Concept", "Organization", "Disease", "Treatment", "Symptom"]


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "__")


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
                log.debug(f"[TIMING] {response.request.method} {response.request.url} → {response.status_code} in {dur:.2f}s")
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
                log.info(f"[STATS] Last 60s: count={len(snap)}, min={min(snap):.2f}s, max={max(snap):.2f}s, avg={sum(snap)/len(snap):.2f}s")


# ── Dataset Loader ──

class DatasetLoader:
    @staticmethod
    def load_kevin_scott(base_path: str) -> DatasetResult:
        result = DatasetResult(name="kevin_scott")
        input_dir = os.path.join(base_path, "kevinScott", "input")
        try:
            files = glob.glob(os.path.join(input_dir, "*.txt"))
            episodes: Dict[str, Dict[int, str]] = {}
            for f in files:
                fname = os.path.basename(f)
                m = re.match(r"(.+)-part(\d+)\.txt$", fname)
                if m:
                    ep_name, part_num = m.group(1), int(m.group(2))
                else:
                    ep_name, part_num = fname.replace(".txt", ""), 0
                episodes.setdefault(ep_name, {})[part_num] = open(f, encoding="utf-8").read()
            for ep_name, parts in episodes.items():
                result.documents[ep_name] = "\n".join(parts[k] for k in sorted(parts))
            csv_path = os.path.join(base_path, "kevinScott", "Kevin Scott Questions.csv")
            result.questions = DatasetLoader._load_csv(csv_path, "question_text", "question_id")
        except Exception as e:
            log.warning(f"Failed to load kevin_scott: {e}")
        return result

    @staticmethod
    def load_msft(base_path: str, question_type: str) -> DatasetResult:
        name = f"msft_{question_type}"
        result = DatasetResult(name=name)
        try:
            txt_dir = os.path.join(base_path, "MSFT", "txt")
            for f in glob.glob(os.path.join(txt_dir, "*.txt")):
                doc_name = os.path.basename(f).replace(".txt", "")
                result.documents[doc_name] = open(f, encoding="utf-8").read()
            csv_name = f"MSFT {'Multi' if question_type == 'multi' else 'Single'} Transcript Questions.csv"
            csv_path = os.path.join(base_path, "MSFT", csv_name)
            result.questions = DatasetLoader._load_csv(csv_path, "question_text", "question_id")
        except Exception as e:
            log.warning(f"Failed to load {name}: {e}")
        return result

    @staticmethod
    def load_hotpotqa(base_path: str) -> DatasetResult:
        result = DatasetResult(name="hotpotqa")
        try:
            input_dir = os.path.join(base_path, "HotPotQA", "input")
            for f in glob.glob(os.path.join(input_dir, "test_*.txt")):
                doc_name = os.path.basename(f).replace(".txt", "")
                result.documents[doc_name] = open(f, encoding="utf-8").read()
            csv_path = os.path.join(base_path, "HotPotQA", "HotPotQA Filtered Questions.csv")
            result.questions = DatasetLoader._load_csv(csv_path, "question_text", "question_id")
        except Exception as e:
            log.warning(f"Failed to load hotpotqa: {e}")
        return result

    @staticmethod
    def _load_csv(path: str, q_col: str, id_col: str) -> List[Dict[str, str]]:
        questions = []
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({"question_id": row.get(id_col, ""), "question_text": row.get(q_col, "")})
        return questions


# ── Benchmark Runner ──

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._concurrency = AdaptiveConcurrencyController()
        self._concurrency.install_hooks()

    def create_graphrag_instance(self, model_config: EmbeddingModelConfig, workspace: str) -> GraphRAG:
        emb = OpenAIEmbeddingService(
            model=model_config.name, base_url=self.config.api_base_url,
            api_key="no-key", embedding_dim=model_config.dim,
        )
        llm = OpenAILLMService(
            model=self.config.llm_model, base_url=self.config.api_base_url,
            api_key="no-key",
        )
        return GraphRAG(
            working_dir=workspace, domain=DOMAIN,
            example_queries="\n".join(EXAMPLE_QUERIES),
            entity_types=ENTITY_TYPES,
            config=GraphRAG.Config(llm_service=llm, embedding_service=emb),
        )

    def check_model_availability(self, model: EmbeddingModelConfig) -> bool:
        try:
            import httpx as hx
            r = hx.post(f"{self.config.api_base_url}/embeddings",
                        json={"model": model.name, "input": "test"}, timeout=30)
            return r.status_code == 200
        except Exception as e:
            log.warning(f"Model {model.display_name} unavailable: {e}")
            return False

    def run_single_model(self, model: EmbeddingModelConfig, dataset: DatasetResult) -> ModelResult:
        mr = ModelResult(model=model, dataset_name=dataset.name)
        safe_name = sanitize_model_name(model.name)
        workspace = os.path.join(self.config.output_dir, "workspace", safe_name, dataset.name)
        os.makedirs(workspace, exist_ok=True)
        try:
            grag = self.create_graphrag_instance(model, workspace)
            # Index
            log.info(f"Indexing {dataset.name} with {model.display_name} ({len(dataset.documents)} docs)")
            t0 = time.time()
            for doc_name, text in dataset.documents.items():
                if text.strip():
                    grag.insert(text)
            mr.index_time_seconds = time.time() - t0
            log.info(f"Indexing done in {mr.index_time_seconds:.1f}s")
            # Query
            questions = dataset.questions
            if self.config.sample_size and self.config.sample_size < len(questions):
                questions = questions[:self.config.sample_size]
            log.info(f"Querying {len(questions)} questions")
            t0 = time.time()
            for q in tqdm(questions, desc=f"{model.display_name}/{dataset.name}"):
                try:
                    resp = grag.query(q["question_text"])
                    resp_dict = resp.to_dict()
                    ctx_chunks = resp_dict.get("context", {}).get("chunks", [])
                    contexts = [item[0]["content"] for item in ctx_chunks] if ctx_chunks else []
                    mr.predictions.append(PredictionItem(
                        id=q["question_id"], question=q["question_text"],
                        source=dataset.name, context=contexts,
                        generated_answer=resp.response, ground_truth=q.get("answer", ""),
                    ))
                except Exception as e:
                    log.error(f"Query failed: {e}")
                    mr.predictions.append(PredictionItem(
                        id=q["question_id"], question=q["question_text"],
                        source=dataset.name, context=[], generated_answer="",
                        ground_truth=q.get("answer", ""), error=str(e),
                    ))
            mr.query_time_seconds = time.time() - t0
        except Exception as e:
            log.error(f"Model {model.display_name} failed on {dataset.name}: {e}")
            mr.error = str(e)
        return mr

    def evaluate_predictions(self, predictions: List[PredictionItem], model_name: str, dataset_name: str) -> EvaluationResult:
        from Evaluation.metrics.rouge import compute_rouge_score
        t0 = time.time()
        rouge_scores, acc_scores = [], []
        for p in predictions:
            if p.error:
                continue
            # ROUGE-L
            try:
                score = asyncio.run(compute_rouge_score(p.generated_answer, p.ground_truth))
                rouge_scores.append(score)
            except Exception:
                rouge_scores.append(float("nan"))
            # Answer correctness
            try:
                from Evaluation.metrics.answer_accuracy import compute_answer_correctness
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                llm = ChatOpenAI(model=self.config.llm_model, base_url=self.config.api_base_url, api_key="no-key")
                emb = OpenAIEmbeddings(model="BAAI/bge-m3", base_url=self.config.api_base_url, api_key="no-key")
                score = asyncio.run(compute_answer_correctness(p.question, p.generated_answer, p.ground_truth, llm, emb))
                acc_scores.append(score)
            except Exception:
                acc_scores.append(float("nan"))

        def safe_mean(vals):
            clean = [v for v in vals if not math.isnan(v)]
            return sum(clean) / len(clean) if clean else float("nan")

        return EvaluationResult(
            model_name=model_name, dataset_name=dataset_name,
            rouge_l=safe_mean(rouge_scores), answer_correctness=safe_mean(acc_scores),
            eval_time_seconds=time.time() - t0,
        )

    def _load_datasets(self) -> List[DatasetResult]:
        datasets = []
        base = os.path.join(PROJECT_DIR, self.config.data_root)
        for ds_name in self.config.datasets:
            if ds_name == "kevin_scott":
                datasets.append(DatasetLoader.load_kevin_scott(base))
            elif ds_name == "msft_multi":
                datasets.append(DatasetLoader.load_msft(base, "multi"))
            elif ds_name == "msft_single":
                datasets.append(DatasetLoader.load_msft(base, "single"))
            elif ds_name == "hotpotqa":
                datasets.append(DatasetLoader.load_hotpotqa(base))
            elif ds_name == "all":
                datasets.append(DatasetLoader.load_kevin_scott(base))
                datasets.append(DatasetLoader.load_msft(base, "multi"))
                datasets.append(DatasetLoader.load_msft(base, "single"))
                datasets.append(DatasetLoader.load_hotpotqa(base))
        return [d for d in datasets if d.documents]

    def run(self) -> BenchmarkSummary:
        summary = BenchmarkSummary(config=self.config, start_time=datetime.now().isoformat())
        t_start = time.time()
        os.makedirs(os.path.join(self.config.output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "evaluations"), exist_ok=True)

        datasets = self._load_datasets()
        log.info(f"Loaded {len(datasets)} datasets")

        for model in self.config.models:
            log.info(f"Checking model: {model.display_name}")
            if not self.check_model_availability(model):
                log.warning(f"Model {model.display_name} unavailable, skipping")
                for ds in datasets:
                    summary.results.append(ModelResult(model=model, dataset_name=ds.name, error="Model unavailable"))
                continue

            for ds in datasets:
                log.info(f"=== {model.display_name} × {ds.name} ===")
                mr = self.run_single_model(model, ds)

                # Save predictions
                safe = sanitize_model_name(model.name)
                pred_path = os.path.join(self.config.output_dir, "predictions", f"{safe}__{ds.name}.json")
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(p) for p in mr.predictions], f, indent=2, ensure_ascii=False)

                # Evaluate
                if not mr.error and mr.predictions:
                    log.info(f"Evaluating {model.display_name}/{ds.name}")
                    try:
                        mr.evaluation = self.evaluate_predictions(mr.predictions, model.display_name, ds.name)
                        eval_path = os.path.join(self.config.output_dir, "evaluations", f"{safe}__{ds.name}.json")
                        with open(eval_path, "w", encoding="utf-8") as f:
                            json.dump(asdict(mr.evaluation), f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        log.error(f"Evaluation failed: {e}")

                summary.results.append(mr)

        summary.end_time = datetime.now().isoformat()
        summary.total_time_seconds = time.time() - t_start
        return summary


# ── Report Generator ──

class ReportGenerator:
    @staticmethod
    def generate_markdown(summary: BenchmarkSummary) -> str:
        lines = ["# Embedding 模型基准测试报告\n"]
        lines.append(f"> 生成时间: {summary.end_time}\n")

        # 测试环境信息
        lines.append("## 1. 测试环境信息\n")
        lines.append("| 项目 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| API 地址 | {summary.config.api_base_url} |")
        lines.append(f"| LLM 模型 | {summary.config.llm_model} |")
        lines.append(f"| 数据集 | {', '.join(summary.config.datasets)} |")
        lines.append(f"| 采样数量 | {summary.config.sample_size} |")
        lines.append(f"| 总耗时 | {summary.total_time_seconds:.1f}s |")
        lines.append("")

        # 模型清单
        lines.append("## 2. 模型清单\n")
        lines.append("| 模型 | 维度 | 最大Token | 状态 |")
        lines.append("|------|------|----------|------|")
        for m in summary.config.models:
            status = "✅"
            for r in summary.results:
                if r.model.name == m.name and r.error == "Model unavailable":
                    status = "❌ 不可用"
                    break
            lines.append(f"| {m.display_name} | {m.dim} | {m.max_tokens} | {status} |")
        lines.append("")

        # Per-dataset metrics
        ds_names = list(set(r.dataset_name for r in summary.results))
        for ds in ds_names:
            ds_results = [r for r in summary.results if r.dataset_name == ds and r.evaluation]
            if not ds_results:
                continue
            lines.append(f"## 3. 指标对比表 - {ds}\n")
            lines.append("| 模型 | ROUGE-L | Answer Correctness | 索引耗时 | 查询耗时 |")
            lines.append("|------|---------|-------------------|---------|---------|")
            rouges = [r.evaluation.rouge_l for r in ds_results if r.evaluation]
            accs = [r.evaluation.answer_correctness for r in ds_results if r.evaluation]
            best_rouge = max((v for v in rouges if not math.isnan(v)), default=None)
            best_acc = max((v for v in accs if not math.isnan(v)), default=None)
            for r in ds_results:
                ev = r.evaluation
                rv = f"{ev.rouge_l:.4f}" if not math.isnan(ev.rouge_l) else "NaN"
                av = f"{ev.answer_correctness:.4f}" if not math.isnan(ev.answer_correctness) else "NaN"
                if best_rouge is not None and not math.isnan(ev.rouge_l) and ev.rouge_l == best_rouge:
                    rv = f"**{rv}**"
                if best_acc is not None and not math.isnan(ev.answer_correctness) and ev.answer_correctness == best_acc:
                    av = f"**{av}**"
                lines.append(f"| {r.model.display_name} | {rv} | {av} | {r.index_time_seconds:.1f}s | {r.query_time_seconds:.1f}s |")
            lines.append("")

        # BGE-M3 模式对比
        bge_results = [r for r in summary.results if r.model.name.startswith("BAAI/bge-m3") and r.evaluation]
        if bge_results:
            lines.append("## 4. BGE-M3 模式对比\n")
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

        # 总体排名
        lines.append("## 5. 总体排名\n")
        model_scores: Dict[str, List[float]] = {}
        for r in summary.results:
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

        # 耗时统计
        lines.append("## 6. 耗时统计\n")
        lines.append("| 模型 | 数据集 | 索引时间 | 查询时间 | 评估时间 | 总时间 |")
        lines.append("|------|--------|---------|---------|---------|--------|")
        for r in summary.results:
            et = r.evaluation.eval_time_seconds if r.evaluation else 0
            total = r.index_time_seconds + r.query_time_seconds + et
            lines.append(f"| {r.model.display_name} | {r.dataset_name} | {r.index_time_seconds:.1f}s | {r.query_time_seconds:.1f}s | {et:.1f}s | {total:.1f}s |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def generate_summary_json(summary: BenchmarkSummary) -> dict:
        results = []
        for r in summary.results:
            entry = {
                "model": r.model.display_name, "model_name": r.model.name,
                "dataset": r.dataset_name, "error": r.error,
                "index_time": r.index_time_seconds, "query_time": r.query_time_seconds,
                "num_predictions": len(r.predictions),
            }
            if r.evaluation:
                entry["rouge_l"] = r.evaluation.rouge_l
                entry["answer_correctness"] = r.evaluation.answer_correctness
                entry["eval_time"] = r.evaluation.eval_time_seconds
            results.append(entry)
        return {
            "config": {"api_base_url": summary.config.api_base_url, "llm_model": summary.config.llm_model,
                       "sample_size": summary.config.sample_size, "datasets": summary.config.datasets},
            "start_time": summary.start_time, "end_time": summary.end_time,
            "total_time_seconds": summary.total_time_seconds, "results": results,
        }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Embedding 模型基准测试")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--dataset", default="kevin_scott",
                        choices=["kevin_scott", "msft_multi", "msft_single", "hotpotqa", "all"])
    parser.add_argument("--models", default="", help="Comma-separated model names, default=all")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "benchmark_results"))
    parser.add_argument("--data-root", default=os.path.join(PROJECT_DIR, "graphrag-benchmarking-datasets", "data"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dual logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG,
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
        sample_size=args.sample, datasets=datasets, models=models,
        output_dir=args.output_dir, data_root=args.data_root,
    )

    runner = BenchmarkRunner(config)
    summary = runner.run()

    # Generate report
    report = ReportGenerator.generate_markdown(summary)
    report_path = os.path.join(args.output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report saved to {report_path}")

    # Generate summary JSON
    summary_data = ReportGenerator.generate_summary_json(summary)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
