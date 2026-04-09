#!/usr/bin/env python3
"""
fast-graphrag 测试脚本 —— 使用远程 OpenAI 兼容 API 作为 LLM 和 Embedding。
绕过原脚本对本地 HuggingFace 模型的依赖。
"""
import asyncio
import argparse
import json
import logging
import os
import sys
import time
import threading
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

# fast-graphrag imports
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, OpenAIEmbeddingService

# ── HTTP 请求计时与统计 ──
import httpx
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_fast_graphrag_test.log", mode="w", encoding="utf-8"),
    ],
)

log = logging.getLogger(__name__)

_req_start_times: Dict[int, float] = {}  # request id -> start time
_stats_lock = threading.Lock()
_stats: List[float] = []  # durations collected in current window

# ── 自适应并发控制 ──
_CONCURRENCY_MIN = 2
_CONCURRENCY_MAX = 50
_CONCURRENCY_INIT = 10
_concurrency_lock = threading.Lock()
_current_concurrency = _CONCURRENCY_INIT
_adaptive_semaphore: asyncio.Semaphore | None = None
_success_streak = 0  # 连续成功计数，每 5 次成功升 1


def _get_semaphore() -> asyncio.Semaphore:
    global _adaptive_semaphore
    if _adaptive_semaphore is None:
        _adaptive_semaphore = asyncio.Semaphore(_current_concurrency)
    return _adaptive_semaphore


def _adjust_concurrency(is_error: bool):
    global _current_concurrency, _adaptive_semaphore, _success_streak
    with _concurrency_lock:
        old = _current_concurrency
        if is_error:
            _success_streak = 0
            _current_concurrency = max(_CONCURRENCY_MIN, _current_concurrency // 2)
        else:
            _success_streak += 1
            if _success_streak >= 5:
                _success_streak = 0
                _current_concurrency = min(_CONCURRENCY_MAX, _current_concurrency + 1)
        if _current_concurrency != old:
            log.info(f"[ADAPTIVE] concurrency {old} → {_current_concurrency}")
            _adaptive_semaphore = asyncio.Semaphore(_current_concurrency)


async def _on_request(request: httpx.Request):
    _req_start_times[id(request)] = time.time()


async def _on_response(response: httpx.Response):
    start = _req_start_times.pop(id(response.request), None)
    if start is None:
        return
    duration = time.time() - start
    method = response.request.method
    url = response.request.url
    status = response.status_code
    log.info(f"[TIMING] {method} {url} → {status} in {duration:.2f}s")
    with _stats_lock:
        _stats.append(duration)
    _adjust_concurrency(status >= 500)


# Monkey-patch httpx.AsyncClient.send to enforce adaptive semaphore
_orig_client_init = httpx.AsyncClient.__init__
_orig_send = httpx.AsyncClient.send


async def _throttled_send(self, request, *args, **kwargs):
    sem = _get_semaphore()
    async with sem:
        return await _orig_send(self, request, *args, **kwargs)


def _patched_client_init(self, *args, **kwargs):
    hooks = kwargs.get("event_hooks", {})
    hooks.setdefault("request", []).append(_on_request)
    hooks.setdefault("response", []).append(_on_response)
    kwargs["event_hooks"] = hooks
    _orig_client_init(self, *args, **kwargs)


httpx.AsyncClient.__init__ = _patched_client_init
httpx.AsyncClient.send = _throttled_send


async def _stats_printer():
    """每 60 秒打印一次统计信息"""
    while True:
        await asyncio.sleep(60)
        with _stats_lock:
            snapshot = _stats.copy()
            _stats.clear()
        if not snapshot:
            log.info("[STATS] No requests in the last 60s")
            continue
        log.info(
            f"[STATS] Last 60s: count={len(snapshot)}, "
            f"min={min(snapshot):.2f}s, max={max(snapshot):.2f}s, "
            f"avg={sum(snapshot)/len(snapshot):.2f}s"
        )


def _start_stats_thread():
    """在后台线程中运行统计打印"""
    def _run():
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_stats_printer())
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── fast-graphrag 配置常量 ──
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


def group_questions_by_source(questions: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for q in questions:
        grouped.setdefault(q["source"], []).append(q)
    return grouped


def process_corpus(
    corpus_name: str,
    context: str,
    args: argparse.Namespace,
    questions: Dict[str, List[dict]],
) -> List[dict]:
    """Index one corpus and answer its questions."""
    log.info(f"Processing corpus: {corpus_name}")

    llm_service = OpenAILLMService(
        model=args.llm_model,
        base_url=args.base_url,
        api_key="no-key",
        max_requests_concurrent=20,
        rate_limit_per_second=True,
        max_requests_per_second=20,
    )
    embedding_service = OpenAIEmbeddingService(
        model=args.embed_model,
        base_url=args.base_url,
        api_key="no-key",
        embedding_dim=args.embed_dim,
    )

    grag = GraphRAG(
        working_dir=os.path.join(args.workspace, corpus_name),
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service,
        ),
    )

    # Index
    if args.skip_index:
        log.info(f"Skipping index for corpus: {corpus_name} (--skip-index)")
    else:
        log.info(f"Indexing corpus: {corpus_name} ({len(context.split())} words)")
        grag.insert(context)
        log.info(f"Indexing done: {corpus_name}")

    # Query
    corpus_questions = questions.get(corpus_name, [])
    if args.sample and args.sample < len(corpus_questions):
        corpus_questions = corpus_questions[: args.sample]
    log.info(f"Answering {len(corpus_questions)} questions for {corpus_name}")

    results = []
    for q in tqdm(corpus_questions, desc=corpus_name):
        try:
            response = grag.query(q["question"])
            resp_dict = response.to_dict()
            context_chunks = resp_dict.get("context", {}).get("chunks", [])
            contexts = [item[0]["content"] for item in context_chunks] if context_chunks else []
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": contexts,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": response.response,
                "ground_truth": q.get("answer", ""),
            })
        except Exception as e:
            log.error(f"Question {q['id']} failed: {e}")
            results.append({"id": q["id"], "question": q["question"], "error": str(e)})
    return results


def main():
    _start_stats_thread()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["medical", "novel"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--llm_model", required=True)
    parser.add_argument("--embed_model", required=True)
    parser.add_argument("--embed_dim", type=int, default=4096)
    parser.add_argument("--workspace", default="./test_workspace")
    parser.add_argument("--output", default="./predictions.json")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing, only run queries")
    args = parser.parse_args()

    benchmark = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GraphRAG-Benchmark")
    subset_paths = {
        "medical": {
            "corpus": os.path.join(benchmark, "Datasets/Corpus/medical.parquet"),
            "questions": os.path.join(benchmark, "Datasets/Questions/medical_questions.parquet"),
        },
        "novel": {
            "corpus": os.path.join(benchmark, "Datasets/Corpus/novel.parquet"),
            "questions": os.path.join(benchmark, "Datasets/Questions/novel_questions.parquet"),
        },
    }

    paths = subset_paths[args.subset]

    # Load corpus
    log.info(f"Loading corpus from {paths['corpus']}")
    corpus_ds = load_dataset("parquet", data_files=paths["corpus"], split="train")
    corpus_data = [{"corpus_name": r["corpus_name"], "context": r["context"]} for r in corpus_ds]
    log.info(f"Loaded {len(corpus_data)} corpus documents")

    if args.sample:
        corpus_data = corpus_data[:1]

    # Load questions
    log.info(f"Loading questions from {paths['questions']}")
    q_ds = load_dataset("parquet", data_files=paths["questions"], split="train")
    question_data = [{
        "id": r["id"], "source": r["source"], "question": r["question"],
        "answer": r["answer"], "question_type": r["question_type"], "evidence": r["evidence"],
    } for r in q_ds]
    grouped = group_questions_by_source(question_data)
    log.info(f"Loaded {len(question_data)} questions")

    # Process
    all_results = []
    for item in corpus_data:
        results = process_corpus(item["corpus_name"], item["context"], args, grouped)
        all_results.extend(results)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(all_results)} results to {args.output}")

    # Final stats
    with _stats_lock:
        snapshot = _stats.copy()
        _stats.clear()
    if snapshot:
        log.info(
            f"[STATS] Final: count={len(snapshot)}, "
            f"min={min(snapshot):.2f}s, max={max(snapshot):.2f}s, "
            f"avg={sum(snapshot)/len(snapshot):.2f}s"
        )


if __name__ == "__main__":
    main()
