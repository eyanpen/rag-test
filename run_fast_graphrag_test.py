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
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

# fast-graphrag imports
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, OpenAIEmbeddingService

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["medical", "novel"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--llm_model", required=True)
    parser.add_argument("--embed_model", required=True)
    parser.add_argument("--embed_dim", type=int, default=4096)
    parser.add_argument("--workspace", default="./test_workspace")
    parser.add_argument("--output", default="./predictions.json")
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


if __name__ == "__main__":
    main()
