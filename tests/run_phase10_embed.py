#!/usr/bin/env python3
"""Phase 10 Embedding + Graph Sync 专用脚本

仅执行 Phase 10（向量嵌入）× N 模型 × (dual-tower if supported) + graph sync。
前提：Phase 1-9 的 parquet 产物已存在于 workspace/{dataset}/output/ 下。
查询和评估由后续脚本完成。

用法:
  python tests/run_phase10_embed.py --dataset medical --models "BAAI/bge-m3"
  python tests/run_phase10_embed.py --dataset all --dual-tower both
"""
import asyncio
import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import litellm
litellm.drop_params = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "GraphRAG-Benchmark"))
sys.path.insert(0, SCRIPT_DIR)

from models import (
    BenchmarkConfig,
    DualTowerMode,
    EMBEDDING_MODELS,
    EmbeddingModelConfig,
    make_graph_name,
    sanitize_name,
)
from concurrency import AdaptiveConcurrencyController
from graph_sync import sync_graph_to_falkordb

log = logging.getLogger(__name__)

# ── Global state for dual-tower prefix injection ──
_embedding_prefix: str = ""


def _install_prefix_hook():
    """Extend httpx hooks to inject text prefix into /embeddings requests."""
    _orig_async_send = httpx.AsyncClient.send
    _orig_sync_send = httpx.Client.send

    async def _async_send_with_prefix(self_client, request, *args, **kwargs):
        _inject_prefix(request)
        return await _orig_async_send(self_client, request, *args, **kwargs)

    def _sync_send_with_prefix(self_client, request, *args, **kwargs):
        _inject_prefix(request)
        return _orig_sync_send(self_client, request, *args, **kwargs)

    httpx.AsyncClient.send = _async_send_with_prefix
    httpx.Client.send = _sync_send_with_prefix


def _inject_prefix(request):
    """If _embedding_prefix is set, prepend it to all text inputs in /embeddings requests."""
    global _embedding_prefix
    if not _embedding_prefix:
        return
    if request.method != "POST" or not request.content:
        return
    try:
        url = str(request.url)
        if "/embeddings" not in url:
            return
        body = json.loads(request.content)
        inp = body.get("input")
        if inp is None:
            return
        if isinstance(inp, list):
            body["input"] = [_embedding_prefix + t if isinstance(t, str) else t for t in inp]
        elif isinstance(inp, str):
            body["input"] = _embedding_prefix + inp
        else:
            return
        new_content = json.dumps(body).encode("utf-8")
        request._content = new_content
        request.headers["content-length"] = str(len(new_content))
    except Exception:
        pass


def set_embedding_prefix(prefix: str):
    global _embedding_prefix
    _embedding_prefix = prefix
    if prefix:
        log.info(f"[PREFIX] embedding prefix set to: {prefix!r}")
    else:
        log.info("[PREFIX] embedding prefix cleared")


# ── GraphRAG config builder (copied core from run_embedding_benchmark.py) ──


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

    return GraphRagConfig(
        completion_models=completion_models,
        embedding_models=embedding_models,
        input_storage=StorageConfig(base_dir=str(Path(input_dir).resolve())),
        output_storage=StorageConfig(base_dir=str(Path(output_dir).resolve())),
        update_output_storage=StorageConfig(base_dir=str(Path(output_dir).resolve()) + "/update"),
        cache=CacheConfig(storage=StorageConfig(base_dir=str(Path(cache_dir).resolve()))),
        table_provider=TableProviderConfig(),
        vector_store=vector_store,
    )


# ── Phase 10 execution ──


async def run_phase_10(
    config: BenchmarkConfig,
    dataset_name: str,
    workspace_dir: str,
    model_config: EmbeddingModelConfig,
    graph_name: str,
) -> float:
    """Execute Phase 10 for a specific embedding model. Returns elapsed seconds."""
    from graphrag.index.workflows.generate_text_embeddings import run_workflow
    from graphrag.index.run.utils import create_run_context
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider
    from graphrag_cache import create_cache
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

    input_dir = os.path.join(workspace_dir, "input")
    output_dir = os.path.join(workspace_dir, "output")
    cache_dir = os.path.join(workspace_dir, "cache")

    # Clear embedding cache to force re-embed
    emb_cache = os.path.join(cache_dir, "text_embedding")
    if os.path.isdir(emb_cache):
        shutil.rmtree(emb_cache)

    grc = _build_graphrag_config(
        config, dataset_name, input_dir, output_dir, cache_dir,
        embedding_model_name=model_config.name,
        embedding_dim=model_config.dim,
        graph_name=graph_name,
    )

    output_storage = create_storage(grc.output_storage)
    output_table_provider = create_table_provider(grc.table_provider, output_storage)
    cache = create_cache(grc.cache)

    context = create_run_context(
        output_storage=output_storage,
        output_table_provider=output_table_provider,
        cache=cache,
        callbacks=NoopWorkflowCallbacks(),
    )

    import falkordb_vector_store  # noqa: F401 — registers "falkordb" type

    t0 = time.time()
    await run_workflow(grc, context)
    elapsed = time.time() - t0
    log.info(f"Phase 10 done: {model_config.display_name}/{dataset_name} → graph '{graph_name}' in {elapsed:.1f}s")
    return elapsed


# ── Model availability ──


def check_model_availability(api_base_url: str, model: EmbeddingModelConfig) -> bool:
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


# ── Job definition ──


def build_jobs(
    models: List[EmbeddingModelConfig],
    datasets: List[str],
    dual_tower_mode: str,
) -> List[Tuple[EmbeddingModelConfig, str, DualTowerMode]]:
    """Build list of (model, dataset, mode) jobs to execute.

    dual_tower_mode: "both" | "symmetric" | "asymmetric"
    """
    jobs = []
    for ds in datasets:
        for model in models:
            if dual_tower_mode in ("both", "symmetric"):
                jobs.append((model, ds, DualTowerMode.SYMMETRIC))
            if dual_tower_mode in ("both", "asymmetric") and model.supports_dual_tower:
                jobs.append((model, ds, DualTowerMode.ASYMMETRIC))
    return jobs


# ── Main ──


async def async_main(config: BenchmarkConfig, dual_tower_mode: str):
    t_start = time.time()
    results = []

    # Verify Phase 1-9 parquet exists
    required_files = [
        "entities.parquet", "relationships.parquet", "text_units.parquet",
        "community_reports.parquet", "communities.parquet",
    ]
    valid_datasets = []
    for ds_name in config.datasets:
        output_dir = os.path.join(config.output_dir, "workspace", ds_name, "output")
        missing = [f for f in required_files if not os.path.isfile(os.path.join(output_dir, f))]
        if missing:
            log.error(f"Dataset {ds_name}: missing Phase 1-9 parquet: {missing}, skipping")
            continue
        valid_datasets.append(ds_name)

    if not valid_datasets:
        log.error("No valid datasets with Phase 1-9 output. Exiting.")
        return

    # Check model availability
    available_models = []
    for model in config.models:
        if check_model_availability(config.api_base_url, model):
            available_models.append(model)
            log.info(f"✓ {model.display_name} available")
        else:
            log.warning(f"✗ {model.display_name} unavailable, skipping")

    if not available_models:
        log.error("No available models. Exiting.")
        return

    # Build job list
    jobs = build_jobs(available_models, valid_datasets, dual_tower_mode)
    log.info(f"=== {len(jobs)} jobs to execute ===")
    for i, (model, ds, mode) in enumerate(jobs, 1):
        mode_tag = " [dual-tower]" if mode == DualTowerMode.ASYMMETRIC else ""
        log.info(f"  {i}. {model.display_name} × {ds}{mode_tag}")

    # Execute jobs
    for i, (model, ds_name, mode) in enumerate(jobs, 1):
        is_dual = mode == DualTowerMode.ASYMMETRIC
        mode_label = "dual-tower" if is_dual else "symmetric"
        graph_name = make_graph_name(model.name, ds_name, dual_tower=is_dual)
        workspace_dir = os.path.join(config.output_dir, "workspace", ds_name)

        log.info(f"=== [{i}/{len(jobs)}] {model.display_name} × {ds_name} ({mode_label}) → {graph_name} ===")

        result = {
            "model": model.name,
            "display_name": model.display_name,
            "dataset": ds_name,
            "mode": mode_label,
            "graph_name": graph_name,
            "phase10_time": 0.0,
            "graph_sync_time": 0.0,
            "error": None,
        }

        # Set prefix for dual-tower document encoding
        if is_dual:
            set_embedding_prefix(model.document_prefix)
            log.info(f"  Document prefix: {model.document_prefix!r}")
        else:
            set_embedding_prefix("")

        # Phase 10
        try:
            result["phase10_time"] = await run_phase_10(
                config, ds_name, workspace_dir, model, graph_name,
            )
        except Exception as e:
            log.error(f"  Phase 10 FAILED: {e}")
            result["error"] = str(e)
            results.append(result)
            continue
        finally:
            set_embedding_prefix("")  # always clear

        # Graph sync
        try:
            t0 = time.time()
            sync_graph_to_falkordb(
                output_dir=os.path.join(workspace_dir, "output"),
                host=config.falkordb_host,
                port=config.falkordb_port,
                graph_name=graph_name,
            )
            result["graph_sync_time"] = time.time() - t0
        except Exception as e:
            log.warning(f"  Graph sync failed (non-fatal): {e}")

        results.append(result)
        log.info(f"  Done: phase10={result['phase10_time']:.1f}s, sync={result['graph_sync_time']:.1f}s")

    # Save results
    total_time = time.time() - t_start
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "dual_tower_mode": dual_tower_mode,
        "jobs": results,
    }
    out_path = os.path.join(config.output_dir, "phase10_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"\n=== All done in {total_time:.1f}s ===")
    log.info(f"Results: {out_path}")

    # Summary table
    log.info(f"\n{'Model':<30} {'Dataset':<10} {'Mode':<12} {'Phase10':>10} {'Sync':>8} {'Status'}")
    log.info("-" * 85)
    for r in results:
        status = "✓" if not r["error"] else f"✗ {r['error'][:30]}"
        log.info(f"{r['display_name']:<30} {r['dataset']:<10} {r['mode']:<12} "
                 f"{r['phase10_time']:>9.1f}s {r['graph_sync_time']:>7.1f}s {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 10 Embedding + Graph Sync（不含查询和评估）")
    parser.add_argument("--dataset", default="all", choices=["medical", "novel", "all"])
    parser.add_argument("--models", default="", help="逗号分隔模型名，默认全部")
    parser.add_argument("--dual-tower", default="both",
                        choices=["both", "symmetric", "asymmetric"],
                        help="both=对称+双塔, symmetric=仅对称, asymmetric=仅双塔")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "benchmark_results"))
    parser.add_argument("--graphrag-root",
                        default="/home/eyanpen/sourceCode/rnd-ai-engine-features/graphrag")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "phase10_embed.log"),
                mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logging.getLogger("LiteLLM").setLevel(logging.INFO)
    logging.getLogger("litellm").setLevel(logging.INFO)

    models = list(EMBEDDING_MODELS)
    if args.models:
        names = [n.strip() for n in args.models.split(",")]
        models = [m for m in EMBEDDING_MODELS if m.name in names]
        if not models:
            print(f"No matching models for: {args.models}")
            print(f"Available: {[m.name for m in EMBEDDING_MODELS]}")
            sys.exit(1)

    datasets = ["medical", "novel"] if args.dataset == "all" else [args.dataset]

    config = BenchmarkConfig(
        datasets=datasets,
        models=models,
        output_dir=args.output_dir,
        graphrag_root=args.graphrag_root,
    )

    ac = AdaptiveConcurrencyController()
    ac.install_hooks()
    _install_prefix_hook()

    asyncio.run(async_main(config, args.dual_tower))


if __name__ == "__main__":
    main()
