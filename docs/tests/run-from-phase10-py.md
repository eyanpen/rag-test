# tests/run_from_phase10.py 代码解读

> 源文件：`tests/run_from_phase10.py`
> 用途：从 Phase 10 开始执行 Embedding 基准测试（跳过 Phase 1-9），对多个 Embedding 模型进行对比评测

---

## 1. 文件定位与设计意图

这个脚本是 `run_embedding_benchmark.py`（完整流水线）的**加速版本**。完整流水线需要先跑 Phase 1-9（文本分块、实体抽取、社区检测等），耗时很长且与 Embedding 模型无关。当 Phase 1-9 的 parquet 产物已经存在时，本脚本直接从 Phase 10（生成文本向量嵌入）开始，大幅缩短迭代周期。

**核心流程**：

```
Phase 10 (Embedding) × N 个模型 → Graph Sync → Query → Evaluate → Report
```

---

## 2. 依赖模块关系

```
run_from_phase10.py
├── models.py              # 数据结构定义（BenchmarkConfig, ModelResult, EMBEDDING_MODELS 等）
├── concurrency.py         # 自适应并发控制器，monkey-patch httpx 实现限流
├── dataset_loader.py      # 加载 medical/novel 数据集的语料和问题
├── graph_sync.py          # 将 parquet 中的实体/关系同步到 FalkorDB 图数据库
├── report_generator.py    # 生成 Markdown 报告和 JSON 摘要
└── run_embedding_benchmark.py  # 复用其中的核心函数：
    ├── run_phase_10()           # 执行 GraphRAG Phase 10 向量嵌入
    ├── check_model_availability() # 检查模型 API 是否可用
    ├── run_queries()            # 用 GraphRAG local search 执行问答
    └── evaluate_predictions()   # 计算 ROUGE-L 和 Answer Correctness
```

---

## 3. 执行流程详解

### 3.1 入口 `main()`

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--dataset", default="all", choices=["medical", "novel", "all"])
    parser.add_argument("--models", default="", help="逗号分隔模型名，默认全部")
    ...
```

- 解析命令行参数（采样数、数据集、模型列表、输出目录等）
- 配置日志：控制台 INFO 级别 + 文件 DEBUG 级别
- 安装 `AdaptiveConcurrencyController` 的 httpx hooks（自适应限流 + 去除 `encoding_format=None`）
- 调用 `asyncio.run(async_main(config))`

**关键设计**：`litellm.drop_params = True` 在文件顶部设置，让 litellm 自动丢弃目标 API 不支持的参数，避免 400 错误。

### 3.2 Step 1：加载问题 & 验证 Phase 1-9 产物

```python
required = ["entities.parquet", "relationships.parquet", "text_units.parquet",
             "community_reports.parquet", "communities.parquet"]
missing = [f for f in required if not os.path.isfile(os.path.join(output_dir, f))]
```

- 遍历每个数据集（medical / novel），检查 `workspace/{dataset}/output/` 下是否存在 5 个必需的 parquet 文件
- 缺失则记录错误并跳过该数据集
- 通过 `DatasetLoader.prepare_medical()` / `prepare_novel()` 加载问题集
- 如果设置了 `sample_size`，截取前 N 个问题

### 3.3 Step 2：模型可用性检查

```python
for model in config.models:
    if check_model_availability(config.api_base_url, model):
        available_models.append(model)
```

对 `EMBEDDING_MODELS` 列表中的每个模型调用 API 检查是否可用。不可用的模型直接标记为 "Model unavailable" 并跳过。

当前支持的模型（定义在 `models.py` 的 `EMBEDDING_MODELS`）：

| 模型 | 维度 | 最大 Token | 双塔支持 |
|------|------|-----------|---------|
| BAAI/bge-m3 (default/heavy/interactive) | 1024 | 8192 | ✅ query:/passage: |
| intfloat/e5-mistral-7b-instruct | 4096 | 4096 | ✅ Instruct prefix |
| intfloat/multilingual-e5-large-instruct | 1024 | 512 | ✅ query:/passage: |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 8192 | ✅ search_query:/search_document: |
| Qwen/Qwen3-Embedding-8B | 4096 | 8192 | ✅ Instruct prefix |
| Qwen/Qwen3-Embedding-8B-Alt | 4096 | 32768 | ✅ (当前 API 返回 500) |

### 3.4 Step 3：Phase 10 + Query + Evaluate（核心循环）

对每个 `(数据集, 模型)` 组合执行三步：

#### Phase 10：生成向量嵌入

```python
mr.embedding_time_seconds = await run_phase_10(config, ds_name, workspace_dir, model)
```

- 调用 GraphRAG 的 `generate_text_embeddings` workflow
- 清除旧的 embedding 缓存（`cache/text_embedding/`），确保用新模型重新计算
- 使用 `FalkorDBVectorStore` 作为向量存储后端
- 返回耗时（秒）

#### Graph Sync：同步到 FalkorDB

```python
sync_graph_to_falkordb(
    output_dir=...,
    host=config.falkordb_host, port=config.falkordb_port,
    graph_name=make_graph_name(model.name, ds_name),
)
```

- 读取 `entities.parquet` 和 `relationships.parquet`
- 用 Cypher `MERGE` 语句将节点和边写入 FalkorDB（幂等操作）
- 图名格式：`{dataset}_{sanitized_model_name}`，如 `medical_baai-bge-m3`

#### Query：执行问答

```python
mr.predictions = await run_queries(config, ds_name, workspace_dir, model, questions)
```

- 构建 GraphRAG local search engine
- 从 FalkorDB 向量存储中检索相关实体
- 对每个问题执行 `search_engine.search()`，收集生成的答案和上下文
- 结果保存为 `PredictionItem` 列表

#### Evaluate：评估质量

```python
mr.evaluation, question_details = await evaluate_predictions(
    config, mr.predictions, model.display_name, ds_name)
```

两个评估指标：
- **ROUGE-L**：生成答案与 ground truth 的文本重叠度
- **Answer Correctness**：基于 LLM 的语义正确性评估（使用 DeepSeek-V3.2 做 judge，BGE-M3 做 embedding 相似度）

评估结果保存到 `evaluations/{model}__{dataset}.json` 和 `_details.json`。

### 3.5 Step 4：生成报告

```python
report = ReportGenerator.generate_markdown(summary)
summary_data = ReportGenerator.generate_summary_json(summary)
```

输出两个文件：
- `benchmark_report.md`：包含测试环境、模型清单、指标对比表、BGE-M3 模式对比、总体排名、耗时统计
- `summary.json`：结构化的 JSON 摘要，便于程序化消费

---

## 4. 关键设计决策

### 4.1 为什么跳过 Phase 1-9？

Phase 1-9（文本分块 → 实体抽取 → 关系抽取 → 社区检测 → 社区报告生成）是 GraphRAG 的知识图谱构建阶段，使用 LLM 而非 Embedding 模型。这些阶段的产物（parquet 文件）对所有 Embedding 模型是共享的。跳过它们可以：
- 节省大量 LLM 调用时间和成本
- 专注于 Embedding 模型的对比评测
- 支持快速迭代（改模型配置后秒级重跑）

### 4.2 自适应并发控制

`AdaptiveConcurrencyController` 通过 monkey-patch `httpx.AsyncClient` 实现：
- 初始并发 10，范围 [2, 50]
- 连续 5 次成功 → 并发 +1
- 遇到 5xx 错误 → 并发 -1
- 同时修复 `encoding_format=None` 问题（某些 Embedding API 不接受此参数）

### 4.3 图名命名策略

```python
def make_graph_name(model_name, dataset_name, dual_tower=False):
    base = sanitize_name(model_name)  # "/" → "-", 转小写
    return dataset_name + "_" + base  # e.g. "medical_baai-bge-m3"
```

每个 `(模型, 数据集)` 组合对应一个独立的 FalkorDB 图，避免不同模型的向量数据互相污染。

---

## 5. 输出目录结构

```
benchmark_results/
├── benchmark.log                          # 完整 DEBUG 日志
├── benchmark_report.md                    # Markdown 报告
├── summary.json                           # JSON 摘要
├── predictions/
│   ├── baai-bge-m3__medical.json         # 每个模型×数据集的预测结果
│   └── ...
├── evaluations/
│   ├── baai-bge-m3__medical.json         # 评估指标
│   ├── baai-bge-m3__medical_details.json # 逐题评估详情
│   └── ...
└── workspace/
    ├── medical/
    │   ├── input/                         # 语料文本
    │   ├── output/                        # Phase 1-9 parquet + Phase 10 embedding
    │   └── cache/
    └── novel/
        └── ...
```

---

## 6. 命令行用法

```bash
# 默认：全部模型 × 全部数据集，采样 5 题
python tests/run_from_phase10.py

# 指定模型和数据集
python tests/run_from_phase10.py --models "BAAI/bge-m3,Qwen/Qwen3-Embedding-8B" --dataset medical --sample 10

# 自定义输出目录
python tests/run_from_phase10.py --output-dir /tmp/bench --sample 20
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample` | 5 | 每个数据集采样的问题数 |
| `--dataset` | all | 数据集选择：medical / novel / all |
| `--models` | 全部 | 逗号分隔的模型名（对应 `EmbeddingModelConfig.name`） |
| `--output-dir` | `tests/benchmark_results` | 输出根目录 |
| `--data-root` | `GraphRAG-Benchmark/Datasets` | 数据集根目录 |
| `--graphrag-root` | `/home/.../graphrag` | GraphRAG 安装路径 |
