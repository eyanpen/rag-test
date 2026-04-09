# GraphRAG-Benchmark 系统设计文档

> 版本：1.0 | 日期：2026-04-09 | 基于 GraphRAG-Bench (ICLR'26)

---

## 1. 项目概述

### 1.1 项目背景

GraphRAG-Benchmark（GraphRAG-Bench）是一个综合性基准测试框架，用于评估 **Graph Retrieval-Augmented Generation（图检索增强生成）** 模型的性能。该项目源自论文 *"When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation"*（arXiv:2506.05690），已被 ICLR'26 接收。

核心研究问题：**GraphRAG 在哪些场景下真正优于传统 RAG？图结构在什么条件下能为 RAG 系统带来可衡量的收益？**

### 1.2 系统目标

- 提供标准化的多领域（医学、小说）评测数据集
- 支持多种 GraphRAG 框架的统一索引构建与批量推理
- 覆盖从图构建（Indexing）→ 知识检索（Retrieval）→ 最终生成（Generation）的全流程评估
- 提供多层次难度的问题类型，从事实检索到创意生成

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     GraphRAG-Benchmark                          │
├─────────────┬──────────────────┬────────────────────────────────┤
│  Datasets   │    Examples      │         Evaluation             │
│  (数据层)    │   (框架适配层)    │         (评估层)                │
├─────────────┼──────────────────┼────────────────────────────────┤
│ Corpus/     │ run_lightrag.py  │ generation_eval.py             │
│  medical.*  │ run_fast-grag.py │ retrieval_eval.py              │
│  novel.*    │ run_hipporag2.py │ indexing_eval.py               │
│ Questions/  │ run_digimon.py   │ metrics/                       │
│  *_ques.*   │                  │   answer_accuracy.py           │
│             │                  │   coverage.py                  │
│             │                  │   faithfulness.py              │
│             │                  │   context_relevance.py         │
│             │                  │   context_relevance_v2.py      │
│             │                  │   evidence_recall.py           │
│             │                  │   rouge.py                     │
│             │                  │   utils.py                     │
│             │                  │ llm/                           │
│             │                  │   ollama_client.py             │
└─────────────┴──────────────────┴────────────────────────────────┘
```

### 2.2 三层架构详解

#### 第一层：数据层（Datasets/）

负责存储和管理评测数据，包含两个领域的语料库和对应问题集。

| 组件 | 路径 | 格式 | 说明 |
|------|------|------|------|
| 医学语料库 | `Datasets/Corpus/medical.*` | parquet / json | 医学领域文本语料 |
| 小说语料库 | `Datasets/Corpus/novel.*` | parquet / json | 文学/小说领域文本语料 |
| 医学问题集 | `Datasets/Questions/medical_questions.*` | parquet / json | 医学领域评测问题 |
| 小说问题集 | `Datasets/Questions/novel_questions.*` | parquet / json | 小说领域评测问题 |

**语料库数据结构：**

```json
{
  "corpus_name": "语料名称（用于分组和关联问题）",
  "context": "完整文本内容"
}
```

**问题集数据结构：**

```json
{
  "id": "问题唯一标识",
  "source": "对应语料名称（与 corpus_name 关联）",
  "question": "问题文本",
  "answer": "标准答案（ground truth）",
  "question_type": "问题类型（4种之一）",
  "evidence": ["支撑证据列表"]
}
```

**四种问题类型（难度递增）：**

| 级别 | 类型 | 说明 | 示例 |
|------|------|------|------|
| L1 | Fact Retrieval | 事实检索，直接从文本中找答案 | "Mont St. Michel 位于法国哪个地区？" |
| L2 | Complex Reasoning | 复杂推理，需要跨段落关联信息 | "Hinze 与 Felicia 的协议如何影响对英格兰统治者的看法？" |
| L3 | Contextual Summarize | 上下文摘要，需要理解并概括 | "John Curgenven 作为康沃尔船夫扮演什么角色？" |
| L4 | Creative Generation | 创意生成，需要基于事实进行创作 | "将亚瑟王与 John Curgenven 的比较改写为新闻报道" |

#### 第二层：框架适配层（Examples/）

为每个 GraphRAG 框架提供统一的适配脚本，完成索引构建和批量推理。

| 框架 | 脚本 | 版本要求 | 特殊适配 |
|------|------|----------|----------|
| LightRAG | `run_lightrag.py` | v1.2.5 | 需修改源码使 `kg_query` 和 `aquery` 返回 context |
| fast-graphrag | `run_fast-graphrag.py` | - | 需在 `_llm/` 下新增 `_hf.py` 添加 HuggingFace Embedding 支持 |
| HippoRAG2 | `run_hipporag2.py` | v1.0.0 | 需在 `embedding_model/` 下新增 `BGE.py` 添加 BGE Embedding 支持 |
| DIGIMON | `run_digimon.py` | - | 需移入 DIGIMON 项目目录，配合其 YAML 配置文件使用 |

**统一输出格式（所有框架必须遵循）：**

```json
{
  "id": "问题ID",
  "question": "问题文本",
  "source": "语料名称",
  "context": ["检索到的上下文片段列表"],
  "evidence": ["标准证据列表"],
  "question_type": "问题类型",
  "generated_answer": "模型生成的答案",
  "ground_truth": "标准答案"
}
```

这个统一格式是整个系统的核心契约——只要框架输出符合此格式，评估代码即可直接运行。

#### 第三层：评估层（Evaluation/）

提供标准化的评估流程，覆盖三个维度：图构建质量、检索质量、生成质量。

---

## 3. 评估体系设计

### 3.1 三维评估框架

```
                    ┌──────────────────┐
                    │   Indexing Eval   │  ← 图结构质量（纯图论指标，不依赖 LLM）
                    │  (indexing_eval)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Retrieval Eval   │  ← 检索质量（LLM 辅助评估）
                    │ (retrieval_eval)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Generation Eval   │  ← 生成质量（LLM + Embedding 评估）
                    │(generation_eval)  │
                    └──────────────────┘
```

### 3.2 Indexing 评估（图构建质量）

**文件：** `Evaluation/indexing_eval.py`

分析知识图谱的结构特征，纯图论计算，不依赖 LLM。

| 指标 | 说明 |
|------|------|
| `num_nodes` / `num_edges` | 节点数 / 边数 |
| `average_degree` | 平均度 |
| `density` | 图密度 |
| `num_components` | 连通分量数 |
| `largest_component_size` | 最大连通分量大小 |
| `average_clustering_coefficient` | 平均聚类系数 |
| `diameter` | 图直径（不连通时为 ∞） |
| `average_component_size` | 平均连通分量大小（排除孤立节点） |
| `median_component_size` | 连通分量大小中位数 |
| `trimmed_mean_component_size` | 截尾均值（去掉最大最小值） |
| `geometric_mean_component_size` | 几何均值 |
| `harmonic_mean_component_size` | 调和均值 |
| `num_isolated_nodes` | 孤立节点数 |
| `num_nodes_degree_above_1/2/3` | 度 > 1/2/3 的节点数 |

**支持的图格式加载：**

| 框架 | 图文件格式 | 加载函数 |
|------|-----------|---------|
| Microsoft GraphRAG | `entities.parquet` + `relationships.parquet` | `load_graph_from_parquet()` |
| LightRAG | `graph_chunk_entity_relation.graphml` | `load_graph_from_graphml()` |
| fast-graphrag | `graph_igraph_data.pklz` | `load_graph_from_picklez()` |
| HippoRAG2 | `graph.pickle` | `load_graph_from_pickle()` |
| 通用 GraphML | `*.graphml` | `load_graph_from_graphml()` |

### 3.3 Retrieval 评估（检索质量）

**文件：** `Evaluation/retrieval_eval.py`

| 指标 | 实现文件 | 说明 |
|------|---------|------|
| Context Relevancy | `metrics/context_relevance.py` | LLM 对上下文评分（0-2 量化），执行两次独立评分取均值，归一化到 0-1 |
| Context Relevancy v2 | `metrics/context_relevance_v2.py` | 增强版，基于 Evidence 评估，支持长文本自动分块（>5000 字符时按 3000 分块） |
| Evidence Recall | `metrics/evidence_recall.py` | LLM 逐条判断标准证据是否被检索上下文覆盖，计算覆盖率 |

**Context Relevancy 评分标准：**
- 0 分：上下文不包含任何与问题相关的信息
- 1 分：上下文部分包含相关信息
- 2 分：上下文完全包含回答问题所需的信息

### 3.4 Generation 评估（生成质量）

**文件：** `Evaluation/generation_eval.py`

不同问题类型使用不同指标组合：

| 问题类型 | 评估指标 |
|----------|---------|
| Fact Retrieval | ROUGE-L + Answer Correctness |
| Complex Reasoning | ROUGE-L + Answer Correctness |
| Contextual Summarize | Answer Correctness + Coverage Score |
| Creative Generation | Answer Correctness + Coverage Score + Faithfulness |

**各指标详解：**

| 指标 | 实现文件 | 计算方式 |
|------|---------|---------|
| **ROUGE-L** | `metrics/rouge.py` | 基于 `rouge_score` 库计算生成答案与标准答案的 ROUGE-L F-measure，纯文本匹配，不依赖 LLM |
| **Answer Correctness** | `metrics/answer_accuracy.py` | 加权组合：**75% 事实性** + **25% 语义相似度**。事实性：LLM 将答案和标准答案拆分为原子陈述，分类为 TP/FP/FN，计算 F-beta 分数。语义相似度：Embedding 余弦相似度归一化到 [0,1] |
| **Coverage Score** | `metrics/coverage.py` | 两步 LLM 评估：(1) 从标准答案提取事实列表 (2) 判断生成答案覆盖了多少事实，计算覆盖率 |
| **Faithfulness** | `metrics/faithfulness.py` | 两步 LLM 评估：(1) 将生成答案拆分为原子陈述 (2) 判断每条陈述是否被检索上下文支持，计算支持率 |

---

## 4. LLM 接入设计

### 4.1 双模式架构

```
┌─────────────────────────────────────┐
│           LLM 接入层                 │
├──────────────┬──────────────────────┤
│   API 模式    │    Ollama 模式       │
│ (OpenAI兼容)  │   (本地部署)          │
├──────────────┼──────────────────────┤
│ ChatOpenAI   │ OllamaClient         │
│ (langchain)  │ OllamaWrapper        │
│              │ (自定义 aiohttp)      │
└──────────────┴──────────────────────┘
```

### 4.2 API 模式

- 使用 `langchain_openai.ChatOpenAI`
- 兼容所有 OpenAI API 格式的服务（OpenAI、Azure、本地 vLLM 等）
- 通过环境变量 `LLM_API_KEY`（或 `OPENAI_API_KEY`）传递密钥
- 默认参数：`temperature=0.0`, `max_retries=3`, `timeout=30`, `seed=42`

### 4.3 Ollama 模式

- 自定义 `OllamaClient`（基于 `aiohttp`），位于 `Evaluation/llm/ollama_client.py`
- 异步 HTTP 调用 Ollama `/api/chat` 端点
- 指数退避重试机制：最多 3 次，基础延迟 2s，按 `2^attempt` 递增
- 连接池管理：总连接上限 10，单主机上限 5，DNS 缓存 300s
- 超时配置：总超时 180s，连接超时 15s，读取超时 120s
- `OllamaWrapper` 封装为与 LangChain 兼容的 `ainvoke` 接口

### 4.4 Embedding 模型

| 使用场景 | Embedding 方案 |
|---------|---------------|
| 评估（API 模式） | `HuggingFaceBgeEmbeddings`（本地加载 HuggingFace 模型） |
| 评估（Ollama 模式） | `OllamaEmbeddings`（调用 Ollama Embedding API） |
| LightRAG 框架 | `hf_embed`（HuggingFace）或 `ollama_embed` |
| fast-graphrag 框架 | 自定义 `HuggingFaceEmbeddingService`（需手动添加） |
| HippoRAG2 框架 | 自定义 `BGEEmbeddingModel`（需手动添加） |

---

## 5. 数据流设计

### 5.1 完整评测流程

```
[1] 加载数据集
     │
     ├── Corpus (parquet) ──→ 按 corpus_name 分组
     └── Questions (parquet) ──→ 按 source 分组（group_questions_by_source）
     │
[2] 索引构建 (Index)
     │
     ├── 各框架适配脚本加载语料文本
     ├── 调用框架 insert/index API 构建知识图谱
     └── 图数据持久化到 workspace/<corpus_name>/ 目录
     │
[3] 批量推理 (Inference)
     │
     ├── 遍历问题集（tqdm 进度条）
     ├── 调用框架 query API 获取 (response, context)
     ├── 收集结果为统一 JSON 格式
     └── 保存到 ./results/<framework>/<corpus_name>/predictions_*.json
     │
[4] 评估 (Evaluation)
     │
     ├── Indexing Eval ──→ 遍历 workspace 目录分析图结构
     ├── Retrieval Eval ──→ 按问题类型分组 → Context Relevancy + Evidence Recall
     └── Generation Eval ──→ 按问题类型分组 → 选择对应指标组合评估
     │
[5] 结果输出
     └── JSON 格式评估报告（支持 detailed_output 模式）
```

### 5.2 并发模型

| 阶段 | 并发策略 | 说明 |
|------|---------|------|
| 索引构建 | `asyncio.gather` | 多个语料并发构建索引 |
| 批量推理 | 串行 + 框架内部异步 | LightRAG 支持 `llm_model_max_async=4` |
| 评估 | `asyncio.Semaphore` | Generation 默认 `max_concurrent=3`，Retrieval 默认 `max_concurrent=1` |

---

## 6. JSON 容错设计

系统通过 `Evaluation/metrics/utils.py` 中的 `JSONHandler` 类实现多层级 JSON 解析容错，这对于处理 LLM 输出的不规范 JSON 至关重要：

```
解析策略（按优先级依次尝试）：
┌─────────────────────────────────────┐
│ 1. json.loads()        标准解析      │
│ 2. json5.loads()       宽松 JSON     │
│ 3. json_repair()       自动修复      │
│ 4. 正则提取 JSON 块    {...} 匹配    │
│ 5. 正则提取数组        [...] 回退    │
│ 6. LLM 自愈（可选）    重新生成      │
└─────────────────────────────────────┘
```

- 步骤 1-3：尝试直接解析原始文本
- 步骤 4：用正则 `\{[\s\S]*\}` 提取第一个 JSON 块后重新解析
- 步骤 5：用正则提取数组内容作为最后回退
- 步骤 6：当 `self_healing=True` 时，将无效输出发给 LLM 要求返回合法 JSON

---

## 7. 完整目录结构

```
GraphRAG-Benchmark/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python 依赖清单
├── LICENSE                            # MIT 许可证
├── pipeline.jpg                       # 评测流程图
├── RAGvsGraphRAG.jpg                  # RAG vs GraphRAG 对比图
│
├── Datasets/                          # 评测数据集
│   ├── Corpus/                        # 语料库
│   │   ├── medical.parquet            # 医学语料（Parquet 格式）
│   │   ├── medical.json               # 医学语料（JSON 格式）
│   │   ├── novel.parquet              # 小说语料（Parquet 格式）
│   │   └── novel.json                 # 小说语料（JSON 格式）
│   └── Questions/                     # 问题集
│       ├── medical_questions.parquet  # 医学问题（Parquet 格式）
│       ├── medical_questions.json     # 医学问题（JSON 格式）
│       ├── novel_questions.parquet    # 小说问题（Parquet 格式）
│       └── novel_questions.json       # 小说问题（JSON 格式）
│
├── Examples/                          # 框架适配脚本
│   ├── README.md                      # 框架使用说明
│   ├── run_lightrag.py                # LightRAG 适配（异步）
│   ├── run_fast-graphrag.py           # fast-graphrag 适配（线程池）
│   ├── run_hipporag2.py               # HippoRAG2 适配（线程池）
│   └── run_digimon.py                 # DIGIMON 适配（异步）
│
└── Evaluation/                        # 评估模块
    ├── __init__.py
    ├── README.md                      # 评估使用说明
    ├── generation_eval.py             # 生成质量评估入口
    ├── retrieval_eval.py              # 检索质量评估入口
    ├── indexing_eval.py               # 图构建质量评估入口
    ├── metrics/                       # 评估指标实现
    │   ├── __init__.py                # 指标导出注册
    │   ├── answer_accuracy.py         # 答案正确性（F-beta + 语义相似度）
    │   ├── coverage.py                # 事实覆盖率
    │   ├── faithfulness.py            # 忠实度（上下文支持率）
    │   ├── context_relevance.py       # 上下文相关性 v1
    │   ├── context_relevance_v2.py    # 上下文相关性 v2（基于 Evidence）
    │   ├── evidence_recall.py         # 证据召回率
    │   ├── rouge.py                   # ROUGE-L 分数
    │   └── utils.py                   # JSON 容错解析工具（JSONHandler）
    └── llm/                           # LLM 客户端封装
        ├── __init__.py                # 导出 OllamaClient, OllamaWrapper
        └── ollama_client.py           # Ollama 异步客户端（aiohttp）
```

---

## 8. 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.10+ | 全项目 |
| 异步框架 | asyncio, aiohttp | 并发推理与评估、Ollama 客户端 |
| LLM 集成 | LangChain (langchain, langchain_openai, langchain_ollama) | API 模式 LLM 调用 |
| 数据处理 | datasets (HuggingFace), numpy | 加载 Parquet 数据集 |
| 图分析 | igraph, pandas | 知识图谱结构分析 |
| 评估指标 | ragas, rouge_score | RAG 评估基础库 |
| 数据验证 | pydantic | 数据模型定义与校验 |
| JSON 容错 | json5, json_repair | LLM 输出解析容错 |
| Embedding | transformers, torch | 本地 Embedding 模型加载 |

---

## 9. 扩展性设计

### 9.1 新增 GraphRAG 框架

只需在 `Examples/` 下编写新的 `run_xxx.py` 适配脚本，确保输出符合第 2.2 节定义的统一 JSON 格式。评估代码无需任何修改。

### 9.2 新增评估指标

1. 在 `Evaluation/metrics/` 下新增指标实现文件（异步函数）
2. 在 `Evaluation/metrics/__init__.py` 中注册导出
3. 在 `generation_eval.py` 或 `retrieval_eval.py` 的 `metric_config` 中配置使用

### 9.3 新增数据集领域

1. 在 `Datasets/Corpus/` 下添加新领域的 `.parquet` 和 `.json` 文件
2. 在 `Datasets/Questions/` 下添加对应问题集
3. 在各 `run_*.py` 的 `SUBSET_PATHS` 字典中注册新路径
4. 在 `argparse` 的 `choices` 中添加新的 subset 选项
