# Tests 代码解读：双塔模式分析与测试对比改动规划

## 1. 代码架构总览

### 1.1 测试系统组成

```
tests/
├── run_embedding_benchmark.sh    # Shell 入口：依赖检查、API 连通性、调用 Python
├── run_embedding_benchmark.py    # 主执行器：Pipeline 编排、查询、评估
├── models.py                     # 数据模型 + EMBEDDING_MODELS 常量
├── falkordb_vector_store.py      # FalkorDB 自定义 VectorStore（GraphRAG 插件）
├── concurrency.py                # 自适应并发控制 + httpx monkey-patch
├── dataset_loader.py             # 数据集加载（medical / novel）
├── graph_sync.py                 # 将 parquet 实体/关系同步到 FalkorDB 图
├── report_generator.py           # Markdown + JSON 报告生成
├── rerun_evaluation.py           # 仅重跑评估（复用已有 predictions）
└── test_rootcause_embedding_input_format.py  # Root Cause 验证测试
```

### 1.2 执行流程

```
Shell 入口 → Python 主执行器
  ├── 每个数据集:
  │   ├── DatasetLoader 加载语料 + 问题
  │   ├── Prompt-tune（生成提取 prompt）
  │   └── Phase 1-9（GraphRAG pipeline 共享阶段，只执行一次）
  │
  └── 每个模型 × 数据集:
      ├── Phase 10（generate_text_embeddings，切换 embedding 模型）
      ├── Graph Sync（实体/关系写入 FalkorDB）
      ├── GraphRAG Local Search（查询问题，生成答案）
      └── 评估（ROUGE-L + Answer Correctness）
```

## 2. 查询方式分析：当前只用了 Local Search

### 2.1 GraphRAG 提供的四种查询方式

GraphRAG 框架在 `graphrag/query/factory.py` 中提供了四种查询引擎：

| 查询方式 | 工厂函数 | 核心机制 | 使用的 embedding 索引 | 使用的图谱特性 |
|---------|---------|---------|---------------------|--------------|
| **Local Search** | `get_local_search_engine()` | 向量检索实体 → 图遍历扩展上下文 → LLM 生成 | `entity_description` | 实体、关系、社区报告、text units |
| **Global Search** | `get_global_search_engine()` | Map-Reduce 遍历社区报告 → LLM 聚合 | 无（不用 embedding 检索） | 社区报告、社区层级 |
| **DRIFT Search** | `get_drift_search_engine()` | 社区报告引导 → 多轮 Local Search 迭代 | `entity_description` | 社区报告 + Local Search 全部特性 |
| **Basic Search** | `get_basic_search_engine()` | 纯向量检索 text units → LLM 生成 | `text_unit_text` | **无**（纯 RAG，不用图谱） |

### 2.2 当前测试代码只使用了 Local Search

```python
# run_embedding_benchmark.py → run_queries()
from graphrag.query.factory import get_local_search_engine

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
result = await search_engine.search(qtext)
```

### 2.3 Local Search 并不是"简单 VectorDB"

你的担心有道理但不完全准确。通过阅读 `LocalSearchMixedContext.build_context()` 源码，Local Search 的上下文构建过程是：

**Step 1: 向量检索入口（这一步确实是 VectorDB）**

```python
# entity_extraction.py → map_query_to_entities()
search_results = text_embedding_vectorstore.similarity_search_by_text(
    text=query,
    text_embedder=lambda t: text_embedder.embedding(input=[t]).first_embedding,
    k=k * oversample_scaler,  # 默认 top_k=10, oversample=2, 实际取 top 20
)
```

用 query 的 embedding 在 `entity_description` 向量索引中找到最相关的实体。**这一步确实是纯向量搜索。**

**Step 2: 图遍历扩展上下文（这是 GraphRAG 的核心价值）**

找到种子实体后，`LocalSearchMixedContext` 按 token 预算比例构建三层上下文：

```
总 token 预算 = max_context_tokens (默认 8000)
├── community_prop (25%) → 社区报告上下文
│   找到种子实体所属的社区 → 按匹配实体数排序 → 填入社区报告全文
│
├── local_prop (25%) → 实体/关系/协变量上下文
│   逐个添加种子实体 → 查找该实体的所有关系 → 构建关系表
│   → 查找关联的 covariates → 直到 token 超限
│
└── text_unit_prop (50%) → 原文 text unit 上下文
    找到种子实体关联的 text units → 按实体排名和关系数排序 → 填入原文
```

**关键点**: Step 2 中的社区报告、关系遍历、text unit 关联都是**图谱操作**，不是向量搜索。这些信息是 Phase 4-9 通过 LLM 提取和 Leiden 聚类构建的知识图谱结构。

**Step 3: LLM 生成答案**

将三层上下文拼接后，用 `LOCAL_SEARCH_SYSTEM_PROMPT` 模板构建 prompt，调用 LLM 生成最终答案。

### 2.4 但 Local Search 确实没有充分利用 GraphRAG

虽然 Local Search 不是纯 VectorDB，但它有明显局限：

1. **入口仍然是向量检索**: 如果 embedding 质量差导致种子实体选错，后续的图遍历再好也没用
2. **没有用到 Global Search 的 Map-Reduce**: Global Search 遍历所有社区报告做 Map-Reduce，适合"总结性"问题（如"这个领域的主要趋势是什么？"）
3. **没有用到 DRIFT Search 的多轮迭代**: DRIFT Search 先用社区报告做 primer，再多轮 Local Search 迭代深入，适合复杂推理问题
4. **没有用到 Basic Search 做对照**: Basic Search 是纯 RAG（向量检索 text units），可以作为 baseline 对比 GraphRAG 的增益

### 2.5 对 Embedding 模型评测的影响

不同查询方式对 embedding 的依赖程度不同：

| 查询方式 | embedding 影响程度 | 原因 |
|---------|------------------|------|
| **Basic Search** | ⭐⭐⭐⭐⭐ 最高 | 纯向量检索，embedding 质量直接决定检索质量 |
| **Local Search** | ⭐⭐⭐⭐ 高 | 向量检索是入口，但图遍历可以补偿部分检索误差 |
| **DRIFT Search** | ⭐⭐⭐ 中 | 社区报告 primer 降低了对向量检索的依赖 |
| **Global Search** | ⭐ 最低 | 不使用 embedding 检索，遍历所有社区报告 |

**结论**: 如果目标是评测 embedding 模型的效果差异，应该至少同时使用 **Basic Search**（纯 embedding 依赖）和 **Local Search**（embedding + 图谱），这样才能区分：
- embedding 本身的检索质量（Basic Search 分数）
- embedding + 图谱结合的效果（Local Search 分数）
- 图谱带来的增益（Local - Basic 的差值）

## 3. Embedding 使用位置分析

整个测试系统中，embedding 模型在以下三个完全独立的环节被使用：

### 3.1 索引阶段（Phase 10）— 被测 embedding 模型

**位置**: `run_embedding_benchmark.py` → `run_phase_10()`

```python
async def run_phase_10(config, dataset_name, workspace_dir, model_config):
    grc = _build_graphrag_config(
        ...,
        embedding_model_name=model_config.name,  # ← 被测模型
        embedding_dim=model_config.dim,
        graph_name=graph_name,
    )
    await run_workflow(grc, context)  # GraphRAG generate_text_embeddings
```

Phase 10 对三类文本生成向量嵌入：
1. `text_unit.text` — TextUnit 原文（Basic Search 使用）
2. `entity.title + ":" + description` — 实体描述（Local/DRIFT Search 使用）
3. `community_report.full_content` — 社区报告全文（DRIFT Search 使用）

**调用方式**: 通过 GraphRAG 内部的 `create_embedding()` 工厂方法，最终调用 OpenAI 兼容的 `/embeddings` API。

**关键点**: 这里的 embedding 调用是**单文本输入**（`input: ["text1", "text2", ...]`），不区分 query/document，属于**对称编码**方式。

### 3.2 查询阶段（Local Search）— 被测 embedding 模型

**位置**: `run_embedding_benchmark.py` → `run_queries()`

Local Search 内部通过 `map_query_to_entities()` 将 query 编码为向量，在 FalkorDB 的 `entity_description` 索引中做相似度搜索：

```python
# entity_extraction.py
search_results = text_embedding_vectorstore.similarity_search_by_text(
    text=query,
    text_embedder=lambda t: text_embedder.embedding(input=[t]).first_embedding,
    k=k * oversample_scaler,
)
```

`FalkorDBVectorStore.similarity_search_by_vector()` 在 Python 端计算余弦相似度（全量扫描，非 ANN）。

### 3.3 评估阶段 — 固定使用 BGE-M3 default

**位置**: `run_embedding_benchmark.py` → `evaluate_predictions()`

```python
eval_emb = OpenAIEmbeddings(
    model="BAAI/bge-m3",           # ← 硬编码，固定使用 BGE-M3 default
    base_url=config.api_base_url,
    api_key="no-key",
    check_embedding_ctx_length=False,
)
```

评估中的 embedding 用于 `answer_accuracy.py` 的 `calculate_semantic_similarity()`：

```python
async def calculate_semantic_similarity(embeddings, answer, ground_truth):
    a_embed, gt_embed = await asyncio.gather(
        embeddings.aembed_query(answer),
        embeddings.aembed_query(ground_truth),
    )
    cosine_sim = np.dot(a_embed, gt_embed) / (np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
    return (cosine_sim + 1) / 2
```

**关键点**: 评估阶段的 embedding 是**固定的 BGE-M3 default**，不随被测模型变化。这里用 `aembed_query()` 对 answer 和 ground_truth 都做 query 编码（对称使用），计算语义相似度作为 Answer Correctness 的一个分量（权重 0.25）。

## 4. 是否使用了双塔模式？

### 4.1 结论：当前代码未使用双塔模式

**当前代码中所有 embedding 调用都是"对称编码"模式**，即 query 和 document 使用相同的编码方式。具体表现：

| 环节 | 调用方式 | 是否双塔 |
|------|---------|---------|
| Phase 10 索引 | GraphRAG `create_embedding()` → `/embeddings` API | ❌ 单塔（统一编码） |
| Local Search 查询 | GraphRAG search engine 内部编码 query → 与索引向量比较 | ❌ 单塔（query 和 doc 用同一模型同一方式编码） |
| 评估 semantic similarity | `aembed_query(answer)` + `aembed_query(ground_truth)` | ❌ 单塔（两个文本都用 query 编码） |

### 4.2 双塔模式的含义

双塔模式（Bi-Encoder / Dual-Tower）指的是：
- **Query Tower**: 对查询文本使用专门的编码方式
- **Document Tower**: 对文档文本使用专门的编码方式
- 两个 tower 产出的向量在同一空间中，但编码路径不同

对于 BGE-M3 模型，API 服务器通过不同的 endpoint 路径区分模式：
- `BAAI/bge-m3` — 默认模式（单塔，对称编码）
- `BAAI/bge-m3/heavy` — heavy 模式（双塔变体，更重的交互计算）
- `BAAI/bge-m3/interactive` — interactive 模式（双塔变体，交叉注意力）

### 4.3 当前代码的问题

虽然 `models.py` 中定义了三种 BGE-M3 模式：

```python
EMBEDDING_MODELS = [
    EmbeddingModelConfig("BAAI/bge-m3", 1024, 8192, "BGE-M3 (default)"),
    EmbeddingModelConfig("BAAI/bge-m3/heavy", 1024, 8192, "BGE-M3 (heavy)"),
    EmbeddingModelConfig("BAAI/bge-m3/interactive", 1024, 8192, "BGE-M3 (interactive)"),
    ...
]
```

但这三种模式在代码中的使用方式完全相同——都是通过 `/embeddings` API 传入模型名称，**没有区分 query 编码和 document 编码**。

真正的双塔模式测试应该：
1. **索引时**: 用 document 编码方式对文档文本生成向量
2. **查询时**: 用 query 编码方式对查询文本生成向量
3. 两种编码方式可能使用不同的 prompt prefix、不同的模型路径、或不同的 API 参数

## 5. 双塔 vs 非双塔对比测试的改动规划

### 5.1 总体思路

为支持双塔模式的模型（BGE-M3 heavy/interactive、Qwen3-Embedding 等）分别进行：
- **非双塔测试**（现有行为）：query 和 document 使用相同编码
- **双塔测试**（新增）：索引时用 document 编码，查询时用 query 编码

### 5.2 需要确认的前置问题

在实施前需要确认 API 服务器的双塔支持方式：

1. **API 层面如何区分 query/document 编码？** 可能的方式：
   - 通过不同的模型名称（如 `BAAI/bge-m3/heavy/query` vs `BAAI/bge-m3/heavy/document`）
   - 通过请求参数（如 `encoding_type: "query"` vs `encoding_type: "document"`）
   - 通过 prompt prefix（如 `"query: "` + text vs `"passage: "` + text）
   - 通过 vLLM/RayLLM 的特定参数

2. **哪些模型支持双塔？** 需要确认：
   - `BAAI/bge-m3/heavy` 和 `BAAI/bge-m3/interactive` 是否真正支持双塔
   - `Qwen/Qwen3-Embedding-8B` 是否支持 query/document 区分
   - `intfloat/e5-mistral-7b-instruct` 是否通过 instruction prefix 实现双塔

### 5.3 数据模型改动

#### 5.3.1 扩展 `EmbeddingModelConfig`

```python
# tests/models.py

@dataclass
class EmbeddingModelConfig:
    name: str               # 模型名称
    dim: int                # Embedding 维度
    max_tokens: int         # 最大 Token 数
    display_name: str       # 报告中显示的名称
    # ── 新增字段 ──
    supports_dual_tower: bool = False          # 是否支持双塔模式
    query_model_name: str | None = None        # 双塔模式下 query 编码的模型名称（如不同于 name）
    document_model_name: str | None = None     # 双塔模式下 document 编码的模型名称
    query_prefix: str = ""                     # query 编码的 prompt prefix（如 "query: "）
    document_prefix: str = ""                  # document 编码的 prompt prefix（如 "passage: "）
    dual_tower_params: dict | None = None      # 双塔模式的额外 API 参数
```

#### 5.3.2 扩展 `EMBEDDING_MODELS` 列表

```python
EMBEDDING_MODELS = [
    # 非双塔模型
    EmbeddingModelConfig("BAAI/bge-m3", 1024, 8192, "BGE-M3 (default)"),
    EmbeddingModelConfig("nomic-ai/nomic-embed-text-v1.5", 768, 8192, "Nomic-v1.5"),

    # 支持双塔的模型
    EmbeddingModelConfig(
        "BAAI/bge-m3/heavy", 1024, 8192, "BGE-M3 (heavy)",
        supports_dual_tower=True,
        query_prefix="",           # 待确认 API 实际要求
        document_prefix="",        # 待确认
    ),
    EmbeddingModelConfig(
        "BAAI/bge-m3/interactive", 1024, 8192, "BGE-M3 (interactive)",
        supports_dual_tower=True,
    ),
    EmbeddingModelConfig(
        "intfloat/e5-mistral-7b-instruct", 4096, 4096, "E5-Mistral-7B",
        supports_dual_tower=True,
        query_prefix="Instruct: Retrieve relevant passages\nQuery: ",
        document_prefix="",
    ),
    EmbeddingModelConfig(
        "Qwen/Qwen3-Embedding-8B", 4096, 8192, "Qwen3-Emb-8B",
        supports_dual_tower=True,
        query_prefix="query: ",
        document_prefix="document: ",
    ),
    ...
]
```

#### 5.3.3 新增 `DualTowerMode` 枚举

```python
from enum import Enum

class DualTowerMode(str, Enum):
    SYMMETRIC = "symmetric"     # 非双塔：query 和 document 使用相同编码
    ASYMMETRIC = "asymmetric"   # 双塔：query 和 document 使用不同编码
```

#### 5.3.4 扩展 `ModelResult`

```python
@dataclass
class ModelResult:
    model: EmbeddingModelConfig
    dataset_name: str
    dual_tower_mode: DualTowerMode = DualTowerMode.SYMMETRIC  # 新增
    predictions: List[PredictionItem] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    embedding_time_seconds: float = 0.0
    query_time_seconds: float = 0.0
    error: Optional[str] = None
```

### 5.4 索引阶段改动（Phase 10）

#### 5.4.1 问题

当前 Phase 10 通过 GraphRAG 内部的 `run_workflow()` 执行，embedding 调用在 GraphRAG 框架内部完成。要实现双塔模式的 document 编码，需要在 embedding 调用时添加 document prefix。

#### 5.4.2 方案 A：通过 httpx hook 注入 prefix（推荐）

利用已有的 `AdaptiveConcurrencyController.install_hooks()` 中的 httpx monkey-patch 机制，在请求发送前检测 embedding 请求并注入 prefix：

```python
# tests/concurrency.py — 扩展 _on_request hook

# 全局状态：当前 embedding 模式
_current_embedding_mode: DualTowerMode = DualTowerMode.SYMMETRIC
_current_document_prefix: str = ""
_current_query_prefix: str = ""
_current_encoding_phase: str = "document"  # "document" 或 "query"

async def _on_request(request):
    if request.method == "POST" and "/embeddings" in str(request.url):
        body = json.loads(request.content)
        if _current_embedding_mode == DualTowerMode.ASYMMETRIC:
            prefix = (_current_document_prefix
                      if _current_encoding_phase == "document"
                      else _current_query_prefix)
            if prefix and "input" in body:
                if isinstance(body["input"], list):
                    body["input"] = [prefix + t if isinstance(t, str) else t
                                     for t in body["input"]]
                elif isinstance(body["input"], str):
                    body["input"] = prefix + body["input"]
                request._content = json.dumps(body).encode("utf-8")
                request.headers["content-length"] = str(len(request._content))
```

#### 5.4.3 方案 B：自定义 Embedding 包装器

创建一个包装 GraphRAG embedding 接口的类，在调用时自动添加 prefix：

```python
# tests/dual_tower_embedding.py

class DualTowerEmbeddingWrapper:
    """包装 GraphRAG 的 embedding 调用，支持 query/document prefix 注入"""

    def __init__(self, base_embedding, mode: DualTowerMode, model_config: EmbeddingModelConfig):
        self._base = base_embedding
        self._mode = mode
        self._config = model_config
        self._phase = "document"  # 默认 document 编码

    def set_phase(self, phase: str):
        """切换编码阶段: 'document' 或 'query'"""
        self._phase = phase

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._mode == DualTowerMode.ASYMMETRIC:
            prefix = (self._config.document_prefix
                      if self._phase == "document"
                      else self._config.query_prefix)
            texts = [prefix + t for t in texts]
        return await self._base.embed(texts)
```

#### 5.4.4 Phase 10 执行改动

```python
async def run_phase_10(config, dataset_name, workspace_dir, model_config, dual_tower_mode):
    # 双塔模式下，Phase 10 索引使用 document 编码
    if dual_tower_mode == DualTowerMode.ASYMMETRIC:
        set_encoding_phase("document")
        set_document_prefix(model_config.document_prefix)

    # ... 现有 Phase 10 逻辑 ...

    await run_workflow(grc, context)
```

### 5.5 查询阶段改动（Local Search）

#### 5.5.1 问题

GraphRAG Local Search 内部会将 query 文本编码为向量。双塔模式下，query 编码需要使用 query prefix。

#### 5.5.2 改动

```python
async def run_queries(config, dataset_name, workspace_dir, model_config, questions, dual_tower_mode):
    # 双塔模式下，查询使用 query 编码
    if dual_tower_mode == DualTowerMode.ASYMMETRIC:
        set_encoding_phase("query")
        set_query_prefix(model_config.query_prefix)

    # ... 现有查询逻辑 ...

    for q in questions:
        result = await search_engine.search(qtext)
        # ...
```

### 5.6 评估阶段改动

评估阶段的 embedding（`calculate_semantic_similarity`）使用固定的 BGE-M3 default，**不需要改动**。评估的目的是衡量生成答案与 ground truth 的语义相似度，与被测模型无关。

### 5.7 主流程编排改动

#### 5.7.1 `async_main()` 改动

```python
async def async_main(config: BenchmarkConfig):
    # ... Phase 1-9 不变 ...

    for ds_name, questions in dataset_questions.items():
        for model in available_models:
            # ── 非双塔测试（所有模型都跑） ──
            mr_sym = ModelResult(model=model, dataset_name=ds_name,
                                dual_tower_mode=DualTowerMode.SYMMETRIC)
            # ... 现有 Phase 10 + 查询 + 评估逻辑 ...
            summary.model_results.append(mr_sym)

            # ── 双塔测试（仅支持双塔的模型） ──
            if model.supports_dual_tower:
                mr_asym = ModelResult(model=model, dataset_name=ds_name,
                                     dual_tower_mode=DualTowerMode.ASYMMETRIC)
                # Phase 10 使用 document 编码
                mr_asym.embedding_time_seconds = await run_phase_10(
                    config, ds_name, workspace_dir, model,
                    dual_tower_mode=DualTowerMode.ASYMMETRIC,
                )
                # 查询使用 query 编码
                mr_asym.predictions = await run_queries(
                    config, ds_name, workspace_dir, model, questions,
                    dual_tower_mode=DualTowerMode.ASYMMETRIC,
                )
                # 评估不变
                mr_asym.evaluation, details = await evaluate_predictions(
                    config, mr_asym.predictions, model.display_name, ds_name,
                )
                summary.model_results.append(mr_asym)
```

#### 5.7.2 FalkorDB 图命名扩展

双塔模式需要独立的 FalkorDB 图，避免与非双塔模式的向量混淆：

```python
def make_graph_name(model_name: str, dataset_name: str, dual_tower: bool = False) -> str:
    base = sanitize_name(model_name)
    if dual_tower:
        base += "_dualtower"
    return base + "_" + dataset_name
```

### 5.8 报告生成改动

#### 5.8.1 新增"双塔 vs 非双塔对比"章节

```python
# report_generator.py

def generate_markdown(summary):
    # ... 现有章节 ...

    # 新增章节：双塔 vs 非双塔对比
    dual_tower_models = set()
    for r in summary.model_results:
        if r.model.supports_dual_tower:
            dual_tower_models.add(r.model.name)

    if dual_tower_models:
        lines.append("## N. 双塔 vs 非双塔模式对比\n")
        for model_name in sorted(dual_tower_models):
            lines.append(f"### {model_name}\n")
            lines.append("| 模式 | 数据集 | ROUGE-L | Answer Correctness | Phase 10 耗时 | 查询耗时 |")
            lines.append("|------|--------|---------|-------------------|-------------|---------|")

            for r in summary.model_results:
                if r.model.name == model_name and r.evaluation:
                    mode_label = "双塔" if r.dual_tower_mode == DualTowerMode.ASYMMETRIC else "非双塔"
                    # ... 输出指标行 ...
            lines.append("")
```

#### 5.8.2 总体排名中区分模式

在总体排名中，同一模型的双塔和非双塔结果作为独立条目参与排名：

```python
# 排名 key 改为 (display_name, mode)
rank_key = f"{r.model.display_name} ({r.dual_tower_mode.value})"
```

### 5.9 命令行参数扩展

```python
parser.add_argument("--dual-tower", choices=["both", "symmetric", "asymmetric"],
                    default="both",
                    help="双塔测试模式: both=两种都跑, symmetric=仅非双塔, asymmetric=仅双塔")
```

### 5.10 Shell 脚本改动

```bash
# run_embedding_benchmark.sh
DUAL_TOWER="both"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dual-tower) DUAL_TOWER="$2"; shift 2 ;;
        # ... 现有参数 ...
    esac
done

ARGS+=(--dual-tower "$DUAL_TOWER")
```

## 6. 改动文件清单与优先级

### 6.1 必须改动的文件

| 文件 | 改动内容 | 优先级 |
|------|---------|--------|
| `tests/models.py` | 扩展 `EmbeddingModelConfig`（新增双塔字段）、新增 `DualTowerMode` 枚举、扩展 `ModelResult`、更新 `EMBEDDING_MODELS`、扩展 `make_graph_name()` | P0 |
| `tests/concurrency.py` | 在 httpx hook 中注入 query/document prefix 逻辑 | P0 |
| `tests/run_embedding_benchmark.py` | `run_phase_10()` 支持 dual_tower_mode 参数、`run_queries()` 支持 dual_tower_mode、`async_main()` 编排双塔/非双塔两轮测试 | P0 |
| `tests/report_generator.py` | 新增"双塔 vs 非双塔对比"章节、总体排名区分模式 | P1 |
| `tests/run_embedding_benchmark.sh` | 新增 `--dual-tower` 参数 | P1 |

### 6.2 可选新增的文件

| 文件 | 内容 | 优先级 |
|------|------|--------|
| `tests/dual_tower_embedding.py` | 双塔 embedding 包装器（方案 B，如果方案 A 不够灵活） | P2 |
| `tests/test_dual_tower.py` | 双塔模式的单元测试和集成测试 | P2 |

### 6.3 不需要改动的文件

| 文件 | 原因 |
|------|------|
| `tests/falkordb_vector_store.py` | 向量存储层不关心编码方式，只存/取向量 |
| `tests/dataset_loader.py` | 数据集加载与 embedding 模式无关 |
| `tests/graph_sync.py` | 图结构同步与 embedding 模式无关 |
| `GraphRAG-Benchmark/Evaluation/metrics/*` | 评估指标使用固定的 BGE-M3 default，不受被测模型影响 |

## 7. 实施步骤建议

### Step 1: 确认 API 双塔支持方式（前置）
- 向 API 服务器发送测试请求，确认 `BAAI/bge-m3/heavy` 和 `BAAI/bge-m3/interactive` 的双塔编码方式
- 确认是通过 model name 区分、prompt prefix 区分、还是 API 参数区分
- 确认 Qwen3-Embedding 和 E5-Mistral 的双塔支持方式

### Step 2: 数据模型扩展
- 修改 `tests/models.py`，添加双塔相关字段和枚举

### Step 3: 实现 prefix 注入机制
- 修改 `tests/concurrency.py`，在 httpx hook 中实现 query/document prefix 注入
- 或创建 `tests/dual_tower_embedding.py` 包装器

### Step 4: 修改主执行器
- 修改 `tests/run_embedding_benchmark.py`，支持双塔/非双塔两轮测试

### Step 5: 修改报告生成
- 修改 `tests/report_generator.py`，新增对比章节

### Step 6: 端到端验证
- 使用 `--sample 1 --dataset medical --models "BAAI/bge-m3,BAAI/bge-m3/heavy" --dual-tower both` 验证

## 8. 问题采样与公平性分析

### 8.1 问题不是随机抽取，每个模型用的完全一样

```python
# async_main() 中，问题在数据集级别加载一次
questions = DatasetLoader.prepare_medical(data_root, input_dir)

# 按顺序截取前 N 条（不是随机）
if config.sample_size and config.sample_size < len(questions):
    questions = questions[:config.sample_size]

dataset_questions[ds_name] = questions  # 存起来，所有模型共用
```

- 问题是**固定取前 N 条**（`questions[:sample_size]`），不是随机采样
- 所有模型对同一个数据集用的是**完全相同的问题列表**，保证了公平对比
- `--sample 5` 就是取 JSON 文件中前 5 个问题

### 8.2 Medical 数据集的问题类型分布

2062 个问题分四种类型，对 Basic Search vs GraphRAG 的需求完全不同：

| 类型 | 数量 | 占比 | 平均 evidence 条数 | Basic Search 能否回答 |
|------|------|------|-------------------|---------------------|
| **Fact Retrieval** | 1098 | 53% | 2.3 | ✅ 大部分可以。469 个单 evidence 问题，检索到一个 text chunk 就够 |
| **Complex Reasoning** | 509 | 25% | 3.5 | ⚠️ 部分困难。需要组合多个事实推理，需要跨实体关联 |
| **Contextual Summarize** | 289 | 14% | 6.1 | ❌ 很难。需要聚合 5-10+ 条 evidence 做综合总结 |
| **Creative Generation** | 166 | 8% | 13.2 | ❌ 几乎不可能。需要 10+ 条 evidence 生成结构化文档 |

### 8.3 当前采样的问题

`--sample 5` 取前 5 条，而 JSON 文件开头全是 Fact Retrieval 类型的简单单事实问题。这意味着：
- 只测了最简单的单事实检索题
- Local Search 的图遍历优势根本体现不出来
- 不同 embedding 模型的差异被压缩

**理想的采样策略**应该按 `question_type` 分层采样，确保四种类型都有覆盖。

## 9. ROUGE-L 分数极低的根因分析

### 9.1 实际测试结果

所有模型的 ROUGE-L 都在 0.005~0.10 之间，Answer Correctness 大面积 NaN：

| 模型 | 数据集 | ROUGE-L | Answer Correctness |
|------|--------|---------|-------------------|
| BGE-M3 (default) | medical | 0.102 | NaN |
| BGE-M3 (heavy) | medical | 0.075 | 0.404 |
| BGE-M3 (interactive) | medical | 0.087 | 0.378 |
| Qwen3-Emb-8B | medical | 0.103 | NaN |
| E5-Mistral-7B | medical | 0.079 | NaN |
| Nomic-v1.5 | medical | 0.085 | NaN |

### 9.2 根因：答案内容正确，但长度严重不匹配

对比 BGE-M3 default 的 5 个 medical predictions：

| 问题 | Ground Truth | 生成答案 | 长度倍数 |
|------|-------------|---------|---------|
| 最常见皮肤癌？ | 12 词 | 325 词 | **31x** |
| BCC 来自哪种细胞？ | 14 词 | 105 词 | **9.5x** |
| BCC 常见部位？ | 14 词 | 274 词 | **22x** |
| BCC 主要风险因素？ | 10 词 | 160 词 | **19x** |
| 白皮肤如何影响？ | 7 词 | 175 词 | **31x** |

生成的答案语义上完全正确（如 "Basal cell carcinoma (BCC) is the most common type of skin cancer" 完美匹配 ground truth），但 LLM 生成了大量额外内容：详细解释、引用标注 `[Data: Entities (0)]`、markdown 标题等。

### 9.3 ROUGE-L 为什么惩罚长答案

ROUGE-L 基于最长公共子序列（LCS），其 F1 分数 = 2 × Precision × Recall / (Precision + Recall)：
- **Recall** = LCS长度 / ground_truth长度 → 较高（答案包含了 GT 的关键词）
- **Precision** = LCS长度 / 生成答案长度 → **极低**（答案是 GT 的 20-30 倍长）
- F1 被 Precision 拖垮

### 9.4 两个叠加因素

1. **`response_type="Multiple Paragraphs"` 硬编码**: 在 `run_queries()` 中设置了让 LLM 生成多段落回答，而 ground truth 是一句话
2. **GraphRAG Local Search 的 system prompt** 本身就鼓励详细回答并包含引用标注

### 9.5 Answer Correctness 大面积 NaN 的原因

`summary.json` 中所有模型都有 `"error": "No module named 'json5'"`。`json5` 是 GraphRAG 的依赖之一，缺失导致评估流程中断。**已安装 `json5 0.14.0` 修复此问题。**

### 9.6 修复方向

| 方向 | 具体做法 | 效果 |
|------|---------|------|
| **改 response_type** | 把 `"Multiple Paragraphs"` 改成 `"A single concise sentence"` 或 `"A brief answer"` | 让 LLM 生成简短回答，ROUGE-L 会显著提升 |
| **换/加评估指标** | 增加 Semantic Similarity（不受长度影响）、Coverage Score（检查 GT 事实是否被覆盖，不惩罚额外内容） | 更公平地评估长答案 |
| **重跑评估** | 安装 json5 后用 `rerun_evaluation.py` 重跑 | 修复 Answer Correctness NaN 问题 |

## 10. 建议：扩展查询方式以充分评测 Embedding

当前只用 Local Search 的问题在于：Local Search 的图遍历（社区报告、关系扩展）会"稀释" embedding 质量差异——即使 embedding 检索到的种子实体不太准确，图遍历仍可能补偿回来。这导致不同 embedding 模型的分数差异可能被压缩。

建议至少增加 **Basic Search** 作为对照：

```python
# 新增 Basic Search 查询
from graphrag.query.factory import get_basic_search_engine

# 需要额外创建 text_unit_text 的 VectorStore
text_unit_store = FalkorDBVectorStore(
    host=config.falkordb_host,
    port=config.falkordb_port,
    graph_name=graph_name,
    index_name="text_unit_text_embedding",  # Phase 10 生成的 text unit 向量索引
    vector_size=model_config.dim,
)
text_unit_store.connect()

basic_engine = get_basic_search_engine(
    text_units=text_units,
    text_unit_embeddings=text_unit_store,
    config=grc,
    response_type="Multiple Paragraphs",
)
basic_result = await basic_engine.search(qtext)
```

这样报告中可以展示：

| 模型 | Basic Search ROUGE-L | Local Search ROUGE-L | 图谱增益 |
|------|---------------------|---------------------|---------|
| BGE-M3 | 0.35 | 0.52 | +0.17 |
| E5-Mistral | 0.41 | 0.55 | +0.14 |

**图谱增益 = Local - Basic**，这个差值才是 GraphRAG 知识图谱真正带来的价值。

## 11. 风险与注意事项

1. **API 兼容性**: 双塔模式的实现高度依赖 API 服务器的支持方式，需要先确认
2. **Phase 10 缓存冲突**: 双塔和非双塔使用同一 workspace 的 Phase 1-9 产物，但 Phase 10 的 embedding cache 需要隔离（当前代码已在每次 Phase 10 前清除 `text_embedding` 缓存目录）
3. **FalkorDB 图数量翻倍**: 支持双塔的模型会产生两个图（`model_dataset` 和 `model_dataset_dualtower`），需要注意存储空间
4. **测试时间增加**: 支持双塔的模型需要跑两次 Phase 10 + 查询 + 评估，总时间约增加 40-60%
5. **GraphRAG 内部编码**: GraphRAG search engine 内部的 query 编码路径需要确认是否可以注入 prefix，如果不行可能需要 monkey-patch GraphRAG 的 embedding 调用
