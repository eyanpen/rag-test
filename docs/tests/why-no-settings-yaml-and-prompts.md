# 为什么 tests 目录下没有 settings.yaml 和 prompts 文件？

## 结论

测试脚本**完全绕过了 `graphrag init`**，通过 Python 代码直接构造 `GraphRagConfig` 对象，因此不需要 `settings.yaml` 和 `prompts/` 目录。

## 详细分析

### 1. 正常的 GraphRAG 使用流程

标准流程是：

```
graphrag init  →  生成 settings.yaml + prompts/*.txt  →  graphrag index
```

`graphrag init` 会在项目目录下创建：
- `settings.yaml` — 包含 LLM、embedding、storage、chunking 等全部配置
- `prompts/extract_graph.txt` — 图谱抽取 prompt
- `prompts/summarize_descriptions.txt` — 描述摘要 prompt
- `prompts/extract_claims.txt` — 声明抽取 prompt
- `prompts/community_report_graph.txt` — 社区报告 prompt
- 等等

### 2. 测试脚本的做法：纯代码构建配置

`tests/run_embedding_benchmark.py` 中的 `_build_graphrag_config()` 函数（第 346-410 行）直接用 Python 代码构造了 `GraphRagConfig` 对象：

```python
from graphrag.config.models.graph_rag_config import GraphRagConfig

grc = GraphRagConfig(
    completion_models={...},
    embedding_models={...},
    input_storage=StorageConfig(base_dir=...),
    output_storage=StorageConfig(base_dir=...),
    update_output_storage=StorageConfig(base_dir=...),
    cache=CacheConfig(storage=StorageConfig(base_dir=...)),
    table_provider=TableProviderConfig(),
    vector_store=VectorStoreConfig(type="falkordb", ...),
)
```

这是因为 `GraphRagConfig` 是一个 Pydantic `BaseModel`，所有字段都有默认值，可以直接实例化。

### 3. Prompts 为什么也不需要？

`GraphRagConfig` 中各子配置（`ExtractGraphConfig`、`SummarizeDescriptionsConfig`、`CommunityReportsConfig` 等）的 `prompt` 字段默认值为 `None`。当 prompt 为 `None` 时，GraphRAG 运行时会 fallback 到代码中硬编码的默认 prompt 常量，例如：

| 配置类 | 默认 prompt 来源 |
|--------|-----------------|
| `ExtractGraphConfig` | `graphrag.prompts.index.extract_graph.GRAPH_EXTRACTION_PROMPT` |
| `SummarizeDescriptionsConfig` | `graphrag.prompts.index.summarize_descriptions.SUMMARIZE_PROMPT` |
| `CommunityReportsConfig` | `graphrag.prompts.index.community_report.COMMUNITY_REPORT_PROMPT` |
| `ExtractClaimsConfig` | `graphrag.prompts.index.extract_claims.EXTRACT_CLAIMS_PROMPT` |

所以 `prompts/*.txt` 文件只是 `graphrag init` 为了方便用户自定义而导出的副本，并非运行必需。

### 4. 为什么这样设计？

测试脚本的目标是 **benchmark 不同 embedding 模型**，需要：
- 动态切换 embedding 模型名称和维度
- 动态指定 FalkorDB 连接参数和 graph_name
- 同一数据集的 Phase 1-9 只跑一次，Phase 10 按模型数量跑 N 次

用 `settings.yaml` 文件驱动的话，每次切换模型都要改文件再重新加载，不如直接在代码里构造 `GraphRagConfig` 灵活。

### 5. 总结对比

| | 标准流程 | 测试脚本 |
|---|---------|---------|
| 初始化 | `graphrag init` | 无 |
| 配置来源 | `settings.yaml` 文件 | Python 代码直接构造 `GraphRagConfig` |
| Prompts | `prompts/*.txt` 文件 | 使用代码内置默认 prompt |
| 灵活性 | 静态配置 | 可动态参数化 |
| 适用场景 | 单次 indexing | 批量 benchmark、自动化测试 |
