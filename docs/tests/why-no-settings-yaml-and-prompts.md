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

---

## 能否用 prompt-tune 定制 prompts 后再 Index？

> 参考：[docs/graphrag/graphrag-prompt-tune-deep-dive.md](../graphrag/graphrag-prompt-tune-deep-dive.md)

**完全可以。** 而且不需要 `settings.yaml`，纯代码方式同样支持注入定制 prompts。

### 原理：`resolved_prompts()` 的双路径机制

每个配置类的 `prompt` 字段都遵循相同的 fallback 逻辑：

```python
# ExtractGraphConfig.resolved_prompts()
def resolved_prompts(self) -> ExtractGraphPrompts:
    return ExtractGraphPrompts(
        extraction_prompt=Path(self.prompt).read_text(encoding="utf-8")
        if self.prompt          # ← 不为 None 时，从文件路径读取
        else GRAPH_EXTRACTION_PROMPT,  # ← 为 None 时，用硬编码默认值
    )
```

所以只要给 `prompt` 字段赋一个文件路径，pipeline 运行时就会读取该文件内容作为 prompt。

### 方案：在 `_build_graphrag_config()` 中注入定制 prompts

#### 第 1 步：运行 prompt-tune 生成定制 prompts

```bash
graphrag prompt-tune \
  --root ./tests/benchmark_results/workspace/kevin_scott \
  --domain "technology industry" \
  --language "English" \
  --output ./tests/prompts/kevin_scott
```

产出：
```
tests/prompts/kevin_scott/
├── extract_graph.txt
├── summarize_descriptions.txt
└── community_report_graph.txt
```

#### 第 2 步：修改 `_build_graphrag_config()` 传入 prompt 路径

```python
from graphrag.config.models.extract_graph_config import ExtractGraphConfig
from graphrag.config.models.summarize_descriptions_config import SummarizeDescriptionsConfig
from graphrag.config.models.community_reports_config import CommunityReportsConfig

# 如果存在定制 prompts 目录，则使用；否则保持 None（用默认值）
prompts_dir = os.path.join(workspace_dir, "prompts")

grc = GraphRagConfig(
    completion_models=completion_models,
    embedding_models=embedding_models,
    input_storage=StorageConfig(base_dir=...),
    output_storage=StorageConfig(base_dir=...),
    # ...其他配置不变...

    # 注入定制 prompts
    extract_graph=ExtractGraphConfig(
        prompt=os.path.join(prompts_dir, "extract_graph.txt")
        if os.path.isfile(os.path.join(prompts_dir, "extract_graph.txt"))
        else None,
    ),
    summarize_descriptions=SummarizeDescriptionsConfig(
        prompt=os.path.join(prompts_dir, "summarize_descriptions.txt")
        if os.path.isfile(os.path.join(prompts_dir, "summarize_descriptions.txt"))
        else None,
    ),
    community_reports=CommunityReportsConfig(
        graph_prompt=os.path.join(prompts_dir, "community_report_graph.txt")
        if os.path.isfile(os.path.join(prompts_dir, "community_report_graph.txt"))
        else None,
    ),
)
```

### 三个 prompt 的注入点对照

| 配置类 | 字段名 | prompt-tune 产出文件 | resolved_prompts() fallback |
|--------|--------|---------------------|---------------------------|
| `ExtractGraphConfig` | `prompt` | `extract_graph.txt` | `GRAPH_EXTRACTION_PROMPT` |
| `SummarizeDescriptionsConfig` | `prompt` | `summarize_descriptions.txt` | `SUMMARIZE_PROMPT` |
| `CommunityReportsConfig` | `graph_prompt` | `community_report_graph.txt` | `COMMUNITY_REPORT_PROMPT` |
| `CommunityReportsConfig` | `text_prompt` | （prompt-tune 不生成） | `COMMUNITY_REPORT_TEXT_PROMPT` |

注意：`CommunityReportsConfig` 有两个 prompt 字段（`graph_prompt` 和 `text_prompt`），prompt-tune 只生成 `graph_prompt` 对应的文件。`text_prompt` 用于基于 text units 的社区报告生成，始终使用默认值。

### 定制 prompts 对 benchmark 的影响

| 维度 | 默认 prompt | prompt-tune 定制后 |
|------|------------|-------------------|
| Entity Types | 固定 `organization, person, geo, event` | 从数据自动发现，如 `technology, company, product, executive` |
| Few-shot Examples | 3 个硬编码虚构场景 | 从真实数据生成的提取示例 |
| 语言匹配 | 固定英文 | 自动检测数据语言 |
| 专家人设 | 无 | 针对数据领域的专家角色 |

对于 benchmark 场景，定制 prompts 主要影响 **Phase 1-9 的图谱质量**（实体/关系提取精度、社区报告质量），而 **Phase 10 的 embedding 质量对比不受影响**（embedding 只处理已提取的文本，不依赖 prompt）。

### 建议

- 如果 benchmark 目标是**对比 embedding 模型**（当前场景）：默认 prompts 足够，因为所有模型共享同一份 Phase 1-9 输出，prompt 质量对模型间的相对排名影响不大。
- 如果 benchmark 目标是**评估端到端 RAG 质量**：强烈建议对每个数据集运行 prompt-tune，否则图谱质量差会拉低所有模型的绝对分数。
