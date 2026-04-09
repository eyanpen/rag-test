# GraphRAG-Benchmark 配置参考文档

> 版本：1.0 | 日期：2026-04-09

---

## 1. 环境变量配置

| 环境变量 | 使用场景 | 说明 |
|---------|---------|------|
| `LLM_API_KEY` | `run_lightrag.py`, `run_fast-graphrag.py`, `generation_eval.py`, `retrieval_eval.py` | OpenAI 兼容 API 的密钥。所有 API 模式的推理和评估脚本共用此变量 |
| `OPENAI_API_KEY` | `run_hipporag2.py` | HippoRAG2 专用的 API 密钥。注意与 `LLM_API_KEY` 不同 |
| `CUDA_VISIBLE_DEVICES` | `run_hipporag2.py`（硬编码 `"5"`）, `run_digimon.py`（硬编码 `"0"`） | 指定使用的 GPU 设备编号。在脚本中硬编码，如需修改请直接编辑源码 |

**设置示例：**

```bash
# API 模式（大多数脚本）
export LLM_API_KEY=sk-xxxxxxxxxxxxxxxx

# HippoRAG2 专用
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# 如需覆盖 GPU 设备（在运行脚本前）
export CUDA_VISIBLE_DEVICES=0,1
```

---

## 2. 框架运行参数

### 2.1 run_lightrag.py

LightRAG 框架的索引构建与批量推理脚本。

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--subset` | str | ✅ | - | 数据子集选择。可选值：`medical`（医学）、`novel`（小说） |
| `--mode` | str | ✅ | - | LLM 调用模式。`API`：OpenAI 兼容 API；`ollama`：本地 Ollama 服务 |
| `--base_dir` | str | ❌ | `./lightrag_workspace` | 工作目录，索引文件和图数据存储位置。每个 corpus 会在此下创建子目录 |
| `--model_name` | str | ❌ | `qwen2.5-14b-instruct` | LLM 模型标识符。API 模式下为模型名（如 `gpt-4o-mini`），Ollama 模式下为已拉取的模型名 |
| `--embed_model` | str | ❌ | `bge-base-en` | Embedding 模型名称或路径。API 模式使用 HuggingFace 本地模型，Ollama 模式使用 Ollama Embedding |
| `--retrieve_topk` | int | ❌ | `5` | 检索时返回的 top-k 文档数量。影响 context 的丰富度和噪声 |
| `--sample` | int | ❌ | `None` | 每个语料采样的问题数量。设置后仅处理前 N 个问题，用于快速测试。同时语料也仅取第 1 个 |
| `--llm_base_url` | str | ❌ | `https://api.openai.com/v1` | LLM API 的基础 URL。Ollama 模式下应设为 `http://localhost:11434` |
| `--llm_api_key` | str | ❌ | `""` | LLM API 密钥。优先使用此参数，为空时回退到 `LLM_API_KEY` 环境变量 |

**LightRAG 内部固定参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `llm_model_max_async` | 4 | LLM 最大并发请求数 |
| `llm_model_max_token_size` | 32768 | LLM 最大 token 数 |
| `chunk_token_size` | 1200 | 文本分块大小（token） |
| `chunk_overlap_token_size` | 100 | 分块重叠大小（token） |
| `embedding_dim` | 1024 | Embedding 维度 |
| `max_token_size`（Embedding） | 8192 | Embedding 最大 token 数 |
| `query_mode` | `hybrid` | 查询模式（混合本地+全局） |
| `max_token_for_text_unit` | 4000 | 查询时文本单元最大 token |
| `max_token_for_global_context` | 4000 | 查询时全局上下文最大 token |
| `max_token_for_local_context` | 4000 | 查询时本地上下文最大 token |

---

### 2.2 run_fast-graphrag.py

fast-graphrag 框架的索引构建与批量推理脚本。

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--subset` | str | ✅ | - | 数据子集：`medical` / `novel` |
| `--mode` | str | ❌ | `API` | LLM 模式：`API` / `ollama` |
| `--base_dir` | str | ❌ | `./Examples/graphrag_workspace` | 工作目录 |
| `--model_name` | str | ❌ | `qwen2.5-14b-instruct` | LLM 模型标识符 |
| `--embed_model_path` | str | ❌ | `/home/xzs/data/model/bge-large-en-v1.5` | Embedding 模型的**本地路径**。必须指向已下载的 HuggingFace 模型目录 |
| `--sample` | int | ❌ | `None` | 采样问题数量 |
| `--llm_base_url` | str | ❌ | `https://api.openai.com/v1` | LLM API 基础 URL |
| `--llm_api_key` | str | ❌ | `""` | API 密钥（回退到 `LLM_API_KEY` 环境变量） |

**fast-graphrag 内部固定参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `DOMAIN` | `"Analyze this story and identify the characters..."` | GraphRAG 领域描述，指导实体提取方向 |
| `EXAMPLE_QUERIES` | 5 个关于《圣诞颂歌》的示例问题 | 帮助模型理解查询风格 |
| `ENTITY_TYPES` | `["Character", "Animal", "Place", "Object", "Activity", "Event"]` | 实体类型列表，限定知识图谱的实体分类 |
| `embedding_dim` | 1024 | Embedding 维度 |
| `max_token_size` | 8192 | Embedding 最大 token 数 |

> ⚠️ `DOMAIN`、`EXAMPLE_QUERIES`、`ENTITY_TYPES` 是硬编码的，针对小说领域优化。如果评测医学领域，建议修改这些参数以获得更好效果。

---

### 2.3 run_hipporag2.py

HippoRAG2 框架的索引构建与批量推理脚本。

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--subset` | str | ✅ | - | 数据子集：`medical` / `novel` |
| `--mode` | str | ❌ | `API` | LLM 模式：`API` / `ollama` |
| `--base_dir` | str | ❌ | `./hipporag2_workspace` | 工作目录 |
| `--model_name` | str | ❌ | `gpt-4o-mini` | LLM 模型标识符 |
| `--embed_model_path` | str | ❌ | `/home/xzs/data/model/contriever` | Embedding 模型本地路径。默认值为 Contriever，建议改为 BGE 模型路径 |
| `--sample` | int | ❌ | `None` | 采样问题数量 |
| `--llm_base_url` | str | ❌ | `https://api.openai.com/v1` | LLM API 基础 URL |
| `--llm_api_key` | str | ❌ | `""` | API 密钥（回退到 `OPENAI_API_KEY` 环境变量） |

**HippoRAG2 BaseConfig 内部参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `force_index_from_scratch` | `True` | 每次运行强制重建索引 |
| `force_openie_from_scratch` | `True` | 每次运行强制重新执行开放信息抽取 |
| `retrieval_top_k` | `5` | 检索阶段返回的 top-k 结果数 |
| `linking_top_k` | `5` | 实体链接阶段的 top-k |
| `max_qa_steps` | `3` | QA 推理的最大步数 |
| `qa_top_k` | `5` | QA 阶段的 top-k |
| `graph_type` | `facts_and_sim_passage_node_unidirectional` | 图构建类型：基于事实和相似段落的单向图 |
| `embedding_batch_size` | `8` | Embedding 批处理大小 |
| `max_new_tokens` | `None` | 生成最大 token 数（None 表示不限制） |
| `openie_mode` | `online` | 开放信息抽取模式：在线（调用 LLM） |
| `rerank_dspy_file_path` | `hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json` | 重排序 DSPy 提示文件路径 |

**文本分块参数（`split_text` 函数）：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_token_size` | `256` | 每个文本块的 token 大小 |
| `chunk_overlap_token_size` | `32` | 相邻块的重叠 token 数 |

---

### 2.4 run_digimon.py

DIGIMON 统一框架的适配脚本。

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--subset` | str | ✅ | - | 数据子集：`medical` / `novel` |
| `--config` | str | ❌ | `./config.yml` | DIGIMON 配置 YAML 文件路径。需按 DIGIMON 文档编写 |
| `--output_dir` | str | ❌ | `./results/GraphRAG` | 结果输出目录 |
| `--mode` | str | ❌ | `config` | 运行模式：`config`（使用配置文件）/ `ollama`（使用 Ollama） |
| `--model_name` | str | ❌ | `qwen2.5-14b-instruct` | LLM 模型标识符（Ollama 模式下使用） |
| `--llm_base_url` | str | ❌ | `http://localhost:11434` | LLM API 基础 URL（Ollama 模式下使用） |
| `--llm_api_key` | str | ❌ | `""` | API 密钥（Ollama 模式下使用） |
| `--sample` | int | ❌ | `None` | 采样问题数量 |

> ⚠️ DIGIMON 的 `--option` 参数（如 `./Option/Method/HippoRAG.yaml`）在 README 示例中出现，但实际脚本使用 `--config`。具体取决于 DIGIMON 项目版本。

---

## 3. 评估参数

### 3.1 generation_eval.py（生成质量评估）

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--mode` | str | ✅ | `API` | LLM 模式：`API` / `ollama` |
| `--model` | str | ❌ | `gpt-4o-mini` | 评估用 LLM 模型。建议使用高质量模型以确保评估准确性 |
| `--base_url` | str | ❌ | `https://api.openai.com/v1` | LLM API 基础 URL |
| `--embedding_model` | str | ❌ | `BAAI/bge-large-en-v1.5` | Embedding 模型（用于 Answer Correctness 的语义相似度计算） |
| `--data_file` | str | ✅ | - | 输入 JSON 文件路径（框架推理输出的结果文件） |
| `--output_file` | str | ❌ | `evaluation_results.json` | 评估结果输出路径 |
| `--num_samples` | int | ❌ | `None` | 每种问题类型的采样数量。用于快速评估 |
| `--detailed_output` | flag | ❌ | `False` | 是否输出每条问题的详细评分。开启后输出包含 `average_scores` 和 `detailed` 两部分 |

**内部固定参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `SEED` | `42` | 随机种子，确保评估可复现 |
| `temperature` | `0.0` | LLM 温度，设为 0 确保确定性输出 |
| `max_retries` | `3` | LLM 调用最大重试次数 |
| `timeout` | `30` | LLM 调用超时时间（秒） |
| `max_concurrent` | `3` | 最大并发评估样本数（通过 Semaphore 控制） |
| `top_p` | `1` | 核采样参数 |
| `presence_penalty` | `0` | 存在惩罚 |
| `frequency_penalty` | `0` | 频率惩罚 |

**各问题类型的指标配置（`metric_config`）：**

```python
{
    'Fact Retrieval':       ["rouge_score", "answer_correctness"],
    'Complex Reasoning':    ["rouge_score", "answer_correctness"],
    'Contextual Summarize': ["answer_correctness", "coverage_score"],
    'Creative Generation':  ["answer_correctness", "coverage_score", "faithfulness"]
}
```

---

### 3.2 retrieval_eval.py（检索质量评估）

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--mode` | str | ✅ | `API` | LLM 模式：`API` / `ollama` |
| `--model` | str | ❌ | `gpt-4o-mini` | 评估用 LLM 模型 |
| `--base_url` | str | ❌ | `https://api.openai.com/v1` | LLM API 基础 URL |
| `--embedding_model` | str | ❌ | `BAAI/bge-large-en-v1.5` | Embedding 模型 |
| `--data_file` | str | ✅ | - | 输入 JSON 文件路径 |
| `--output_file` | str | ❌ | `retrieval_results.json` | 评估结果输出路径 |
| `--num_samples` | int | ❌ | `None` | 每种问题类型的采样数量 |
| `--detailed_output` | flag | ❌ | `False` | 是否输出详细评分 |

**内部固定参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `SEED` | `42` | 随机种子 |
| `max_concurrent` | `1` | 最大并发数。**注意比 generation_eval 低**，因为检索评估的 LLM 调用更密集 |

**评估指标（固定）：**
- `context_relevancy`：上下文相关性
- `evidence_recall`：证据召回率

---

### 3.3 indexing_eval.py（图构建质量评估）

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--framework` | str | ✅ | - | 框架名称。可选值：`microsoft_graphrag`, `lightrag`, `fast_graphrag`, `hipporag2`, `graphml` |
| `--base_path` | str | ✅ | - | 图数据根目录路径（框架 workspace 目录） |
| `--folder_name` | str | ❌ | `None` | 子目录名称。`hipporag2` 框架**必须**指定（如 `graph_store`） |
| `--output` | str | ❌ | `None` | 输出文件路径。不指定则打印到标准输出 |

> 此评估不需要 LLM 和 API Key，纯图论计算。

---

## 4. OllamaClient 内部配置

`Evaluation/llm/ollama_client.py` 中的连接和超时参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `total` timeout | `180s` | 单次请求总超时时间（3 分钟） |
| `connect` timeout | `15s` | TCP 连接建立超时 |
| `sock_read` timeout | `120s` | Socket 读取超时（等待 LLM 响应，2 分钟） |
| `limit`（连接池） | `10` | 总连接数上限 |
| `limit_per_host` | `5` | 单主机连接数上限 |
| `ttl_dns_cache` | `300s` | DNS 缓存有效期（5 分钟） |
| `max_retries` | `3` | 最大重试次数 |
| `base_delay` | `2s` | 重试基础延迟，按 `2^attempt` 指数递增（2s → 4s → 8s） |

**默认生成参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `temperature` | `0.0` | 温度（确定性输出） |
| `num_ctx` | `32768` | 上下文窗口大小 |
| `top_p` | `1` | 核采样 |
| `seed` | `None` | 随机种子（可通过 kwargs 传入） |

---

## 5. 评估指标内部参数

### 5.1 Answer Correctness（`answer_accuracy.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `weights` | `[0.75, 0.25]` | 事实性权重 75%，语义相似度权重 25% |
| `beta` | `1.0` | F-beta 分数的 beta 值（1.0 即 F1） |

### 5.2 Coverage Score（`coverage.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_retries` | `2` | LLM 调用重试次数 |
| 参考答案截断 | `3000` 字符 | 避免过长 prompt |
| 响应截断 | `3000` 字符 | 避免过长 prompt |

### 5.3 Faithfulness（`faithfulness.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_retries` | `2` | LLM 调用重试次数 |
| 问题截断 | `500` 字符 | 避免过长 prompt |
| 答案截断 | `3000` 字符 | 避免过长 prompt |
| 上下文截断 | `10000` 字符 | 避免过长 prompt |
| 陈述截断 | `5000` 字符 | 避免过长 prompt |

### 5.4 Context Relevance（`context_relevance.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_retries` | `2` | LLM 调用重试次数 |
| 评分次数 | `2` | 双次独立评分取均值，提高稳定性 |
| 评分范围 | `0-2` | 0=不相关，1=部分相关，2=完全相关 |
| 上下文截断 | `20000` 字符 | 避免过长 prompt |

### 5.5 Context Relevance v2（`context_relevance_v2.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_retries` | `3` | LLM 调用重试次数 |
| `max_length` | `21000` 字符 | 超长上下文截断阈值 |
| `chunk_size` | `3000` 字符 | 长文本分块大小 |
| 分块阈值 | `5000` 字符 | 超过此长度才分块 |

### 5.6 Evidence Recall（`evidence_recall.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_retries` | `2` | LLM 调用重试次数 |
| 上下文截断 | `20000` 字符 | 避免过长 prompt |

---

## 6. 配置最佳实践

### 6.1 快速测试配置

首次运行建议使用 `--sample` 参数限制数据量：

```bash
# 仅处理 10 个问题，快速验证流程
python Examples/run_lightrag.py --subset medical --mode API --sample 10 ...

# 评估时也可采样
python -m Evaluation.generation_eval --num_samples 5 ...
```

### 6.2 生产评估配置

- 使用高质量 LLM（如 `gpt-4o`）作为评估模型
- 不设置 `--sample` 和 `--num_samples`，评估全量数据
- 开启 `--detailed_output` 保留每条评分明细
- 使用 `BAAI/bge-large-en-v1.5`（1024 维）作为 Embedding 模型

### 6.3 Ollama 本地部署配置

```bash
# 拉取推荐模型
ollama pull qwen2.5:14b          # 推理用
ollama pull qwen2.5:72b          # 评估用（更准确）

# 所有脚本统一使用
--mode ollama --base_url http://localhost:11434
```

### 6.4 注意事项

1. **`embed_model` vs `embed_model_path`**：LightRAG 使用 `--embed_model`（HuggingFace 名称），fast-graphrag 和 HippoRAG2 使用 `--embed_model_path`（本地路径）
2. **API Key 变量名不统一**：大多数脚本用 `LLM_API_KEY`，HippoRAG2 用 `OPENAI_API_KEY`
3. **CUDA 设备硬编码**：`run_hipporag2.py` 硬编码 `CUDA_VISIBLE_DEVICES="5"`，`run_digimon.py` 硬编码 `"0"`，需根据实际环境修改
4. **fast-graphrag 的 DOMAIN/ENTITY_TYPES**：硬编码为小说领域，评测医学领域时建议修改
5. **Retrieval 评估并发低**：`retrieval_eval.py` 的 `max_concurrent=1`，评估速度较慢，这是有意为之以避免 LLM 过载
