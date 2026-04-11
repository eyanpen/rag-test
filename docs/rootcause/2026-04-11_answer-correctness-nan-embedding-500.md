# 🔍 Root Cause 分析报告

## 问题概述
- **问题描述**: Benchmark 评估中所有模型的 `answer_correctness` 均为 NaN，embedding API 返回 500 Internal Server Error
- **错误信息**: `httpx.HTTPStatusError: Server error '500 Internal Server Error' for url 'http://10.210.156.69:8633/embeddings'`
- **影响范围**: 所有 12 个成功运行的模型×数据集组合的 answer_correctness 指标全部缺失
- **严重程度**: High
- **分析日期**: 2026-04-11

---

## 1. 现象描述

### 1.1 错误表现
1. `summary.json` 和所有 `evaluations/*.json` 中 `answer_correctness` 字段均为 `NaN`
2. `benchmark.log` 中出现 60 次 `Answer correctness failed for <id>: Internal Server Error`（Medical 35 次 × 7 模型 = 35，Novel 25 次 × 5 模型 = 25）
3. Embedding API (`http://10.210.156.69:8633/embeddings`) 在 evaluation 阶段持续返回 HTTP 500

### 1.2 触发条件
- 在 `run_embedding_benchmark.py` 的 evaluation 阶段，`compute_answer_correctness()` 调用 `calculate_semantic_similarity()` 时，`OpenAIEmbeddings` 向 embedding API 发送请求触发 500 错误

### 1.3 复现步骤
1. 运行 benchmark：所有模型的 Phase 10（embedding 构建）正常完成
2. 进入 evaluation 阶段，计算 `answer_correctness`
3. `compute_answer_correctness()` 内部调用 `calculate_semantic_similarity()` → `embeddings.aembed_query()` → 发送 embedding 请求
4. 服务端返回 500，异常被 catch，`acc_scores` 全部为 `nan`
5. `safe_mean([nan, nan, ...])` 返回 `nan`

---

## 2. 信息收集

### 2.1 日志证据

**证据 1：Embedding 500 错误（benchmark.log:140240）**
```
2026-04-10 06:33:35,005 DEBUG HTTP Response: POST http://10.210.156.69:8633/embeddings "500 Internal Server Error"
  content-length: 21, content-type: text/plain; charset=utf-8
```

**证据 2：失败请求发送的是 token IDs 数组（benchmark.log:140225）**
```python
'json_data': {
    'input': [[2, 7648, 7874, 4078, 315, 28049, 26211, ...]],  # ← token IDs!
    'model': 'BAAI/bge-m3',
    'encoding_format': 'base64'
}
```

**证据 3：Phase 10 成功请求发送的是文本字符串（benchmark.log:138134）**
```python
'json_data': {
    'input': ['TUMORS:Tumors are abnormal tissue masses...'],  # ← 文本字符串
    'model': 'BAAI/bge-m3',
    'encoding_format': None
}
```

**证据 4：所有 answer_correctness 评估均失败（共 60 次）**
```
Medical: 5 questions × 7 models = 35 次失败
Novel:   5 questions × 5 models = 25 次失败
```

### 2.2 代码分析

**调用链：**
```
run_embedding_benchmark.py:451
  → compute_answer_correctness(question, answer, ground_truth, eval_llm, eval_emb)
    → calculate_semantic_similarity(embeddings, answer, ground_truth)  # answer_accuracy.py:170
      → embeddings.aembed_query(answer)  # langchain OpenAIEmbeddings
        → POST /embeddings  # 发送 token IDs → 500
```

**eval_emb 初始化代码（run_embedding_benchmark.py:448-453）：**
```python
eval_emb = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url=config.api_base_url,  # http://10.210.156.69:8633
    api_key="no-key",
)
```

**关键差异：**
- Phase 10 的 embedding 调用（GraphRAG 构建阶段）使用 litellm 直接发送**文本字符串** → 成功（200 OK）
- Evaluation 阶段的 embedding 调用使用 `langchain_openai.OpenAIEmbeddings` → 内部先 tokenize 再发送 **token IDs 数组** → 失败（500）

### 2.3 环境信息
- API 服务端：`http://10.210.156.69:8633`（基于 RayLLM/vLLM 的推理服务）
- Embedding 模型：`BAAI/bge-m3`
- 服务端返回 `content-length: 21`（极短响应体），说明服务端无法处理 token IDs 格式的输入

---

## 3. Root Cause

### 3.1 根因描述

**三个独立问题：**

**问题 1（answer_correctness = NaN）：** `langchain_openai.OpenAIEmbeddings` 默认会先将文本 tokenize 为 token IDs 数组再发送给 API。但 `http://10.210.156.69:8633` 上的 RayLLM/vLLM embedding 服务不支持 token IDs 格式的输入（只接受文本字符串），因此返回 500 Internal Server Error。所有 5 个 sample 的 `answer_correctness` 计算全部失败，`safe_mean([nan, nan, nan, nan, nan])` 返回 `nan`。

**问题 2（mE5-Large 400 错误）：** `intfloat/multilingual-e5-large-instruct` 模型的最大输入长度仅 512 tokens，GraphRAG 构建阶段的 entity description 文本超过了这个限制，导致 `PromptTooLongError: Input too long. The maximum input length is 512 tokens`。

**问题 3（Qwen3-Emb-8B / Alt 在 novel 数据集失败）：** FalkorDB (Redis) 磁盘持久化失败，`MISCONF Redis is configured to save RDB snapshots, but it's currently unable to persist to disk`，导致写入操作被拒绝。

### 3.2 因果链

**问题 1（核心问题）：**
```
OpenAIEmbeddings 默认 tokenize 行为
  → 将文本转为 token IDs 数组 [[2, 7648, 7874, ...]]
    → POST /embeddings 发送 token IDs
      → RayLLM/vLLM 不支持此格式
        → 返回 500 Internal Server Error
          → compute_answer_correctness 异常被 catch
            → acc_scores 全部为 nan
              → safe_mean 返回 nan
                → answer_correctness = NaN
```

### 3.3 证据汇总

| # | 证据 | 来源 | 支持结论 |
|---|------|------|---------|
| 1 | 失败请求 input 为 `[[2, 7648, ...]]`（token IDs） | benchmark.log:140225 | OpenAIEmbeddings 发送了 token IDs 而非文本 |
| 2 | 成功请求 input 为 `['TUMORS:Tumors are...']`（文本） | benchmark.log:138134 | 同一 API 接受文本字符串格式 |
| 3 | 500 响应 content-length 仅 21 字节 | benchmark.log:140241 | 服务端无法解析请求，返回简短错误 |
| 4 | 60 次 "Answer correctness failed: Internal Server Error" | benchmark.log 全文 | 所有评估均因此失败 |
| 5 | Phase 10 embedding 调用全部成功（200 OK） | benchmark.log:138157 | 证明 API 本身正常，问题在请求格式 |

### 3.4 测试验证
- **测试文件**: `tests/test_rootcause_embedding_input_format.py`
- **运行方式**:
  ```bash
  cd /home/eyanpen/sourceCode/rag-test
  source venv/bin/activate
  python -m pytest tests/test_rootcause_embedding_input_format.py -v
  ```
- **测试结果**（2026-04-11 实际运行，2 passed）:
  - ✅ `test_token_ids_input_returns_500` — 发送 token IDs `[[2, 7648, ...]]` → 500（复现问题）
  - ✅ `test_text_string_input_returns_200` — 发送文本字符串 → 200 + 正确 embedding 数据（验证修复方向）

### 3.5 Tokenize vs 直接发送文本 技术分析

#### 先 Tokenize 再发送 Token IDs（OpenAIEmbeddings 默认行为）

`check_embedding_ctx_length=True`（默认）时的流程：
```
文本 → tiktoken tokenize → [[2, 7648, 7874, ...]] → POST /embeddings
```

**优点：**
- 客户端可在发送前检测文本是否超过模型 max_tokens 限制
- 超长时自动分块 → 分别 embed → 取平均，避免服务端报错
- 对 OpenAI 官方 API 完全兼容（同时支持文本和 token IDs 两种输入）

**缺点：**
- tiktoken 是 OpenAI 模型的 tokenizer，与 BAAI/bge-m3 等开源模型的 tokenizer 不一致（token IDs 对不上）
- 很多非 OpenAI 推理服务（RayLLM、vLLM、TEI 等）的 /embeddings 端点不支持 token IDs 输入
- 增加客户端计算开销（需加载 tokenizer）

#### 直接发送文本字符串（`check_embedding_ctx_length=False`）

```
文本 → ["TUMORS: Tumors are..."] → POST /embeddings
```

**优点：**
- 兼容性最好，所有 embedding 服务都支持文本字符串输入
- 服务端用自己模型的 tokenizer 处理，token 化结果准确
- 无额外依赖，不需要客户端加载 tiktoken
- 更简单，减少出错环节

**缺点：**
- 客户端无法提前知道文本是否超长，超长时会收到服务端 400/413 错误
- 不会自动分块处理超长文本

#### 本场景的核心矛盾

对于非 OpenAI 模型，先 tokenize 不仅没有好处，反而有两个问题：
1. **格式不兼容** — 服务端不接受 token IDs，返回 500
2. **语义不正确** — 即使服务端接受，tiktoken 的 token IDs 用 bge-m3 的 tokenizer 解码会得到乱码

**结论：** 对于非 OpenAI 推理服务 + 开源 embedding 模型，直接发送文本字符串是唯一正确的选择。

---

## 4. 排除项

### 已排查但排除的假设

| # | 假设 | 排除原因 |
|---|------|---------|
| 1 | Embedding API 服务宕机 | Phase 10 阶段同一 API 正常返回 200，且 500 错误响应时间极短（0.01s），说明服务在线但拒绝了请求 |
| 2 | BAAI/bge-m3 模型不可用 | Phase 10 使用同一模型成功生成了大量 embedding |
| 3 | 网络问题 | 所有请求都有响应，只是状态码为 500 |
| 4 | LLM 部分（statement generation）失败 | LLM 调用（chat/completions）在 answer_correctness 失败后继续正常工作 |

---

## 5. 修复方案

### 方案 A: 禁用 OpenAIEmbeddings 的 tokenize 行为 ✅ 已采纳

**描述**: 在创建 `OpenAIEmbeddings` 时设置 `check_embedding_ctx_length=False`，阻止 langchain 先 tokenize 再发送 token IDs 的行为，改为直接发送原始文本字符串。

**改动范围**:
- `tests/run_embedding_benchmark.py`: 修改 `eval_emb` 初始化参数

**代码变更**:
```python
# Before
eval_emb = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url=config.api_base_url,
    api_key="no-key",
)

# After
eval_emb = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url=config.api_base_url,
    api_key="no-key",
    check_embedding_ctx_length=False,  # 禁用 tokenize，直接发送文本
)
```

**优点**:
- 改动最小（1 行）
- 直接解决根因
- 不影响其他功能

**缺点**:
- 如果输入文本超过模型 max_tokens，不会自动截断（但 bge-m3 支持 8192 tokens，一般够用）

**风险**: 极低
**工作量**: 小
**状态**: ✅ 已于 2026-04-11 02:39 应用到 `tests/run_embedding_benchmark.py`

---

### 方案 B: 替换为 litellm 的 embedding 调用（未采纳）

**描述**: 不使用 `langchain_openai.OpenAIEmbeddings`，改用 litellm 的 `aembedding()` 函数（与 Phase 10 相同的调用方式），确保发送文本字符串。

**改动范围**:
- `tests/run_embedding_benchmark.py`: 替换 eval_emb 的创建和使用方式
- `GraphRAG-Benchmark/Evaluation/metrics/answer_accuracy.py`: 修改 `calculate_semantic_similarity` 接口

**优点**: 与 Phase 10 使用相同的调用路径，一致性好
**缺点**: 改动较大，需要修改多文件接口
**风险**: 中等
**工作量**: 中

---

## 6. 方案对比

| 维度 | 方案 A ✅ | 方案 B |
|------|--------|--------|
| 改动范围 | 1 行 | 多文件 |
| 实现复杂度 | 极低 | 中 |
| 风险等级 | 极低 | 中 |
| 向后兼容 | 完全兼容 | 需改接口 |
| 性能影响 | 无 | 无 |
| 可维护性 | 好 | 好 |
| 推荐度 | ⭐⭐⭐ | ⭐⭐ |

## 7. 建议

**已采纳方案**: 方案 A

**理由**: 只需加一个参数 `check_embedding_ctx_length=False` 即可解决。这是 langchain `OpenAIEmbeddings` 的已知行为——当 `check_embedding_ctx_length=True`（默认值）时，它会先用 tiktoken tokenize 文本，然后发送 token IDs 数组给 API。但 RayLLM/vLLM 的 embedding endpoint 不支持 token IDs 输入格式，只接受文本字符串。

**附加建议**:
- mE5-Large 的 512 token 限制问题：需要在 embedding 前对文本做截断，或从 benchmark 模型列表中移除该模型
- Redis MISCONF 问题：检查 FalkorDB 所在机器的磁盘空间，或设置 `config set stop-writes-on-bgsave-error no`

---

## Q&A 追踪

### Q1 (2026-04-11 02:23)
**用户问题**: 写测试 case 验证 token IDs 格式和文本字符串格式分别调用 embedding API 的结果

**回答**: 编写了 `tests/test_rootcause_embedding_input_format.py`，包含两个测试：
- `test_token_ids_input_returns_500`: 发送 `[[2, 7648, 7874, ...]]` → 确认返回 500
- `test_text_string_input_returns_200`: 发送 `['TUMORS: Tumors are...']` → 确认返回 200 + 正确 embedding 数据

两个测试均通过，验证了根因。测试加了 API 可达性检查，不可达时自动 skip。

**影响**: 新增测试验证证据，强化了 Root Cause 结论。已更新到报告 3.4 节。

---

### Q2 (2026-04-11 02:31)
**用户问题**: 再跑一次 test_token_ids_input_returns_500

**回答**: 该次运行时 API `10.210.156.69:8633` 不可达，测试被 skip（health check 超时）。服务端当时未运行或网络不通。

**影响**: 无，不影响结论。API 可用性是外部因素。

---

### Q3 (2026-04-11 02:35)
**用户问题**: 再跑一次

**回答**: API 恢复，2 passed。结果与首次一致：token IDs → 500，文本字符串 → 200。

**影响**: 二次验证通过，结论稳定。

---

### Q4 (2026-04-11 02:37)
**用户问题**: 详细说说 tokenize 和直接发送文本有什么优劣

**回答**:

**先 Tokenize（默认行为 `check_embedding_ctx_length=True`）：**
- 优点：客户端可提前检测超长文本并自动分块；对 OpenAI 官方 API 完全兼容
- 缺点：tiktoken 是 OpenAI tokenizer，与 bge-m3 等开源模型 tokenizer 不一致；很多非 OpenAI 推理服务不支持 token IDs 输入；增加客户端开销

**直接发送文本（`check_embedding_ctx_length=False`）：**
- 优点：兼容性最好；服务端用正确的 tokenizer 处理；无额外依赖；更简单
- 缺点：不会自动截断超长文本

核心矛盾：对非 OpenAI 模型，tiktoken 的 token IDs 即使被服务端接受，解码出来也是乱码。所以直接发送文本是唯一正确选择。

**影响**: 补充了技术决策依据。已更新到报告 3.5 节。

---

### Q5 (2026-04-11 02:39)
**用户问题**: 更新到文档，并采用方案 A

**回答**: 已执行方案 A，在 `tests/run_embedding_benchmark.py` 的 `OpenAIEmbeddings` 初始化中加入 `check_embedding_ctx_length=False`。同时更新 RCA 报告，标记方案 A 为已采纳，补充测试验证和技术分析章节。

**影响**: 修复已应用。报告更新为最终版本。

---

### Q6 (2026-04-11 02:44)
**用户问题**: git comment

**回答**: 分析工作区改动（1 个修改文件 + 2 个新增文件），生成 commit message 并提交。
```
bb92201 fix(eval): 修复 answer_correctness 全为 NaN 的问题
```
3 files changed, 401 insertions。

**影响**: 修复代码和分析文档已入库。

---

### Q7 (2026-04-11 02:44)
**用户问题**: 重新运行 Phase 10 及以后的步骤，确保 answer_correctness 有值。如果因 embedding maxtoken 限制，可保持为空。

**回答**: 正在执行，见下方调试过程记录。
