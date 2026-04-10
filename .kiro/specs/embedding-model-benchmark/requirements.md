# 需求文档：Embedding 模型基准测试脚本

## 简介

创建一键测试脚本，放置于 `tests/` 目录下，用于系统性比较不同 Embedding 模型在 GraphRAG 场景中的效果。脚本使用 `GraphRAG-Benchmark/Datasets/` 下的 medical 和 novel 数据集（含完整 ground_truth 标准答案），通过 Microsoft GraphRAG pipeline 进行索引构建和查询，利用 GraphRAG-Benchmark 评估框架计算多维度指标，并生成结构化 Markdown 对比报告。对于支持多模式的模型（如 BAAI/bge-m3 的默认/heavy/interactive 模式），还需比较不同模式间的效果差异。

## 术语表

- **Benchmark_Script**: 放置于 `tests/` 目录下的一键测试入口脚本（Shell 脚本），负责编排整个测试流程
- **Test_Runner**: Python 测试执行器，负责遍历 Embedding 模型、构建索引、执行查询、收集结果
- **Embedding_Model**: 通过 OpenAI 兼容 API 提供的向量化模型，将文本转换为稠密向量表示
- **Bi_Encoder_Mode**: 双塔模式，BAAI/bge-m3 模型的一种变体，query 和 document 分别编码；包括默认模式、heavy 模式和 interactive 模式
- **GraphRAG_Index**: Microsoft GraphRAG 框架构建的知识图谱索引，用于后续查询
- **Evaluation_Pipeline**: GraphRAG-Benchmark 提供的评估流水线，包含 ROUGE-L、Answer Correctness、Coverage Score、Faithfulness、Context Relevancy、Evidence Recall 等指标
- **Benchmark_Report**: 最终生成的 Markdown 格式对比报告，包含所有模型在所有数据集上的评估指标对比
- **API_Server**: 位于 `http://10.210.156.69:8633` 的 OpenAI 兼容 API 服务器，提供 LLM 和 Embedding 服务
- **Dataset_Loader**: 数据集加载模块，负责从 `GraphRAG-Benchmark/Datasets/` 加载语料文本和带标准答案的问题

## 可用 Embedding 模型清单

以下为 API 服务器上可用的 Embedding 类型模型：

| 模型名称 | 类型 | 最大 Token | 备注 |
|---------|------|-----------|------|
| BAAI/bge-m3 | embedding | 8192 | 默认模式 |
| BAAI/bge-m3/heavy | embedding | 8192 | heavy 模式（双塔变体） |
| BAAI/bge-m3/interactive | embedding | 8192 | interactive 模式（双塔变体） |
| intfloat/e5-mistral-7b-instruct | embedding | 4096 | 指令微调 E5 模型 |
| intfloat/multilingual-e5-large-instruct | embedding | 512 | 多语言 E5 模型 |
| nomic-ai/nomic-embed-text-v1.5 | embedding | 8192 | Nomic 文本嵌入模型 |
| Qwen/Qwen3-Embedding-8B | embedding | 8192 | Qwen3 嵌入模型 |
| Qwen/Qwen3-Embedding-8B-Alt | embedding | 32768 | Qwen3 嵌入模型（长上下文变体） |

Reranker 模型（可选用于检索增强）：

| 模型名称 | 类型 | 最大 Token |
|---------|------|-----------|
| BAAI/bge-reranker-v2-m3 | rerank | 8192 |

## 可用数据集

数据集根目录：`GraphRAG-Benchmark/Datasets/`

| 数据集 | 语料文件 | 问题文件 | 问题数量 | 含 ground_truth | 状态 |
|--------|---------|---------|---------|----------------|------|
| Medical | `Corpus/medical.json` (单文档, 1MB) | `Questions/medical_questions.json` | 2062 | ✅ answer + evidence | 可用 |
| Novel | `Corpus/novel.json` (20 篇小说) | `Questions/novel_questions.json` | 2010 | ✅ answer + evidence | 可用 |

问题 JSON 字段：`id`, `source`, `question`, `answer`, `question_type`, `evidence`, `evidence_relations`/`evidence_triple`

## 需求

### 需求 1：一键测试入口脚本

**用户故事：** 作为开发者，我希望在 `tests/` 目录下有一个一键运行的 Shell 脚本，以便快速启动完整的 Embedding 模型基准测试流程。

#### 验收标准

1. THE Benchmark_Script SHALL 位于 `tests/` 目录下，文件名为 `run_embedding_benchmark.sh`
2. WHEN 用户执行 `bash tests/run_embedding_benchmark.sh` 时，THE Benchmark_Script SHALL 自动完成依赖检查、API 连通性验证、测试执行和报告生成的完整流程
3. THE Benchmark_Script SHALL 支持通过命令行参数指定采样数量（`--sample`），默认值为 5，用于快速验证
4. THE Benchmark_Script SHALL 支持通过命令行参数指定要测试的数据集（`--dataset`），可选值为 `medical`、`novel`、`all`，默认值为 `medical`
5. THE Benchmark_Script SHALL 支持通过命令行参数指定要测试的 Embedding 模型列表（`--models`），默认测试所有可用 Embedding 模型
6. IF 依赖检查失败，THEN THE Benchmark_Script SHALL 输出缺失依赖列表并尝试自动安装
7. IF API 连通性检查失败，THEN THE Benchmark_Script SHALL 输出错误信息并终止执行，返回非零退出码
8. THE Benchmark_Script SHALL 将所有中间结果保存到 `tests/benchmark_results/` 目录下
9. THE Benchmark_Script SHALL 在执行过程中输出带时间戳的进度日志

### 需求 2：Embedding 模型遍历与索引构建

**用户故事：** 作为开发者，我希望测试脚本能自动遍历所有指定的 Embedding 模型，为每个模型独立构建 GraphRAG 索引，以便公平比较不同模型的效果。

#### 验收标准

1. THE Test_Runner SHALL 遍历所有指定的 Embedding_Model，为每个模型独立执行索引构建和查询流程
2. THE Test_Runner SHALL 使用 fast-graphrag 框架的 `OpenAIEmbeddingService` 连接远程 API（`http://10.210.156.69:8633`）进行 Embedding 计算
3. THE Test_Runner SHALL 使用 `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` 作为 LLM 模型，通过 `OpenAILLMService` 连接同一 API 服务器
4. THE Test_Runner SHALL 为每个 Embedding_Model 创建独立的工作目录（`tests/benchmark_results/workspace/{model_name}/`），避免索引数据互相干扰
5. WHEN 某个 Embedding_Model 的索引构建或查询过程失败时，THE Test_Runner SHALL 记录错误信息并继续处理下一个模型
6. THE Test_Runner SHALL 记录每个模型的索引构建耗时和查询耗时

### 需求 3：数据集加载与问题查询

**用户故事：** 作为开发者，我希望测试脚本能正确加载 `GraphRAG-Benchmark/Datasets/` 下的 medical 和 novel 数据集（含语料和带标准答案的问题），以便使用标准化数据进行评测。

#### 验收标准

1. THE Dataset_Loader SHALL 从 `GraphRAG-Benchmark/Datasets/Corpus/medical.json` 加载 Medical 语料文本，写入 GraphRAG input 目录
2. THE Dataset_Loader SHALL 从 `GraphRAG-Benchmark/Datasets/Questions/medical_questions.json` 加载问题列表，包含 `question`、`answer`（ground_truth）、`evidence` 字段
3. THE Dataset_Loader SHALL 从 `GraphRAG-Benchmark/Datasets/Corpus/novel.json` 加载 Novel 语料（20 篇小说），每篇写入独立 txt 文件
4. THE Dataset_Loader SHALL 从 `GraphRAG-Benchmark/Datasets/Questions/novel_questions.json` 加载问题列表，包含 `question`、`answer`（ground_truth）、`evidence` 字段
5. THE Dataset_Loader SHALL 将问题的 `answer` 字段作为 ground_truth 传递给评估流程
6. WHEN 数据集文件不存在或格式异常时，THE Dataset_Loader SHALL 输出警告信息并跳过该数据集
7. THE Test_Runner SHALL 对每个 Embedding_Model 和每个数据集的组合执行查询，收集生成的答案和检索到的上下文

### 需求 4：BGE-M3 多模式对比

**用户故事：** 作为开发者，我希望测试脚本能比较 BAAI/bge-m3 的默认模式、heavy 模式和 interactive 模式的效果差异，以便评估不同编码策略的优劣。

#### 验收标准

1. THE Test_Runner SHALL 将 `BAAI/bge-m3`、`BAAI/bge-m3/heavy`、`BAAI/bge-m3/interactive` 作为三个独立的 Embedding 模型分别进行测试
2. THE Benchmark_Report SHALL 包含一个专门的 BGE-M3 模式对比章节，将三种模式的指标并排展示
3. THE Benchmark_Report SHALL 在 BGE-M3 模式对比章节中标注各指标的最优模式

### 需求 5：评估指标计算

**用户故事：** 作为开发者，我希望测试脚本能使用 GraphRAG-Benchmark 的评估框架计算多维度指标，以便全面衡量不同 Embedding 模型的效果。

#### 验收标准

1. THE Evaluation_Pipeline SHALL 计算以下生成质量指标：ROUGE-L、Answer Correctness
2. THE Evaluation_Pipeline SHALL 使用 `GraphRAG-Benchmark/Evaluation/metrics/` 下的指标实现进行计算
3. THE Evaluation_Pipeline SHALL 使用同一 LLM（`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`）作为评估用 LLM
4. IF 某个指标计算失败，THEN THE Evaluation_Pipeline SHALL 将该指标记录为 NaN 并继续计算其他指标
5. THE Evaluation_Pipeline SHALL 记录每个模型的评估耗时

### 需求 6：测试报告生成

**用户故事：** 作为开发者，我希望测试完成后自动生成结构化的 Markdown 对比报告，以便直观比较不同 Embedding 模型的效果。

#### 验收标准

1. THE Benchmark_Report SHALL 以 Markdown 格式生成，保存到 `tests/benchmark_results/benchmark_report.md`
2. THE Benchmark_Report SHALL 包含以下章节：测试环境信息、模型清单、各数据集的指标对比表、BGE-M3 模式对比、总体排名、耗时统计
3. THE Benchmark_Report SHALL 在指标对比表中使用表格格式，每行为一个 Embedding 模型，每列为一个评估指标
4. THE Benchmark_Report SHALL 在每个指标列中标注最优值（加粗显示）
5. THE Benchmark_Report SHALL 包含总体排名章节，按所有指标的加权平均分对模型进行排序
6. THE Benchmark_Report SHALL 包含测试环境信息：API 地址、LLM 模型、数据集、采样数量、测试时间
7. THE Benchmark_Report SHALL 包含耗时统计：每个模型的索引构建时间、查询时间、评估时间、总时间

### 需求 7：错误处理与容错

**用户故事：** 作为开发者，我希望测试脚本具备良好的错误处理能力，以便在部分模型或数据集失败时仍能完成其余测试并生成报告。

#### 验收标准

1. IF 某个 Embedding_Model 的 API 调用返回错误，THEN THE Test_Runner SHALL 记录错误详情、跳过该模型并继续测试其他模型
2. IF 某个数据集的加载失败，THEN THE Test_Runner SHALL 记录错误详情、跳过该数据集并继续测试其他数据集
3. THE Test_Runner SHALL 在测试开始前验证 API 服务器对每个 Embedding_Model 的可用性
4. IF API 服务器对某个 Embedding_Model 返回不可用，THEN THE Test_Runner SHALL 将该模型标记为"不可用"并在报告中注明
5. THE Benchmark_Script SHALL 将完整运行日志保存到 `tests/benchmark_results/benchmark.log`

### 需求 8：自适应并发控制

**用户故事：** 作为开发者，我希望测试脚本能根据 API 服务器的响应状态自动调节并发请求数，以便在高负载时避免雪崩，在低负载时充分利用吞吐。

#### 验收标准

1. THE Test_Runner SHALL 实现自适应并发控制，初始并发数为 10，最小为 2，最大为 50
2. WHEN API 服务器返回 5xx 状态码时，THE Test_Runner SHALL 将当前并发数减 1（不低于最小值 2）
3. WHEN 连续 5 次请求成功时，THE Test_Runner SHALL 将当前并发数加 1（不超过最大值 50）
4. THE Test_Runner SHALL 通过 Monkey-patch `httpx.AsyncClient.send` 实现全局请求节流，使用 `asyncio.Semaphore` 控制并发
5. THE Test_Runner SHALL 在并发数变化时输出 INFO 级别日志到控制台（格式：`[ADAPTIVE] concurrency {old} → {new}`）
6. THE Test_Runner SHALL 配置双日志输出：控制台输出 INFO 级别，文件输出 DEBUG 级别（日志文件保存到 `tests/benchmark_results/benchmark.log`）
7. THE Test_Runner SHALL 记录每个 HTTP 请求的耗时，每 60 秒输出一次统计摘要（请求数、最小/最大/平均耗时）

### 需求 9：结果持久化与可复现性

**用户故事：** 作为开发者，我希望测试的中间结果和最终结果都被持久化保存，以便后续分析和复现。

#### 验收标准

1. THE Test_Runner SHALL 将每个模型的推理结果保存为 JSON 文件（`tests/benchmark_results/predictions/{model_name}.json`），格式与 GraphRAG-Benchmark 统一输出格式一致
2. THE Test_Runner SHALL 将每个模型的评估结果保存为 JSON 文件（`tests/benchmark_results/evaluations/{model_name}.json`）
3. THE Test_Runner SHALL 将所有模型的汇总评估结果保存为 JSON 文件（`tests/benchmark_results/summary.json`）
4. THE Benchmark_Script SHALL 在报告头部记录完整的测试参数，包括 API 地址、LLM 模型、Embedding 模型列表、数据集、采样数量
