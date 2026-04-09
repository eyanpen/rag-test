# 实施计划：Embedding 模型基准测试

## 概述

将设计文档中的 Embedding 模型基准测试系统转化为可执行的编码任务。实现分为 6 个阶段：数据模型与工具函数、自适应并发控制、数据集加载器、基准测试执行器、报告生成器、Shell 入口脚本。每个阶段包含核心实现和可选测试子任务，所有属性测试使用 `pytest` + `hypothesis` 框架。

## Tasks

- [ ] 1. 搭建项目结构与核心数据模型
  - [ ] 1.1 创建 `tests/run_embedding_benchmark.py`，定义所有数据模型（`EmbeddingModelConfig`、`BenchmarkConfig`、`DatasetResult`、`QuestionItem`、`PredictionItem`、`EvaluationResult`、`ModelResult`、`BenchmarkSummary`）和常量（`EMBEDDING_MODELS` 列表、`DOMAIN`、`EXAMPLE_QUERIES`、`ENTITY_TYPES`）
    - 使用 `@dataclass` 定义所有数据类
    - 实现 `sanitize_model_name()` 工具函数，将模型名称中的 `/` 替换为 `__`，确保生成合法文件系统路径
    - 定义 8 个 Embedding 模型配置（含 BGE-M3 三种模式）
    - _需求: 2.1, 2.4, 9.1_

  - [ ]* 1.2 编写 Property 1 属性测试：工作目录唯一性
    - **Property 1: 工作目录唯一性**
    - 在 `tests/test_embedding_benchmark.py` 中创建测试文件
    - 使用 `hypothesis` 生成随机模型名称字符串（含 `/`、空格、特殊字符），验证 `sanitize_model_name()` 输出不含 `/` 且不同输入产生不同输出
    - **验证: 需求 2.4**

  - [ ]* 1.3 编写 Property 11 属性测试：推理结果格式合规性
    - **Property 11: 推理结果格式合规性**
    - 使用 `hypothesis` 生成随机 `PredictionItem` 实例，验证序列化为 JSON 后包含所有必需字段（`id`、`question`、`source`、`context`、`generated_answer`、`ground_truth`）且类型正确
    - **验证: 需求 9.1**

- [ ] 2. 实现自适应并发控制器
  - [ ] 2.1 在 `tests/run_embedding_benchmark.py` 中实现 `AdaptiveConcurrencyController` 类
    - 实现 `__init__(self, init, min_val, max_val)` 构造函数
    - 实现 `adjust(self, is_error)` 方法：5xx 时并发 -1，连续 5 次成功时并发 +1
    - 实现 `install_hooks(self)` 方法：Monkey-patch `httpx.AsyncClient.send` 注入节流和计时
    - 实现 HTTP 请求计时和每 60 秒统计摘要输出
    - 配置双日志输出：控制台 INFO + 文件 DEBUG（`tests/benchmark_results/benchmark.log`）
    - _需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 2.2 编写 Property 10 属性测试：自适应并发控制不变量
    - **Property 10: 自适应并发控制不变量**
    - 使用 `hypothesis` 生成随机 `(init, min, max)` 配置和随机 success/error 序列，验证：初始并发数等于 init；任意调用序列后 min ≤ current ≤ max；单次错误并发 -1；连续 5 次成功并发 +1；错误重置连续成功计数
    - **验证: 需求 8.1, 8.2, 8.3**

- [ ] 3. 实现数据集加载器
  - [ ] 3.1 在 `tests/run_embedding_benchmark.py` 中实现 `DatasetLoader` 类
    - 实现 `load_kevin_scott(base_path)` 静态方法：加载 `kevinScott/input/*.txt`，合并同一 Episode 的 part 文件（按 part 编号升序），加载 `Kevin Scott Questions.csv`
    - 实现 `load_msft(base_path, question_type)` 静态方法：加载 `MSFT/txt/*.txt`，根据 `question_type` 加载对应 CSV 问题文件
    - 实现 `load_hotpotqa(base_path)` 静态方法：加载 `HotPotQA/input/test_*.txt`，加载 `HotPotQA Filtered Questions.csv`
    - 所有加载方法在文件不存在或格式异常时输出警告并返回空结果，不抛出异常
    - _需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 3.2 编写 Property 3 属性测试：Episode 分片合并正确性
    - **Property 3: Episode 分片合并正确性**
    - 使用 `hypothesis` 生成随机 Episode 名称和随机数量的 part 文件内容，验证合并后文本按 part 编号升序包含所有分片，且长度等于所有分片长度之和加分隔符
    - **验证: 需求 3.1**

  - [ ]* 3.3 编写 Property 4 属性测试：CSV 问题加载完整性
    - **Property 4: CSV 问题加载完整性**
    - 使用 `hypothesis` 生成随机 CSV 内容（随机行数、随机文本），验证返回的问题列表长度等于 CSV 数据行数，且每个问题的字段与 CSV 对应行一致
    - **验证: 需求 3.2**

- [ ] 4. 检查点 - 确保数据模型、并发控制和数据集加载器正确
  - 确保所有测试通过，如有问题请向用户确认。

- [ ] 5. 实现基准测试执行器
  - [ ] 5.1 在 `tests/run_embedding_benchmark.py` 中实现 `BenchmarkRunner` 类
    - 实现 `__init__(self, config)` 构造函数，接收 `BenchmarkConfig`
    - 实现 `create_graphrag_instance(model_config, workspace)` 方法：使用 `OpenAIEmbeddingService` 和 `OpenAILLMService` 创建 `GraphRAG` 实例
    - 实现 `run_single_model(self, model, dataset)` 方法：对单个模型执行索引构建 + 查询，记录耗时，捕获异常并记录到 `ModelResult.error`
    - 实现 `evaluate_predictions(self, predictions, model_name)` 方法：调用 `GraphRAG-Benchmark/Evaluation/metrics/` 的 `compute_rouge_score` 和 `compute_answer_correctness`，单指标失败记为 NaN
    - 实现 `run(self)` 方法：遍历所有模型和数据集组合，编排完整流程，保存 predictions 和 evaluations JSON 文件
    - 实现模型可用性预检查：测试开始前对每个模型发送测试 embedding 请求
    - _需求: 2.1, 2.2, 2.3, 2.5, 2.6, 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.3, 7.4, 9.1, 9.2_

  - [ ]* 5.2 编写 Property 2 属性测试：模型故障隔离
    - **Property 2: 模型故障隔离**
    - 使用 mock 注入随机模型失败，验证未失败模型仍生成完整结果
    - **验证: 需求 2.5, 7.1**

  - [ ]* 5.3 编写 Property 5 属性测试：数据集故障隔离
    - **Property 5: 数据集故障隔离**
    - 使用 mock 注入随机数据集加载失败，验证可正常加载的数据集仍完成测试
    - **验证: 需求 3.6, 7.2**

  - [ ]* 5.4 编写 Property 6 属性测试：指标计算故障降级
    - **Property 6: 指标计算故障降级**
    - 使用 mock 注入随机指标计算异常，验证失败指标记为 NaN，其他指标正常返回
    - **验证: 需求 5.4**

- [ ] 6. 实现报告生成器
  - [ ] 6.1 在 `tests/run_embedding_benchmark.py` 中实现 `ReportGenerator` 类
    - 实现 `generate_markdown(summary)` 静态方法：生成包含测试环境信息、模型清单、各数据集指标对比表、BGE-M3 模式对比、总体排名（按加权平均分降序）、耗时统计的 Markdown 报告
    - 实现 `generate_summary_json(summary)` 静态方法：生成 JSON 格式汇总数据
    - 在指标对比表中对每列最优值加粗标注（`**value**`）
    - 在 BGE-M3 模式对比章节中标注各指标最优模式
    - _需求: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 9.3_

  - [ ]* 6.2 编写 Property 7 属性测试：报告章节完整性
    - **Property 7: 报告章节完整性**
    - 使用 `hypothesis` 生成随机 `BenchmarkSummary`，验证生成的 Markdown 包含所有必需章节标题
    - **验证: 需求 6.2**

  - [ ]* 6.3 编写 Property 8 属性测试：最优值标注正确性
    - **Property 8: 最优值标注正确性**
    - 使用 `hypothesis` 生成随机数值列表，验证仅最大值被加粗标注
    - **验证: 需求 6.4, 4.3**

  - [ ]* 6.4 编写 Property 9 属性测试：排名顺序正确性
    - **Property 9: 排名顺序正确性**
    - 使用 `hypothesis` 生成随机模型结果集，验证排名严格按加权平均分降序排列
    - **验证: 需求 6.5**

- [ ] 7. 检查点 - 确保执行器和报告生成器正确
  - 确保所有测试通过，如有问题请向用户确认。

- [ ] 8. 实现 Shell 入口脚本与主函数集成
  - [ ] 8.1 在 `tests/run_embedding_benchmark.py` 中实现 `main()` 入口函数
    - 解析命令行参数（`--sample`、`--dataset`、`--models`、`--output-dir`、`--data-root`）
    - 构建 `BenchmarkConfig`，创建 `BenchmarkRunner` 并调用 `run()`
    - 调用 `ReportGenerator` 生成 Markdown 报告和 summary.json
    - 创建输出目录结构（`predictions/`、`evaluations/`、`workspace/`）
    - _需求: 1.3, 1.4, 1.5, 1.8, 9.1, 9.2, 9.3, 9.4_

  - [ ] 8.2 创建 `tests/run_embedding_benchmark.sh` Shell 入口脚本
    - 实现参数解析（`--sample`、`--dataset`、`--models`）并传递给 Python 执行器
    - 实现依赖检查（`fast_graphrag`、`httpx`、`hypothesis`、`pytest`、`langchain_openai` 等），缺失时尝试 `pip install`
    - 实现 API 连通性检查（LLM + Embedding），失败时 `exit 1`
    - 输出带时间戳的进度日志
    - 调用 `python3 tests/run_embedding_benchmark.py` 执行测试
    - _需求: 1.1, 1.2, 1.6, 1.7, 1.9, 7.5_

- [ ] 9. 最终检查点 - 确保所有组件集成正确
  - 确保所有测试通过，如有问题请向用户确认。

## 备注

- 标记 `*` 的子任务为可选任务，可跳过以加速 MVP 交付
- 每个任务引用了具体的需求编号，确保可追溯性
- 属性测试验证设计文档中定义的 11 个正确性属性
- 单元测试验证具体场景和边界条件
- 检查点确保增量验证，及时发现问题
