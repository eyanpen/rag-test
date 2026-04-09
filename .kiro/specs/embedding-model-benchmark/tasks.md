# 实施计划：Embedding 模型基准测试

## 概述

基于 Microsoft GraphRAG pipeline 分割策略实现 Embedding 模型基准测试系统。核心思路：Phase 1-9（文档加载→社区报告）每个数据集只执行一次，Phase 10（向量嵌入）按 Embedding 模型数量执行 N 次。向量存入 FalkorDB（10.210.156.69:6379），图命名为 `<model>_<dataset>`。

## Tasks

- [ ] 1. 搭建项目结构与核心数据模型
  - [x] 1.1 创建 `tests/run_embedding_benchmark.py`，定义数据模型和常量
    - 定义 `EmbeddingModelConfig`、`BenchmarkConfig`、`DatasetPhaseResult`、`PredictionItem`、`EvaluationResult`、`ModelResult`、`BenchmarkSummary` 数据类
    - `BenchmarkConfig` 包含 `falkordb_host`/`falkordb_port`、`graphrag_root` 字段
    - 定义 `EMBEDDING_MODELS` 列表（8 个模型含 BGE-M3 三种模式）
    - 实现 `sanitize_name(model_name)` 工具函数：`/` → `-`，转小写，用于 FalkorDB 图命名
    - 实现 `make_graph_name(model_name, dataset_name)` 函数：`sanitize(model) + "_" + dataset`
    - _需求: 2.1, 2.4, 9.1_

  - [ ]* 1.2 编写 Property 1 + Property 11 属性测试
    - **Property 1: FalkorDB 图命名唯一性** — 不同模型名称生成不同图名，不含非法字符
    - **Property 11: 推理结果格式合规性** — PredictionItem JSON 包含所有必需字段
    - 测试文件：`tests/test_embedding_benchmark.py`
    - **验证: 需求 2.4, 9.1**

- [ ] 2. 实现 FalkorDB 自定义 VectorStore
  - [~] 2.1 创建 `tests/falkordb_vector_store.py`，实现 `FalkorDBVectorStore` 类
    - 继承 `graphrag_vectors.VectorStore` 抽象基类
    - 实现 `connect()`：连接 FalkorDB（host=10.210.156.69, port=6379, 无密码），选择图 `self.graph_name`
    - 实现 `load_documents(documents, overwrite)`：将向量数据写入 FalkorDB 图节点
    - 实现 `similarity_search_by_vector(query_embedding, k)`：向量相似度搜索
    - 实现 `similarity_search_by_text(text, text_embedder, k)`：文本→向量→搜索
    - 调用 `register_vector_store("falkordb", FalkorDBVectorStore)` 注册到 GraphRAG 工厂
    - _需求: 2.2, 设计: FalkorDB 集成_

- [ ] 3. 实现自适应并发控制器
  - [~] 3.1 在 `tests/run_embedding_benchmark.py` 中实现 `AdaptiveConcurrencyController`
    - `__init__(init=10, min_val=2, max_val=50)`
    - `adjust(is_error)`：5xx → 并发 -1，连续 5 成功 → +1
    - `install_hooks()`：Monkey-patch `httpx.AsyncClient.send` 注入节流和计时
    - 每 60 秒输出请求统计摘要
    - 双日志：控制台 INFO + 文件 DEBUG（`tests/benchmark_results/benchmark.log`）
    - _需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 3.2 编写 Property 10 属性测试：自适应并发控制不变量
    - **Property 10** — 随机 (init, min, max) + 随机 success/error 序列，验证不变量
    - **验证: 需求 8.1, 8.2, 8.3**

- [ ] 4. 实现数据集加载器
  - [~] 4.1 在 `tests/run_embedding_benchmark.py` 中实现 `DatasetLoader`
    - `prepare_kevin_scott(data_root, output_dir)`：合并 Episode part 文件（按 part 编号升序），写入 GraphRAG input 目录，加载 `Kevin Scott Questions.csv`
    - `prepare_msft(data_root, question_type, output_dir)`：准备 MSFT txt 文件，加载对应 CSV
    - `prepare_hotpotqa(data_root, output_dir)`：准备 HotPotQA `test_*.txt` 文件，加载 CSV
    - 所有方法在文件不存在/格式异常时输出警告并返回 None
    - _需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 4.2 编写 Property 3 + Property 4 属性测试
    - **Property 3: Episode 分片合并正确性** — 随机 part 文件，验证升序合并
    - **Property 4: CSV 问题加载完整性** — 随机 CSV，验证行数一致
    - **验证: 需求 3.1, 3.2**

- [ ] 5. 检查点 — 验证基础组件
  - 确保数据模型、FalkorDB VectorStore、并发控制、数据集加载器正确。如有问题请向用户确认。

- [ ] 6. 实现 GraphRAG Pipeline 分割执行器
  - [~] 6.1 实现 Phase 1-9 共享执行逻辑
    - 使用 `PipelineFactory.register_pipeline("pre_embedding", [...])` 注册不含 embedding 的 pipeline
    - 为每个数据集生成 `settings.yaml`（配置 completion_models、input_storage、output_storage、cache、table_provider）
    - 使用 `run_pipeline()` 执行 Phase 1-9，产物写入 `tests/benchmark_results/workspace/{dataset}/output/`
    - 记录 Phase 1-9 耗时到 `DatasetPhaseResult`
    - _需求: 2.2, 2.3, 设计: Pipeline 分割_

  - [~] 6.2 实现 Phase 10 按模型执行逻辑
    - 动态修改 `GraphRagConfig.embedding_models` 切换 Embedding 模型
    - 配置 `VectorStoreConfig(type="falkordb", host=..., port=..., graph_name=make_graph_name(model, dataset))`
    - 调用 `generate_text_embeddings.run_workflow()` 执行 Phase 10
    - 对三类文本生成向量：text_unit.text、entity.description、community_report.full_content
    - 记录 Phase 10 耗时
    - _需求: 2.1, 2.2, 2.4, 2.6, 设计: Phase 10 按模型执行_

  - [ ]* 6.3 编写 Property 6 属性测试：Phase 1-9 共享正确性
    - **Property 6** — 验证同一数据集的 parquet 产物只生成一次
    - **验证: 设计决策 — Pipeline 分割优化**

- [ ] 7. 实现查询与评估流程
  - [~] 7.1 实现 GraphRAG Search 查询逻辑
    - 从 `output_storage` 读取 parquet 表（entities, relationships, text_units, community_reports）
    - 从 FalkorDB 创建 `description_embedding_store`
    - 使用 `create_local_search()` 或 `create_global_search()` 创建 search engine
    - 对每个问题执行查询，收集 `PredictionItem`（含 context 和 generated_answer）
    - 模型可用性预检查：测试开始前对每个模型发送测试 embedding 请求
    - _需求: 2.5, 7.1, 7.3, 7.4_

  - [~] 7.2 实现评估指标计算
    - 调用 `GraphRAG-Benchmark/Evaluation/metrics/rouge.py` 的 `compute_rouge_score`
    - 调用 `GraphRAG-Benchmark/Evaluation/metrics/answer_accuracy.py` 的 `compute_answer_correctness`
    - 评估用 LLM 使用 `langchain_openai.ChatOpenAI` 连接 `http://10.210.156.69:8633`
    - 单指标失败记为 NaN，不影响其他指标
    - _需求: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 7.3 编写 Property 2 + Property 5 + Property 7 属性测试
    - **Property 2: 模型故障隔离** — mock 注入模型失败，验证其他模型正常
    - **Property 5: 数据集故障隔离** — mock 注入数据集失败，验证其他数据集正常
    - **Property 7: 指标计算故障降级** — mock 注入指标异常，验证 NaN 降级
    - **验证: 需求 2.5, 3.6, 5.4, 7.1, 7.2**

- [ ] 8. 检查点 — 验证 Pipeline 执行和查询评估
  - 确保 Phase 1-9 共享执行、Phase 10 按模型执行、查询和评估流程正确。如有问题请向用户确认。

- [ ] 9. 实现报告生成器
  - [~] 9.1 在 `tests/run_embedding_benchmark.py` 中实现 `ReportGenerator`
    - `generate_markdown(summary)`：生成 Markdown 报告，含测试环境、模型清单、指标对比表、BGE-M3 模式对比、总体排名（加权平均分降序）、耗时统计（含 Phase 1-9 共享耗时 + Phase 10 各模型耗时）
    - `generate_summary_json(summary)`：生成 JSON 汇总
    - 指标对比表中最优值加粗（`**value**`）
    - BGE-M3 模式对比章节标注各指标最优模式
    - _需求: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 9.3_

  - [ ]* 9.2 编写 Property 8 + Property 9 属性测试
    - **Property 8: 最优值标注正确性** — 随机数值列表，验证仅最大值加粗
    - **Property 9: 排名顺序正确性** — 随机结果集，验证降序排列
    - **验证: 需求 6.4, 6.5, 4.3**

- [ ] 10. 实现 Shell 入口脚本与主函数
  - [~] 10.1 在 `tests/run_embedding_benchmark.py` 中实现 `main()` 入口
    - 解析命令行参数（`--sample`、`--dataset`、`--models`、`--output-dir`、`--data-root`、`--graphrag-root`）
    - 构建 `BenchmarkConfig`，编排完整流程：数据集加载 → Phase 1-9 → Phase 10 × N → 查询 → 评估 → 报告
    - 创建输出目录结构
    - 保存 predictions/evaluations JSON 和 summary.json
    - _需求: 1.3, 1.4, 1.5, 1.8, 9.1, 9.2, 9.3, 9.4_

  - [~] 10.2 创建 `tests/run_embedding_benchmark.sh` Shell 入口脚本
    - 参数解析（`--sample`、`--dataset`、`--models`）传递给 Python
    - 依赖检查（`graphrag`、`falkordb`、`httpx`、`langchain_openai` 等），缺失时 `pip install`
    - API 连通性检查（LLM + Embedding）+ FalkorDB 连通性检查
    - 带时间戳进度日志
    - _需求: 1.1, 1.2, 1.6, 1.7, 1.9, 7.5_

- [ ] 11. 最终检查点 — 端到端验证
  - 使用 `--sample 1 --dataset kevin_scott --models "BAAI/bge-m3"` 运行端到端测试
  - 验证 FalkorDB 中图 `bge-m3_kevin_scott` 已创建且包含向量数据
  - 验证 `benchmark_report.md` 和 `summary.json` 正确生成
  - 如有问题请向用户确认。

## 备注

- 标记 `*` 的子任务为可选属性测试，可跳过以加速 MVP 交付
- GraphRAG 代码位于 `/home/eyanpen/sourceCode/rnd-ai-engine-features/graphrag/`
- FalkorDB 地址：`10.210.156.69:6379`，无密码
- Pipeline 参考文档：`docs/graphrag/graphrag-index-pipeline-deep-dive.md`
- Phase 1-9 产物（parquet 表）在 `workspace/{dataset}/output/` 下共享
- Phase 10 为每个 Embedding 模型创建独立的 FalkorDB 图
