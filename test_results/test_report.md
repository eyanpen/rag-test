# fast-graphrag 测试报告

> 生成时间: 2026-04-09 14:49:35

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| Python | Python 3.12.3 |
| OS | Linux 6.8.0-79-generic |
| API Base URL | http://10.210.156.69:8633 |
| LLM 模型 | Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 |
| Embedding 模型 | Qwen/Qwen3-Embedding-8B |
| Embedding 维度 | 4096 |

## 2. 测试配置

| 项目 | 值 |
|------|-----|
| 数据子集 | medical |
| 采样数量 | 5 |
| 工作目录 | /home/eyanpen/sourceCode/rag-test/test_fast_graphrag_workspace |

## 3. 连通性检查

| 服务 | 状态 |
|------|------|
| LLM API | ✅ 通过 |
| Embedding API | ✅ 通过 |

## 4. 推理结果

| 指标 | 值 |
|------|-----|
| 总问题数 | 5 |
| 成功 | 5 |
| 失败 | 0 |
| 成功率 | 100.0% |
| 总耗时 | 0s |
| 平均每题 | 0.0s |

## 5. LLM-as-Judge 评分

| 指标 | 值 |
|------|-----|
| 评分数量 | 5 |
| 平均分 | 9.2/10 |
| 最高分 | 10/10 |
| 最低分 | 8/10 |

| ID | 分数 | 理由 |
|-----|------|------|
| Medical-73586ddc | 9/10 | 生成答案的核心含义正确，都指出基底细胞癌是最常见的皮肤癌类型。但标准答案提供了更准确的医学术语'Basal cell carcinoma (BCC)'，而生成答 |
| Medical-a8bad1cf | 10/10 | 语义完全一致，关键信息无遗漏。生成答案准确描述了基底细胞癌起源于表皮下层的基底细胞，并正确指出了基底细胞层的位置。 |
| Medical-422500d5 | 8/10 | 生成答案核心含义正确，都指出BCC主要影响阳光暴露区域。关键信息'face'、'neck'基本一致，但generated answer额外包含了'scalp,  |
| Medical-6d2a190d | 9/10 | 生成答案的核心含义正确，都指出了紫外线/阳光暴露是基底细胞癌的主要风险因素。但标准答案更简洁准确地表述为'UV radiation exposure'，而生成答 |
| Medical-5ad931db | 10/10 | 语义完全一致，generated answer不仅包含了ground truth的核心信息'fair skin increases the risk of BC |

## 6. 结果样例

```json
[
  {
    "id": "Medical-73586ddc",
    "question": "What is the most common type of skin cancer?",
    "source": "Medical",
    "context": [
      ". This book is only about basal cell skin cancer. For more information on squamous cell skin cancer or melanoma, read the NCCN Guidelines for Patients, available at NCCN.org/patientguidelines and on t...",
      "About basal cell skin cancer What is basal cell skin cancer? How is basal cell skin cancer treated? What can you do to get the best care? Basal cell skin cancer, also known as basal cell carcinoma (BC..."
    ],
    "evidence": "Basal cell carcinoma (BCC) is the most common type of skin cancer.",
    "question_type": "Fact Retrieval",
    "generated_answer": "The most common type of skin cancer is basal cell skin cancer.",
    "ground_truth": "Basal cell carcinoma (BCC) is the most common type of skin cancer.",
    "judge_score": 9,
    "judge_reason": "生成答案的核心含义正确，都指出基底细胞癌是最常见的皮肤癌类型。但标准答案提供了更准确的医学术语'Basal cell carcinoma (BCC)'，而生成答案使用了相对简单的表述'basal cell skin cancer'，在专业性上略有不足。"
  },
  {
    "id": "Medical-a8bad1cf",
    "question": "From which cell type does basal cell carcinoma arise?",
    "source": "Medical",
    "context": [
      ". To find out, your doctor will use a needle biopsy to take a sample from the lymph node for testing. Squamous cell skin cancer that has not spread to nearby lymph nodes is referred to as “local” squa...",
      ". Types of surgery for squamous cell skin cancer include curettage and electrodesiccation, PDEMA/Mohs surgery, and standard surgical excision. Radiation therapy is a treatment option for people who ca..."
    ],
    "evidence": "Basal cell carcinoma arises from basal cells.; Basal cells are located in the lower part of the epidermis.",
    "question_type": "Fact Retrieval",
    "generated_answer": "Basal cell carcinoma arises from basal cells, which are found in the lower part of the epidermis, called the basal cell layer.",
    "ground_truth": "Basal cell carcinoma arises from basal cells in the lower part of the epidermis.",
    "judge_score": 10,
    "judge_reason": "语义完全一致，关键信息无遗漏。生成答案准确描述了基底细胞癌起源于表皮下层的基底细胞，并正确指出了基底细胞层的位置。"
  }
]
```

## 7. 问题类型分布

| 类型 | 数量 |
|------|------|
| Fact Retrieval | 5 |

## 8. 运行日志（末尾）

```
2026-04-09 11:07:51,325 INFO [TIMING] POST http://10.210.156.69:8633/chat/completions → 200 in 0.36s
2026-04-09 11:07:51,406 INFO HTTP Request: POST http://10.210.156.69:8633/embeddings "HTTP/1.1 200 OK"
2026-04-09 11:07:51,406 INFO [TIMING] POST http://10.210.156.69:8633/embeddings → 200 in 0.08s
2026-04-09 11:07:52,795 INFO HTTP Request: POST http://10.210.156.69:8633/chat/completions "HTTP/1.1 200 OK"
2026-04-09 11:07:52,796 INFO [TIMING] POST http://10.210.156.69:8633/chat/completions → 200 in 1.34s
Medical:  60%|██████    | 3/5 [00:04<00:03,  1.66s/it]2026-04-09 11:07:53,227 INFO HTTP Request: POST http://10.210.156.69:8633/chat/completions "HTTP/1.1 200 OK"
2026-04-09 11:07:53,227 INFO [TIMING] POST http://10.210.156.69:8633/chat/completions → 200 in 0.34s
2026-04-09 11:07:53,309 INFO HTTP Request: POST http://10.210.156.69:8633/embeddings "HTTP/1.1 200 OK"
2026-04-09 11:07:53,309 INFO [TIMING] POST http://10.210.156.69:8633/embeddings → 200 in 0.08s
2026-04-09 11:07:54,103 INFO HTTP Request: POST http://10.210.156.69:8633/chat/completions "HTTP/1.1 200 OK"
2026-04-09 11:07:54,103 INFO [TIMING] POST http://10.210.156.69:8633/chat/completions → 200 in 0.75s
Medical:  80%|████████  | 4/5 [00:05<00:01,  1.52s/it]2026-04-09 11:07:54,579 INFO HTTP Request: POST http://10.210.156.69:8633/chat/completions "HTTP/1.1 200 OK"
2026-04-09 11:07:54,579 INFO [TIMING] POST http://10.210.156.69:8633/chat/completions → 200 in 0.38s
2026-04-09 11:07:54,665 INFO HTTP Request: POST http://10.210.156.69:8633/embeddings "HTTP/1.1 200 OK"
2026-04-09 11:07:54,665 INFO [TIMING] POST http://10.210.156.69:8633/embeddings → 200 in 0.08s
2026-04-09 11:07:55,600 INFO HTTP Request: POST http://10.210.156.69:8633/chat/completions "HTTP/1.1 200 OK"
2026-04-09 11:07:55,600 INFO [TIMING] POST http://10.210.156.69:8633/chat/completions → 200 in 0.88s
Medical: 100%|██████████| 5/5 [00:07<00:00,  1.51s/it]Medical: 100%|██████████| 5/5 [00:07<00:00,  1.50s/it]
2026-04-09 11:07:55,616 INFO Saved 5 results to /home/eyanpen/sourceCode/rag-test/test_results/predictions.json
2026-04-09 11:07:55,617 INFO [STATS] Final: count=19, min=0.08s, max=47.41s, avg=9.73s
```
