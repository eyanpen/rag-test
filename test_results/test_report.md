# fast-graphrag 测试报告

> 生成时间: 2026-04-09 11:07:57

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
| 总耗时 | 2487s |
| 平均每题 | 497.4s |

## 5. 结果样例

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
    "ground_truth": "Basal cell carcinoma (BCC) is the most common type of skin cancer."
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
    "ground_truth": "Basal cell carcinoma arises from basal cells in the lower part of the epidermis."
  }
]
```

## 6. 问题类型分布

| 类型 | 数量 |
|------|------|
| Fact Retrieval | 5 |

## 7. 运行日志（末尾）

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
