# Answer Correctness 评测机制深度解读

## 1. 核心问题：为什么需要 Embedding？

`answer_correctness` 的评测**不是**简单地"用 LLM 比较两个答案"，而是采用了一种**双通道加权融合**的评分策略：

```
最终分数 = 0.75 × factuality_score(LLM分类) + 0.25 × similarity_score(Embedding余弦相似度)
```

这两个通道各自解决不同的问题：

| 通道 | 工具 | 解决的问题 | 权重 |
|------|------|-----------|------|
| Factuality（事实性） | LLM | 答案中的每个事实陈述是否正确？是否遗漏了关键信息？ | 75% |
| Semantic Similarity（语义相似度） | Embedding | 答案和标准答案在整体语义空间中有多接近？ | 25% |

### 为什么不能只用 LLM？

LLM 分类通道（factuality）是**离散的**——它把陈述分为 TP/FP/FN 三类，然后算 F-beta 分数。这种方式有两个局限：

1. **粒度粗糙**：一个陈述要么对要么错，无法表达"部分正确"或"表述不同但意思接近"
2. **依赖 LLM 输出质量**：如果 LLM 的 JSON 解析失败（`json.JSONDecodeError`），factuality 直接返回 0.0

Embedding 通道作为**连续的语义补偿**：
- 即使 LLM 分类出错，embedding 余弦相似度仍能给出一个合理的基线分数
- 它捕捉的是"整体语义距离"，对措辞不同但含义相近的答案更宽容

## 2. 完整评测流程（四步）

### Step 1: 陈述拆分（Statement Generation）

对**生成答案**和**标准答案**分别调用 LLM，将完整文本拆分为独立的事实陈述。

Prompt 模板：
```
Given a question and an answer, analyze the complexity of each sentence 
in the answer. Break down each sentence into one or more fully 
understandable statements. Ensure that no pronouns are used in any statement.
```

示例输入：
```
Question: Who was Albert Einstein and what is he best known for?
Answer: He was a German-born theoretical physicist, widely acknowledged 
to be one of the greatest and most influential physicists of all time.
```

示例输出：
```json
[
  "Albert Einstein was a German-born theoretical physicist.",
  "Albert Einstein is recognized as one of the greatest and most influential physicists of all time."
]
```

关键点：**消除代词**，让每个陈述独立可理解。

### Step 2: 事实性分类（Factuality Classification）

用 LLM 将答案陈述和标准答案陈述进行交叉比对，分为三类：

| 类别 | 含义 | 示例 |
|------|------|------|
| TP (True Positive) | 答案中的陈述被标准答案支持 | 答案说"水的沸点是100°C"，标准答案也说了 |
| FP (False Positive) | 答案中的陈述在标准答案中找不到支持 | 答案说"太阳靠核裂变驱动"，但标准答案说是核聚变 |
| FN (False Negative) | 标准答案中有但答案遗漏的陈述 | 标准答案提到"沸点随海拔变化"，但答案没提 |

然后计算 F-beta 分数：
```python
precision = TP / (TP + FP)    # 答案中正确陈述的比例
recall    = TP / (TP + FN)    # 标准答案被覆盖的比例
F_beta    = (1 + β²) × (precision × recall) / (β² × precision + recall)
```

默认 β=1.0，即 precision 和 recall 等权。

### Step 3: 语义相似度（Semantic Similarity）

用 Embedding 模型分别对**完整的生成答案**和**完整的标准答案**做向量化，然后计算余弦相似度：

```python
cosine_sim = dot(a_embed, gt_embed) / (norm(a_embed) * norm(gt_embed))
similarity_score = (cosine_sim + 1) / 2   # 映射到 [0, 1]
```

注意：这里是对**整段文本**做 embedding，不是对单个陈述。

### Step 4: 加权融合

```python
final_score = 0.75 × factuality_score + 0.25 × similarity_score
```

## 3. 具体举例

### 例子：太阳的能量来源

**问题**: What powers the sun and what is its primary function?

**标准答案** (Ground Truth):
> The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium. This fusion process in the sun's core releases a tremendous amount of energy. The energy from the sun provides heat and light, which are essential for life on Earth.

**生成答案** (Generated Answer):
> The sun is powered by nuclear fission, similar to nuclear reactors on Earth. The primary function of the sun is to provide light to the solar system.

#### Step 1 拆分后：

答案陈述：
1. "The sun is powered by nuclear fission, similar to nuclear reactors on Earth."
2. "The primary function of the sun is to provide light to the solar system."

标准答案陈述：
1. "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium."
2. "This fusion process in the sun's core releases a tremendous amount of energy."
3. "The energy from the sun provides heat and light, which are essential for life on Earth."
4. "The sun's light plays a critical role in Earth's climate system."
5. "Sunlight helps to drive the weather and ocean currents."

#### Step 2 分类结果：

- **TP = 1**: "The primary function of the sun is to provide light to the solar system." — 被标准答案部分支持
- **FP = 1**: "The sun is powered by nuclear fission..." — 错误，标准答案说的是 fusion
- **FN = 4**: 标准答案中关于核聚变、能量释放、气候系统、天气洋流的陈述都被遗漏

```
precision = 1/(1+1) = 0.5
recall    = 1/(1+4) = 0.2
F1        = 2 × 0.5 × 0.2 / (0.5 + 0.2) = 0.286
```

**factuality_score ≈ 0.286**

#### Step 3 语义相似度：

虽然答案说错了核裂变/核聚变，但两段文本都在讨论"太阳的能量和功能"这个主题，embedding 向量会有一定的相似度。假设余弦相似度 = 0.72：

```
similarity_score = (0.72 + 1) / 2 = 0.86
```

#### Step 4 最终分数：

```
final = 0.75 × 0.286 + 0.25 × 0.86 = 0.215 + 0.215 = 0.430
```

如果**没有 embedding 通道**，分数只有 0.286。embedding 的 25% 权重在这里起到了"语义兜底"的作用——虽然事实有误，但答案至少在讨论正确的主题。

### 对比：一个高分的例子

**问题**: What is the boiling point of water?

**标准答案**: The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level. The boiling point of water can change with altitude.

**生成答案**: The boiling point of water is 100 degrees Celsius at sea level.

分类结果：
- TP = 1（100°C at sea level）
- FP = 0
- FN = 1（海拔影响沸点）

```
factuality = F1(1, 0, 1) = 2×(1.0×0.5)/(1.0+0.5) = 0.667
similarity ≈ 0.95（几乎在说同一件事）
final = 0.75 × 0.667 + 0.25 × 0.95 = 0.500 + 0.238 = 0.738
```

## 4. tests/ 代码中的调用链

在 `tests/run_embedding_benchmark.py` 的 `evaluate_predictions()` 函数中：

```python
# 初始化 LLM 和 Embedding 客户端
eval_llm = ChatOpenAI(model=config.llm_model, base_url=config.api_base_url, ...)
eval_emb = OpenAIEmbeddings(model="BAAI/bge-m3", base_url=config.api_base_url, ...)

# 对每个预测调用 answer_correctness
score = await compute_answer_correctness(
    p.question, p.generated_answer, p.ground_truth, 
    eval_llm,   # 用于陈述拆分 + 事实性分类
    eval_emb    # 用于语义相似度计算
)
```

Embedding 模型固定使用 `BAAI/bge-m3`（1024维），而不是被测试的 embedding 模型。这是因为评测指标本身需要一个**稳定的基准 embedding**，不能用被测对象来评测自己。

## 5. 设计哲学总结

| 设计决策 | 原因 |
|---------|------|
| 双通道而非单通道 | LLM 分类是离散的，embedding 提供连续的语义补偿 |
| 75/25 权重分配 | 事实正确性是主要指标，语义相似度是辅助兜底 |
| 评测 embedding 固定用 bge-m3 | 评测基准必须稳定，不能用被测模型评测自己 |
| 陈述拆分消除代词 | 让每个陈述独立可判断，避免上下文依赖导致误分类 |
| JSON 解析多级 fallback | LLM 输出格式不稳定，需要 json → json5 → json_repair → LLM self-healing 多级容错 |
