# FalkorDB 图数据结构解读：Nodes、Edges 与 GraphRAG 数据流

本文档基于 `tests/graph_sync.py`、`tests/falkordb_vector_store.py` 和 GraphRAG `extract_graph` 模块的代码分析，解读 FalkorDB 中图数据的构成、来源和写入机制。

---

## 1. Nodes 的构成（以 medical 数据集为例）

FalkorDB 中一个图（如 `medical_baai-bge-m3`）的节点总数 **7,433**，来自**两个独立的写入者**：

### 1.1 Phase 10 Embedding 写入（`falkordb_vector_store.py`）

GraphRAG Phase 10（`generate_text_embeddings`）对三类文本生成向量嵌入，通过 `FalkorDBVectorStore._insert_one()` 写入 FalkorDB：

| Label | 来源 | 数量 | 说明 |
|-------|------|------|------|
| `entity_description` | `entities.parquet` | 3,299 | 实体 title + description 的向量 |
| `community_full_content` | `community_reports.parquet` | 636 | 社区报告全文的向量 |
| `text_unit_text` | `text_units.parquet` | 199 | 文本单元原文的向量 |
| **小计** | | **4,134** | |

这些节点的核心属性是 `vector`（JSON 序列化的浮点数组），用于向量相似度搜索。

写入代码（`falkordb_vector_store.py`）：

```python
label = self.index_name  # 如 "entity_description"
query = f"CREATE (n:{label} {{{props_str}}})"
self.graph.query(query, props)  # props 包含 id, vector, 及其他 data 字段
```

### 1.2 Graph Sync 写入（`graph_sync.py`）

`graph_sync.py` 的 `sync_graph_to_falkordb()` 在 Phase 10 完成后被调用，将 `entities.parquet` 中的实体作为**图结构节点**写入，label 为实体类型（如 `DISEASE`、`ORGAN`）：

| Label | 数量 | Label | 数量 |
|-------|------|-------|------|
| DISEASE | 531 | HEALTHCARE_PROFESSIONAL | 134 |
| BODY_PART | 251 | TREATMENT | 130 |
| MEDICAL_PROCEDURE | 235 | PATHOLOGY | 129 |
| DRUG | 222 | TISSUE | 110 |
| SYMPTOM | 218 | CELL_TYPE | 110 |
| MEDICAL_TEST | 218 | 其他 (~45 种) | 961 |
| **小计** | | | **3,299** |

写入代码（`graph_sync.py`）：

```python
entity_type = _safe_label(raw_type)  # 正则清理非法字符
graph.query(
    f"MERGE (n:{entity_type} {{id: $eid}}) "
    f"SET n.title = $title, n.type = $type, n.description = $description, n.degree = $degree",
    params,
)
```

### 1.3 两套节点的关系

同一个实体（如 id=xxx 的 "BASAL CELL SKIN CANCER"）在 FalkorDB 中存在**两个独立节点**：

```
(:entity_description {id: "xxx", vector: "[0.12, -0.34, ...]"})   ← 向量搜索用
(:DISEASE {id: "xxx", title: "BASAL CELL SKIN CANCER", type: "DISEASE"})  ← 图遍历用
```

两者互不关联，服务于不同的查询路径。

---

## 2. Edges 的构成

### 2.1 为什么只有 RELATED 一种类型

GraphRAG 的 LLM 提取 prompt 中，关系的输出格式为：

```
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
```

**没有 `relationship_type` 字段**。LLM 只输出自然语言 description 和数值 strength，不输出分类标签。

对应的解析代码（`graph_extractor.py`）：

```python
relationships.append({
    "source": source,
    "target": target,
    "description": edge_description,  # 自然语言描述
    "weight": weight,                 # 关系强度
    # 没有 type 字段
})
```

因此 `relationships.parquet` 中没有关系类型列，`graph_sync.py` 写入时统一使用 `RELATED`：

```cypher
MERGE (a)-[r:RELATED {id: $rid}]->(b)
```

### 2.2 GraphRAG 为什么这样设计

1. **关系语义靠 description 承载** — `"Basal cell skin cancer develops in the skin organ"` 比 `DEVELOPS_IN` 信息量大得多
2. **GraphRAG 的检索不走图遍历** — 走向量相似度搜索 + 社区报告，不需要 `MATCH (a)-[:TREATS]->(b)` 这种按类型查询
3. **LLM 提取的关系类型不稳定** — 同一种关系可能被标注为 `TREATS`、`IS_TREATMENT_FOR`、`USED_TO_TREAT`，强行分类引入噪声

---

## 3. 关系的合并与方向

### 3.1 合并流程

GraphRAG 对从多个 text chunk 提取的关系做两轮处理：

**第一轮：`_merge_relationships()`**（extract_graph 阶段）

```python
all_relationships.groupby(["source", "target"], sort=False).agg(
    description=("description", list),    # description 聚合成列表
    text_unit_ids=("source_id", list),
    weight=("weight", "sum"),             # weight 求和
)
```

按 `(source, target)` 精确匹配合并。**不做方向归一化**，即 `(A→B)` 和 `(B→A)` 是不同的 key。

**第二轮：`finalize_relationships()`**（finalize 阶段）

按 `(source, target)` 去重，补充 `combined_degree`、`id` 等字段。同样不做方向归一化。

### 3.2 双向边的实际情况

以 medical 数据集为例，6,312 条边中有 **169 对是双向的**：

```
PET SCAN → TRACER
  desc: "A PET SCAN is a diagnostic imaging procedure that utilizes a radioactive TRACER"

TRACER → PET SCAN
  desc: "Tracers are essential components of PET scans, enabling visualization of cellular activity"
```

两个方向各自有独立的 description，视角不同但语义相关。

### 3.3 写入 FalkorDB 后的表现

双向边在 FalkorDB 中保留为 **2 条独立的有向边**，因为 MERGE 的去重依据是 `{id: $rid}`，每条边有唯一 id。

---

## 4. MERGE 语句详解

```cypher
MATCH (a {title: $src}), (b {title: $tgt})
MERGE (a)-[r:RELATED {id: $rid}]->(b)
SET r.description = $desc, r.weight = $weight
```

### 4.1 逐部分拆解

| 部分 | 含义 |
|------|------|
| `MATCH (a {title: $src}), (b {title: $tgt})` | 找到 title 匹配的源节点和目标节点 |
| `MERGE` | "有则匹配，无则创建" |
| `(a)-[...]->(b)` | 从 a 到 b 的有向边 |
| `r:RELATED` | 边的类型为 RELATED |
| `{id: $rid}` | MERGE 的匹配条件：id 必须等于参数值 |
| `SET r.description = ...` | 设置/更新边的属性 |

### 4.2 MERGE 的判断逻辑

MERGE 检查整个 pattern 是否已存在：

> 从节点 a 到节点 b，是否已存在一条类型为 RELATED 且 id = $rid 的有向边？

- **存在** → 匹配到这条边，执行 SET 更新属性（幂等）
- **不存在** → 创建新边

### 4.3 举例

假设 parquet 中有 3 条记录：

```
id=aaa  A → B  "A causes B"
id=bbb  B → A  "B is caused by A"
id=aaa  A → B  "A causes B"        ← 重复
```

| 执行顺序 | MERGE 检查 | 结果 |
|---------|-----------|------|
| 第 1 次 (id=aaa, A→B) | A→B 有 RELATED{id:aaa}？**无** | 创建新边 |
| 第 2 次 (id=bbb, B→A) | B→A 有 RELATED{id:bbb}？**无** | 创建新边（反方向） |
| 第 3 次 (id=aaa, A→B) | A→B 有 RELATED{id:aaa}？**有** | 匹配已有边，SET 更新 |

最终：**2 条边**（aaa 正向 + bbb 反向），第 3 次重复被 MERGE 去重。

### 4.4 `{id: $rid}` 的作用

如果去掉 `{id: $rid}`：

```cypher
MERGE (a)-[r:RELATED]->(b)
```

MERGE 只检查"a→b 之间有没有任何 RELATED 边"，同方向的多条边会被合并为一条，不同 description 会被覆盖。加了 `{id: $rid}` 才能让每条关系保持独立。

---

## 5. Entity Type Label 的作用与设计

### 5.1 当前作用

在当前项目中，entity type label（如 `DISEASE`、`ORGAN`）的唯一实际作用是**图可视化分类**——在 FalkorDB Browser 中不同 label 显示不同颜色。

当前没有用于：查询过滤、索引加速、Schema 约束。

### 5.2 未来价值

| 场景 | 示例 |
|------|------|
| 查询加速 | `MATCH (n:PERSON)` 只扫描 PERSON 桶，比全图扫描快数量级 |
| 图模式匹配 | `MATCH (p:PERSON)-[:RELATED]->(l:LOCATION)<-[:RELATED]-(e:HISTORICAL_EVENT)` |
| Schema 约束 | `CREATE CONSTRAINT ON (n:PERSON) ASSERT n.title IS NOT NULL` |
| Type-aware 检索 | 用户问"有哪些组织"→ `MATCH (n:ORGANIZATION) RETURN n.title` |
| 图分析算法 | 限定子图跑 PageRank、社区检测 |

### 5.3 Label 清理策略

由于 Cypher label 只允许字母、数字、下划线，`graph_sync.py` 使用正则清理：

```python
def _safe_label(raw_type: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", raw_type)
```

同时将原始 type 存为节点属性 `n.type`，保留完整信息：

```cypher
MERGE (n:WORK_OF_LITERATURE__IMPLIED_ENTITY_TYPE__CULTURAL_PRACTICE_ {id: $eid})
SET n.type = 'WORK OF LITERATURE (IMPLIED ENTITY TYPE: CULTURAL PRACTICE)'
```

Label 用于可视化和快速过滤，`n.type` 用于精确查询。
