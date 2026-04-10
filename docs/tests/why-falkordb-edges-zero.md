# 为什么 FalkorDB 中 Edges(0) 没有数据？

## 结论

FalkorDB 在这个项目中**只被用作向量存储（Vector Store）**，不是用来存图结构的。GraphRAG 的图数据（entities + relationships）存在 Parquet 文件里，不在 FalkorDB 中。

## 详细分析

### 1. FalkorDB 里实际存了什么？

看 `falkordb_vector_store.py` 的 `_insert_one()` 方法：

```python
query = f"CREATE (n:{label} {{{props_str}}})"
```

它只执行 `CREATE (n:label {...})`——**只创建节点（Node），从不创建边（Edge/Relationship）**。

整个 `FalkorDBVectorStore` 类中没有任何 `CREATE ()-[]->()`、`MERGE ... -[]->`、或任何涉及边的 Cypher 语句。

FalkorDB 中的节点按 `index_name` 作为 label 分类，存储的是三类 embedding 向量：

| Label（index_name） | 来源表 | 内容 |
|---------------------|--------|------|
| `entity_description_embedding` | entities.parquet | 实体的 title + description 的向量 |
| `text_unit_text_embedding` | text_units.parquet | 文本块的向量 |
| `community_full_content_embedding` | community_reports.parquet | 社区报告全文的向量 |

所以 FalkorDB 里只有 Nodes，没有 Edges，是正常的。

### 2. 图结构数据存在哪里？

GraphRAG pipeline 的图数据存储在 `workspace/<dataset>/output/` 下的 Parquet 文件中：

```
output/
├── entities.parquet          ← 图的节点（实体）
├── relationships.parquet     ← 图的边（关系）
├── communities.parquet       ← 社区划分结果
├── community_reports.parquet ← 社区报告
└── text_units.parquet        ← 文本块
```

`extract_graph` workflow 提取实体和关系后，直接写入 Parquet：

```python
# graphrag/index/workflows/extract_graph.py
await context.output_table_provider.write_dataframe("entities", entities)
await context.output_table_provider.write_dataframe("relationships", relationships)
```

`create_communities` workflow 用 `relationships.parquet` 做 Leiden 社区检测，结果也写回 Parquet。

**整个 pipeline 中，图的拓扑结构（谁连接谁）始终在 Parquet/DataFrame 层面处理，从未写入 FalkorDB。**

### 3. 为什么不把图结构也存入 FalkorDB？

GraphRAG 的 Vector Store 接口（`graphrag_vectors.VectorStore`）设计上就是一个纯向量检索接口，只有：
- `load_documents()` / `insert()` — 写入带向量的文档
- `similarity_search_by_vector()` — 向量相似度搜索
- `search_by_id()` — 按 ID 查找

它没有"创建关系"、"图遍历"等图操作的抽象。GraphRAG 把 FalkorDB 当成了一个和 LanceDB 等价的向量数据库来用，没有利用 FalkorDB 的图能力。

### 4. 查询时图结构怎么用的？

`run_queries()` 中的 local search 流程：

```python
# 从 Parquet 读取图结构
entities_df = pd.read_parquet("output/entities.parquet")
relationships_df = pd.read_parquet("output/relationships.parquet")
communities_df = pd.read_parquet("output/communities.parquet")
community_reports_df = pd.read_parquet("output/community_reports.parquet")

# 从 FalkorDB 只读取向量（用于实体描述的相似度搜索）
description_store = FalkorDBVectorStore(...)
description_store.connect()

# 搜索引擎组合两者
search_engine = get_local_search_engine(
    entities=entities,                          # ← 来自 Parquet
    relationships=relationships,                # ← 来自 Parquet
    description_embedding_store=description_store,  # ← 来自 FalkorDB
)
```

查询时先用 FalkorDB 做向量相似度搜索找到相关实体，再用 Parquet 中的 relationships 做图上下文扩展。

### 5. 总结

```
┌─────────────────────────────────────────────────┐
│              GraphRAG 数据存储架构                │
├─────────────────────┬───────────────────────────┤
│   Parquet 文件       │   FalkorDB                │
├─────────────────────┼───────────────────────────┤
│ entities (节点)      │ entity vectors (向量)      │
│ relationships (边)   │ text_unit vectors (向量)   │
│ communities (社区)   │ community vectors (向量)   │
│ community_reports   │                           │
│ text_units          │ Nodes: ✅  Edges: ❌       │
├─────────────────────┼───────────────────────────┤
│ 存储：图拓扑 + 属性   │ 存储：纯向量嵌入           │
│ 用途：图遍历、社区检测 │ 用途：相似度搜索           │
└─────────────────────┴───────────────────────────┘
```

FalkorDB 的 Edges(0) 是完全正常的——它在这里不是图数据库，而是向量数据库。

---

## 已实现：将图结构同步到 FalkorDB

为了让 FalkorDB 成为真正的图数据库（同时具备向量检索和图遍历能力），新增了 `tests/graph_sync.py` 模块，在 Phase 10 完成后自动将 Parquet 中的图结构写入 FalkorDB。

### 实现方式

`sync_graph_to_falkordb()` 函数读取 `entities.parquet` 和 `relationships.parquet`，通过 Cypher 写入 FalkorDB：

```python
# 写入节点 — entity type 作为 label（如 ORGANIZATION、PERSON）
MERGE (n:ORGANIZATION {id: $eid})
SET n.title = $title, n.description = $description, n.degree = $degree

# 写入边 — 通过 title 匹配源/目标节点
MATCH (a {title: $src}), (b {title: $tgt})
MERGE (a)-[r:RELATED {id: $rid}]->(b)
SET r.description = $desc, r.weight = $weight
```

### 调用时机

在 `run_embedding_benchmark.py` 的 Step 3 中，每个 model × dataset 组合的 Phase 10 完成后立即调用：

```
Phase 10 (embedding) → graph_sync → Query → Evaluate
```

每个 embedding 模型有独立的 `graph_name`（如 `baai-bge-m3_medical`），图结构写入对应的图中。

### 同步后的 FalkorDB 数据结构

以 kevin_scott 数据集为例：

```
同步前: Nodes(N)    Edges(0)     ← 只有向量节点
同步后: Nodes(N+3353) Edges(5776)  ← 向量节点 + 实体节点 + 关系边
```

节点按 entity type 分 label：

| Label | 示例 |
|-------|------|
| `ORGANIZATION` | Microsoft, OpenAI |
| `PERSON` | Kevin Scott, Sam Altman |
| `EVENT` | Build Conference |
| `GEO` | Silicon Valley |

所有关系统一使用 `RELATED` 类型，携带 `description` 和 `weight` 属性。

### 同步后可执行的图查询示例

```cypher
-- 查看 Kevin Scott 的直接关联
MATCH (k:PERSON {title: 'KEVIN SCOTT'})-[r:RELATED]->(t)
RETURN t.title, r.weight, r.description
ORDER BY r.weight DESC LIMIT 10

-- 两跳路径发现
MATCH p = (a {title: 'KEVIN SCOTT'})-[:RELATED*1..2]->(b)
RETURN p LIMIT 20

-- 结合向量搜索 + 图遍历（先向量找相关实体，再图上扩展邻居）
MATCH (n:entity_description_embedding)
WHERE n.id = $entity_id
WITH n
MATCH (e {title: n.data_title})-[:RELATED]->(neighbor)
RETURN neighbor.title, neighbor.description
```

### 设计选择

| 决策 | 原因 |
|------|------|
| 用 `MERGE` 而非 `CREATE` | 重跑幂等，不产生重复数据 |
| 放在 Phase 10 之后 | 每个 embedding 模型有独立 graph_name，需写入对应图 |
| sync 失败只 warning | 不影响 benchmark 主流程 |
| description 截断 5000 字符 | 避免超长文本导致 Cypher 执行问题 |
| 图结构与向量节点共存同一 graph | 可在同一图内组合向量搜索和图遍历 |

### 同步后的完整架构

```
┌──────────────────────────────────────────────────────────┐
│                FalkorDB (同步后)                          │
├──────────────────────────────────────────────────────────┤
│  向量节点 (原有)                                          │
│  ├── entity_description_embedding  ← embedding 向量      │
│  ├── text_unit_text_embedding      ← embedding 向量      │
│  └── community_full_content_embedding ← embedding 向量   │
│                                                          │
│  图结构节点 (新增)                                        │
│  ├── ORGANIZATION  ← entities.parquet                    │
│  ├── PERSON        ← entities.parquet                    │
│  ├── EVENT         ← entities.parquet                    │
│  └── ...                                                 │
│                                                          │
│  关系边 (新增)                                            │
│  └── -[:RELATED]-> ← relationships.parquet               │
│                                                          │
│  Nodes: ✅  Edges: ✅  Vectors: ✅                        │
└──────────────────────────────────────────────────────────┘
```
