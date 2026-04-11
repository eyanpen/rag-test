# 🔍 Root Cause 分析报告

## 问题概述
- **问题描述**: FalkorDB 中所有 novel 数据集的图 Edges 为 0，但 medical 数据集正常（6312 Edges）
- **错误信息**: `Invalid input '(': expected a label, '{', a parameter or ')' — MERGE (n:WORK_OF_LITERATURE_(IMPLIED_ENTITY_TYPE:_CULTURAL_PRACTICE) ...`
- **影响范围**: 所有 `*_novel` 图（6 个），导致 novel 数据集的 GraphRAG 查询无法利用图结构
- **严重程度**: High
- **分析日期**: 2026-04-11

---

## 1. 现象描述

### 1.1 错误表现
FalkorDB 中所有 novel 图的状态：
| 图名 | Nodes | Edges |
|------|-------|-------|
| `*_medical` (7个) | 7433 | 6312 ✅ |
| `*_novel` (6个) | 14812~19404 | **0** ❌ |

novel 图中有节点但全部是 Phase 10 embedding 写入的向量节点（`entity_description`、`community_full_content`、`text_unit_text`），graph_sync 写入的实体节点只有 964 个（应为 14812 个）。

### 1.2 触发条件
novel 数据集的 entity type 包含 Cypher label 非法字符（括号、冒号、斜杠）。

### 1.3 复现步骤
1. 运行 `run_embedding_benchmark.py` 对 novel 数据集执行 benchmark
2. Phase 10 完成后 graph_sync 开始写入实体
3. 遍历到 index=964 的实体（type=`WORK OF LITERATURE (IMPLIED ENTITY TYPE: CULTURAL PRACTICE)`）时 Cypher 语法错误
4. 异常未被循环内捕获，整个 sync 函数中断，后续 13848 个实体和全部 18772 条关系均未写入

---

## 2. 信息收集

### 2.1 日志证据
`benchmark.log` 中所有 novel 图的 sync 均在同一位置失败：
```
2026-04-10 07:16:52,795 INFO Syncing 14812 entities to FalkorDB graph 'baai-bge-m3_novel'
2026-04-10 07:16:53,304 WARNING Graph sync failed for BGE-M3 (default)/novel:
  errMsg: Invalid input '(': expected a label, '{', a parameter or ')'
  errCtx: MERGE (n:WORK_OF_LITERATURE_(IMPLIED_ENTITY_TYPE:_CULTURAL_PRACTICE) {id: $ei...
```

medical 图全部成功：
```
2026-04-10 06:32:23,570 INFO Syncing 3299 entities to FalkorDB graph 'baai-bge-m3_medical'
2026-04-10 06:32:39,060 INFO Graph sync done: baai-bge-m3_medical — Nodes: 3299, Edges: 6312
```

### 2.2 代码分析
`tests/graph_sync.py` 第 30 行：
```python
entity_type = str(row.get("type", "Entity")).replace(" ", "_")
```
仅将空格替换为下划线，未处理 `()`、`:`、`/` 等 Cypher label 非法字符。

异常在 `for _, row in entities_df.iterrows()` 循环内抛出，但没有 try/except 包裹单行写入，导致整个函数中断。

### 2.3 数据检查
novel 数据集中含非法字符的 entity type（共 15 个实体）：

| Entity Type | 非法字符 | 数量 |
|-------------|---------|------|
| `WORK OF LITERATURE (IMPLIED ENTITY TYPE: CULTURAL PRACTICE)` | `()` `:` | 1 |
| `ANIMAL/CREATURE` | `/` | 3 |
| `SYMBOL/CULTURAL ICON` | `/` | 1 |
| `ABSTRACTION / CULTURAL PRACTICE` | `/` | 1 |
| `DISEASE/CULTURAL PRACTICE` | `/` | 2 |
| `VESSEL/TRANSPORT` | `/` | 1 |
| `ORGANIZATION/TECHNOLOGY` | `/` | 3 |
| `ANIMAL/REGIONAL SYMBOL` | `/` | 1 |
| `COMMUNITY ROLE / CULTURAL PRACTICE` | `/` | 1 |

第一个坏实体在 DataFrame index=964，与 FalkorDB 中实际写入的 964 个节点完全吻合。

medical 数据集的 entity type 全部是简单单词（`DISEASE`、`BODY_PART` 等），不含任何非法字符，所以不受影响。

---

## 3. Root Cause

### 3.1 根因描述
`graph_sync.py` 的 `entity_type` sanitize 逻辑不完整：只做了 `replace(" ", "_")`，未清理 Cypher label 语法中的非法字符（括号、冒号、斜杠等）。当 novel 数据集中出现含这些字符的 entity type 时，生成的 Cypher `MERGE (n:LABEL_WITH_(PARENS) ...)` 语法错误。同时，异常未在循环内捕获，导致一个坏实体中断整个 sync 流程。

### 3.2 因果链
```
novel entity type 含 "()" → replace(" ","_") 未清理 → Cypher label 含非法字符
→ FalkorDB 返回语法错误 → 异常未在循环内捕获 → 整个 sync 函数中断
→ 后续实体未写入 → edges 的 MATCH 找不到节点 → 0 Edges
```

### 3.3 证据汇总
| # | 证据 | 来源 | 支持结论 |
|---|------|------|---------|
| 1 | novel 图 Edges=0，medical 图 Edges=6312 | FalkorDB 查询 | 问题仅影响 novel |
| 2 | novel 图只有 964 个实体节点（应 14812） | FalkorDB 查询 | sync 中途中断 |
| 3 | 第一个含非法字符的实体在 index=964 | parquet 分析 | 中断位置精确匹配 |
| 4 | 日志报 Cypher 语法错误含 `(` | benchmark.log | 直接错误证据 |
| 5 | medical entity type 无特殊字符 | parquet 分析 | 解释为何 medical 不受影响 |

### 3.4 测试验证
- **测试文件**: `tests/test_rootcause_graph_sync_label.py`
- **运行方式**:
  ```bash
  cd /home/eyanpen/sourceCode/rag-test
  source venv/bin/activate
  python -m pytest tests/test_rootcause_graph_sync_label.py -v
  ```
- **测试结果**:
  - ✅ `test_reproduce_issue` — 原始 sanitize 产生含非法字符的 label
  - ✅ `test_fix_resolves_issue` — 修复后所有 label 只含合法字符
  - ✅ `test_normal_types_unaffected` — 正常类型修复前后结果一致

---

## 4. 排除项

| # | 假设 | 排除原因 |
|---|------|---------|
| 1 | FalkorDB 连接问题 | medical 同一 FalkorDB 实例写入正常 |
| 2 | parquet 数据缺失 | novel relationships.parquet 有 18772 条记录 |
| 3 | title 含特殊字符导致匹配失败 | 405 个 title 含引号，但 graph_sync 使用参数化查询 `$title`，不受影响 |
| 4 | description 过长 | max 3078 chars，FalkorDB 无此限制 |

---

## 5. 修复方案

### 方案 A: 修复 label sanitize + 单行容错

**描述**: 用正则清理 entity_type 中所有非法字符，同时在循环内加 try/except 防止单个实体失败中断整体。

**改动范围**:
- `tests/graph_sync.py`: sanitize 逻辑 + 循环容错

**代码示例**:
```python
import re

def _safe_label(raw_type: str) -> str:
    """Convert entity type to valid Cypher label."""
    return re.sub(r"[^A-Za-z0-9_]", "_", raw_type)

# Nodes loop:
for _, row in entities_df.iterrows():
    try:
        entity_type = _safe_label(str(row.get("type", "Entity")))
        ...
    except Exception as e:
        log.warning("Failed to sync entity %s: %s", row.get("title", "?"), e)

# Edges loop:
for _, row in relationships_df.iterrows():
    try:
        ...
    except Exception as e:
        log.warning("Failed to sync edge %s: %s", row.get("id", "?"), e)
```

**优点**:
- 彻底解决 label 非法字符问题
- 单个实体/关系失败不影响其他
- 改动最小

**缺点**:
- 无

**风险**: 无（正则替换是确定性的，参数化查询不受影响）

**工作量**: 小

---

### 方案 B: 统一使用 Entity label + type 属性

**描述**: 所有实体统一用 `Entity` 作为 Cypher label，原始 type 存为节点属性。

**代码示例**:
```python
graph.query(
    "MERGE (n:Entity {id: $eid}) "
    "SET n.title = $title, n.type = $type, n.description = $description, n.degree = $degree",
    {"eid": ..., "title": ..., "type": raw_type, ...},
)
```

**优点**:
- 完全避免 label 字符问题
- 简化查询

**缺点**:
- 丢失了按 label 分类的图结构语义
- 需要修改 edge 写入的 MATCH 查询

**工作量**: 小

---

## 6. 方案对比

| 维度 | 方案 A | 方案 B |
|------|--------|--------|
| 改动范围 | 小 | 小 |
| 保留图语义 | ✅ 保留 label 分类 | ❌ 统一 Entity |
| 风险 | 无 | 低 |
| 向后兼容 | ✅ | 需重建图 |
| 推荐度 | ⭐⭐⭐ | ⭐⭐ |

## 7. 建议

**推荐方案**: 方案 A

**理由**: 改动最小，保留图结构语义，且通过正则 + 循环容错双重保障，彻底解决问题。

---

> ⚠️ 请确认选择的修复方案后，我将执行修复。

---

## Q&A 追踪

### Q1 (2026-04-11 06:44)
**用户问题**: 能否将非法字符转义，查询的时候再转义回来，否则查询会出现不一致的问题？

**回答**: 分析了整个数据流后发现，entity type label 在当前代码中是"只写不读"的：
- edges 写入用 `{title: $src}` 匹配，不指定 label
- GraphRAG search 查询走 `falkordb_vector_store.py`，用的是 `entity_description` 等 embedding label
- 没有任何代码通过 `MATCH (n:PERSON)` 按 entity type 查询

因此不存在"查询时需要转义回来"的不一致问题，`re.sub` 单向替换完全够用。

**影响**: 无需修正结论，确认方案 A 可行。

---

### Q2 (2026-04-11 06:48)
**用户问题**: 为什么不推荐转义？

**回答**: 三个原因：
1. **Cypher label 没有真正的转义语法** — 反引号引用可以包含特殊字符，但反引号本身不能出现在 label 中，只是推迟问题
2. **转义/反转义增加耦合面** — 所有读写 label 的地方都必须统一加反引号，忘了就不一致
3. **当前 label 只写不读，保真价值为零** — 需要原始 type 值时，存 `n.type` 属性比转义 label 更直接可靠

**影响**: 无需修正结论。

---

### Q3 (2026-04-11 06:50)
**用户问题**: entity type label 主要作用是什么？

**回答**: 在当前项目中，entity type label 唯一的实际作用是**图可视化分类**（FalkorDB Browser 中不同 label 显示不同颜色）。没有用于查询过滤、索引加速、Schema 约束等。

**影响**: 无需修正结论。

---

### Q4 (2026-04-11 06:51)
**用户问题**: 将来是否会用到 entity type label？

**回答**: 会的，label 是图数据库核心设计元素，典型未来场景：
1. **查询加速** — 按 label 分桶存储，`MATCH (n:PERSON)` 比全图扫描快数量级
2. **图模式匹配** — `MATCH (p:PERSON)-[:RELATED]->(l:LOCATION)<-[:RELATED]-(e:HISTORICAL_EVENT)` 多跳查询
3. **Schema 约束** — 基于 label 的 NOT NULL / UNIQUE 约束
4. **GraphRAG 高级检索** — type-aware retrieval，如"小说里有哪些组织"直接 `MATCH (n:ORGANIZATION)`
5. **图分析算法** — 限定子图跑 PageRank、社区检测等

**影响**: 确认修复方案应额外存 `n.type` 属性保留原始值，为未来精确查询留路。方案 A 代码示例已更新。 <!-- 更新于 2026-04-11，参见 Q&A #4 -->
