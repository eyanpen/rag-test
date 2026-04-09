# 🔍 Root Cause 分析报告

## 问题概述
- **问题描述**: fast-graphrag 索引构建过程中大量 HTTP 500 Internal Server Error
- **错误信息**: `HTTP/1.1 500 Internal Server Error` + `ReadTimeout(TimeoutError())`
- **影响范围**: fast-graphrag 索引构建完全失败，无法完成知识图谱构建
- **严重程度**: High
- **分析日期**: 2026-04-09

---

## 1. 现象描述

### 1.1 错误表现
fast-graphrag 在索引构建阶段向 `http://10.210.156.69:8633/chat/completions` 发送大量并发 LLM 请求，在约 50 秒后收到大量 500 Internal Server Error，并触发疯狂重试（同一秒内 9+ 次重试），形成雪崩效应。

### 1.2 触发条件
fast-graphrag 的 `grag.insert(context)` 调用，会将语料分块后并发发送实体提取请求给 LLM。

### 1.3 复现步骤
1. 启动 ericai proxy（`start_proxy_0000.py`，端口 8633）
2. 运行 fast-graphrag 索引构建（`run_fast-graphrag.py --subset medical`）
3. 约 50 秒后开始出现大量 500 错误

---

## 2. 信息收集

### 2.1 日志证据

**证据 1：并发连接爆发**

proxy.log 显示在 `06:30:45` 的 1 秒内建立了 **100 个** TCP 连接：

```
06:30:45  →  100 个 connect_tcp.started
06:31:47  →   81 个 connect_tcp.started（第二波）
06:32:50  →   73 个 connect_tcp.started（第三波）
06:33:14  →   96 个 connect_tcp.started（第四波）
```

**证据 2：ReadTimeout 爆发**

在 `06:31:47`（第一波连接发出约 62 秒后），出现 **80 个** ReadTimeout：

```
2026-04-09 06:31:47,610-621  →  80 个 ReadTimeout(TimeoutError())
2026-04-09 06:32:49,809-819  →  73 个 ReadTimeout(TimeoutError())
2026-04-09 06:33:50,546-702  →  104 个 ReadTimeout(TimeoutError())
```

总计 **257 个 ReadTimeout**。

**证据 3：超时时间吻合**

- 第一波连接：`06:30:45` → 超时：`06:31:47` = **62 秒**
- 第二波连接：`06:31:48` → 超时：`06:32:50` = **62 秒**
- 第三波连接：`06:32:50` → 超时：`06:33:50` = **60 秒**

与 proxy 中 `httpx.Timeout(60.0)` 配置完全吻合。

**证据 4：成功请求的响应时间**

成功的请求（200 OK）响应时间在 **27-57 秒**之间：

```
06:30:45 发送 → 06:31:13 收到 = 28 秒
06:30:45 发送 → 06:31:42 收到 = 57 秒
06:30:46 发送 → 06:31:44 收到 = 58 秒（已接近 60 秒超时）
```

### 2.2 代码分析

**fast-graphrag 默认并发配置（`BaseLLMService`）：**

```python
max_requests_concurrent = 1024   # 最大并发请求数！
max_requests_per_minute = 500
max_requests_per_second = 60
rate_limit_concurrency = True
rate_limit_per_minute = False    # 每分钟限流未启用
rate_limit_per_second = False    # 每秒限流未启用
```

fast-graphrag 默认允许 **1024 个并发请求**，且每秒/每分钟限流都未启用。

**ericai proxy 配置（`ericai_proxy.py`）：**

```python
# 第 103 行：单个 httpx.AsyncClient，无连接池限制
async with httpx.AsyncClient(base_url=ERICSSON_GENAI_SSO_BASE, verify=_precombined_ca_bundle) as client:

# 第 151 行：60 秒超时
timeout=httpx.Timeout(60.0)
```

- httpx.AsyncClient 默认连接池：`max_connections=100, max_keepalive_connections=20`
- 超时：60 秒（固定，无读超时单独配置）

**后端 LLM 服务（ray.sero.gic.ericsson.se）：**

Qwen3-Coder-480B-A35B-Instruct-FP8 是一个 480B 参数的模型，单次推理本身就慢（27-57 秒），当 100 个请求同时涌入时，后端排队导致大量请求超过 60 秒超时。

### 2.3 环境信息

| 组件 | 配置 |
|------|------|
| Proxy | ericai proxy on `0.0.0.0:8633`，httpx.Timeout=60s |
| 后端 | `ray.sero.gic.ericsson.se`（HTTPS），vLLM 服务 |
| LLM 模型 | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`（8×H200 GPU） |
| 模型状态 | HEALTHY，1 replica running |

---

## 3. Root Cause

### 3.1 根因描述

**fast-graphrag 默认并发 1024 + 无限流** 与 **proxy 60 秒硬超时** + **480B 大模型单次推理 30-60 秒** 三者叠加，导致：

1. fast-graphrag 瞬间发出 ~100 个并发 LLM 请求
2. 后端 LLM 服务（单 replica）无法同时处理这么多请求，排队等待
3. proxy 的 60 秒超时到期，大量请求被 proxy 主动断开（ReadTimeout）
4. proxy 向 fast-graphrag 返回 500 错误
5. fast-graphrag 的 OpenAI SDK 自动重试，在同一秒内再次发出大量请求
6. 形成 **超时→500→重试→再超时** 的雪崩循环

### 3.2 因果链

```
fast-graphrag insert() 
  → 文本分块 → 并发发送 ~100 个实体提取请求（max_concurrent=1024）
    → ericai proxy 转发到后端 LLM
      → 480B 模型单 replica 排队处理，单次 30-60s
        → 排在后面的请求等待 > 60s
          → proxy httpx.Timeout(60s) 触发 ReadTimeout
            → proxy 返回 500 给 fast-graphrag
              → OpenAI SDK 自动重试（~1s 间隔）
                → 新请求再次涌入 → 雪崩
```

### 3.3 证据汇总

| # | 证据 | 来源 | 支持结论 |
|---|------|------|---------|
| 1 | 1 秒内 100 个 TCP 连接 | proxy.log connect_tcp 统计 | fast-graphrag 并发极高 |
| 2 | 257 个 ReadTimeout | proxy.log ReadTimeout 统计 | proxy 超时是 500 的直接原因 |
| 3 | 超时间隔 = 60s | 连接时间 vs 超时时间差 | 与 proxy Timeout(60.0) 吻合 |
| 4 | 成功请求耗时 27-57s | proxy.log 200 OK 响应时间 | 大模型推理本身接近超时边界 |
| 5 | max_requests_concurrent=1024 | fast-graphrag 源码 | 默认无并发限制 |
| 6 | httpx.Timeout(60.0) | ericai_proxy.py 第 151 行 | proxy 超时硬编码 60s |
| 7 | 重试间隔 ~0.8-1.0s | 用户提供的 500 错误日志 | OpenAI SDK 自动重试加剧问题 |

---

## 4. 排除项

| # | 假设 | 排除原因 |
|---|------|---------|
| 1 | 后端 LLM 服务宕机 | proxy.log 显示所有到达后端的请求都返回 200 OK，共 45 个成功 |
| 2 | 网络连接问题 | TCP 连接全部成功建立，TLS 握手正常 |
| 3 | API Key / 认证问题 | Azure SSO 认证成功（200），后续请求也正常 |
| 4 | Embedding 服务问题 | Embedding 请求正常返回 200 OK |

---

## 5. 修复方案

### 方案 A: 降低 fast-graphrag 并发数（推荐）

**描述**: 在创建 `OpenAILLMService` 时限制并发请求数

**改动范围**:
- `run_fast_graphrag_test.py`（或 `run_fast-graphrag.py`）

**代码示例**:
```python
llm_service = OpenAILLMService(
    model=args.llm_model,
    base_url=args.base_url,
    api_key="no-key",
    max_requests_concurrent=4,     # 从 1024 降到 4
    rate_limit_per_second=True,
    max_requests_per_second=2,     # 每秒最多 2 个请求
)
```

**优点**:
- 改动最小，只需加参数
- 从源头解决并发过高问题
- 不需要修改 proxy 或后端

**缺点**:
- 索引构建速度会变慢（但能成功完成）

**风险**: 低
**工作量**: 小

---

### 方案 B: 增大 proxy 超时时间

**描述**: 将 ericai proxy 的超时从 60s 增大到 300s

**改动范围**:
- `ericai-client/ericai/ericai_proxy.py` 第 151 行

**代码示例**:
```python
# 原始
timeout=httpx.Timeout(60.0)

# 修改为
timeout=httpx.Timeout(300.0)  # 5 分钟
```

**优点**:
- 改动极小
- 允许大模型有更多时间处理

**缺点**:
- 不解决并发过高的根本问题
- 100 个并发请求仍会让后端排队，只是等更久
- 可能导致 proxy 内存占用过高（大量长连接）

**风险**: 中
**工作量**: 小

---

### 方案 C: 方案 A + B 组合

**描述**: 同时降低并发 + 增大超时

**代码示例**:
```python
# fast-graphrag 侧
llm_service = OpenAILLMService(
    max_requests_concurrent=8,
    rate_limit_per_second=True,
    max_requests_per_second=4,
)

# proxy 侧
timeout=httpx.Timeout(180.0)  # 3 分钟
```

**优点**:
- 双重保障，最稳健
- 并发合理 + 超时宽裕

**缺点**:
- 需要改两处

**风险**: 低
**工作量**: 小

---

## 6. 方案对比

| 维度 | 方案 A | 方案 B | 方案 C |
|------|--------|--------|--------|
| 改动范围 | 1 个文件 | 1 个文件 | 2 个文件 |
| 解决根因 | ✅ 是 | ❌ 否 | ✅ 是 |
| 实现复杂度 | 低 | 低 | 低 |
| 风险等级 | 低 | 中 | 低 |
| 索引速度 | 较慢 | 不变 | 适中 |
| 推荐度 | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |

## 7. 建议

**推荐方案**: 方案 A（降低 fast-graphrag 并发数）

**理由**:
1. 480B 大模型单次推理就需要 30-60 秒，高并发毫无意义——后端只有 1 个 replica，请求只会排队
2. 并发 4-8 个请求已经能充分利用后端 GPU 的 batch 能力
3. 改动最小，不影响其他使用 proxy 的服务
4. 如果后续仍偶尔超时，可以额外增大 proxy 超时（升级为方案 C）

---

> ⚠️ 请确认选择的修复方案后，我将执行修复。
