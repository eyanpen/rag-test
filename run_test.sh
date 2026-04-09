#!/usr/bin/env bash
# ============================================================
# GraphRAG-Benchmark fast-graphrag 一键测试脚本
# 使用远程 OpenAI 兼容 API（LLM + Embedding）
# ============================================================
set -euo pipefail

STOP_AFTER=""
FULL_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-after) STOP_AFTER="$2"; shift 2 ;;
    --full) FULL_RUN=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BASE_URL="http://10.210.156.69:8633"
LLM_MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
EMBED_DIM=4096
SUBSET="medical"
SAMPLE=5
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR/GraphRAG-Benchmark"
WORKSPACE="$SCRIPT_DIR/test_fast_graphrag_workspace"
RESULTS_DIR="$SCRIPT_DIR/test_results"
REPORT="$RESULTS_DIR/test_report.md"

mkdir -p "$RESULTS_DIR" "$WORKSPACE"

# ── 辅助函数 ──
log()  { echo -e "\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
err()  { echo -e "\033[1;31m[$(date +%H:%M:%S)] ERROR: $*\033[0m"; }
ts()   { date +%s; }

# ── 判断是否可以跳过 index 及之前步骤 ──
SKIP_INDEX=false
LLM_OK=false
EMBED_OK=false
if ! $FULL_RUN && ls "$WORKSPACE"/*/graph_igraph_data.pklz &>/dev/null; then
  SKIP_INDEX=true
  LLM_OK=true
  EMBED_OK=true
  log "检测到已有 index，跳过 Step 1-2 及 index，直接跑 query (使用 --full 强制全流程)"
fi

if ! $SKIP_INDEX; then
# ── 1. 安装依赖 ──
log "Step 1/6: 检查 Python 依赖..."
if python3 -c "import datasets, fast_graphrag, openai, aiohttp, tqdm, transformers, torch" 2>/dev/null; then
  log "  所有依赖已安装，跳过安装步骤 ✅"
else
  log "  部分依赖缺失，开始安装..."
  pip install -q datasets fast-graphrag openai aiohttp tqdm transformers torch 2>&1 | tail -3
fi

# ── 2. 连通性检查 ──
log "Step 2/6: 检查 API 连通性..."
LLM_OK=false
EMBED_OK=false

if curl -sf --max-time 10 "$BASE_URL/chat/completions" \
  -X POST -H "Content-Type: application/json" \
  -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
  -o /dev/null; then
  LLM_OK=true
  log "  LLM API ✅"
else
  err "  LLM API ❌ ($BASE_URL)"
fi

if curl -sf --max-time 10 "$BASE_URL/embeddings" \
  -X POST -H "Content-Type: application/json" \
  -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"test\"}" \
  -o /dev/null; then
  EMBED_OK=true
  log "  Embedding API ✅"
else
  err "  Embedding API ❌ ($BASE_URL)"
fi

if ! $LLM_OK || ! $EMBED_OK; then
  err "API 连通性检查失败，终止测试"
  exit 1
fi

if [[ "$STOP_AFTER" == "2" ]]; then
  log "--stop-after 2: 连通性检查完成，退出"
  exit 0
fi
fi # end SKIP_INDEX

# ── 3. 运行测试 ──
SKIP_INDEX_FLAG=""
if $SKIP_INDEX; then
  log "Step 3/6: 跳过 index，直接跑 query (subset=$SUBSET, sample=$SAMPLE)..."
  SKIP_INDEX_FLAG="--skip-index"
else
  log "Step 3/6: 运行 fast-graphrag 完整测试 (subset=$SUBSET, sample=$SAMPLE)..."
fi

T_START=$(ts)

python3 "$SCRIPT_DIR/run_fast_graphrag_test.py" \
  --subset "$SUBSET" \
  --sample "$SAMPLE" \
  --base_url "$BASE_URL" \
  --llm_model "$LLM_MODEL" \
  --embed_model "$EMBED_MODEL" \
  --embed_dim "$EMBED_DIM" \
  --workspace "$WORKSPACE" \
  --output "$RESULTS_DIR/predictions.json" \
  $SKIP_INDEX_FLAG \
  2>&1 | tee "$RESULTS_DIR/run.log"

T_END=$(ts)
DURATION=$((T_END - T_START))

# ── 4. 检查结果 ──
log "Step 4/6: 验证输出..."
if [ ! -f "$RESULTS_DIR/predictions.json" ]; then
  err "predictions.json 未生成"
  exit 1
fi

TOTAL=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/predictions.json'))))")
ERRORS=$(python3 -c "import json; print(sum(1 for x in json.load(open('$RESULTS_DIR/predictions.json')) if 'error' in x))")
SUCCESS=$((TOTAL - ERRORS))

log "  总计: $TOTAL, 成功: $SUCCESS, 失败: $ERRORS"

# ── 5. 简单质量抽检 ──
log "Step 5/6: 质量抽检..."
python3 -c "
import json
data = json.load(open('$RESULTS_DIR/predictions.json'))
for item in data[:3]:
    if 'error' in item:
        continue
    q = item.get('question','')[:80]
    a = item.get('generated_answer','')[:120]
    ctx_n = len(item.get('context', []))
    print(f'  Q: {q}')
    print(f'  A: {a}')
    print(f'  Context chunks: {ctx_n}')
    print()
"

# ── 6. 生成报告 ──
log "Step 6/6: 生成测试报告..."

cat > "$REPORT" << REPORT_EOF
# fast-graphrag 测试报告

> 生成时间: $(date '+%Y-%m-%d %H:%M:%S')

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| Python | $(python3 --version 2>&1) |
| OS | $(uname -s -r) |
| API Base URL | $BASE_URL |
| LLM 模型 | $LLM_MODEL |
| Embedding 模型 | $EMBED_MODEL |
| Embedding 维度 | $EMBED_DIM |

## 2. 测试配置

| 项目 | 值 |
|------|-----|
| 数据子集 | $SUBSET |
| 采样数量 | $SAMPLE |
| 工作目录 | $WORKSPACE |

## 3. 连通性检查

| 服务 | 状态 |
|------|------|
| LLM API | $($LLM_OK && echo "✅ 通过" || echo "❌ 失败") |
| Embedding API | $($EMBED_OK && echo "✅ 通过" || echo "❌ 失败") |

## 4. 推理结果

| 指标 | 值 |
|------|-----|
| 总问题数 | $TOTAL |
| 成功 | $SUCCESS |
| 失败 | $ERRORS |
| 成功率 | $(python3 -c "print(f'{$SUCCESS/$TOTAL*100:.1f}%' if $TOTAL>0 else 'N/A')") |
| 总耗时 | ${DURATION}s |
| 平均每题 | $(python3 -c "print(f'{$DURATION/$TOTAL:.1f}s' if $TOTAL>0 else 'N/A')") |

## 5. 结果样例

\`\`\`json
$(python3 -c "
import json
data = json.load(open('$RESULTS_DIR/predictions.json'))
sample = data[:2]
for s in sample:
    if 'context' in s and isinstance(s['context'], list):
        s['context'] = [c[:200]+'...' if len(c)>200 else c for c in s['context'][:2]]
    if 'generated_answer' in s:
        s['generated_answer'] = s['generated_answer'][:300]
    if 'ground_truth' in s:
        s['ground_truth'] = s['ground_truth'][:300]
print(json.dumps(sample, indent=2, ensure_ascii=False))
")
\`\`\`

## 6. 问题类型分布

$(python3 -c "
import json
from collections import Counter
data = json.load(open('$RESULTS_DIR/predictions.json'))
types = Counter(x.get('question_type','unknown') for x in data if 'error' not in x)
print('| 类型 | 数量 |')
print('|------|------|')
for t, c in types.most_common():
    print(f'| {t} | {c} |')
")

## 7. 运行日志（末尾）

\`\`\`
$(tail -20 "$RESULTS_DIR/run.log")
\`\`\`
REPORT_EOF

log "✅ 测试完成！报告已生成: $REPORT"
log "   预测结果: $RESULTS_DIR/predictions.json"
log "   运行日志: $RESULTS_DIR/run.log"
