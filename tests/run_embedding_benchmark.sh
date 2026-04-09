#!/usr/bin/env bash
# Embedding 模型基准测试入口脚本
set -euo pipefail

BASE_URL="http://10.210.156.69:8633"
LLM_MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
EMBED_TEST_MODEL="BAAI/bge-m3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/benchmark_results"

SAMPLE=5
DATASET="kevin_scott"
MODELS=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --sample) SAMPLE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

log()  { echo -e "\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
err()  { echo -e "\033[1;31m[$(date +%H:%M:%S)] ERROR: $*\033[0m"; }

mkdir -p "$OUTPUT_DIR"

# Step 1: Dependencies
log "Step 1/4: 检查 Python 依赖..."
if python3 -c "import fast_graphrag, httpx, rouge_score, tqdm, langchain_openai" 2>/dev/null; then
  log "  依赖已安装 ✅"
else
  log "  安装缺失依赖..."
  pip install -q fast-graphrag httpx rouge-score tqdm langchain-openai 2>&1 | tail -3
fi

# Step 2: API connectivity
log "Step 2/4: 检查 API 连通性..."
LLM_OK=false
EMBED_OK=false

if curl -sf --max-time 15 "$BASE_URL/chat/completions" \
  -X POST -H "Content-Type: application/json" \
  -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
  -o /dev/null; then
  LLM_OK=true; log "  LLM API ✅"
else
  err "  LLM API ❌"
fi

if curl -sf --max-time 15 "$BASE_URL/embeddings" \
  -X POST -H "Content-Type: application/json" \
  -d "{\"model\":\"$EMBED_TEST_MODEL\",\"input\":\"test\"}" \
  -o /dev/null; then
  EMBED_OK=true; log "  Embedding API ✅"
else
  err "  Embedding API ❌"
fi

if ! $LLM_OK || ! $EMBED_OK; then
  err "API 连通性检查失败"; exit 1
fi

# Step 3: Run benchmark
log "Step 3/4: 运行基准测试 (sample=$SAMPLE, dataset=$DATASET)..."

CMD="python3 $SCRIPT_DIR/run_embedding_benchmark.py --sample $SAMPLE --dataset $DATASET --output-dir $OUTPUT_DIR"
if [ -n "$MODELS" ]; then
  CMD="$CMD --models $MODELS"
fi

eval "$CMD" 2>&1 | tee "$OUTPUT_DIR/benchmark.log"

# Step 4: Check results
log "Step 4/4: 验证结果..."
if [ -f "$OUTPUT_DIR/benchmark_report.md" ]; then
  log "  报告已生成: $OUTPUT_DIR/benchmark_report.md ✅"
else
  err "  报告未生成 ❌"
fi
if [ -f "$OUTPUT_DIR/summary.json" ]; then
  log "  汇总已生成: $OUTPUT_DIR/summary.json ✅"
else
  err "  汇总未生成 ❌"
fi

log "✅ 基准测试完成！"
