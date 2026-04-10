#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Embedding 模型基准测试 — Shell 入口脚本
# 用法: bash tests/run_embedding_benchmark.sh [OPTIONS]
#   --sample N          每个数据集采样问题数 (默认 5)
#   --dataset NAME      medical|novel|all (默认 all)
#   --models "M1,M2"    逗号分隔的模型列表 (默认全部)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$SCRIPT_DIR/benchmark_results"
DATA_ROOT="$PROJECT_DIR/GraphRAG-Benchmark/Datasets"

# Defaults
SAMPLE=5
DATASET="all"
MODELS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)  SAMPLE="$2";  shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --models)  MODELS="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

mkdir -p "$OUTPUT_DIR"

# ── Dependency check ──
log "Checking Python dependencies..."
DEPS=(graphrag falkordb httpx langchain_openai pandas rouge_score)
MISSING=()
for pkg in "${DEPS[@]}"; do
    python3 -c "import $pkg" 2>/dev/null || MISSING+=("$pkg")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    log "Missing packages: ${MISSING[*]}"
    log "Attempting pip install..."
    pip install "${MISSING[@]}" || { log "ERROR: pip install failed"; exit 1; }
fi
log "All dependencies OK"

# ── API connectivity check ──
log "Checking API connectivity (LLM)..."
LLM_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "http://10.210.156.69:8633/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    --connect-timeout 10 || echo "000")
if [[ "$LLM_STATUS" != "200" ]]; then
    log "ERROR: LLM API unreachable (status=$LLM_STATUS)"
    exit 1
fi
log "LLM API OK"

log "Checking API connectivity (Embedding)..."
EMB_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "http://10.210.156.69:8633/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"model":"BAAI/bge-m3","input":"test"}' \
    --connect-timeout 10 || echo "000")
if [[ "$EMB_STATUS" != "200" ]]; then
    log "ERROR: Embedding API unreachable (status=$EMB_STATUS)"
    exit 1
fi
log "Embedding API OK"

# ── FalkorDB connectivity check ──
log "Checking FalkorDB connectivity..."
python3 -c "
from falkordb import FalkorDB
db = FalkorDB(host='10.210.156.69', port=6379)
db.list_graphs()
print('FalkorDB OK')
" || { log "ERROR: FalkorDB unreachable"; exit 1; }

# ── Run benchmark ──
log "Starting benchmark: sample=$SAMPLE dataset=$DATASET models=${MODELS:-all}"

ARGS=(
    --sample "$SAMPLE"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --data-root "$DATA_ROOT"
)
if [[ -n "$MODELS" ]]; then
    ARGS+=(--models "$MODELS")
fi

python3 "$SCRIPT_DIR/run_embedding_benchmark.py" "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/benchmark_console.log"

log "Benchmark complete. Results in $OUTPUT_DIR/"
log "  Report:  $OUTPUT_DIR/benchmark_report.md"
log "  Summary: $OUTPUT_DIR/summary.json"
log "  Log:     $OUTPUT_DIR/benchmark.log"
