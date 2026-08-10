#!/bin/bash
# ============================================================================
# 一键 NSYS Profiling — 短促 burst + 精确 GPU kernel 抓取
#
# 用法:
#   ./scripts/server/profile_config.sh 2P_2pD              # PPD, comprehensive burst
#   ./scripts/server/profile_config.sh 2P_2D  sharegpt     # PD, sharegpt burst
#   ./scripts/server/profile_config.sh 2P_2pD comprehensive # PPD, comprehensive burst
#
# 输出: logs/nsys/<config>/
# ============================================================================
set -e

CONFIG="${1:-2P_2pD}"
MODE="${2:-comprehensive}"     # comprehensive | sharegpt
BURST_DURATION="${BURST:-20}"  # burst 秒数 (短=文件小=看得清)
QPS="${QPS:-4}"

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/nsys/${CONFIG}_${MODE}"
NSYS_PATH="/root/miniconda3/nsight-compute-2025.1.1/host/target-linux-x64/nsys"

mkdir -p "$LOG_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║  NSYS Profile: $CONFIG"
echo "║  Mode: $MODE | Burst: ${BURST_DURATION}s | QPS: $QPS"
echo "╚══════════════════════════════════════════╝"

# 清理
cleanup() {
    pkill -TERM -f "[v]llm serve" 2>/dev/null || true
    sleep 8; pkill -9 -f "[v]llm" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

# [1] 启动 nsys 服务
echo ""; echo "=== [1/3] Starting servers (nsys wrapping each GPU) ==="
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Llama-3.1-8B}"
bash "$SCRIPT_DIR/start_nsys_profile.sh" "$CONFIG"

# [2] 跑 burst（15-20s 高密度请求）
echo ""; echo "=== [2/3] Profile burst (${BURST_DURATION}s) ==="
python "$PROJECT_DIR/scripts/profile_burst.py" \
    --mode "$MODE" --duration "$BURST_DURATION" --qps "$QPS" \
    2>&1 | tee "$LOG_DIR/burst.log"

# [3] 停止 + 导出
echo ""; echo "=== [3/3] Stop + export CSV ==="
pkill -TERM -f "[v]llm serve" 2>/dev/null || true
sleep 10; pkill -9 -f "[v]llm" 2>/dev/null || true
sleep 3; trap - EXIT

for f in "$LOG_DIR"/*.nsys-rep; do
    [ -f "$f" ] || continue
    b=$(basename "$f" .nsys-rep)
    echo "  Exporting $b"
    "$NSYS_PATH" stats "$f" --report gpukernsum --format csv \
        -o "$LOG_DIR/${b}_kernsum.csv" 2>/dev/null || true
    "$NSYS_PATH" stats "$f" --report cudaapisum --format csv \
        -o "$LOG_DIR/${b}_cudaapi.csv" 2>/dev/null || true
done

echo ""
echo "============================================"
echo " Done. ls -lh $LOG_DIR/"
echo " nsys-ui $LOG_DIR/*.nsys-rep"
echo ""
echo " Compare:"
echo "   ./scripts/server/profile_config.sh 2P_2D comprehensive"
echo "   ./scripts/server/profile_config.sh 2P_2pD sharegpt"
echo "============================================"
