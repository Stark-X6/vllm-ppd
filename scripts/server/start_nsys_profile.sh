#!/bin/bash
# ============================================================================
# NSYS Profiling: 全量抓取 vLLM serve 进程整个生命周期的 GPU kernel
#
# 用法:
#   ./scripts/server/start_nsys_profile.sh 2P_2pD    # PPD profiling
#   ./scripts/server/start_nsys_profile.sh 2P_2D     # PD baseline
#
# 输出: logs/nsys/<config>/
#   每个 GPU 一个 .nsys-rep + 服务日志
#
# 注意: 采集窗口包含模型加载（~30-60s）和 online 推理
#       分析时在 nsys-ui 里滚动到 benchmark 时间段即可
# ============================================================================

set -e

CONFIG="${1:-2P_2pD}"

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$SCRIPT_DIR/config.sh"

LOG_DIR="$PROJECT_DIR/logs/nsys/$CONFIG"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.nsys-rep 2>/dev/null || true

NSYS_PATH="/root/miniconda3/nsight-compute-2025.1.1/host/target-linux-x64/nsys"

echo "=============================================="
echo "NSYS Profiling: $CONFIG"
echo "  Mode:    全量抓取 (含模型加载 + online 推理)"
echo "  Output:  $LOG_DIR/"
echo "=============================================="

# ── 通用 vLLM serve 参数 ──
SERVE_ARGS=(
    "$MODEL_PATH"
    --host 0.0.0.0
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --trust-remote-code --disable-log-requests
    --enable-prefix-caching
)

# ── nsys 参数 ──
NSYS_ARGS=(
    -o "$LOG_DIR"
    --force-overwrite true
    --cuda-memory-usage=true
    --gpuctx-switch=true
)

# ── NCCL 环境 ──
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=32

# ── 清理旧进程 ──
if [ -f "$SCRIPT_DIR/cleanup_all.sh" ]; then
    bash "$SCRIPT_DIR/cleanup_all.sh" 2>/dev/null || true
else
    pkill -9 -f "[v]llm serve" 2>/dev/null || true
    pkill -9 -f "[c]omprehensive_proxy" 2>/dev/null || true
    sleep 5
fi

# ── KV Config 辅助函数 ──
P_KV() {
    local PORT=$1 KV=$2
    echo '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":'$KV',"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"'$PORT'","send_type":"PUT_ASYNC"}}'
}
D_KV() {
    local PORT=$1 KV=$2
    echo '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":10000000000,"kv_port":'$KV',"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"'$PORT'","send_type":"PUT_ASYNC"}}'
}

# ═══════════════════════════════════════════════════════════════
# Proxy
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[0/5] Starting Proxy..."
python "$PROJECT_DIR/ppd/comprehensive_proxy.py" \
    --config "$CONFIG" --http-port 10001 --zmq-port 30001 \
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

# ═══════════════════════════════════════════════════════════════
# GPU Servers (每个用 nsys 包)
# ═══════════════════════════════════════════════════════════════

case "$CONFIG" in

2P_2pD)
    echo "[1/5] P GPU0 (nsys)"
    CUDA_VISIBLE_DEVICES=0 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8100 --kv-transfer-config "$(P_KV 8100 14579)" \
        > "$LOG_DIR/prefill0.log" 2>&1 &

    echo "[2/5] P GPU1 (nsys)"
    CUDA_VISIBLE_DEVICES=1 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8101 --kv-transfer-config "$(P_KV 8101 14581)" \
        > "$LOG_DIR/prefill1.log" 2>&1 &

    echo "[3/5] pD GPU2 (nsys)"
    CUDA_VISIBLE_DEVICES=2 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8200 --kv-transfer-config "$(D_KV 8200 14580)" \
        > "$LOG_DIR/ppd_decode2.log" 2>&1 &

    echo "[4/5] pD GPU3 (nsys)"
    CUDA_VISIBLE_DEVICES=3 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8201 --kv-transfer-config "$(D_KV 8201 14582)" \
        > "$LOG_DIR/ppd_decode3.log" 2>&1 &
    ;;

2P_2D)
    echo "[1/5] P GPU0 (nsys)"
    CUDA_VISIBLE_DEVICES=0 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8100 --kv-transfer-config "$(P_KV 8100 14579)" \
        > "$LOG_DIR/prefill0.log" 2>&1 &

    echo "[2/5] P GPU1 (nsys)"
    CUDA_VISIBLE_DEVICES=1 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8101 --kv-transfer-config "$(P_KV 8101 14581)" \
        > "$LOG_DIR/prefill1.log" 2>&1 &

    echo "[3/5] D GPU2 (nsys)"
    CUDA_VISIBLE_DEVICES=2 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8200 --kv-transfer-config "$(D_KV 8200 14580)" \
        > "$LOG_DIR/decode2.log" 2>&1 &

    echo "[4/5] D GPU3 (nsys)"
    CUDA_VISIBLE_DEVICES=3 "$NSYS_PATH" profile "${NSYS_ARGS[@]}" \
        python -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
        --port 8201 --kv-transfer-config "$(D_KV 8201 14582)" \
        > "$LOG_DIR/decode3.log" 2>&1 &
    ;;

*)
    echo "ERROR: Unknown config $CONFIG. Supported: 2P_2D, 2P_2pD"
    exit 1
    ;;
esac

# ═══════════════════════════════════════════════════════════════
# 等待就绪
# ═══════════════════════════════════════════════════════════════
echo ""
echo "=== Waiting for servers ==="
MAX_WAIT=300; WAITED=0
for PORT in 8100 8101 8200 8201; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 3; WAITED=$((WAITED + 3))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Timeout $PORT"
            exit 1
        fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  waiting... (${WAITED}s)"
    done
    echo "  Port $PORT: READY"
done

echo ""
echo "=============================================="
echo "Servers ready. nsys is capturing ALL GPU activity."
echo ""
echo "Now run benchmark in another terminal:"
echo "  python scripts/benchmark/comprehensive_benchmark.py \\"
echo "    --config $CONFIG --workload all --qps 1 2 4 --skip-startup"
echo ""
echo "OR sharegpt:"
echo "  python scripts/benchmark/sharegpt_benchmark.py \\"
echo "    --config $CONFIG --num-conversations 500 --qps 4"
echo ""
echo "After benchmark completes: pkill -TERM -f 'vllm serve'"
echo "nsys will save .nsys-rep automatically."
echo "=============================================="
