#!/bin/bash
# 2P+2pD with nsys profiling wrapping each GPU
set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$SCRIPT_DIR/config.sh"

NSYS="/root/miniconda3/nsight-compute-2025.1.1/host/target-linux-x64/nsys"
LOG_DIR="$PROJECT_DIR/logs/nsys/2P_2pD"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.nsys-rep 2>/dev/null || true

echo "=== 2P_2pD NSYS Profiling ==="
echo "Output: $LOG_DIR/"

# Cleanup
if [ -f "$SCRIPT_DIR/cleanup_all.sh" ]; then
    bash "$SCRIPT_DIR/cleanup_all.sh" 2>/dev/null || true
fi
sleep 3

export NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL NCCL_P2P_DISABLE=0 CUDA_DEVICE_MAX_CONNECTIONS=32

SERVE_ARGS=(
    "$MODEL_PATH" --host 0.0.0.0 --max-model-len 8192
    --gpu-memory-utilization 0.75 --trust-remote-code --disable-log-requests
    --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder
)

NSYS_ARGS=(
    -o "$LOG_DIR" --force-overwrite true
    --cuda-memory-usage=true
    --duration 30 --capture-range-end stop
)

# Proxy
echo "[1] Proxy"
python "$PROJECT_DIR/ppd/comprehensive_proxy.py" \
    --config 2P_2pD --http-port 10001 --zmq-port 30001 > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

# GPU0: Prefill
echo "[2] Prefill GPU0 (nsys)"
P_KV='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":14579,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8100","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=0 "$NSYS" profile "${NSYS_ARGS[@]}" \
    python -u -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
    --port 8100 --kv-transfer-config "$P_KV" > "$LOG_DIR/prefill0.log" 2>&1 &

# GPU1: Prefill
echo "[3] Prefill GPU1 (nsys)"
P_KV1='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":14581,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8101","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=1 "$NSYS" profile "${NSYS_ARGS[@]}" \
    python -u -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
    --port 8101 --kv-transfer-config "$P_KV1" > "$LOG_DIR/prefill1.log" 2>&1 &

# GPU2: pD Decode
echo "[4] pD Decode GPU2 (nsys)"
D_KV='{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":10000000000,"kv_port":14580,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8200","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=2 "$NSYS" profile "${NSYS_ARGS[@]}" \
    python -u -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
    --port 8200 --kv-transfer-config "$D_KV" > "$LOG_DIR/ppd_decode2.log" 2>&1 &

# GPU3: pD Decode
echo "[5] pD Decode GPU3 (nsys)"
D_KV1='{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":10000000000,"kv_port":14582,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8201","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=3 "$NSYS" profile "${NSYS_ARGS[@]}" \
    python -u -m vllm.entrypoints.cli.main serve "${SERVE_ARGS[@]}" \
    --port 8201 --kv-transfer-config "$D_KV1" > "$LOG_DIR/ppd_decode3.log" 2>&1 &

# Wait
echo "Waiting..."
MAX_WAIT=300; WAITED=0
for PORT in 8100 8101 8200 8201; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 3; WAITED=$((WAITED+3))
        [ $WAITED -ge $MAX_WAIT ] && echo "Timeout $PORT" && exit 1
        [ $((WAITED%30)) -eq 0 ] && echo "  ...${WAITED}s"
    done
    echo "  $PORT ready"
done

echo ""
echo "=== NSYS profiling active! ==="
echo "Now run UniAgent, then stop servers."
echo "  cd /root/uni-agent && BASE_URL=http://localhost:10001/v1 MODEL=/root/autodl-tmp/Qwen3-8B PPD_TRACE_OUTPUT=/root/vllm-ppd/results/agent_trace.json python examples/inference/parallel_infer_api.py --task-config examples/inference/task_config_ppd.yaml --limit 1"
echo ""
echo "Stop:  pkill -TERM -f 'vllm serve' && sleep 10"
echo "nsys:  ls -lh $LOG_DIR/*.nsys-rep"
