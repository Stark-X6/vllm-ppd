#!/bin/bash
# PPD 最小配置: 1 Prefill + 1 pD Decode (2 GPU)
set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$SCRIPT_DIR/config.sh"

LOG_DIR="$PROJECT_DIR/logs/1P_1pD"
mkdir -p "$LOG_DIR"

echo "Starting 1P_1pD PPD..."

# Cleanup
pkill -9 -f "[v]llm serve" 2>/dev/null || true
pkill -9 -f "[c]omprehensive_proxy" 2>/dev/null || true
sleep 3

# NCCL
export NCCL_DEBUG=INFO VLLM_LOGGING_LEVEL=INFO
export NCCL_P2P_LEVEL=NVL NCCL_P2P_DISABLE=0 CUDA_DEVICE_MAX_CONNECTIONS=32

# Proxy
echo "[1/3] Proxy"
python "$PROJECT_DIR/ppd/comprehensive_proxy.py" \
    --config 1P_1pD --http-port 10001 --zmq-port 30001 \
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

# GPU0: Prefill
echo "[2/3] P GPU0"
CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8100 --max-model-len 8192 \
    --gpu-memory-utilization 0.75 --trust-remote-code \
    --disable-log-requests --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser llama3_json \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":14579,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8100","send_type":"PUT_ASYNC"}}' \
    > "$LOG_DIR/prefill0.log" 2>&1 &

# GPU1: pD Decode
echo "[3/3] pD GPU1"
CUDA_VISIBLE_DEVICES=1 python -u -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8200 --max-model-len 8192 \
    --gpu-memory-utilization 0.75 --trust-remote-code \
    --disable-log-requests --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser llama3_json \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":10000000000,"kv_port":14580,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8200","send_type":"PUT_ASYNC"}}' \
    > "$LOG_DIR/ppd_decode0.log" 2>&1 &

# Wait
echo "Waiting..."
MAX_WAIT=300; WAITED=0
for PORT in 8100 8200; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 3; WAITED=$((WAITED + 3))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  waiting... ${WAITED}s"
    done
    echo "  $PORT ready"
done

echo ""
echo "1P_1pD ready. Proxy: http://localhost:10001"
