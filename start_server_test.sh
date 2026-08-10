#!/bin/bash
export MODEL_PATH="/root/autodl-tmp/Llama-3.1-8B"
CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8300 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --trust-remote-code \
    --disable-log-requests --enable-prefix-caching \
    > /root/vllm-ppd/logs/single_replica/server.log 2>&1 &
SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID"
