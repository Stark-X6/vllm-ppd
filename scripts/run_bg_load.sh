#!/bin/bash
# Quick background load via simple curl loop
# Each iteration sends N concurrent requests via background jobs
CONCURRENCY=${1:-8}
DELAY=${2:-2.0}
URL="${3:-http://localhost:10001/v1/chat/completions}"
MODEL="${4:-/root/autodl-tmp/Qwen3-32B}"

echo "[bg-load] concurrency=$CONCURRENCY delay=${DELAY}s url=$URL"

TOPICS=("ML" "systems" "networks" "databases" "compilers" "graph theory" "crypto" "HPC" "bioinformatics" "security")

send_one() {
    local idx=$1
    local topic="${TOPICS[$((idx % ${#TOPICS[@]}))]}"
    curl -s "$URL" -H "Content-Type: application/json" -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a technical essay about $topic. Seed: $RANDOM\"}],
        \"max_tokens\": 256,
        \"temperature\": 0.7
    }" > /dev/null
}

idx=0
while true; do
    t0=$(date +%s%N)
    for ((i=0; i<CONCURRENCY; i++)); do
        send_one $((idx * CONCURRENCY + i)) &
    done
    wait
    t1=$(date +%s%N)
    elapsed=$(echo "scale=3; ($t1 - $t0) / 1000000000" | bc)
    echo "[bg-load] burst $CONCURRENCY req, elapsed=${elapsed}s"
    idx=$((idx + 1))
    sleep "$DELAY"
done
