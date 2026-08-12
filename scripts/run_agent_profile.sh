#!/bin/bash
# Run an agent workload with one shared PPD run ID and run-scoped artifacts.
#
# Usage:
#   scripts/run_agent_profile.sh -- \
#     bash -lc 'cd /root/uni-agent && NUM_WORKERS=3 BASE_URL=http://localhost:10001/v1 \
#       python examples/inference/parallel_infer_api.py \
#       --task-config examples/inference/task_config_ppd.yaml --limit 3'

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_ID=""
if [ "${1:-}" = "--run-id" ]; then
    RUN_ID="${2:-}"
    shift 2
fi
if [ "${1:-}" = "--" ]; then
    shift
fi
if [ "$#" -eq 0 ]; then
    echo "usage: $0 [--run-id ID] -- <agent command...>" >&2
    exit 2
fi
if [ -z "$RUN_ID" ]; then
    RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
SAFE_RUN_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="$PROJECT_DIR/logs/runs/$SAFE_RUN_ID"
if [ -e "$RUN_DIR" ]; then
    echo "run directory already exists: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"

export PPD_RUN_ID="$RUN_ID"
export PPD_TRACE_OUTPUT="$RUN_DIR/agent_trace.json"
export UNI_AGENT_LOG_DIR="${UNI_AGENT_LOG_DIR:-$RUN_DIR/rollouts}"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'start_ts=%s\n' "$(date +%s.%N)"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
} > "$RUN_DIR/manifest.txt"

python "$PROJECT_DIR/scripts/profile_metrics_poller.py" \
    --run-id "$RUN_ID" \
    --interval "${PPD_METRICS_INTERVAL:-0.5}" \
    --output "$RUN_DIR/metrics.csv" \
    > "$RUN_DIR/metrics_poller.log" 2>&1 &
METRICS_PID=$!

set +e
"$@" 2>&1 | tee "$RUN_DIR/agent.log"
WORKLOAD_STATUS=${PIPESTATUS[0]}
set -e

kill -TERM "$METRICS_PID" 2>/dev/null || true
wait "$METRICS_PID" 2>/dev/null || true
printf 'end_ts=%s\nworkload_status=%s\n' "$(date +%s.%N)" "$WORKLOAD_STATUS" >> "$RUN_DIR/manifest.txt"

python "$PROJECT_DIR/scripts/profile_analyze.py" --run-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/analysis.txt"
ANALYZE_STATUS=${PIPESTATUS[0]}
echo "Profiling artifacts: $RUN_DIR"
if [ "$WORKLOAD_STATUS" -ne 0 ]; then
    exit "$WORKLOAD_STATUS"
fi
exit "$ANALYZE_STATUS"
