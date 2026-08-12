#!/bin/bash
# ============================================================================
# Unified Configuration for vLLM PPD Servers
# ============================================================================
# All model and path configurations should be defined here.
# Other scripts should source this file.
#
# Environment variable overrides (for model scaling experiments):
# - MODEL_PATH: Override model path before sourcing this file
# - MAX_MODEL_LEN: Override max model length
# - GPU_MEMORY_UTILIZATION: Override GPU memory utilization
# ============================================================================

# Model path - set MODEL_PATH env var before sourcing this file
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen3-32B}"

# Qwen3-Coder uses its dedicated structured tool parser; dense Qwen3 uses Hermes.
# TOOL_CALL_PARSER can still be overridden explicitly for another model.
if [ -z "${TOOL_CALL_PARSER:-}" ]; then
    case "$MODEL_PATH" in
        *Qwen3-Coder*) TOOL_CALL_PARSER="qwen3_coder" ;;
        *) TOOL_CALL_PARSER="hermes" ;;
    esac
fi
export TOOL_CALL_PARSER

# Model name for pkill (derived from MODEL_PATH if not set)
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B}"

# Model settings (support environment variable override)
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"

# Conversation-hash salt: one per stack start, so re-running the same task
# instance doesn't inherit turn counters / server affinity from a prior run.
export PPD_PROXY_EPOCH="${PPD_PROXY_EPOCH:-$(date +%s)}"

# Project directory
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Log directory
export LOG_DIR="${PROJECT_DIR}/logs"

# Standard ports
export PROXY_HTTP_PORT=10001
export PROXY_ZMQ_PORT=30001

# GPU server base ports
export PREFILL_BASE_PORT=8100
export DECODE_BASE_PORT=8200
export REPLICA_BASE_PORT=8300
