#!/usr/bin/env python3
"""
Comprehensive Proxy Server for All 17 GPU Configurations.

This proxy handles routing for:
- Pure Replica (4R)
- Pure Disaggregated (1P+3D, 2P+2D, etc.)
- Hybrid (1R+1P+2D, etc.)

Key features:
- Automatic instance discovery via ZMQ
- Cache affinity for multi-turn conversations
- Flexible routing based on server types (P, D, pD, R)
- Statistics collection
- Dynamic PPD mode: Uses benchmark data to decide when PPD is beneficial

Usage:
    python comprehensive_proxy.py --config 2P_2D --http-port 10001 --zmq-port 30001

    # With dynamic PPD mode:
    python comprehensive_proxy.py --config 2P_2D --enable-ppd-mode --ppd-benchmark-path results/comprehensive
"""

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import msgpack
import zmq
from quart import Quart, request, Response

# Add project root to path for optimizer imports
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


# ---- PPD profiling: per-request timing trace (JSONL) ----
# Each request routed by the proxy appends one line with the pipeline split:
#   proxy overhead (ts_route - ts_enter), P prefill (ts_prefill_done - ts_route),
#   decode/stream window (ts_exit - ts_prefill_done), and streamed TTFT.
_TIMING_LOCK = threading.Lock()
_TIMING_OUTPUT = os.environ.get(
    "PPD_PROXY_TIMING_OUTPUT",
    str(Path(PROJECT_DIR) / "logs" / "proxy_timing.jsonl"),
)
_RUN_OUTPUT_ROOT = Path(os.environ.get(
    "PPD_RUN_OUTPUT_ROOT",
    str(Path(PROJECT_DIR) / "logs" / "runs"),
))


def _write_timing(record: Dict[str, Any]) -> None:
    try:
        output_path = Path(_TIMING_OUTPUT)
        run_id = str(record.get("run_id") or "unscoped")
        if run_id != "unscoped":
            safe_run_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
            output_path = _RUN_OUTPUT_ROOT / safe_run_id / "proxy_timing.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with _TIMING_LOCK:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[PROXY] timing write error: {e}")


# Global state - instances by type
prefill_instances: Dict[str, Tuple[str, float]] = {}  # http_addr -> (zmq_addr, stamp)
decode_instances: Dict[str, Tuple[str, float]] = {}   # D instances
ppd_decode_instances: Dict[str, Tuple[str, float]] = {}  # pD instances
replica_instances: Dict[str, float] = {}  # http_addr -> stamp

# Locks
prefill_cv = threading.Condition()
decode_cv = threading.Condition()
ppd_decode_cv = threading.Condition()
replica_cv = threading.Condition()

# Conversation tracking
# Key: conv_hash, Value: (turn_count, last_access, assigned_servers)
# assigned_servers: {"prefill": addr, "decode": addr, "replica": addr}
conversation_state: Dict[str, Tuple[int, float, Dict[str, str]]] = {}
conversation_lock = threading.Lock()

# Statistics
stats = {
    "total_requests": 0,
    "turn1_requests": 0,
    "turn2plus_requests": 0,
    "pd_requests": 0,
    "ppd_requests": 0,
    "ppd_direct_requests": 0,
    "ppd_dynamic_requests": 0,  # PPD mode enabled by dynamic decision engine
    "replica_requests": 0,
    "cache_affinity_hits": 0,
    "cache_affinity_misses": 0,
}
stats_lock = threading.Lock()

# Configuration
CONFIG_NAME = "2P_2D"
CONVERSATION_TTL = 3600
DEFAULT_PING_SECONDS = 5

# Dynamic PPD mode configuration
ENABLE_PPD_MODE = False
PPD_DECISION_ENGINE = None
PPD_DECISION_LOG: List[dict] = []  # Log of PPD decisions for analysis
PPD_LOG_LOCK = threading.Lock()

# Static configuration for replica ports (replicas don't use ZMQ registration)
REPLICA_PORTS_BY_CONFIG = {
    "4R": [8300, 8400, 8500, 8600],
    "1R_1P_2D": [8300],
    "1R_1P_1D_1pD": [8300],
    "1R_1P_2pD": [8300],
    "1R_2P_1D": [8300],
    "1R_2P_1pD": [8300],
    "2R_1P_1D": [8300, 8400],
    "2R_1P_1pD": [8300, 8400],
}

# Explicit decode port mapping: config_name -> {"D": [ports], "pD": [ports]}
# This ensures precise D vs pD distinction for mixed configurations
DECODE_PORTS_BY_CONFIG = {
    # Pure PD mode (all decode servers are D)
    "1P_3D": {"D": [8200, 8201, 8202], "pD": []},
    "2P_2D": {"D": [8200, 8201], "pD": []},
    "3P_1D": {"D": [8200], "pD": []},
    "1R_1P_2D": {"D": [8200, 8201], "pD": []},
    "1R_2P_1D": {"D": [8200], "pD": []},
    "2R_1P_1D": {"D": [8200], "pD": []},

    # Pure PPD mode (all decode servers are pD)
    "1P_3pD": {"D": [], "pD": [8200, 8201, 8202]},
    "2P_2pD": {"D": [], "pD": [8200, 8201]},
    "3P_1pD": {"D": [], "pD": [8200]},
    "1R_1P_2pD": {"D": [], "pD": [8200, 8201]},
    "1R_2P_1pD": {"D": [], "pD": [8200]},
    "2R_1P_1pD": {"D": [], "pD": [8200]},

    # Mixed D/pD configurations
    "1P_1pD": {"D": [], "pD": [8200]},
    "1P_2D_1pD": {"D": [8200, 8201], "pD": [8202]},
    "1P_1D_2pD": {"D": [8200], "pD": [8201, 8202]},
    "2P_1D_1pD": {"D": [8200], "pD": [8201]},
    "1R_1P_1D_1pD": {"D": [8200], "pD": [8201]},
}

app = Quart(__name__)


def _remove_oldest_instances(instances: dict, is_replica: bool = False) -> None:
    """Remove expired instances from the registry."""
    current_time = time.time()
    expired = []
    for key, value in instances.items():
        stamp = value if is_replica else value[1]
        if stamp < current_time:
            expired.append(key)
    for key in expired:
        print(f"[PROXY] Remove expired: {key}")
        instances.pop(key, None)


def _get_port_from_addr(addr: str) -> int:
    """Extract port number from address string (ip:port format)."""
    try:
        return int(addr.split(":")[-1])
    except (ValueError, IndexError):
        return 0


def _is_ppd_port(port: int) -> bool:
    """Check if the given port is configured as pD (prefill-capable decode).

    Uses explicit port mapping from DECODE_PORTS_BY_CONFIG for precise D/pD distinction.
    """
    config = DECODE_PORTS_BY_CONFIG.get(CONFIG_NAME, {})
    ppd_ports = config.get("pD", [])
    return port in ppd_ports


def _listen_for_register(poller, router_socket):
    """Listen for instance registration messages."""
    while True:
        try:
            socks = dict(poller.poll(timeout=1000))
            if router_socket not in socks:
                continue

            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            http_addr = data.get("http_address", "")
            zmq_addr = data.get("zmq_address", "")
            server_type = data.get("type", "")

            stamp = time.time() + DEFAULT_PING_SECONDS

            if server_type == "P":
                with prefill_cv:
                    if http_addr not in prefill_instances:
                        print(f"[PROXY] Add [P] HTTP:{http_addr}, ZMQ:{zmq_addr}")
                    prefill_instances[http_addr] = (zmq_addr, stamp)
                    _remove_oldest_instances(prefill_instances)

            elif server_type == "D":
                # Use explicit port mapping to determine D vs pD
                port = _get_port_from_addr(http_addr)
                is_ppd = _is_ppd_port(port)

                if is_ppd:
                    with ppd_decode_cv:
                        if http_addr not in ppd_decode_instances:
                            print(f"[PROXY] Add [pD] HTTP:{http_addr}, ZMQ:{zmq_addr}")
                        ppd_decode_instances[http_addr] = (zmq_addr, stamp)
                        _remove_oldest_instances(ppd_decode_instances)
                else:
                    with decode_cv:
                        if http_addr not in decode_instances:
                            print(f"[PROXY] Add [D] HTTP:{http_addr}, ZMQ:{zmq_addr}")
                        decode_instances[http_addr] = (zmq_addr, stamp)
                        _remove_oldest_instances(decode_instances)

            elif server_type == "R":
                with replica_cv:
                    if http_addr not in replica_instances:
                        print(f"[PROXY] Add [R] HTTP:{http_addr}")
                    replica_instances[http_addr] = stamp
                    _remove_oldest_instances(replica_instances, is_replica=True)

        except Exception as e:
            print(f"[PROXY] Registration error: {e}")


def start_service_discovery(hostname: str, port: int):
    """Start the ZMQ service discovery listener."""
    if not hostname:
        hostname = socket.gethostname()

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    thread = threading.Thread(
        target=_listen_for_register,
        args=[poller, router_socket],
        daemon=True
    )
    thread.start()
    return thread


def register_static_replicas():
    """Register replica instances based on static configuration.

    Replicas don't use KV transfer, so they don't register via ZMQ.
    We register them statically based on the config name.
    """
    replica_ports = REPLICA_PORTS_BY_CONFIG.get(CONFIG_NAME, [])
    if not replica_ports:
        return

    # Get local IP
    local_ip = socket.gethostbyname(socket.gethostname())

    with replica_cv:
        for port in replica_ports:
            http_addr = f"{local_ip}:{port}"
            # Use a far-future timestamp so they don't expire
            replica_instances[http_addr] = time.time() + 86400 * 365
            print(f"[PROXY] Add [R] HTTP:{http_addr} (static)")


def _message_value_text(value: Any) -> str:
    """Return a stable textual representation for token estimation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def estimate_token_count(text: Any) -> int:
    """Estimate token count from text (rough approximation: ~4 chars per token)."""
    serialized = _message_value_text(text)
    if not serialized:
        return 0
    return len(serialized) // 4


def estimate_message_token_count(message: dict) -> int:
    """Estimate one chat message, including structured tool-call payloads.

    The estimate intentionally remains lightweight and tokenizer-free, matching
    the original PPD proxy. Structured fields matter in agent transcripts: an
    assistant tool call is reusable context, while tool_call_id/name and the
    tool observation are part of the next append-prefill.
    """
    token_count = estimate_token_count(message.get("content"))
    token_count += estimate_token_count(message.get("tool_calls"))
    token_count += estimate_token_count(message.get("name"))
    token_count += estimate_token_count(message.get("tool_call_id"))
    # Approximate chat-template role/separator overhead.
    return token_count + 4


def _extract_request_features_with_metadata(
    data: dict,
) -> Tuple[int, int, int, Dict[str, Any]]:
    """Extract PPD decision features plus profiling metadata.

    For chat requests, the current append consists of every message after the
    most recent assistant response. This supports both regular chat (the new
    user message) and ReAct agents (one or more tool observations). Everything
    through that assistant response is reusable accumulated context.
    """
    output_tokens = data.get("max_tokens", 128)
    if "max_completion_tokens" in data:
        output_tokens = data.get("max_completion_tokens", output_tokens)
    try:
        output_tokens = max(0, int(output_tokens))
    except (TypeError, ValueError):
        output_tokens = 128

    if "messages" in data:
        messages = [msg for msg in (data.get("messages") or []) if isinstance(msg, dict)]
        last_assistant = next(
            (index for index in range(len(messages) - 1, -1, -1)
             if messages[index].get("role") == "assistant"),
            None,
        )

        if last_assistant is not None:
            split_index = last_assistant + 1
            source = "chat_after_last_assistant"
        else:
            # First turn: the latest user message is the new input; system and
            # any preceding setup messages form its context.
            last_user = next(
                (index for index in range(len(messages) - 1, -1, -1)
                 if messages[index].get("role") == "user"),
                len(messages),
            )
            split_index = last_user
            source = "chat_first_user_turn"

        context_messages = messages[:split_index]
        append_messages = messages[split_index:]
        context_tokens = sum(estimate_message_token_count(msg) for msg in context_messages)
        current_input_tokens = sum(estimate_message_token_count(msg) for msg in append_messages)
        metadata = {
            "feature_source": source,
            "feature_token_estimator": "chars_per_4_with_chat_overhead_v1",
            "last_message_role": messages[-1].get("role") if messages else None,
            "append_message_roles": [msg.get("role") for msg in append_messages],
        }
        return current_input_tokens, output_tokens, context_tokens, metadata

    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(part) for part in prompt)

    # For continuation prompts, estimate based on "User:" markers.
    if "User:" in prompt:
        parts = prompt.rsplit("User:", 1)
        if len(parts) == 2:
            context_part, current_part = parts
            if "Assistant:" in current_part:
                current_part = current_part.split("Assistant:", 1)[0]
            return (
                estimate_token_count(current_part),
                output_tokens,
                estimate_token_count(context_part),
                {
                    "feature_source": "completion_last_user_marker",
                    "feature_token_estimator": "chars_per_4_v1",
                    "last_message_role": None,
                    "append_message_roles": [],
                },
            )

    # Fallback: treat entire prompt as context plus a small input, preserving
    # the original proxy behavior for legacy completion clients.
    total_tokens = estimate_token_count(prompt)
    return (
        min(total_tokens, 256),
        output_tokens,
        max(0, total_tokens - 256),
        {
            "feature_source": "completion_fallback",
            "feature_token_estimator": "chars_per_4_v1",
            "last_message_role": None,
            "append_message_roles": [],
        },
    )


def extract_request_features(data: dict) -> Tuple[int, int, int]:
    """Extract request features for the PPD decision engine.

    Returns:
        (input_tokens, output_tokens, context_tokens)
        - input_tokens: estimated tokens appended since the latest assistant turn
        - output_tokens: configured generation cap (current PPD-compatible proxy behavior)
        - context_tokens: estimated reusable tokens through the latest assistant turn
    """
    input_tokens, output_tokens, context_tokens, _ = _extract_request_features_with_metadata(data)
    return input_tokens, output_tokens, context_tokens


def log_ppd_decision(
    conv_hash: str,
    turn: int,
    input_tokens: int,
    output_tokens: int,
    context_tokens: int,
    current_qps: float,
    decision: bool,
    routing_mode: str,
):
    """Log a PPD decision for later analysis."""
    global PPD_DECISION_LOG

    entry = {
        "timestamp": time.time(),
        "conv_hash": conv_hash[:8],
        "turn": turn,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": context_tokens,
        "current_qps": current_qps,
        "ppd_decision": decision,
        "actual_mode": routing_mode,
    }

    with PPD_LOG_LOCK:
        PPD_DECISION_LOG.append(entry)
        # Keep only last 10000 entries
        if len(PPD_DECISION_LOG) > 10000:
            PPD_DECISION_LOG = PPD_DECISION_LOG[-10000:]


def get_conversation_hash(
    data: dict, *, conversation_id: Optional[str] = None, run_id: Optional[str] = None
) -> str:
    """Generate a hash to identify a conversation."""
    if conversation_id:
        history_str = f"{run_id or 'unscoped'}|{conversation_id}"
    elif "messages" in data:
        messages = data.get("messages", [])
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user:
            history_str = first_user.get("content", "")
        else:
            history_str = ""
    else:
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)

        if "User:" in prompt:
            start = prompt.find("User:")
            end = prompt.find("Assistant:", start)
            if end > start:
                history_str = prompt[start:end].strip()
            else:
                history_str = prompt[start:start+200]
        else:
            history_str = prompt[:200]

    # PPD experiment: salt the hash so back-to-back runs of the same task
    # instance (identical first user message) don't collide on the in-memory
    # turn counter / server affinity. Set PPD_PROXY_EPOCH once per stack start.
    history_str += os.environ.get("PPD_PROXY_EPOCH", "")

    return hashlib.md5(history_str.encode()).hexdigest()


def get_conversation_info(conv_hash: str) -> Tuple[int, Dict[str, str]]:
    """Get turn number and assigned servers for a conversation."""
    with conversation_lock:
        if conv_hash in conversation_state:
            turn_count, _, assigned = conversation_state[conv_hash]
            return turn_count, assigned.copy()
        return 0, {}


def update_conversation(
    conv_hash: str,
    assigned: Optional[Dict[str, str]] = None
) -> Tuple[int, Dict[str, str]]:
    """Update conversation state and return new turn number."""
    with conversation_lock:
        # Clean up expired conversations
        current_time = time.time()
        expired = [k for k, v in conversation_state.items()
                   if current_time - v[1] > CONVERSATION_TTL]
        for k in expired:
            del conversation_state[k]

        if conv_hash in conversation_state:
            turn_count, _, existing_assigned = conversation_state[conv_hash]
            turn_count += 1
            # Merge new assignments with existing
            if assigned:
                existing_assigned.update(assigned)
            final_assigned = existing_assigned
        else:
            turn_count = 1
            final_assigned = assigned or {}

        conversation_state[conv_hash] = (turn_count, current_time, final_assigned)
        return turn_count, final_assigned


def select_servers(
    conv_hash: str,
    current_turn: int,
    assigned: Dict[str, str],
    use_ppd_dynamic: bool = False,
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Select servers based on configuration and cache affinity.

    Args:
        conv_hash: Conversation hash for cache affinity
        current_turn: Current turn number (1, 2, 3, ...)
        assigned: Previously assigned servers for this conversation
        use_ppd_dynamic: If True, enable PPD mode for Turn 2+ even on D servers
                         (determined by PPD decision engine)

    Returns: (prefill_addr, decode_addr, replica_addr, routing_mode)
    routing_mode: "pd", "ppd", "ppd_direct", "ppd_dynamic", "replica"
    - "pd": Standard prefill-decode mode with KV transfer every turn
    - "ppd": PPD mode Turn 1 (KV transfer to pD server)
    - "ppd_direct": PPD mode Turn 2+ (direct to pD, uses prefix cache)
    - "ppd_dynamic": Dynamic PPD mode (Turn 2+ direct to D, decision engine enabled)
    - "replica": Request routed to replica server

    Routing Strategy:
    1. Pure Replica (4R): All requests to replica, hash-based selection
    2. Pure Disagg (1P_3D, 2P_2pD, etc.): All requests through P->D/pD
    3. Hybrid (1R_1P_2pD, etc.): Distribute requests between replica and disagg
       - Use conversation hash to deterministically assign mode
       - Once assigned, cache affinity keeps same mode for Turn 2+
    4. Dynamic PPD: When enabled, Turn 2+ requests can bypass prefill even on D servers
    """
    prefill_addr = None
    decode_addr = None
    replica_addr = None
    routing_mode = "pd"

    # Check available instances
    with prefill_cv:
        prefill_list = list(prefill_instances.keys())
    with decode_cv:
        decode_list = list(decode_instances.keys())
    with ppd_decode_cv:
        ppd_list = list(ppd_decode_instances.keys())
    with replica_cv:
        replica_list = list(replica_instances.keys())

    has_prefill = len(prefill_list) > 0
    has_decode = len(decode_list) > 0 or len(ppd_list) > 0
    has_replica = len(replica_list) > 0

    # Check if this conversation already has an assigned mode
    cache_affinity_hit = False
    if assigned:
        if assigned.get("replica") and assigned["replica"] in replica_list:
            # Continue with replica mode
            routing_mode = "replica"
            replica_addr = assigned["replica"]
            cache_affinity_hit = True

        elif assigned.get("decode"):
            # Continue with disagg mode
            all_decode = decode_list + ppd_list
            if assigned["decode"] in all_decode:
                decode_addr = assigned["decode"]
                if assigned.get("prefill") and assigned["prefill"] in prefill_list:
                    prefill_addr = assigned["prefill"]
                else:
                    prefill_addr = prefill_list[hash(conv_hash) % len(prefill_list)] if prefill_list else None

                if decode_addr in ppd_list:
                    routing_mode = "ppd_direct" if current_turn > 1 else "ppd"
                else:
                    routing_mode = "pd"
                cache_affinity_hit = True

    # New conversation - determine routing mode (only if no cache affinity hit)
    if not cache_affinity_hit:
        if has_replica and not has_prefill and not has_decode:
            # Pure replica mode (4R)
            routing_mode = "replica"
            idx = hash(conv_hash) % len(replica_list)
            replica_addr = replica_list[idx]

        elif has_prefill and has_decode and not has_replica:
            # Pure disaggregated mode (1P_3D, 2P_2pD, 1P_2D_1pD, etc.)
            # Select prefill
            idx = hash(conv_hash) % len(prefill_list)
            prefill_addr = prefill_list[idx]

            # Select decode - distribute across ALL decode servers (D + pD) by capacity
            # This ensures mixed configs like 1P_2D_1pD use all 3 servers, not just pD
            all_decode = decode_list + ppd_list
            decode_idx = hash(conv_hash) % len(all_decode)
            decode_addr = all_decode[decode_idx]

            if decode_addr in ppd_list:
                routing_mode = "ppd"
            else:
                routing_mode = "pd"

        elif has_prefill and has_decode and has_replica:
            # Hybrid mode (1R_1P_2pD, 2R_1P_1D, etc.)
            # Distribute requests between replica and disagg based on capacity
            # Calculate total capacity slots
            num_replica_slots = len(replica_list)
            all_decode = decode_list + ppd_list
            num_disagg_slots = len(all_decode)  # One slot per decode server
            total_slots = num_replica_slots + num_disagg_slots

            # Use hash to select slot
            slot = hash(conv_hash) % total_slots

            if slot < num_replica_slots:
                # Route to replica
                routing_mode = "replica"
                replica_addr = replica_list[slot]
            else:
                # Route to disagg
                decode_idx = slot - num_replica_slots
                if decode_idx < len(ppd_list):
                    decode_addr = ppd_list[decode_idx]
                    routing_mode = "ppd"
                else:
                    decode_addr = decode_list[decode_idx - len(ppd_list)]
                    routing_mode = "pd"

                # Select prefill
                prefill_addr = prefill_list[hash(conv_hash) % len(prefill_list)]

        elif has_replica:
            # Only replica available
            routing_mode = "replica"
            idx = hash(conv_hash) % len(replica_list)
            replica_addr = replica_list[idx]

    # For pD nodes: Turn 2+ always goes directly to pD (static x=1)
    # pD nodes always use local prefix cache, no decision engine needed
    if routing_mode == "ppd" and current_turn > 1:
        routing_mode = "ppd_direct"

    # For D nodes with dynamic PPD (--enable-ppd on D configs):
    # Decision engine determines if Turn 2+ should bypass prefill
    # and use local prefix cache instead of routing through P
    if routing_mode == "pd" and use_ppd_dynamic and current_turn > 1:
        routing_mode = "ppd_dynamic"

    return prefill_addr, decode_addr, replica_addr, routing_mode


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


async def forward_request(
    url: str, data: dict, request_id: str, response_meta: Optional[Dict[str, Any]] = None
):
    """Forward request to vLLM with streaming support."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response_meta is not None:
                response_meta["status"] = response.status
            # Always yield the response content, even for error status
            async for chunk_bytes in response.content.iter_any():
                if chunk_bytes:
                    yield chunk_bytes


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """Handle inference requests with automatic routing."""
    t_enter = time.time()
    t_prefill_done = None
    response_usage: Dict[str, Any] = {}
    feature_metadata: Dict[str, Any] = {}
    decision_qps_est = 0.0
    prefill_response_meta: Dict[str, Any] = {}
    decode_response_meta: Dict[str, Any] = {}
    profile_context = {
        "schema_version": 2,
        "run_id": request.headers.get("X-PPD-Run-ID", "unscoped"),
        "episode_id": request.headers.get("X-PPD-Episode-ID"),
        "conversation_id": request.headers.get("X-PPD-Conversation-ID"),
        "step_id": request.headers.get("X-PPD-Step-ID"),
        "client_request_id": request.headers.get("X-PPD-Client-Request-ID"),
        "request_phase": request.headers.get("X-PPD-Request-Phase", "default"),
        "request_attempt": request.headers.get("X-PPD-Request-Attempt", "1"),
    }

    def _timing_record(t_first: Optional[float]) -> Dict[str, Any]:
        record = {
            **profile_context,
            "ts_enter": round(t_enter, 4),
            "ts_route": round(t_route, 4),
            "ts_prefill_done": round(t_prefill_done, 4) if t_prefill_done else None,
            "ts_first_chunk": round(t_first, 4) if t_first else None,
            "ts_exit": round(time.time(), 4),
            "route": routing_mode,
            "turn": turn_number,
            "conv": conv_hash[:8],
            "request_id": request_id,
            # Backward-compatible aliases used by existing analyzers.
            "in_tokens": input_tokens,
            "out_tokens": output_tokens,
            "ctx_tokens": context_tokens,
            # Explicit names distinguish decision-time estimates from the exact
            # usage fields populated after vLLM tokenizes/generates the request.
            "decision_input_tokens_est": input_tokens,
            "decision_output_tokens_est": output_tokens,
            "decision_context_tokens_est": context_tokens,
            "decision_qps_est": decision_qps_est,
            "ppd_mode_enabled": ENABLE_PPD_MODE,
            "ppd_decision": ppd_decision,
            **feature_metadata,
            "prefill_addr": prefill_addr,
            "decode_addr": decode_addr,
            "t_route_s": round(t_route - t_enter, 4),
            "t_prefill_s": round(t_prefill_done - t_route, 4) if t_prefill_done else None,
            # First transport chunk is not necessarily the first generated token.
            "ttfc_s": round(t_first - t_enter, 4) if t_first else None,
            "prefill_status": prefill_response_meta.get("status"),
            "decode_status": decode_response_meta.get("status"),
            "t_total_s": round(time.time() - t_enter, 4),
        }
        record.update(response_usage)
        return record

    try:
        data = await request.get_json()
        conv_hash = get_conversation_hash(
            data,
            conversation_id=profile_context["conversation_id"],
            run_id=profile_context["run_id"],
        )

        # Get conversation state
        current_turn, assigned = get_conversation_info(conv_hash)

        # Dynamic PPD decision
        use_ppd_dynamic = False
        ppd_decision = None
        input_tokens, output_tokens, context_tokens, feature_metadata = (
            _extract_request_features_with_metadata(data)
        )

        with stats_lock:
            decision_qps_est = stats.get("total_requests", 0) / max(
                1, time.time() - stats.get("start_time", time.time())
            )

        if ENABLE_PPD_MODE and PPD_DECISION_ENGINE is not None:
            # Call decision engine (returns False for Turn 1).
            ppd_decision = PPD_DECISION_ENGINE.should_use_ppd(
                turn=current_turn + 1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                current_qps=decision_qps_est,
                context_tokens=context_tokens,
            )
            use_ppd_dynamic = ppd_decision

        # Select servers
        prefill_addr, decode_addr, replica_addr, routing_mode = select_servers(
            conv_hash, current_turn + 1, assigned, use_ppd_dynamic=use_ppd_dynamic
        )
        t_route = time.time()

        # Update conversation state
        new_assigned = {}
        if prefill_addr:
            new_assigned["prefill"] = prefill_addr
        if decode_addr:
            new_assigned["decode"] = decode_addr
        if replica_addr:
            new_assigned["replica"] = replica_addr

        turn_number, final_assigned = update_conversation(conv_hash, new_assigned)

        # Update statistics
        with stats_lock:
            stats["total_requests"] += 1
            if turn_number == 1:
                stats["turn1_requests"] += 1
            else:
                stats["turn2plus_requests"] += 1
                if assigned:
                    stats["cache_affinity_hits"] += 1
                else:
                    stats["cache_affinity_misses"] += 1

            if routing_mode == "pd":
                stats["pd_requests"] += 1
            elif routing_mode == "ppd":
                stats["ppd_requests"] += 1
            elif routing_mode == "ppd_direct":
                stats["ppd_direct_requests"] += 1
            elif routing_mode == "ppd_dynamic":
                stats["ppd_dynamic_requests"] += 1
            elif routing_mode == "replica":
                stats["replica_requests"] += 1

        # Log PPD decision for analysis
        if ENABLE_PPD_MODE and ppd_decision is not None:
            log_ppd_decision(
                conv_hash=conv_hash,
                turn=turn_number,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_tokens=context_tokens,
                current_qps=decision_qps_est,
                decision=ppd_decision,
                routing_mode=routing_mode,
            )

        is_streaming = data.get("stream", False)

        # Route based on mode
        if routing_mode == "replica":
            # Direct to replica
            request_id = f"replica_{uuid.uuid4().hex}"
            target_url = f"http://{replica_addr}{request.path}"

            print(f"[PROXY] Replica: turn={turn_number}, conv={conv_hash[:8]}, "
                  f"-> R:{replica_addr}")

        elif routing_mode == "ppd_direct":
            # Direct to pD (no prefill needed)
            request_id = f"ppd_direct_{uuid.uuid4().hex}"
            target_url = f"http://{decode_addr}{request.path}"

            print(f"[PROXY] PPD-Direct: turn={turn_number}, conv={conv_hash[:8]}, "
                  f"-> pD:{decode_addr}")

        elif routing_mode == "ppd_dynamic":
            # Dynamic PPD mode - direct to D (treating it like pD for Turn 2+)
            request_id = f"ppd_dynamic_{uuid.uuid4().hex}"
            target_url = f"http://{decode_addr}{request.path}"

            print(f"[PROXY] PPD-Dynamic: turn={turn_number}, conv={conv_hash[:8]}, "
                  f"in={input_tokens}, out={output_tokens}, ctx={context_tokens}, "
                  f"-> D:{decode_addr}")

        else:
            # PD or PPD mode - need prefill first
            with prefill_cv:
                prefill_zmq = prefill_instances.get(prefill_addr, ("", 0))[0]
            with decode_cv:
                decode_zmq = decode_instances.get(decode_addr, ("", 0))[0]
            if not decode_zmq:
                with ppd_decode_cv:
                    decode_zmq = ppd_decode_instances.get(decode_addr, ("", 0))[0]

            request_id = (
                f"___prefill_addr_{prefill_zmq}___decode_addr_"
                f"{decode_zmq}_{uuid.uuid4().hex}"
            )

            # Prefill request
            prefill_request = data.copy()
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            print(f"[PROXY] {routing_mode.upper()}: turn={turn_number}, "
                  f"conv={conv_hash[:8]}, P:{prefill_addr} -> D:{decode_addr}")

            # Execute prefill
            async for _ in forward_request(
                f"http://{prefill_addr}{request.path}",
                prefill_request,
                request_id,
                prefill_response_meta,
            ):
                continue
            t_prefill_done = time.time()  # P-node prefill complete (KV transfer async)

            target_url = f"http://{decode_addr}{request.path}"

        # Execute decode/generation
        if is_streaming:
            async def _timed_stream():
                # PPD profiling: record first transport chunk (TTFC) and total.
                t_first = None
                async for chunk in forward_request(
                    target_url, data, request_id, decode_response_meta
                ):
                    if t_first is None:
                        t_first = time.time()
                    yield chunk
                _write_timing(_timing_record(t_first))

            return Response(
                _timed_stream(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked",
                }
            )
        else:
            response_bytes = b""
            async for chunk in forward_request(
                target_url, data, request_id, decode_response_meta
            ):
                response_bytes += chunk
            try:
                response_json = json.loads(response_bytes)
                usage = response_json.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                response_usage.update({
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cached_tokens": details.get("cached_tokens"),
                })
                if response_json.get("error"):
                    response_usage["upstream_error"] = str(response_json["error"])[:1000]
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            _write_timing(_timing_record(None))
            return response_bytes, decode_response_meta.get("status", 502), {
                "Content-Type": "application/json"
            }

    except Exception as e:
        import traceback
        print(f"[PROXY] Error: {e}")
        traceback.print_exc()
        _write_timing({
            **profile_context,
            "ts_enter": round(t_enter, 4),
            "ts_exit": round(time.time(), 4),
            "route": None,
            "error": f"{type(e).__name__}: {e}",
        })
        return {"error": str(e)}, 500


@app.route("/status", methods=["GET"])
async def get_status():
    """Get proxy status."""
    with prefill_cv:
        p_list = list(prefill_instances.keys())
    with decode_cv:
        d_list = list(decode_instances.keys())
    with ppd_decode_cv:
        pd_list = list(ppd_decode_instances.keys())
    with replica_cv:
        r_list = list(replica_instances.keys())

    with stats_lock:
        current_stats = stats.copy()

    return {
        "config": CONFIG_NAME,
        "instances": {
            "prefill": p_list,
            "decode": d_list,
            "ppd_decode": pd_list,
            "replica": r_list,
        },
        "conversations": len(conversation_state),
        "stats": current_stats,
    }


@app.route("/stats", methods=["GET"])
async def get_stats():
    """Get detailed statistics."""
    with stats_lock:
        return stats.copy()


@app.route("/conversations", methods=["GET"])
async def get_conversations():
    """Get active conversations."""
    with conversation_lock:
        return {
            "count": len(conversation_state),
            "conversations": [
                {
                    "hash": k[:8],
                    "turn": v[0],
                    "age": int(time.time() - v[1]),
                    "assigned": v[2],
                }
                for k, v in list(conversation_state.items())[:50]
            ]
        }


@app.route("/ppd-decisions", methods=["GET"])
async def get_ppd_decisions():
    """Get PPD decision log for analysis."""
    with PPD_LOG_LOCK:
        all_decisions = PPD_DECISION_LOG.copy()

    # Compute summary statistics using ALL decisions (not just last 100)
    total = len(all_decisions)
    ppd_count = sum(1 for d in all_decisions if d.get("ppd_decision"))
    pd_count = total - ppd_count

    # Only return recent decisions for display (last 100)
    recent_decisions = all_decisions[-100:] if len(all_decisions) > 100 else all_decisions

    return {
        "ppd_mode_enabled": ENABLE_PPD_MODE,
        "total_logged": total,
        "ppd_decisions": ppd_count,
        "pd_decisions": pd_count,
        "ppd_ratio": ppd_count / max(1, total),
        "recent_decisions": recent_decisions,
        "engine_stats": PPD_DECISION_ENGINE.get_decision_stats() if PPD_DECISION_ENGINE else None,
    }


@app.route("/ppd-decisions/reset", methods=["POST"])
async def reset_ppd_decisions():
    """Reset PPD decision log. Used to clear stats between test runs."""
    global PPD_DECISION_LOG

    with PPD_LOG_LOCK:
        old_count = len(PPD_DECISION_LOG)
        PPD_DECISION_LOG = []

    # Also reset engine stats if available
    if PPD_DECISION_ENGINE is not None:
        PPD_DECISION_ENGINE.stats = type(PPD_DECISION_ENGINE.stats)()

    print(f"[PROXY] PPD decision log reset (cleared {old_count} entries)")

    return {
        "status": "reset",
        "cleared_entries": old_count,
    }


@app.route("/ppd-performance", methods=["GET"])
async def get_ppd_performance():
    """Get PPD performance comparison from the decision engine."""
    if PPD_DECISION_ENGINE is None:
        return {"error": "PPD mode not enabled"}, 400

    return {
        "base_config": PPD_DECISION_ENGINE.base_config,
        "ppd_config": PPD_DECISION_ENGINE.ppd_config,
        "comparison": PPD_DECISION_ENGINE.get_performance_comparison(),
    }


def main():
    global CONFIG_NAME, ENABLE_PPD_MODE, PPD_DECISION_ENGINE

    parser = argparse.ArgumentParser(description="Comprehensive Proxy Server")
    parser.add_argument("--config", type=str, default="2P_2D",
                        help="Configuration name (e.g., 2P_2D, 1P_3D, 4R)")
    parser.add_argument("--http-port", type=int, default=10001,
                        help="HTTP port for the proxy")
    parser.add_argument("--zmq-port", type=int, default=30001,
                        help="ZMQ port for service discovery")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")

    # Dynamic PPD mode arguments
    parser.add_argument("--enable-ppd-mode", action="store_true",
                        help="Enable dynamic PPD routing with decision engine")
    parser.add_argument("--ppd-benchmark-path", type=str,
                        default=os.path.join(PROJECT_DIR, "results", "comprehensive"),
                        help="Path to benchmark results for PPD decision engine")
    parser.add_argument("--w-ttft", type=float, default=1.0,
                        help="Weight for TTFT improvement in PPD decision (default: 1.0)")
    parser.add_argument("--w-tpot", type=float, default=1.0,
                        help="Weight for TPOT degradation penalty in PPD decision (default: 1.0)")

    args = parser.parse_args()
    CONFIG_NAME = args.config

    # Get decode port configuration
    decode_config = DECODE_PORTS_BY_CONFIG.get(CONFIG_NAME, {"D": [], "pD": []})
    d_ports = decode_config.get("D", [])
    ppd_ports = decode_config.get("pD", [])

    print(f"[PROXY] Starting Comprehensive Proxy")
    print(f"[PROXY] Config: {CONFIG_NAME}")
    print(f"[PROXY] Decode ports: D={d_ports}, pD={ppd_ports}")
    print(f"[PROXY] HTTP: {args.host}:{args.http_port}")
    print(f"[PROXY] ZMQ: {args.host}:{args.zmq_port}")

    # Initialize dynamic PPD mode if enabled
    if args.enable_ppd_mode:
        print(f"[PROXY] Dynamic PPD mode: ENABLED")
        print(f"[PROXY] Benchmark path: {args.ppd_benchmark_path}")
        print(f"[PROXY] Weights: w_ttft={args.w_ttft}, w_tpot={args.w_tpot}")

        try:
            from ppd.optimizer.ppd_decision_engine import PPDDecisionEngine
            # PPD mode is designed for D-node configs (e.g., 3P_1D)
            # The decision engine compares PD (base) vs pD (ppd) configs
            PPD_DECISION_ENGINE = PPDDecisionEngine(
                benchmark_data_path=args.ppd_benchmark_path,
                base_config=CONFIG_NAME,
                w_ttft=args.w_ttft,
                w_tpot=args.w_tpot,
            )
            ENABLE_PPD_MODE = True
            print(f"[PROXY] PPD Decision Engine initialized successfully")
            print(f"[PROXY] Comparing: {PPD_DECISION_ENGINE.base_config} vs {PPD_DECISION_ENGINE.ppd_config}")
        except Exception as e:
            print(f"[PROXY] ERROR: Failed to initialize PPD Decision Engine: {e}")
            print(f"[PROXY] Continuing without dynamic PPD mode")
            ENABLE_PPD_MODE = False
    else:
        print(f"[PROXY] Dynamic PPD mode: DISABLED")

    # Initialize stats start time
    with stats_lock:
        stats["start_time"] = time.time()

    # Start service discovery
    start_service_discovery(args.host, args.zmq_port)

    # Register static replicas (if any in this config)
    register_static_replicas()

    # Run the proxy
    app.run(host=args.host, port=args.http_port)


if __name__ == "__main__":
    main()
