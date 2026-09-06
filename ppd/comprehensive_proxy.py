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
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import msgpack
import zmq
from quart import Quart, request, Response

# Add project root to path for optimizer imports
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from ppd.routing_contract import ROUTING_MODES, RoutingContract


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
# Key: (state_key, client_request_id). A record freezes the logical turn,
# decision and server assignment across transport/tool-choice retries.
logical_request_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
conversation_lock = threading.Lock()

# In-flight proxy counters are request-level (not TTL-resident episode counts).
active_request_lock = threading.Lock()
active_requests_by_phase: Dict[str, int] = defaultdict(int)
active_decode_requests: Dict[str, int] = defaultdict(int)
active_prefill_requests: Dict[str, int] = defaultdict(int)
active_decode_conversations: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

# Logical arrivals support both the paper-compatible lifetime average and
# observable sliding-window rates. Entries older than 60 seconds are pruned.
logical_arrivals: deque[float] = deque()

# Statistics
stats = {
    # Backward-compatible total now means logical requests, not HTTP attempts.
    "total_requests": 0,
    "logical_requests": 0,
    "physical_requests": 0,
    "retry_requests": 0,
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

# Stable by default. ``legacy_hash`` exists only to reproduce pre-fix runs.
ROUTING_CONTRACT = RoutingContract()

# Dynamic PPD mode configuration
ENABLE_PPD_MODE = False
PPD_DECISION_ENGINE = None
PPD_DECISION_LOG: List[dict] = []  # Log of PPD decisions for analysis
PPD_LOG_LOCK = threading.Lock()

# Optional tokenizer used only for the newly appended message suffix. Encoding
# the suffix avoids re-tokenizing a 100K+ context on every Agent turn.
_FEATURE_TOKENIZER: Any = None
_FEATURE_TOKENIZER_PATH: Optional[str] = None
_FEATURE_TOKENIZER_ERROR: Optional[str] = None

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


def _template_token_count(
    tokenizer: Any,
    messages: List[dict],
    *,
    tools: Optional[List[dict]],
    add_generation_prompt: bool,
) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids", [])
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def _validate_suffix_tokenizer(tokenizer: Any) -> None:
    """Prove that suffix-only tokenization matches full-template subtraction."""
    tools = [{
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Probe tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    context = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "reason",
            "tool_calls": [{
                "id": "call-probe",
                "type": "function",
                "function": {"name": "probe", "arguments": "{}"},
            }],
        },
    ]
    append = [{
        "role": "tool",
        "tool_call_id": "call-probe",
        "name": "probe",
        "content": "result",
    }]
    full = _template_token_count(
        tokenizer, context + append, tools=tools, add_generation_prompt=True
    )
    prefix = _template_token_count(
        tokenizer, context, tools=tools, add_generation_prompt=False
    )
    suffix = _template_token_count(
        tokenizer, append, tools=None, add_generation_prompt=True
    )
    if full - prefix != suffix:
        raise ValueError(
            "chat template is not suffix-additive for assistant/tool turns "
            f"(full-prefix={full - prefix}, suffix={suffix})"
        )


def configure_feature_tokenizer(path: Optional[str]) -> bool:
    """Load and validate the tokenizer used for exact append-token features."""
    global _FEATURE_TOKENIZER, _FEATURE_TOKENIZER_PATH, _FEATURE_TOKENIZER_ERROR
    _FEATURE_TOKENIZER = None
    _FEATURE_TOKENIZER_PATH = path
    _FEATURE_TOKENIZER_ERROR = None
    if not path:
        return False
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("tokenizer has no chat_template")
        _validate_suffix_tokenizer(tokenizer)
        _FEATURE_TOKENIZER = tokenizer
        print(f"[PROXY] Exact append tokenizer enabled: {path}")
        return True
    except Exception as exc:
        _FEATURE_TOKENIZER_ERROR = f"{type(exc).__name__}: {exc}"
        print(
            "[PROXY] WARNING: exact append tokenizer unavailable; "
            f"falling back to chars/4: {_FEATURE_TOKENIZER_ERROR}"
        )
        return False


def _exact_append_token_count(messages: List[dict]) -> Optional[int]:
    """Tokenize only the suffix that is new after the latest assistant turn."""
    global _FEATURE_TOKENIZER_ERROR
    if _FEATURE_TOKENIZER is None:
        return None
    if not messages:
        return 0
    try:
        return _template_token_count(
            _FEATURE_TOKENIZER,
            messages,
            tools=None,
            add_generation_prompt=True,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if error != _FEATURE_TOKENIZER_ERROR:
            print(
                "[PROXY] WARNING: append suffix tokenization failed; "
                f"falling back to chars/4 for this request: {error}"
            )
        _FEATURE_TOKENIZER_ERROR = error
        return None


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
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    token_count += estimate_token_count(reasoning)
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
        fallback_input_tokens = sum(
            estimate_message_token_count(msg) for msg in append_messages
        )
        exact_input_tokens = _exact_append_token_count(append_messages)
        if exact_input_tokens is None:
            current_input_tokens = fallback_input_tokens
            token_estimator = "chars_per_4_with_chat_overhead_v1"
        else:
            current_input_tokens = exact_input_tokens
            token_estimator = "hf_suffix_chat_template_v1"
        last_assistant_message = (
            messages[last_assistant] if last_assistant is not None else {}
        )
        metadata = {
            "feature_source": source,
            "feature_token_estimator": token_estimator,
            "context_token_estimator": "chars_per_4_with_chat_overhead_v1",
            "feature_tokenizer_path": _FEATURE_TOKENIZER_PATH,
            "feature_tokenizer_error": _FEATURE_TOKENIZER_ERROR,
            "append_tokens_fallback_est": fallback_input_tokens,
            "assistant_reasoning_replayed": bool(
                last_assistant_message.get("reasoning")
                or last_assistant_message.get("reasoning_content")
            ),
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
    decision_metadata: Optional[Dict[str, Any]] = None,
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
        "decision_source": (decision_metadata or {}).get("decision_source"),
        "selected_calibration_cell": (decision_metadata or {}).get(
            "selected_calibration_cell"
        ),
        "decision_metadata": dict(decision_metadata or {}),
    }

    with PPD_LOG_LOCK:
        PPD_DECISION_LOG.append(entry)
        # Keep only last 10000 entries
        if len(PPD_DECISION_LOG) > 10000:
            PPD_DECISION_LOG = PPD_DECISION_LOG[-10000:]


def get_conversation_hash(
    data: dict, *, conversation_id: Optional[str] = None, run_id: Optional[str] = None
) -> str:
    """Generate the run-scoped state key for turn count/cache affinity."""
    if conversation_id:
        history_str = f"{run_id or 'unscoped'}|{conversation_id}"
    elif "messages" in data:
        messages = data.get("messages", [])
        first_user = next(
            (message for message in messages
             if isinstance(message, dict) and message.get("role") == "user"),
            None,
        )
        history_str = _message_value_text(first_user.get("content")) if first_user else ""
    else:
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(part) for part in prompt)
        if "User:" in prompt:
            start = prompt.find("User:")
            end = prompt.find("Assistant:", start)
            history_str = prompt[start:end].strip() if end > start else prompt[start:start + 200]
        else:
            history_str = prompt[:200]

    # State must be isolated across stack epochs; assignment must not be. The
    # latter is handled separately by get_routing_identity/RoutingContract.
    epoch = os.environ.get("PPD_PROXY_EPOCH", "")
    return hashlib.md5(f"{history_str}|epoch={epoch}".encode()).hexdigest()


def get_routing_identity(
    data: dict, *, explicit_routing_key: Optional[str] = None
) -> Tuple[str, str]:
    """Return a cross-run assignment key and its provenance.

    Instrumented agents send SWE instance_id + rollout index explicitly. The
    first-user/prompt fallback is deterministic for legacy clients, but cannot
    be used with a strict frozen map unless that exact key is present.
    """
    if explicit_routing_key:
        return str(explicit_routing_key), "header"
    if "messages" in data:
        messages = data.get("messages") or []
        first_user = next(
            (message for message in messages
             if isinstance(message, dict) and message.get("role") == "user"),
            None,
        )
        if first_user is not None:
            return _message_value_text(first_user.get("content")), "first_user_fallback"
    prompt = data.get("prompt", "")
    return _message_value_text(prompt), "prompt_fallback"


def get_conversation_info(conv_hash: str) -> Tuple[int, Dict[str, str]]:
    """Get turn number and assigned servers for a conversation."""
    with conversation_lock:
        if conv_hash in conversation_state:
            turn_count, _, assigned = conversation_state[conv_hash]
            return turn_count, assigned.copy()
        return 0, {}


def _cleanup_conversation_state(current_time: float) -> None:
    expired = [
        key for key, value in conversation_state.items()
        if current_time - value[1] > CONVERSATION_TTL
    ]
    for key in expired:
        del conversation_state[key]
        for request_key in [item for item in logical_request_state if item[0] == key]:
            del logical_request_state[request_key]


def update_conversation(
    conv_hash: str,
    assigned: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str]]:
    """Advance one *logical* conversation turn and return its assignment."""
    with conversation_lock:
        current_time = time.time()
        _cleanup_conversation_state(current_time)
        if conv_hash in conversation_state:
            turn_count, _, existing_assigned = conversation_state[conv_hash]
            turn_count += 1
            final_assigned = existing_assigned.copy()
            if assigned:
                final_assigned.update(assigned)
        else:
            turn_count = 1
            final_assigned = dict(assigned or {})
        conversation_state[conv_hash] = (turn_count, current_time, final_assigned)
        return turn_count, final_assigned.copy()


def get_logical_request(
    conv_hash: str, client_request_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Return and touch a previously frozen route for an HTTP retry."""
    if not client_request_id:
        return None
    with conversation_lock:
        _cleanup_conversation_state(time.time())
        record = logical_request_state.get((conv_hash, client_request_id))
        if record is None:
            return None
        record["last_access"] = time.time()
        record["physical_attempts"] = int(record.get("physical_attempts", 1)) + 1
        result = dict(record)
        result["assigned"] = dict(record.get("assigned") or {})
        result["routing_metadata"] = dict(record.get("routing_metadata") or {})
        result["decision_metadata"] = dict(record.get("decision_metadata") or {})
        result["decision_rate"] = dict(record.get("decision_rate") or {})
        return result


def save_logical_request(
    conv_hash: str, client_request_id: Optional[str], record: Dict[str, Any]
) -> None:
    if not client_request_id:
        return
    stored = dict(record)
    stored["last_access"] = time.time()
    stored["physical_attempts"] = 1
    stored["assigned"] = dict(record.get("assigned") or {})
    stored["routing_metadata"] = dict(record.get("routing_metadata") or {})
    stored["decision_metadata"] = dict(record.get("decision_metadata") or {})
    stored["decision_rate"] = dict(record.get("decision_rate") or {})
    with conversation_lock:
        logical_request_state[(conv_hash, client_request_id)] = stored


def record_request_arrival(now: float, *, is_logical: bool) -> Dict[str, Any]:
    """Record one physical attempt and return explicitly-defined QPS views."""
    with stats_lock:
        stats["physical_requests"] += 1
        if not is_logical:
            stats["retry_requests"] += 1
        started = float(stats.get("start_time") or now)
        elapsed = max(1.0, now - started)
        logical_before = int(stats.get("logical_requests", 0))
        # Preserve the original Dynamic input: prior logical requests divided
        # by proxy uptime (with the original one-second minimum denominator).
        decision_qps = logical_before / elapsed
        if is_logical:
            stats["logical_requests"] = logical_before + 1
            stats["total_requests"] = logical_before + 1
            logical_arrivals.append(now)
        cutoff = now - 60.0
        while logical_arrivals and logical_arrivals[0] < cutoff:
            logical_arrivals.popleft()
        result: Dict[str, Any] = {
            "decision_qps_est": decision_qps,
            "decision_qps_source": "prior_logical_requests/proxy_uptime_min_1s",
            "decision_qps_numerator": logical_before,
            "decision_qps_denominator_s": elapsed,
            "logical_lifetime_qps": int(stats.get("logical_requests", 0)) / elapsed,
        }
        for window in (1, 5, 10, 60):
            count = sum(timestamp >= now - window for timestamp in logical_arrivals)
            denominator = max(1e-6, min(float(window), max(now - started, 1e-6)))
            result[f"logical_window_qps_{window}s"] = count / denominator
            result[f"logical_window_count_{window}s"] = count
        return result


def _record_affinity(
    metadata: Dict[str, Any], role: str, address: str, candidates: List[str]
) -> None:
    metadata[f"{role}_slot"] = candidates.index(address)
    metadata[f"{role}_assignment_source"] = "conversation_affinity"


def select_servers(
    conv_hash: str,
    current_turn: int,
    assigned: Dict[str, str],
    use_ppd_dynamic: bool = False,
    *,
    routing_key: Optional[str] = None,
    include_metadata: bool = False,
):
    """Select sorted server slots using the configured routing contract."""
    key = routing_key or conv_hash
    prefill_addr = None
    decode_addr = None
    replica_addr = None
    routing_mode = "pd"

    with prefill_cv:
        prefill_list = ROUTING_CONTRACT.candidate_addresses(prefill_instances.keys())
    with decode_cv:
        decode_list = ROUTING_CONTRACT.candidate_addresses(decode_instances.keys())
    with ppd_decode_cv:
        ppd_list = ROUTING_CONTRACT.candidate_addresses(ppd_decode_instances.keys())
    with replica_cv:
        replica_list = ROUTING_CONTRACT.candidate_addresses(replica_instances.keys())
    all_decode = ROUTING_CONTRACT.candidate_addresses(decode_list + ppd_list)

    metadata: Dict[str, Any] = {
        "routing_contract_version": ROUTING_CONTRACT.descriptor()["contract_version"],
        "routing_mode_config": ROUTING_CONTRACT.mode,
        "routing_seed": ROUTING_CONTRACT.seed,
        "routing_map_fingerprint_sha256": ROUTING_CONTRACT.map_fingerprint,
        "routing_key_sha256": ROUTING_CONTRACT.key_hash(key),
        "prefill_candidates": prefill_list,
        "decode_candidates": all_decode,
        "replica_candidates": replica_list,
    }

    def choose(candidates: List[str], role: str) -> str:
        decision = ROUTING_CONTRACT.choose_slot(
            key, role, len(candidates), legacy_key=conv_hash
        )
        metadata[f"{role}_slot"] = decision.slot
        metadata[f"{role}_assignment_source"] = decision.source
        return candidates[decision.slot]

    has_prefill = bool(prefill_list)
    has_decode = bool(all_decode)
    has_replica = bool(replica_list)
    cache_affinity_hit = False

    if assigned:
        if assigned.get("replica") in replica_list:
            routing_mode = "replica"
            replica_addr = assigned["replica"]
            _record_affinity(metadata, "replica", replica_addr, replica_list)
            cache_affinity_hit = True
        elif assigned.get("decode") in all_decode:
            decode_addr = assigned["decode"]
            _record_affinity(metadata, "decode", decode_addr, all_decode)
            if assigned.get("prefill") in prefill_list:
                prefill_addr = assigned["prefill"]
                _record_affinity(metadata, "prefill", prefill_addr, prefill_list)
            elif prefill_list:
                prefill_addr = choose(prefill_list, "prefill")
            routing_mode = "ppd" if decode_addr in ppd_list else "pd"
            cache_affinity_hit = True

    if not cache_affinity_hit:
        if has_replica and not has_prefill and not has_decode:
            routing_mode = "replica"
            replica_addr = choose(replica_list, "replica")
        elif has_prefill and has_decode and not has_replica:
            prefill_addr = choose(prefill_list, "prefill")
            decode_addr = choose(all_decode, "decode")
            routing_mode = "ppd" if decode_addr in ppd_list else "pd"
        elif has_prefill and has_decode and has_replica:
            service_targets = [
                *(('replica', address) for address in replica_list),
                *(('decode', address) for address in all_decode),
            ]
            service = ROUTING_CONTRACT.choose_slot(
                key, "service", len(service_targets), legacy_key=conv_hash
            )
            metadata["service_slot"] = service.slot
            metadata["service_assignment_source"] = service.source
            kind, address = service_targets[service.slot]
            if kind == "replica":
                routing_mode = "replica"
                replica_addr = address
                _record_affinity(metadata, "replica", address, replica_list)
            else:
                decode_addr = address
                _record_affinity(metadata, "decode", address, all_decode)
                prefill_addr = choose(prefill_list, "prefill")
                routing_mode = "ppd" if decode_addr in ppd_list else "pd"
        elif has_replica:
            routing_mode = "replica"
            replica_addr = choose(replica_list, "replica")

    if routing_mode == "ppd" and current_turn > 1:
        routing_mode = "ppd_direct"
    if routing_mode == "pd" and use_ppd_dynamic and current_turn > 1:
        routing_mode = "ppd_dynamic"

    metadata["cache_affinity_hit"] = cache_affinity_hit
    metadata["selected_prefill_addr"] = prefill_addr
    metadata["selected_decode_addr"] = decode_addr
    metadata["selected_replica_addr"] = replica_addr
    result = (prefill_addr, decode_addr, replica_addr, routing_mode)
    return (*result, metadata) if include_metadata else result


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
ENABLE_ORCA_HEADERS = os.environ.get("PPD_ENABLE_ORCA_HEADERS", "true").lower() in {
    "1", "true", "yes", "on"
}


def _parse_orca_header(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    payload = value[5:] if value.startswith("JSON ") else value
    try:
        document = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return {}
    named = document.get("named_metrics") if isinstance(document, dict) else None
    return dict(named) if isinstance(named, dict) else {}


async def forward_request(
    url: str,
    data: dict,
    request_id: str,
    response_meta: Optional[Dict[str, Any]] = None,
):
    """Forward a request while recording HTTP/ORCA transport boundaries."""
    meta = response_meta if response_meta is not None else {}
    meta["http_start_ts"] = time.time()
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
                "X-Request-Id": request_id,
            }
            if ENABLE_ORCA_HEADERS:
                headers["endpoint-load-metrics-format"] = "JSON"
            async with session.post(url=url, json=data, headers=headers) as response:
                meta["status"] = response.status
                meta["http_headers_ts"] = time.time()
                meta["orca"] = _parse_orca_header(
                    response.headers.get("endpoint-load-metrics")
                )
                async for chunk_bytes in response.content.iter_any():
                    if chunk_bytes:
                        meta.setdefault("http_first_byte_ts", time.time())
                        yield chunk_bytes
    finally:
        meta["http_end_ts"] = time.time()


def _extract_upstream_json(
    response_bytes: bytes, response_meta: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    try:
        document = json.loads(response_bytes)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    request_metrics = document.get("ppd_request_metrics")
    if isinstance(request_metrics, dict):
        response_meta["request_metrics"] = request_metrics
    return document


def _phase_profile_fields(prefix: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    start = meta.get("http_start_ts")
    headers = meta.get("http_headers_ts")
    first = meta.get("http_first_byte_ts")
    end = meta.get("http_end_ts")
    orca = meta.get("orca") or {}
    fields: Dict[str, Any] = {
        f"{prefix}_status": meta.get("status"),
        f"{prefix}_http_start_ts": round(start, 4) if start else None,
        f"{prefix}_http_headers_ts": round(headers, 4) if headers else None,
        f"{prefix}_http_first_byte_ts": round(first, 4) if first else None,
        f"{prefix}_http_end_ts": round(end, 4) if end else None,
        f"{prefix}_http_s": round(end - start, 4) if start and end else None,
        f"{prefix}_http_headers_s": round(headers - start, 4) if start and headers else None,
        f"{prefix}_http_first_byte_s": round(first - start, 4) if start and first else None,
        f"{prefix}_orca_waiting_at_response": orca.get("num_requests_waiting"),
        f"{prefix}_orca_kv_usage_at_response": orca.get("kv_cache_usage_perc"),
        f"{prefix}_request_metrics": meta.get("request_metrics"),
    }
    request_metrics = meta.get("request_metrics") or {}
    for name in (
        "queue_s",
        "prefill_elapsed_s",
        "decode_elapsed_s",
        "inference_elapsed_s",
        "ttft_s",
        "e2e_s",
        "frontend_overhead_s",
        "num_preemptions",
        "prompt_tokens",
        "cached_prompt_tokens",
        "uncached_prompt_tokens",
        "completion_tokens",
        "requested_max_tokens",
    ):
        fields[f"{prefix}_engine_{name}"] = request_metrics.get(name)
    return fields


def _active_load_snapshot(
    prefill_addr: Optional[str], generation_addr: Optional[str], conv_hash: str
) -> Dict[str, Any]:
    """Capture exact proxy in-flight counts immediately before this request."""
    with active_request_lock:
        conversations = active_decode_conversations.get(generation_addr or "", {})
        generation_count = active_decode_requests.get(generation_addr or "", 0)
        return {
            "proxy_active_requests_before": sum(active_requests_by_phase.values()),
            "proxy_active_prefill_requests_before": sum(active_prefill_requests.values()),
            "proxy_active_decode_requests_before": sum(active_decode_requests.values()),
            "proxy_active_requests_on_prefill_before": active_prefill_requests.get(
                prefill_addr or "", 0
            ),
            "proxy_active_requests_on_generation_before": generation_count,
            # Backward-compatible alias: the generation node is a D/pD in the
            # configurations used by the Agent experiment.
            "proxy_active_requests_on_decode_before": generation_count,
            "proxy_other_inflight_conversations_on_decode": sum(
                1 for key, count in conversations.items()
                if key != conv_hash and count > 0
            ),
        }


def _active_request_change(
    phase: str, address: Optional[str], conv_hash: str, delta: int
) -> None:
    address = address or "unknown"
    with active_request_lock:
        active_requests_by_phase[phase] += delta
        if active_requests_by_phase[phase] <= 0:
            active_requests_by_phase.pop(phase, None)
        table = active_prefill_requests if phase == "prefill" else active_decode_requests
        table[address] += delta
        if table[address] <= 0:
            table.pop(address, None)
        if phase == "decode":
            conversations = active_decode_conversations[address]
            conversations[conv_hash] += delta
            if conversations[conv_hash] <= 0:
                conversations.pop(conv_hash, None)
            if not conversations:
                active_decode_conversations.pop(address, None)


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """Handle one physical HTTP attempt under the logical request contract."""
    t_enter = time.time()
    t_route = t_enter
    t_prefill_done = None
    response_usage: Dict[str, Any] = {}
    feature_metadata: Dict[str, Any] = {}
    decision_metadata: Dict[str, Any] = {}
    routing_metadata: Dict[str, Any] = {}
    decision_rate: Dict[str, Any] = {}
    observed_rate: Dict[str, Any] = {}
    load_metadata: Dict[str, Any] = {}
    prefill_response_meta: Dict[str, Any] = {}
    decode_response_meta: Dict[str, Any] = {}
    prefill_addr = decode_addr = replica_addr = generation_addr = None
    routing_mode = None
    turn_number = None
    conv_hash = ""
    request_id = ""
    logical_retry = False
    logical_physical_attempt = 1
    ppd_decision = None
    input_tokens = output_tokens = context_tokens = 0
    decision_input_tokens = decision_output_tokens = decision_context_tokens = 0
    routing_key = ""
    routing_key_source = "unknown"
    prefill_request_contract = None
    profile_context = {
        "schema_version": 4,
        "run_id": request.headers.get("X-PPD-Run-ID", "unscoped"),
        "episode_id": request.headers.get("X-PPD-Episode-ID"),
        "conversation_id": request.headers.get("X-PPD-Conversation-ID"),
        "step_id": request.headers.get("X-PPD-Step-ID"),
        "client_request_id": request.headers.get("X-PPD-Client-Request-ID"),
        "request_phase": request.headers.get("X-PPD-Request-Phase", "default"),
        "request_attempt": request.headers.get("X-PPD-Request-Attempt", "1"),
    }

    def timing_record(t_first: Optional[float]) -> Dict[str, Any]:
        now = time.time()
        record = {
            **profile_context,
            "ts_enter": round(t_enter, 4),
            "ts_route": round(t_route, 4),
            "ts_prefill_done": round(t_prefill_done, 4) if t_prefill_done else None,
            "ts_first_chunk": round(t_first, 4) if t_first else None,
            "ts_exit": round(now, 4),
            "route": routing_mode,
            "turn": turn_number,
            "conv": conv_hash[:8] if conv_hash else None,
            "state_key_sha256": conv_hash or None,
            "request_id": request_id or None,
            "logical_retry": logical_retry,
            "logical_physical_attempt": logical_physical_attempt,
            "routing_key_source": routing_key_source,
            "routing_key_label": routing_key if routing_key_source == "header" else None,
            "in_tokens": decision_input_tokens,
            "out_tokens": decision_output_tokens,
            "ctx_tokens": decision_context_tokens,
            "decision_input_tokens_est": decision_input_tokens,
            "decision_output_tokens_est": decision_output_tokens,
            "decision_context_tokens_est": decision_context_tokens,
            "requested_output_tokens": decision_output_tokens,
            "decision_qps_est": decision_rate.get("decision_qps_est"),
            "decision_qps_source": decision_rate.get("decision_qps_source"),
            "decision_rate_snapshot": decision_rate,
            "arrival_rate_snapshot": observed_rate,
            "ppd_mode_enabled": ENABLE_PPD_MODE,
            "ppd_decision": ppd_decision,
            "decision_source": decision_metadata.get("decision_source"),
            "selected_calibration_cell": decision_metadata.get(
                "selected_calibration_cell"
            ),
            "decision_metadata": decision_metadata,
            **feature_metadata,
            **routing_metadata,
            **load_metadata,
            "prefill_addr": prefill_addr,
            "decode_addr": decode_addr,
            "replica_addr": replica_addr,
            "generation_addr": generation_addr,
            "prefill_request_contract": prefill_request_contract,
            "t_route_s": round(t_route - t_enter, 4),
            "t_prefill_s": round(t_prefill_done - t_route, 4) if t_prefill_done else None,
            # Transport first-byte/chunk, explicitly not model first-token time.
            "ttfc_s": round(t_first - t_enter, 4) if t_first else None,
            "t_total_s": round(now - t_enter, 4),
        }
        record.update(_phase_profile_fields("prefill", prefill_response_meta))
        record.update(_phase_profile_fields("decode", decode_response_meta))
        record.update(response_usage)
        return record

    try:
        data = await request.get_json()
        routing_key, routing_key_source = get_routing_identity(
            data,
            explicit_routing_key=request.headers.get("X-PPD-Routing-Key"),
        )
        conv_hash = get_conversation_hash(
            data,
            conversation_id=profile_context["conversation_id"],
            run_id=profile_context["run_id"],
        )
        logical_record = get_logical_request(
            conv_hash, profile_context["client_request_id"]
        )
        logical_retry = logical_record is not None
        observed_rate = record_request_arrival(t_enter, is_logical=not logical_retry)
        input_tokens, output_tokens, context_tokens, feature_metadata = (
            _extract_request_features_with_metadata(data)
        )

        if logical_record is not None:
            turn_number = int(logical_record["turn"])
            prefill_addr = logical_record.get("prefill_addr")
            decode_addr = logical_record.get("decode_addr")
            replica_addr = logical_record.get("replica_addr")
            routing_mode = logical_record.get("routing_mode")
            ppd_decision = logical_record.get("ppd_decision")
            decision_metadata = dict(logical_record.get("decision_metadata") or {})
            routing_metadata = dict(logical_record.get("routing_metadata") or {})
            decision_rate = dict(logical_record.get("decision_rate") or {})
            decision_input_tokens = int(logical_record.get("decision_input_tokens", input_tokens))
            decision_output_tokens = int(logical_record.get("decision_output_tokens", output_tokens))
            decision_context_tokens = int(logical_record.get("decision_context_tokens", context_tokens))
            logical_physical_attempt = int(logical_record.get("physical_attempts", 2))
        else:
            current_turn, assigned = get_conversation_info(conv_hash)
            candidate_turn = current_turn + 1
            decision_rate = dict(observed_rate)
            decision_input_tokens = input_tokens
            decision_output_tokens = output_tokens
            decision_context_tokens = context_tokens
            use_ppd_dynamic = False
            if ENABLE_PPD_MODE and PPD_DECISION_ENGINE is not None:
                ppd_decision, decision_metadata = (
                    PPD_DECISION_ENGINE.decide_with_metadata(
                        turn=candidate_turn,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        current_qps=float(decision_rate["decision_qps_est"]),
                        context_tokens=context_tokens,
                    )
                )
                use_ppd_dynamic = bool(ppd_decision)
            prefill_addr, decode_addr, replica_addr, routing_mode, routing_metadata = (
                select_servers(
                    conv_hash,
                    candidate_turn,
                    assigned,
                    use_ppd_dynamic=use_ppd_dynamic,
                    routing_key=routing_key,
                    include_metadata=True,
                )
            )
            routing_metadata["routing_key_source"] = routing_key_source
            new_assigned = {
                role: address
                for role, address in (
                    ("prefill", prefill_addr),
                    ("decode", decode_addr),
                    ("replica", replica_addr),
                )
                if address
            }
            turn_number, final_assigned = update_conversation(conv_hash, new_assigned)
            save_logical_request(
                conv_hash,
                profile_context["client_request_id"],
                {
                    "turn": turn_number,
                    "assigned": final_assigned,
                    "prefill_addr": prefill_addr,
                    "decode_addr": decode_addr,
                    "replica_addr": replica_addr,
                    "routing_mode": routing_mode,
                    "ppd_decision": ppd_decision,
                    "decision_metadata": decision_metadata,
                    "routing_metadata": routing_metadata,
                    "decision_rate": decision_rate,
                    "decision_input_tokens": decision_input_tokens,
                    "decision_output_tokens": decision_output_tokens,
                    "decision_context_tokens": decision_context_tokens,
                },
            )

            with stats_lock:
                if turn_number == 1:
                    stats["turn1_requests"] += 1
                else:
                    stats["turn2plus_requests"] += 1
                    if assigned:
                        stats["cache_affinity_hits"] += 1
                    else:
                        stats["cache_affinity_misses"] += 1
                counter = {
                    "pd": "pd_requests",
                    "ppd": "ppd_requests",
                    "ppd_direct": "ppd_direct_requests",
                    "ppd_dynamic": "ppd_dynamic_requests",
                    "replica": "replica_requests",
                }.get(routing_mode)
                if counter:
                    stats[counter] += 1

            if ENABLE_PPD_MODE and ppd_decision is not None:
                log_ppd_decision(
                    conv_hash=conv_hash,
                    turn=turn_number,
                    input_tokens=decision_input_tokens,
                    output_tokens=decision_output_tokens,
                    context_tokens=decision_context_tokens,
                    current_qps=float(decision_rate["decision_qps_est"]),
                    decision=bool(ppd_decision),
                    routing_mode=routing_mode,
                    decision_metadata=decision_metadata,
                )

        t_route = time.time()
        if not routing_mode or (not decode_addr and not replica_addr):
            raise RuntimeError("routing contract found no generation server")
        generation_addr = decode_addr or replica_addr
        load_metadata = _active_load_snapshot(prefill_addr, generation_addr, conv_hash)
        is_streaming = bool(data.get("stream", False))

        if routing_mode == "replica":
            request_id = f"replica_{uuid.uuid4().hex}"
            target_url = f"http://{replica_addr}{request.path}"
            print(
                f"[PROXY] Replica: turn={turn_number}, retry={logical_retry}, "
                f"conv={conv_hash[:8]} -> R:{replica_addr}"
            )
        elif routing_mode == "ppd_direct":
            request_id = f"ppd_direct_{uuid.uuid4().hex}"
            target_url = f"http://{decode_addr}{request.path}"
            print(
                f"[PROXY] PPD-Direct: turn={turn_number}, retry={logical_retry}, "
                f"conv={conv_hash[:8]} -> pD:{decode_addr}"
            )
        elif routing_mode == "ppd_dynamic":
            request_id = f"ppd_dynamic_{uuid.uuid4().hex}"
            target_url = f"http://{decode_addr}{request.path}"
            print(
                f"[PROXY] PPD-Dynamic: turn={turn_number}, retry={logical_retry}, "
                f"conv={conv_hash[:8]}, in={decision_input_tokens}, "
                f"out={decision_output_tokens}, ctx={decision_context_tokens} "
                f"-> D:{decode_addr}"
            )
        else:
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
            prefill_request = data.copy()
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1
            # The prompt/tool schemas remain identical because vLLM excludes
            # tools for tool_choice=none only when a separate opt-in flag is
            # enabled. This changes only output parsing and avoids strict JSON
            # validation of a deliberately one-token prefill-only generation.
            if prefill_request.get("tools"):
                prefill_request["tool_choice"] = "none"
            prefill_request["stream"] = False
            prefill_request.pop("stream_options", None)
            prefill_request_contract = "same_prompt_tool_choice_none_max_tokens_1_v1"
            print(
                f"[PROXY] {routing_mode.upper()}: turn={turn_number}, "
                f"retry={logical_retry}, conv={conv_hash[:8]}, "
                f"P:{prefill_addr} -> D:{decode_addr}"
            )
            prefill_bytes = b""
            _active_request_change("prefill", prefill_addr, conv_hash, 1)
            try:
                async for chunk in forward_request(
                    f"http://{prefill_addr}{request.path}",
                    prefill_request,
                    request_id,
                    prefill_response_meta,
                ):
                    prefill_bytes += chunk
            finally:
                _active_request_change("prefill", prefill_addr, conv_hash, -1)
            _extract_upstream_json(prefill_bytes, prefill_response_meta)
            t_prefill_done = time.time()
            target_url = f"http://{decode_addr}{request.path}"

        if is_streaming:
            async def timed_stream():
                t_first = None
                _active_request_change("decode", generation_addr, conv_hash, 1)
                try:
                    async for chunk in forward_request(
                        target_url, data, request_id, decode_response_meta
                    ):
                        if t_first is None:
                            t_first = time.time()
                        yield chunk
                finally:
                    _active_request_change("decode", generation_addr, conv_hash, -1)
                    _write_timing(timing_record(t_first))

            return Response(
                timed_stream(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked",
                },
            )

        response_bytes = b""
        _active_request_change("decode", generation_addr, conv_hash, 1)
        try:
            async for chunk in forward_request(
                target_url, data, request_id, decode_response_meta
            ):
                response_bytes += chunk
        finally:
            _active_request_change("decode", generation_addr, conv_hash, -1)
        response_json = _extract_upstream_json(response_bytes, decode_response_meta)
        if response_json is not None:
            usage = response_json.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            prompt_tokens = usage.get("prompt_tokens")
            cached_tokens = details.get("cached_tokens")
            response_usage.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": usage.get("completion_tokens"),
                "actual_output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cached_tokens": cached_tokens,
                "uncached_prompt_tokens": (
                    max(0, prompt_tokens - cached_tokens)
                    if isinstance(prompt_tokens, int) and isinstance(cached_tokens, int)
                    else None
                ),
            })
            if response_json.get("error"):
                response_usage["upstream_error"] = str(response_json["error"])[:1000]
        _write_timing(timing_record(None))
        return response_bytes, decode_response_meta.get("status", 502), {
            "Content-Type": "application/json"
        }

    except Exception as exc:
        import traceback
        print(f"[PROXY] Error: {exc}")
        traceback.print_exc()
        error_record = timing_record(None)
        error_record["error"] = f"{type(exc).__name__}: {exc}"
        _write_timing(error_record)
        return {"error": str(exc)}, 500


@app.route("/status", methods=["GET"])
async def get_status():
    """Get proxy status."""
    with prefill_cv:
        p_list = ROUTING_CONTRACT.candidate_addresses(prefill_instances.keys())
    with decode_cv:
        d_list = ROUTING_CONTRACT.candidate_addresses(decode_instances.keys())
    with ppd_decode_cv:
        pd_list = ROUTING_CONTRACT.candidate_addresses(ppd_decode_instances.keys())
    with replica_cv:
        r_list = ROUTING_CONTRACT.candidate_addresses(replica_instances.keys())

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
        "logical_request_records": len(logical_request_state),
        "routing_contract": ROUTING_CONTRACT.descriptor(),
        "orca_response_load_enabled": ENABLE_ORCA_HEADERS,
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
    global CONFIG_NAME, ENABLE_PPD_MODE, PPD_DECISION_ENGINE, ROUTING_CONTRACT

    parser = argparse.ArgumentParser(description="Comprehensive Proxy Server")
    parser.add_argument("--config", type=str, default="2P_2D",
                        help="Configuration name (e.g., 2P_2D, 1P_3D, 4R)")
    parser.add_argument("--http-port", type=int, default=10001,
                        help="HTTP port for the proxy")
    parser.add_argument("--zmq-port", type=int, default=30001,
                        help="ZMQ port for service discovery")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument(
        "--routing-mode",
        choices=ROUTING_MODES,
        default=os.environ.get("PPD_ROUTING_MODE", "stable_hash"),
        help="Stable SHA-256 assignment (default) or pre-fix Python hash behavior.",
    )
    parser.add_argument(
        "--routing-seed",
        default=os.environ.get("PPD_ROUTING_SEED", "agentic-ppd-v1"),
        help="Explicit seed shared by paired experiment configurations.",
    )
    parser.add_argument(
        "--routing-map-path",
        default=os.environ.get("PPD_ROUTING_MAP_PATH") or None,
        help="Optional frozen sample-to-slot JSON map.",
    )
    parser.add_argument(
        "--routing-map-strict",
        action="store_true",
        default=os.environ.get("PPD_ROUTING_MAP_STRICT", "false").lower()
        in {"1", "true", "yes", "on"},
        help="Fail closed when the frozen map lacks a requested key/role.",
    )

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
    parser.add_argument(
        "--feature-tokenizer-path",
        type=str,
        default=os.environ.get("PPD_FEATURE_TOKENIZER_PATH"),
        help="Local tokenizer path for exact append-only Dynamic features; "
             "falls back to chars/4 when unset or incompatible.",
    )

    args = parser.parse_args()
    CONFIG_NAME = args.config
    ROUTING_CONTRACT = RoutingContract(
        mode=args.routing_mode,
        seed=args.routing_seed,
        map_path=args.routing_map_path,
        map_strict=args.routing_map_strict,
    )

    # Get decode port configuration
    decode_config = DECODE_PORTS_BY_CONFIG.get(CONFIG_NAME, {"D": [], "pD": []})
    d_ports = decode_config.get("D", [])
    ppd_ports = decode_config.get("pD", [])

    print(f"[PROXY] Starting Comprehensive Proxy")
    print(f"[PROXY] Config: {CONFIG_NAME}")
    print(f"[PROXY] Decode ports: D={d_ports}, pD={ppd_ports}")
    print(f"[PROXY] HTTP: {args.host}:{args.http_port}")
    print(f"[PROXY] ZMQ: {args.host}:{args.zmq_port}")
    print(f"[PROXY] Routing contract: {json.dumps(ROUTING_CONTRACT.descriptor())}")

    configure_feature_tokenizer(args.feature_tokenizer_path)

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
