#!/usr/bin/env python3
"""Analyze one PPD agent profiling run with request-level joins.

Schema v2 joins agent steps to proxy requests by client_request_id. Legacy
traces (including run9) fall back to a conversation-scoped timestamp match;
they are never joined against every concurrent request in the episode window.
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return sorted(rows, key=lambda row: row.get("ts_enter") or 0)


def load_metrics(path: str) -> list[dict[str, Any]]:
    rows = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for key, value in list(row.items()):
                if value in (None, ""):
                    row[key] = None
                    continue
                if key not in ("run_id", "sample_id", "status"):
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            if isinstance(row.get("ts"), (int, float)):
                rows.append(row)
    return rows


def load_traces(pattern: str) -> list[dict[str, Any]]:
    paths = [pattern] if os.path.isfile(pattern) else sorted(glob.glob(pattern))
    traces = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as file:
                trace = json.load(file)
            trace["_path"] = path
            traces.append(trace)
        except (OSError, json.JSONDecodeError):
            continue
    return traces


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def window_rows(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [row for row in rows if start <= float(row.get("ts", 0)) <= end]


def proxy_overlaps(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if float(row.get("ts_enter") or 0) < end and float(row.get("ts_exit") or 0) > start
    ]


def legacy_conversation(trace: dict[str, Any], proxy: list[dict[str, Any]]) -> str | None:
    nearest_conversations = []
    for step in trace.get("steps_detail", []):
        start = float(step.get("llm_start_ts") or step.get("step_start") or 0)
        candidates = [row for row in proxy if row.get("conv") and row.get("ts_enter")]
        if candidates:
            nearest = min(candidates, key=lambda row: abs(float(row["ts_enter"]) - start))
            if abs(float(nearest["ts_enter"]) - start) <= 1.0:
                nearest_conversations.append(nearest["conv"])
    return Counter(nearest_conversations).most_common(1)[0][0] if nearest_conversations else None


def select_proxy(trace: dict[str, Any], proxy: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    run_id = trace.get("run_id")
    episode_id = trace.get("episode_id")
    if trace.get("schema_version", 1) >= 2 and episode_id:
        rows = [
            row for row in proxy
            if row.get("episode_id") == episode_id
            and (not run_id or row.get("run_id") == run_id)
        ]
        return rows, "exact episode_id/client_request_id"

    conv = legacy_conversation(trace, proxy)
    start = float(trace.get("episode_start") or 0) - 2
    end = float(trace.get("episode_end") or start + trace.get("episode_duration_s", 0)) + 5
    rows = [
        row for row in proxy
        if start <= float(row.get("ts_enter") or 0) <= end
        and (conv is None or row.get("conv") == conv)
    ]
    return rows, f"legacy conversation/timestamp (conv={conv or 'unknown'})"


def requests_for_step(
    trace: dict[str, Any], step: dict[str, Any], proxy: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    client_request_id = step.get("client_request_id")
    if client_request_id:
        return [row for row in proxy if row.get("client_request_id") == client_request_id]
    start = float(step.get("llm_start_ts") or step.get("step_start") or 0)
    candidates = [row for row in proxy if row.get("ts_enter")]
    if not candidates:
        return []
    nearest = min(candidates, key=lambda row: abs(float(row["ts_enter"]) - start))
    return [nearest] if abs(float(nearest["ts_enter"]) - start) <= 1.0 else []


def parse_tool_arguments(tool_call: dict[str, Any]) -> tuple[dict[str, Any], str]:
    raw = str((tool_call.get("function") or {}).get("arguments") or "")
    try:
        value = json.loads(raw)
        return (value if isinstance(value, dict) else {}), raw
    except (TypeError, json.JSONDecodeError):
        return {}, raw


def classify_tool_operation(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    """Return a stable operation label and a short human-readable detail."""
    if name == "shell":
        command = str(arguments.get("command") or "")
        flat = " ".join(command.split())
        # Put test runners before generic python so `python -m pytest` remains
        # comparable with a direct `pytest` invocation.
        patterns = (
            ("pytest", r"(^|\W)pytest(\W|$)"),
            ("unittest", r"(^|\W)unittest(\W|$)"),
            ("tox", r"(^|\W)tox(\W|$)"),
            ("grep", r"(^|\W)(?:rg|grep)(\W|$)"),
            ("find", r"(^|\W)find(\W|$)"),
            ("git", r"(^|\W)git(\W|$)"),
            ("pip", r"(^|\W)(?:pip|pip3)(\W|$)"),
            ("python", r"(^|\W)python(?:3(?:\.\d+)?)?(\W|$)"),
            ("make", r"(^|\W)make(\W|$)"),
        )
        labels = [label for label, pattern in patterns if re.search(pattern, command)]
        if "pytest" in labels:
            family = "pytest"
        elif "unittest" in labels:
            family = "unittest"
        elif "tox" in labels:
            family = "tox"
        else:
            family = "+".join(labels) or "other"
        return f"shell:{family}", flat[:120] or "(empty command)"

    if name == "str_replace_editor":
        operation = str(arguments.get("command") or "unknown")
        path = str(arguments.get("path") or "")
        return f"editor:{operation}", path[:120]

    return name or "unknown", ""


def tool_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for step in trace.get("steps_detail", []):
        assistant = step.get("assistant_message") or step.get("response") or {}
        tool_calls = {
            call.get("id"): call for call in (assistant.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("id")
        }
        cursor = float(
            step.get("tools_start_ts")
            or (float(step.get("step_start") or 0) + float(step.get("llm_time_s") or 0))
        )
        for index, tool in enumerate(step.get("tools", [])):
            start = float(tool.get("start_ts") or cursor)
            duration = float(tool.get("time_s") or 0)
            end = float(tool.get("end_ts") or start + duration)
            tool_call = tool_calls.get(tool.get("tool_call_id"), {})
            arguments, raw_arguments = parse_tool_arguments(tool_call)
            operation, detail = classify_tool_operation(str(tool.get("name") or ""), arguments)
            events.append({
                **tool,
                "step": step.get("step"),
                "tool_index": tool.get("tool_index", index),
                "start_ts": start,
                "end_ts": end,
                "time_s": duration,
                "arguments": arguments,
                "raw_arguments": raw_arguments,
                "operation": operation,
                "detail": detail,
            })
            cursor = end
    return events


def sandbox_rpc_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten non-overlapping raw provider events from a schema-v3 trace."""
    events: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()

    def add(raw_events: Any, *, scope: str, step: Any = None, tool_index: Any = None) -> None:
        if not isinstance(raw_events, list):
            return
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            key = (raw.get("seq"), raw.get("kind"), raw.get("start_ts"))
            if key in seen:
                continue
            seen.add(key)
            events.append({
                **raw,
                "scope": scope,
                "step": step,
                "tool_index": tool_index,
            })

    profile = trace.get("sandbox_profile") or {}
    add(profile.get("pre_agent_events"), scope="sandbox_start")
    add((trace.get("toolbox_setup") or {}).get("sandbox_events"), scope="toolbox_setup")
    for step in trace.get("steps_detail", []):
        for tool in step.get("tools", []):
            add(
                tool.get("sandbox_events"),
                scope="tool",
                step=step.get("step"),
                tool_index=tool.get("tool_index"),
            )
    return sorted(events, key=lambda event: float(event.get("start_ts") or 0))


def sandbox_event_label(event: dict[str, Any]) -> str:
    if event.get("kind") == "modal.exec":
        return str(event.get("operation") or "modal.exec")
    return str(event.get("kind") or "unknown")


def stream_tail_s(event: dict[str, Any]) -> float:
    """Approximate stream completion after process completion.

    Modal stream reads and process.wait run concurrently, so stream-read time
    must not be added to process-wait time. Only the positive tail is a useful
    approximation of post-process output transfer/finalization.
    """
    stream = max(
        float(event.get("stdout_read_s") or 0),
        float(event.get("stderr_read_s") or 0),
    )
    return max(0.0, stream - float(event.get("process_wait_s") or 0))


def tool_sandbox_breakdown(event: dict[str, Any]) -> str:
    rows = [row for row in event.get("sandbox_events", []) if isinstance(row, dict)]
    if not rows:
        return "provider phases=unavailable"
    rpc_total = sum(float(row.get("duration_s") or 0) for row in rows)
    tool_total = float(event.get("time_s") or 0)
    unattributed = max(0.0, tool_total - rpc_total)
    tool_profile = event.get("tool_profile") or {}
    remote_command = tool_profile.get("remote_command_s")
    slow = max(rows, key=lambda row: float(row.get("duration_s") or 0))
    detail = (
        f"provider={rpc_total:.3f}s/{tool_total:.3f}s "
        f"host_unattributed={unattributed:.3f}s "
        f"slow_rpc={sandbox_event_label(slow)}:{float(slow.get('duration_s') or 0):.3f}s"
    )
    if isinstance(remote_command, (int, float)):
        # Remote command time is nested inside one of the provider wait RPCs;
        # subtract it only for this approximate decomposition, never add it to
        # provider_total.
        provider_noncommand = max(0.0, rpc_total - float(remote_command))
        detail += (
            f" remote_command={float(remote_command):.3f}s"
            f" provider_noncommand~={provider_noncommand:.3f}s"
        )
    if slow.get("kind") == "modal.exec":
        detail += (
            f" submit={float(slow.get('submit_s') or 0):.3f}s"
            f" wait={float(slow.get('process_wait_s') or 0):.3f}s"
            f" stream_tail~={stream_tail_s(slow):.3f}s"
        )
    elif slow.get("bytes") is not None:
        detail += f" bytes={int(slow['bytes'])}"
    return detail


def print_sandbox_rpc_summary(events: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}:")
    if not events:
        print("  unavailable (requires a schema-v3 trace from the instrumented Modal sandbox)")
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[sandbox_event_label(event)].append(event)
    print(
        f"  {'phase':<24} {'count':>5} {'total_s':>9} {'p50_s':>8} "
        f"{'p90_s':>8} {'max_s':>8} {'submit90':>9} {'wait90':>8} {'tail90~':>8}"
    )
    for phase, rows in sorted(
        grouped.items(),
        key=lambda item: -sum(float(row.get("duration_s") or 0) for row in item[1]),
    ):
        durations = [float(row.get("duration_s") or 0) for row in rows]
        submits = [float(row.get("submit_s") or 0) for row in rows if row.get("kind") == "modal.exec"]
        waits = [float(row.get("process_wait_s") or 0) for row in rows if row.get("kind") == "modal.exec"]
        tails = [stream_tail_s(row) for row in rows if row.get("kind") == "modal.exec"]
        fmt = lambda values: f"{percentile(values, .9):.3f}" if values else "-"
        print(
            f"  {phase:<24.24} {len(rows):>5} {sum(durations):>9.3f} "
            f"{percentile(durations, .5):>8.3f} {percentile(durations, .9):>8.3f} "
            f"{max(durations):>8.3f} {fmt(submits):>9} {fmt(waits):>8} {fmt(tails):>8}"
        )


def print_tool_operation_summary(events: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}:")
    if not events:
        print("  no tool calls")
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("operation") or event.get("name") or "unknown")].append(event)
    print(f"  {'operation':<24} {'count':>5} {'total_s':>9} {'mean_s':>8} "
          f"{'p50_s':>8} {'p90_s':>8} {'max_s':>8} {'errors':>6}")
    for operation, rows in sorted(grouped.items(), key=lambda item: -sum(row["time_s"] for row in item[1])):
        durations = [float(row.get("time_s") or 0) for row in rows]
        errors = sum(row.get("status") not in (None, "ok", "legacy") for row in rows)
        print(
            f"  {operation:<24.24} {len(rows):>5} {sum(durations):>9.2f} "
            f"{statistics.mean(durations):>8.2f} {percentile(durations, .5):>8.2f} "
            f"{percentile(durations, .9):>8.2f} {max(durations):>8.2f} {errors:>6}"
        )


def cache_source(route: str) -> str:
    if route in ("ppd", "pd"):
        return "xfer"
    if route in ("ppd_direct", "ppd_dynamic"):
        return "local"
    return "unknown"


def response_kind(step: dict[str, Any]) -> str:
    raw_response = step.get("response") or {}
    choices = raw_response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    if not message:
        message = step.get("assistant_message") or {}
    content = str(message.get("content") or "")
    has_reasoning = bool(
        message.get("reasoning")
        or message.get("reasoning_content")
        or "<think>" in content
    )
    has_tools = bool(message.get("tool_calls"))
    if has_reasoning:
        return "think+tool" if has_tools else "think"
    if has_tools:
        return "tool"
    return "text" if content else "empty"


def reusable_prefix_tokens(step: dict[str, Any]) -> float:
    """Estimate the previous request's contiguous prefix present next turn.

    Hidden reasoning is retained in the trace but not replayed into the API
    transcript. That divergence means none of that completion is prefix-reusable.
    """
    prompt = float(step.get("prompt_tokens") or 0)
    completion = float(step.get("completion_tokens") or 0)
    if (
        response_kind(step).startswith("think")
        and not bool(step.get("reasoning_in_next_prompt", False))
    ):
        return prompt
    return prompt + completion


def cache_observation(
    step: dict[str, Any], requests: list[dict[str, Any]]
) -> tuple[int | float, int | float | None, dict[str, Any], bool]:
    """Return the pre-generation cache state for one logical agent step.

    A tool-choice retry reuses the exact prompt just processed by its failed
    first HTTP request. Its final response therefore reports an artificially
    warm cache. Use the first physical request when it has usage data;
    otherwise mark cache state unknown rather than treating the retry as TTL
    retention evidence.
    """
    initial = requests[0] if requests else {}
    final = requests[-1] if requests else {}
    retry = bool(step.get("tool_choice_retry")) or len(requests) > 1
    prompt = step.get("prompt_tokens") or initial.get("prompt_tokens") or final.get("prompt_tokens") or 0
    if retry:
        cached = initial.get("cached_tokens")
    else:
        cached = step.get("cached_tokens")
        if cached is None:
            cached = final.get("cached_tokens")
    return prompt, cached, initial or final, retry


def summarize_load(
    metrics: list[dict[str, Any]], proxy: list[dict[str, Any]], start: float, end: float
) -> str:
    samples = window_rows(metrics, start, end)
    overlap = proxy_overlaps(proxy, start, end)
    if not samples:
        nodes = Counter(str(row.get("decode_addr") or "?").rsplit(":", 1)[-1] for row in overlap)
        physical = ", ".join(f":{port} req={count}" for port, count in sorted(nodes.items()))
        return f"metrics=unknown; overlapping_requests={physical or 'none'}"

    summaries = []
    for port in sorted({str(row.get("port")) for row in samples}):
        node = [row for row in samples if str(row.get("port")) == port and row.get("status") != "error"]
        if not node:
            continue
        running = metric_values(node, "running")
        waiting = metric_values(node, "waiting")
        cache = metric_values(node, "cache_usage_pct")
        gpu = metric_values(node, "gpu_util_pct")
        summaries.append(
            f":{port} run_avg={statistics.mean(running):.2f}" if running else f":{port} run=?"
        )
        if waiting:
            summaries[-1] += f" wait_max={max(waiting):.0f}"
        if cache:
            summaries[-1] += f" kv_max={max(cache):.1%}"
        if gpu:
            summaries[-1] += f" gpu_avg={statistics.mean(gpu):.0f}%"
    return "; ".join(summaries) or "metrics=unavailable"


def print_counter_deltas(metrics: list[dict[str, Any]]) -> None:
    print("\nNode counter deltas during the episode:")
    if not metrics:
        print("  unknown (no metrics samples in the episode window)")
        return
    for port in sorted({str(row.get("port")) for row in metrics}):
        rows = sorted(
            [row for row in metrics if str(row.get("port")) == port and row.get("status") != "error"],
            key=lambda row: row["ts"],
        )
        if len(rows) < 2:
            print(f"  :{port} unknown (fewer than two healthy samples)")
            continue
        fields = {}
        for key in (
            "prefix_hits", "prefix_queries", "external_prefix_hits",
            "external_prefix_queries", "preemptions", "ttft_sum", "ttft_count",
            "tpot_sum", "tpot_count",
        ):
            if isinstance(rows[0].get(key), (int, float)) and isinstance(rows[-1].get(key), (int, float)):
                fields[key] = float(rows[-1][key]) - float(rows[0][key])
        text = " ".join(
            f"{key}={value:.3f}" if key.endswith("_sum") else f"{key}={value:.0f}"
            for key, value in fields.items()
        )
        if fields.get("ttft_count", 0) > 0 and "ttft_sum" in fields:
            text += f" new_ttft_mean_ms={1000 * fields['ttft_sum'] / fields['ttft_count']:.1f}"
        if fields.get("tpot_count", 0) > 0 and "tpot_sum" in fields:
            text += f" new_tpot_mean_ms={1000 * fields['tpot_sum'] / fields['tpot_count']:.1f}"
        print(f"  :{port} {text or 'counters unavailable'}")


def analyze_trace(trace: dict[str, Any], all_proxy: list[dict[str, Any]], all_metrics: list[dict[str, Any]]) -> None:
    steps = trace.get("steps_detail", [])
    start = float(trace.get("episode_start") or (steps[0].get("step_start") if steps else 0) or 0)
    end = float(trace.get("episode_end") or start + float(trace.get("episode_duration_s") or 0))
    proxy, join_mode = select_proxy(trace, all_proxy)
    run_id = trace.get("run_id")
    metrics = [
        row for row in all_metrics
        if start - 2 <= float(row.get("ts") or 0) <= end + 2
        and (not run_id or not row.get("run_id") or row.get("run_id") == run_id)
    ]

    print("=" * 100)
    print(f"trace={trace.get('_path')} run={run_id or 'legacy'} episode={trace.get('episode_id') or 'legacy'}")
    print(
        f"steps={trace.get('steps', len(steps))} tools={trace.get('num_tool_calls', 0)} "
        f"duration={trace.get('episode_duration_s', '?')}s reason={trace.get('termination_reason', 'unknown')}"
    )
    print(f"join={join_mode}; proxy_requests={len(proxy)}; metric_rows={len(metrics)}")

    warnings = []
    healthy_metrics = [row for row in metrics if row.get("status") != "error"]
    if not proxy:
        warnings.append("no proxy records joined")
    if not healthy_metrics:
        warnings.append("no metrics in episode window: load and eviction are UNKNOWN")
    if proxy and not any(row.get("ttft_s") is not None for row in proxy):
        warnings.append("TTFT unavailable (non-streaming requests)")
    if not any(
        step.get("cached_tokens") is not None for step in steps
    ) and not any(row.get("cached_tokens") is not None for row in proxy):
        warnings.append("per-request cached_tokens unavailable: TTL retention is UNKNOWN")
    if warnings:
        print("DATA QUALITY:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    print("\nPer-step LLM requests:")
    print("  cache%=cached/prompt; retain%=cached/previous contiguous reusable prefix")
    print("  hidden reasoning is excluded when it is not replayed; "
          "src xfer=P-to-pD KV transfer, local=pD prefix cache")
    print("  resp_tok=server usage.completion_tokens; includes text/reasoning/tool-call serialization")
    print(f"{'step':>4} {'route':>11} {'turn':>4} {'node':>5} {'phys':>4} {'llm_s':>7} "
          f"{'prompt':>7} {'cached':>7} {'cache%':>7} {'retain%':>7} {'src':>7} "
          f"{'resp_tok':>8} {'kind':>10} {'retry':>5}")
    step_requests = {}
    ordered_steps = sorted(steps, key=lambda item: int(item.get("step", 0)))
    previous_step = None
    for step in ordered_steps:
        requests = requests_for_step(trace, step, proxy)
        step_requests[step.get("step")] = requests
        prompt, cached, initial, retry = cache_observation(step, requests)
        hit = (100 * cached / prompt) if isinstance(cached, (int, float)) and prompt else None
        reusable = (
            reusable_prefix_tokens(previous_step) if previous_step else 0
        )
        retention = (100 * cached / reusable) if isinstance(cached, (int, float)) and reusable else None
        node = str(initial.get("decode_addr") or "-").rsplit(":", 1)[-1]
        route = str(initial.get("route", "-"))
        print(
            f"{int(step.get('step', 0)):>4} {route:>11} "
            f"{str(initial.get('turn', '-')):>4} {node:>5} {len(requests):>4} "
            f"{float(step.get('llm_time_s') or 0):>7.2f} {int(prompt):>7} "
            f"{str(int(cached)) if isinstance(cached, (int, float)) else '-':>7} "
            f"{f'{hit:.2f}' if hit is not None else '-':>7} "
            f"{f'{retention:.2f}' if retention is not None else '-':>7} "
            f"{cache_source(route):>7} "
            f"{int(step.get('completion_tokens') or 0):>8} "
            f"{response_kind(step):>10} {str(retry):>5}"
        )
        previous_step = step

    dynamic_rows = [
        requests[0]
        for step in ordered_steps
        if (requests := step_requests.get(step.get("step"), []))
        and requests[0].get("ppd_decision") is not None
    ]
    if dynamic_rows:
        ppd_count = sum(bool(row.get("ppd_decision")) for row in dynamic_rows)
        qps_values = metric_values(dynamic_rows, "decision_qps_est")
        input_estimates = metric_values(dynamic_rows, "decision_input_tokens_est")
        output_estimates = metric_values(dynamic_rows, "decision_output_tokens_est")
        context_estimates = metric_values(dynamic_rows, "decision_context_tokens_est")
        actual_outputs = metric_values(dynamic_rows, "completion_tokens")
        route_counts = Counter(str(row.get("route") or "unknown") for row in dynamic_rows)
        feature_sources = Counter(str(row.get("feature_source") or "unknown") for row in dynamic_rows)
        feature_estimators = Counter(
            str(row.get("feature_token_estimator") or "unknown") for row in dynamic_rows
        )
        append_estimator_deltas = [
            float(row["decision_input_tokens_est"])
            - float(row["append_tokens_fallback_est"])
            for row in dynamic_rows
            if isinstance(row.get("decision_input_tokens_est"), (int, float))
            and isinstance(row.get("append_tokens_fallback_est"), (int, float))
            and row.get("feature_token_estimator") == "hf_suffix_chat_template_v1"
        ]
        later_turn_rows = [row for row in dynamic_rows if int(row.get("turn") or 0) > 1]
        print("\nDynamic decisions:")
        print(
            f"  total={len(dynamic_rows)} ppd={ppd_count} pd={len(dynamic_rows) - ppd_count} "
            f"routes={dict(route_counts)}"
        )
        if qps_values:
            print(
                f"  qps_est p50={statistics.median(qps_values):.3f} "
                f"p90={percentile(qps_values, .9):.3f} max={max(qps_values):.3f}"
            )
        estimate_parts = []
        for label, values in (
            ("append", input_estimates), ("output_cap", output_estimates),
            ("context", context_estimates), ("actual_output", actual_outputs),
        ):
            if values:
                estimate_parts.append(
                    f"{label}_p50={statistics.median(values):.0f} {label}_max={max(values):.0f}"
                )
        if estimate_parts:
            print("  " + " | ".join(estimate_parts))
        print(
            f"  feature_sources={dict(feature_sources)} "
            f"feature_estimators={dict(feature_estimators)}"
        )
        if append_estimator_deltas:
            print(
                "  exact_append_minus_fallback "
                f"p50={statistics.median(append_estimator_deltas):.0f} "
                f"p90={percentile(append_estimator_deltas, .9):.0f} "
                f"min={min(append_estimator_deltas):.0f} "
                f"max={max(append_estimator_deltas):.0f}"
            )
        if later_turn_rows:
            replayed = sum(bool(row.get("assistant_reasoning_replayed")) for row in later_turn_rows)
            print(f"  assistant_reasoning_replayed={replayed}/{len(later_turn_rows)} later turns")

    events = sorted(tool_events(trace), key=lambda event: event["time_s"], reverse=True)
    print("\nSlowest tool calls and server load during each call:")
    for event in events[:10]:
        load = summarize_load(all_metrics, all_proxy, event["start_ts"], event["end_ts"])
        print(
            f"  step={event['step']:>3} idx={event['tool_index']} "
            f"{str(event.get('operation')):>24} {event['time_s']:>8.3f}s "
            f"status={event.get('status', 'legacy')} detail={event.get('detail') or '-'} | {load}\n"
            f"    {tool_sandbox_breakdown(event)}"
        )
    print_tool_operation_summary(events, "Tool operation breakdown")
    print_sandbox_rpc_summary(
        sandbox_rpc_events(trace), "Modal lifecycle and RPC phase breakdown"
    )

    print("\nTTL evidence (long tool pause -> next logical request cache retention):")
    if not events:
        print("  no tool calls")
    for event in events[:10]:
        next_steps = [step for step in steps if int(step.get("step", 0)) > int(event["step"])]
        next_step = next_steps[0] if next_steps else None
        if not next_step:
            print(f"  step={event['step']} pause={event['time_s']:.3f}s -> no next request")
            continue
        requests = step_requests.get(next_step.get("step"), [])
        next_prompt, cached, next_initial, _ = cache_observation(next_step, requests)
        previous_step = next(
            (step for step in steps if int(step.get("step", 0)) == int(event["step"])),
            {},
        )
        reusable = reusable_prefix_tokens(previous_step)
        retention = cached / reusable if isinstance(cached, (int, float)) and reusable else None
        print(
            f"  step={event['step']} {event.get('name')} pause={event['time_s']:.3f}s "
            f"-> step={next_step.get('step')} cached={cached if cached is not None else 'unknown'} "
            f"reusable~={int(reusable) if reusable else 'unknown'} next_prompt={next_prompt or 'unknown'} "
            f"retention={f'{retention:.1%}' if retention is not None else 'unknown'} "
            f"next_route={next_initial.get('route') or 'unknown'}"
        )
    # Report material loss between consecutive logical requests. The 90%
    # threshold excludes normal cache-block alignment differences.
    print("\nCache loss events (retention < 90%):")
    print(f"  {'steps':>7} {'route transition':>23} {'same':>4} {'node':>5} "
          f"{'reusable':>9} {'cached':>8} {'lost':>8} {'retain':>7} {'interpretation':>20} "
          f"{'other active episodes':>28} {'wait':>5} {'kv_max':>7} {'llm_s':>7}")
    loss_count = 0
    for previous_step, next_step in zip(ordered_steps, ordered_steps[1:]):
        requests = step_requests.get(next_step.get("step"), [])
        _, cached, initial, _ = cache_observation(next_step, requests)
        reusable = reusable_prefix_tokens(previous_step)
        if not reusable or not isinstance(cached, (int, float)):
            continue
        retention = float(cached) / reusable
        if retention >= 0.90:
            continue
        previous_requests = step_requests.get(previous_step.get("step"), [])
        _, _, previous_initial, _ = cache_observation(previous_step, previous_requests)
        previous_route = str(previous_initial.get("route") or "unknown")
        current_route = str(initial.get("route") or "unknown")
        previous_node_addr = str(previous_initial.get("decode_addr") or "")
        node_addr = str(initial.get("decode_addr") or "")
        same_node = bool(previous_node_addr and node_addr and previous_node_addr == node_addr)
        if current_route == "pd":
            interpretation = "policy_pd_recompute"
        elif not same_node:
            interpretation = "node_change"
        elif current_route in ("ppd_dynamic", "ppd_direct"):
            interpretation = "cache_loss_candidate"
        else:
            interpretation = "route_other"
        port = node_addr.rsplit(":", 1)[-1] if node_addr else "-"
        request_start = float(initial.get("ts_enter") or next_step.get("llm_start_ts") or 0)
        request_end = float(initial.get("ts_exit") or next_step.get("llm_end_ts") or request_start)
        pause_start = float(previous_step.get("llm_end_ts") or previous_step.get("step_end") or request_start)
        current_episode = trace.get("episode_id")
        other_active = sorted({
            str(row.get("episode_id")) for row in proxy_overlaps(all_proxy, min(pause_start, request_start), request_end)
            if row.get("episode_id") and row.get("episode_id") != current_episode
            and str(row.get("decode_addr") or "") == node_addr
            and (not run_id or not row.get("run_id") or row.get("run_id") == run_id)
        })
        node_metrics = [
            row for row in window_rows(all_metrics, request_start, request_end)
            if str(int(float(row.get("port")))) == port and row.get("status") != "error"
            and (not run_id or not row.get("run_id") or row.get("run_id") == run_id)
        ]
        waiting = metric_values(node_metrics, "waiting")
        cache = metric_values(node_metrics, "cache_usage_pct")
        active_text = ",".join(episode[:12] for episode in other_active) or "none"
        lost = max(0, round(reusable - float(cached)))
        wait_text = f"{max(waiting):.0f}" if waiting else "?"
        kv_text = f"{max(cache):.1%}" if cache else "?"
        transition = f"{previous_route}->{current_route}"
        print(f"  {int(previous_step.get('step', 0)):>3}->{int(next_step.get('step', 0)):<3} "
              f"{transition:>23} {('yes' if same_node else 'no'):>4} {port:>5} "
              f"{int(reusable):>9} {int(cached):>8} {lost:>8} {retention:>7.1%} "
              f"{interpretation:>20} {active_text:>28} {wait_text:>5} {kv_text:>7} "
              f"{float(next_step.get('llm_time_s') or 0):>7.2f}")
        loss_count += 1
    if not loss_count:
        print("  (none)")

    llm = sum(float(step.get("llm_time_s") or 0) for step in steps)
    tools = sum(float(event.get("time_s") or 0) for event in tool_events(trace))
    setup = float((trace.get("toolbox_setup") or {}).get("duration_s") or 0)
    total = float(trace.get("episode_duration_s") or llm + tools + setup)
    other = max(0.0, total - llm - tools - setup)
    print("\nWall-clock attribution:")
    for label, value in (("Modal/toolbox setup", setup), ("LLM", llm), ("tool execution", tools), ("other", other)):
        print(f"  {label:20} {value:8.2f}s ({100 * value / max(total, 1e-9):5.1f}%)")

    llm_times = [float(step.get("llm_time_s") or 0) for step in steps]
    per_output = [
        1000 * float(step.get("llm_time_s") or 0) / float(step.get("completion_tokens"))
        for step in steps if step.get("completion_tokens")
    ]
    print("\nLatency distribution (descriptive only; total latency is not a cache-miss detector):")
    if llm_times:
        print(f"  LLM median={statistics.median(llm_times):.3f}s p90={percentile(llm_times, .9):.3f}s")
    if per_output:
        print(f"  client total/output-token median={statistics.median(per_output):.1f}ms")
    print_counter_deltas(metrics)


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        return (
            args.proxy_timing or str(run_dir / "proxy_timing.jsonl"),
            args.agent_trace or str(run_dir / "agent_trace*.json"),
            args.metrics or str(run_dir / "metrics.csv"),
        )
    return (
        args.proxy_timing or "/root/vllm-ppd/logs/proxy_timing.jsonl",
        args.agent_trace or "/root/vllm-ppd/results/agent_trace*.json",
        args.metrics or "/root/vllm-ppd/logs/metrics.csv",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Run-scoped directory containing proxy timing, traces, and metrics")
    parser.add_argument("--proxy-timing")
    parser.add_argument("--agent-trace")
    parser.add_argument("--metrics")
    args = parser.parse_args()
    proxy_path, trace_pattern, metrics_path = resolve_paths(args)
    traces = load_traces(trace_pattern)
    if not traces:
        print(f"error: no agent traces matched {trace_pattern}", file=sys.stderr)
        return 2
    proxy = load_jsonl(proxy_path)
    metrics = load_metrics(metrics_path)
    for trace in traces:
        analyze_trace(trace, proxy, metrics)
    print("=" * 100)
    all_tool_events = [event for trace in traces for event in tool_events(trace)]
    print_tool_operation_summary(
        all_tool_events,
        "Run-wide tool operation breakdown",
    )
    print_sandbox_rpc_summary(
        [event for trace in traces for event in sandbox_rpc_events(trace)],
        "Run-wide Modal lifecycle and RPC phase breakdown",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
