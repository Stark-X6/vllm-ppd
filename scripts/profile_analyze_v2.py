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


def tool_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for step in trace.get("steps_detail", []):
        cursor = float(
            step.get("tools_start_ts")
            or (float(step.get("step_start") or 0) + float(step.get("llm_time_s") or 0))
        )
        for index, tool in enumerate(step.get("tools", [])):
            start = float(tool.get("start_ts") or cursor)
            duration = float(tool.get("time_s") or 0)
            end = float(tool.get("end_ts") or start + duration)
            events.append({
                **tool,
                "step": step.get("step"),
                "tool_index": tool.get("tool_index", index),
                "start_ts": start,
                "end_ts": end,
                "time_s": duration,
            })
            cursor = end
    return events


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
    print(f"{'step':>4} {'route':>11} {'turn':>4} {'node':>5} {'phys':>4} {'llm_s':>7} "
          f"{'prompt':>7} {'cached':>7} {'hit%':>6} {'output':>6} {'retry':>5}")
    step_requests = {}
    for step in steps:
        requests = requests_for_step(trace, step, proxy)
        step_requests[step.get("step")] = requests
        final = requests[-1] if requests else {}
        prompt = step.get("prompt_tokens") or final.get("prompt_tokens") or 0
        cached = step.get("cached_tokens")
        if cached is None:
            cached = final.get("cached_tokens")
        hit = (100 * cached / prompt) if isinstance(cached, (int, float)) and prompt else None
        node = str(final.get("decode_addr") or "-").rsplit(":", 1)[-1]
        retry = bool(step.get("tool_choice_retry")) or len(requests) > 1
        print(
            f"{int(step.get('step', 0)):>4} {str(final.get('route', '-')):>11} "
            f"{str(final.get('turn', '-')):>4} {node:>5} {len(requests):>4} "
            f"{float(step.get('llm_time_s') or 0):>7.2f} {int(prompt):>7} "
            f"{str(int(cached)) if isinstance(cached, (int, float)) else '-':>7} "
            f"{f'{hit:.1f}' if hit is not None else '-':>6} "
            f"{int(step.get('completion_tokens') or 0):>6} {str(retry):>5}"
        )

    events = sorted(tool_events(trace), key=lambda event: event["time_s"], reverse=True)
    print("\nSlowest tool calls and server load during each call:")
    for event in events[:10]:
        load = summarize_load(all_metrics, all_proxy, event["start_ts"], event["end_ts"])
        print(
            f"  step={event['step']:>3} idx={event['tool_index']} "
            f"{str(event.get('name')):>20} {event['time_s']:>8.3f}s "
            f"status={event.get('status', 'legacy')} | {load}"
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
        final = requests[-1] if requests else {}
        cached = next_step.get("cached_tokens")
        if cached is None:
            cached = final.get("cached_tokens")
        next_prompt = next_step.get("prompt_tokens") or final.get("prompt_tokens") or 0
        previous_step = next(
            (step for step in steps if int(step.get("step", 0)) == int(event["step"])),
            {},
        )
        reusable = (
            float(previous_step.get("prompt_tokens") or 0)
            + float(previous_step.get("completion_tokens") or 0)
        )
        retention = cached / reusable if isinstance(cached, (int, float)) and reusable else None
        print(
            f"  step={event['step']} {event.get('name')} pause={event['time_s']:.3f}s "
            f"-> step={next_step.get('step')} cached={cached if cached is not None else 'unknown'} "
            f"reusable~={int(reusable) if reusable else 'unknown'} next_prompt={next_prompt or 'unknown'} "
            f"retention={f'{retention:.1%}' if retention is not None else 'unknown'}"
        )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
