#!/usr/bin/env python3
"""Align and analyze PPD profiling data from three sources.

Sources (all wall-clock, same host):
  - logs/proxy_timing.jsonl   per-request pipeline split (written by comprehensive_proxy.py)
  - results/agent_trace.json  per-step LLM/tool timing (written by uni-agent react agent)
  - logs/metrics.csv          1Hz node metrics (written by profile_metrics_poller.py)

Answers the three questions:
  1. Which tool calls hang (and for how long)?
  2. What was the LLM server load while they were hanging?
  3. Where did the time go: LLM prefill/decode, KV transfer, or the tool itself?
Plus the PPD-specific eviction signature: tool-pause window in which a node's
cache_usage drops sharply, followed by a TTFT jump on the next request.

Usage:
    python scripts/profile_analyze.py [--proxy-timing logs/proxy_timing.jsonl]
                                      [--agent-trace results/agent_trace.json]
                                      [--metrics logs/metrics.csv]
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

EVICT_DROP = 0.30      # cache_usage drop within a tool window that counts as eviction
ANOMALY_RATIO = 1.5    # llm_time vs similar-context baseline that counts as suspicious


def load_proxy_timing(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("ts_enter") or 0)
    return rows


def load_agent_trace(path):
    import glob
    # if path is a glob or directory, find the latest trace
    if not os.path.isfile(path):
        candidates = sorted(glob.glob(f"{path}*.json") if "*" not in path else glob.glob(path),
                            key=os.path.getmtime, reverse=True)
        path = candidates[0] if candidates else path
    with open(path) as f:
        data = json.load(f)
    return data


def load_metrics(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in row:
                if key not in ("ts", "port") and row[key] not in (None, ""):
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            try:
                row["ts"] = float(row["ts"])
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows


def cache_series(rows, port):
    """(ts, cache_usage) pairs for one node, sorted by time."""
    return [(r["ts"], r["cache_usage_pct"]) for r in rows
            if str(r.get("port")) == str(port) and r.get("cache_usage_pct") is not None]


def min_in_window(series, t0, t1):
    return min((v for ts, v in series if t0 <= ts <= t1), default=None)


def median_in_window(series, t0, t1):
    vals = [v for ts, v in series if t0 <= ts <= t1]
    return statistics.median(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-timing", default="/root/vllm-ppd/logs/proxy_timing.jsonl")
    ap.add_argument("--agent-trace", default="/root/vllm-ppd/results/agent_trace.json")
    ap.add_argument("--metrics", default="/root/vllm-ppd/logs/metrics.csv")
    args = ap.parse_args()

    proxy = load_proxy_timing(args.proxy_timing)
    trace = load_agent_trace(args.agent_trace)
    metrics = load_metrics(args.metrics)
    steps = trace.get("steps_detail", [])

    # Keep only rows inside the episode's wall-clock window; the raw files
    # accumulate across runs, so without this stale requests pollute the split.
    ep_start = trace.get("episode_start") or (steps[0]["step_start"] - 2 if steps else 0)
    ep_end = trace.get("episode_end") or (ep_start + trace.get("episode_duration_s", 0))
    proxy = [r for r in proxy if (r.get("ts_enter") or 0) >= ep_start - 2 and
             (r.get("ts_enter") or 0) <= ep_end + 5]
    metrics = [m for m in metrics if m.get("ts", 0) >= ep_start - 2 and m.get("ts", 0) <= ep_end + 5]

    print("=" * 78)
    print("PPD agent session analysis")
    print("=" * 78)
    print(f"episode: {trace.get('steps', 0)} steps, "
          f"{trace.get('num_tool_calls', 0)} tool calls, "
          f"{trace.get('episode_duration_s', '?')}s, "
          f"reason={trace.get('termination_reason') or trace.get('error', 'unknown')}")
    print(f"proxy requests: {len(proxy)}, metrics rows: {len(metrics)}")
    print()

    # ---- 1. per-step table ----
    print(f"{'step':>4} {'route':>10} {'turn':>4} {'llm_s':>7} {'tools_s':>7} "
          f"{'prompt_tok':>10} {'tools':>22}")
    for s in steps:
        tools_str = ", ".join(f"{t.get('name')}={t.get('time_s')}s" for t in s.get("tools", []))
        # match the proxy request inside this step's LLM window
        t0, t1 = s["step_start"], s["step_start"] + s.get("llm_time_s", 0)
        reqs = [r for r in proxy if r.get("ts_enter") and t0 - 0.5 <= r["ts_enter"] <= t1 + 0.5]
        route = reqs[0].get("route", "?") if reqs else "-"
        turn = reqs[0].get("turn", "?") if reqs else "-"
        print(f"{s['step']:>4} {str(route):>10} {str(turn):>4} "
              f"{s.get('llm_time_s', 0):>7.2f} {s.get('tools_time_s', 0):>7.2f} "
              f"{s.get('prompt_tokens', 0):>10} {tools_str:>22}")

    # ---- 2. pipeline split from the proxy ----
    print()
    print("Proxy per-request split (PPD: t_prefill = P-node prefill; "
          "ttft - prefill = transfer + first decode):")
    for r in proxy:
        if r.get("route") is None:
            print(f"  [error] {r.get('ts_exit', '?')} {r.get('error', '?')}")
            continue
        prefill = f"{r.get('t_prefill_s', 0):.2f}s" if r.get("t_prefill_s") else "     -"
        print(f"  turn={r.get('turn', '?'):>2} {str(r.get('route')):>11} "
              f"ctx={r.get('ctx_tokens', '?'):>5} "
              f"route_s={r.get('t_route_s', 0):.3f} prefill_s={prefill} "
              f"ttft_s={r.get('ttft_s', '-')} total_s={r.get('t_total_s', 0):.2f}")

    # ---- 3. slowest tools ----
    print()
    print("Slowest tool calls (sorted):")
    all_tools = sorted(
        ((s["step"], t.get("name"), t.get("time_s", 0)) for s in steps for t in s.get("tools", [])),
        key=lambda x: x[2], reverse=True,
    )
    for step, name, t in all_tools[:10]:
        # server load during this tool call
        t0 = steps[step - 1]["step_start"] + steps[step - 1].get("llm_time_s", 0)
        t1 = t0 + t
        load = []
        for r in metrics:
            if t0 <= r["ts"] <= t1 and r.get("running") is not None:
                load.append((int(r["port"]), int(r["running"]), r.get("cache_usage_pct")))
        load_str = ", ".join(f":{p} run={run}" for p, run, _ in load[:6]) if load else "no metrics"
        print(f"  step={step:>3} {name:>20} {t:>7.2f}s   during: {load_str}")

    # ---- 4. eviction detection ----
    print()
    print("Eviction check (cache_usage drop > %.0f%% in a tool window):" % (EVICT_DROP * 100))
    ports = sorted({str(r["port"]) for r in metrics})
    series = {p: cache_series(metrics, p) for p in ports}
    evicts = []
    for s in steps:
        t0 = s["step_start"] + s.get("llm_time_s", 0)
        t1 = t0 + s.get("tools_time_s", 0)
        if t1 - t0 < 0.5:
            continue
        for p in ports:
            before = median_in_window(series[p], t0 - 5.0, t0)
            during = min_in_window(series[p], t0, t1)
            if before is None or during is None:
                continue
            drop = before - during
            if drop >= EVICT_DROP:
                evicts.append((s["step"], p, before, during))
                print(f"  ⚠️ step={s['step']}: node :{p} cache {before:.0f}% -> {during:.0f}% "
                      f"(drop {drop:.0f}%) during {s.get('tools_time_s', 0):.1f}s tool pause")
    if not evicts:
        print("  (no drops >= %.0f%%; no eviction pressure observed)" % (EVICT_DROP * 100))

    # ---- 5. LLM time anomaly vs similar-context baseline ----
    print()
    print("LLM-time anomaly check (llm_time > %.0fx similar-context median):" % ANOMALY_RATIO)
    by_ctx = defaultdict(list)
    for s in steps:
        by_ctx[s.get("prompt_tokens", 0) // 500].append(s.get("llm_time_s", 0))
    for s in steps:
        bucket = s.get("prompt_tokens", 0) // 500
        peers = by_ctx.get(bucket, [])
        if len(peers) >= 3:
            base = statistics.median(peers)
        elif bucket > 0 and (bucket - 1) in by_ctx:
            base = statistics.median(by_ctx[bucket - 1])
        else:
            continue
        if base > 0 and s.get("llm_time_s", 0) > ANOMALY_RATIO * base:
            print(f"  ⚠️ step={s['step']}: llm_time={s.get('llm_time_s', 0):.2f}s "
                  f"vs baseline {base:.2f}s (ctx={s.get('prompt_tokens', 0)} tok) "
                  f"-> likely full recompute / contention")

    # ---- 6. where did the wall time go ----
    print()
    total_llm = sum(s.get("llm_time_s", 0) for s in steps)
    total_tool = sum(s.get("tools_time_s", 0) for s in steps)
    total = trace.get("episode_duration_s") or (total_llm + total_tool)
    print("Where the episode wall-clock went:")
    print(f"  LLM requests: {total_llm:.1f}s ({100 * total_llm / max(total, 1e-9):.0f}%)")
    print(f"  Tool calls:   {total_tool:.1f}s ({100 * total_tool / max(total, 1e-9):.0f}%)")
    other = max(0, total - total_llm - total_tool)
    print(f"  Other:        {other:.1f}s ({100 * other / max(total, 1e-9):.0f}%)")

    # turn1 vs turn2+ TTFT: KV-transfer overhead estimate
    ppd = [r for r in proxy if r.get("route") == "ppd" and r.get("ttft_s")]
    direct = [r for r in proxy if r.get("route") == "ppd_direct" and r.get("ttft_s")]
    if ppd and direct:
        print()
        print(f"Turn1 (P prefill + KV transfer + decode) TTFT: "
              f"median {statistics.median(r['ttft_s'] for r in ppd):.2f}s (n={len(ppd)})")
        print(f"Turn2+ (pD append prefill) TTFT:                "
              f"median {statistics.median(r['ttft_s'] for r in direct):.2f}s (n={len(direct)})")


if __name__ == "__main__":
    from profile_analyze_v2 import main as main_v2
    sys.exit(main_v2())
