#!/usr/bin/env python3
"""Poll /metrics on every PPD vLLM node and log key indicators to CSV.

The proxy has no metrics endpoint, so each vLLM instance (P: 8100/8101,
pD: 8200/8201) is polled directly. Output is wall-clock aligned with the
proxy timing JSONL (logs/proxy_timing.jsonl) and the agent trace
(results/agent_trace.json) for bottleneck analysis.

Usage:
    python scripts/profile_metrics_poller.py --ports 8100,8101,8200,8201 \
        --interval 0.5 --output logs/metrics.csv
    # run in background while the agent session executes:
    nohup python scripts/profile_metrics_poller.py > logs/metrics_poller.log 2>&1 &
"""

import argparse
import csv
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# metric name -> regex group for the value (tolerates optional {label} suffix)
METRIC_PATTERNS = {
    "cache_usage_pct": r"vllm:kv_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.e+-]+)",
    "requests_running": r"vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.]+)",
    "requests_waiting": r"vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.]+)",
    "preemptions": r"vllm:num_preemptions_total(?:\{[^}]*\})?\s+([0-9.]+)",
    "ttft_sum": r"vllm:time_to_first_token_seconds_sum(?:\{[^}]*\})?\s+([0-9.e+-]+)",
    "ttft_count": r"vllm:time_to_first_token_seconds_count(?:\{[^}]*\})?\s+([0-9.]+)",
    "tpot_sum": r"vllm:time_per_output_token_seconds_sum(?:\{[^}]*\})?\s+([0-9.e+-]+)",
    "tpot_count": r"vllm:time_per_output_token_seconds_count(?:\{[^}]*\})?\s+([0-9.]+)",
    "prefix_cache_hits": r"vllm:prefix_cache_hits_total(?:\{[^}]*\})?\s+([0-9.]+)",
    "prefix_cache_queries": r"vllm:prefix_cache_queries_total(?:\{[^}]*\})?\s+([0-9.]+)",
}

CSV_HEADER = [
    "ts", "port", "cache_usage_pct", "running", "waiting",
    "ttft_ms", "tpot_ms", "prefix_hits", "prefix_queries", "preemptions",
]


def fetch_metrics(port: int, timeout: float = 2.0) -> dict:
    """Fetch and parse /metrics from one node; all keys None on failure."""
    out = dict.fromkeys(CSV_HEADER, None)
    out["ts"] = round(time.time(), 3)
    out["port"] = port
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return out
    vals = {}
    for key, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, text)
        vals[key] = float(m.group(1)) if m else None
    out["cache_usage_pct"] = vals["cache_usage_pct"]
    out["running"] = vals["requests_running"]
    out["waiting"] = vals["requests_waiting"]
    out["preemptions"] = vals["preemptions"]
    # means of the histogram buckets' sum/count
    if vals["ttft_count"]:
        out["ttft_ms"] = round(1000 * vals["ttft_sum"] / vals["ttft_count"], 3)
    if vals["tpot_count"]:
        out["tpot_ms"] = round(1000 * vals["tpot_sum"] / vals["tpot_count"], 3)
    out["prefix_hits"] = vals["prefix_cache_hits"]
    out["prefix_queries"] = vals["prefix_cache_queries"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", default="8100,8101,8200,8201", help="comma-separated vLLM ports")
    ap.add_argument("--interval", type=float, default=0.5, help="poll interval seconds (default 0.5)")
    ap.add_argument("--output", default="/root/vllm-ppd/logs/metrics.csv", help="CSV output path")
    ap.add_argument("--duration", type=float, default=0, help="run for N seconds; 0 = until Ctrl-C")
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    writer.writeheader()
    csv_file.flush()

    print(f"[poller] ports={ports} interval={args.interval}s -> {args.output}", flush=True)
    deadline = time.time() + args.duration if args.duration > 0 else None
    pool = ThreadPoolExecutor(max_workers=len(ports))
    try:
        while deadline is None or time.time() < deadline:
            t0 = time.time()
            rows = list(pool.map(fetch_metrics, ports))
            for row in rows:
                if row["port"] is not None:
                    writer.writerow(row)
            csv_file.flush()
            elapsed = time.time() - t0
            delay = args.interval - elapsed
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=True)
        csv_file.close()
        print("[poller] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
