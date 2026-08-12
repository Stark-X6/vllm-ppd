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
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

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
    "external_prefix_cache_hits": r"vllm:external_prefix_cache_hits_total(?:\{[^}]*\})?\s+([0-9.]+)",
    "external_prefix_cache_queries": r"vllm:external_prefix_cache_queries_total(?:\{[^}]*\})?\s+([0-9.]+)",
}

CSV_HEADER = [
    "schema_version", "run_id", "sample_id", "ts", "fetch_start_ts", "fetch_latency_ms",
    "port", "status", "cache_usage_pct", "running", "waiting",
    "ttft_ms", "ttft_sum", "ttft_count", "tpot_ms", "tpot_sum", "tpot_count",
    "prefix_hits", "prefix_queries", "external_prefix_hits", "external_prefix_queries",
    "preemptions", "gpu_index", "gpu_util_pct",
    "gpu_memory_util_pct", "gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_power_w",
]


def fetch_metrics(port: int, run_id: str, timeout: float = 2.0) -> dict:
    """Fetch and parse one node's metrics, including explicit collection health."""
    out = dict.fromkeys(CSV_HEADER, None)
    fetch_start = time.time()
    out.update({
        "schema_version": 2,
        "run_id": run_id,
        "sample_id": uuid4().hex,
        "fetch_start_ts": round(fetch_start, 3),
        "port": port,
        "status": "error",
    })
    try:
        # Cloud environments commonly inject HTTP_PROXY. Local vLLM metrics
        # must never leave the host or depend on NO_PROXY spelling.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        out["ts"] = round(time.time(), 3)
        out["fetch_latency_ms"] = round((time.time() - fetch_start) * 1000, 3)
        return out
    vals = {}
    out["status"] = "ok"
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        vals[key] = float(match.group(1)) if match else None
    out["cache_usage_pct"] = vals["cache_usage_pct"]
    out["running"] = vals["requests_running"]
    out["waiting"] = vals["requests_waiting"]
    out["preemptions"] = vals["preemptions"]
    out["ttft_sum"] = vals["ttft_sum"]
    out["ttft_count"] = vals["ttft_count"]
    out["tpot_sum"] = vals["tpot_sum"]
    out["tpot_count"] = vals["tpot_count"]
    # means of the histogram buckets' sum/count
    if vals["ttft_count"] and vals["ttft_sum"] is not None:
        out["ttft_ms"] = round(1000 * vals["ttft_sum"] / vals["ttft_count"], 3)
    if vals["tpot_count"] and vals["tpot_sum"] is not None:
        out["tpot_ms"] = round(1000 * vals["tpot_sum"] / vals["tpot_count"], 3)
    out["prefix_hits"] = vals["prefix_cache_hits"]
    out["prefix_queries"] = vals["prefix_cache_queries"]
    out["external_prefix_hits"] = vals["external_prefix_cache_hits"]
    out["external_prefix_queries"] = vals["external_prefix_cache_queries"]
    out["ts"] = round(time.time(), 3)
    out["fetch_latency_ms"] = round((time.time() - fetch_start) * 1000, 3)
    return out




def fetch_gpu_metrics() -> dict[int, dict[str, float]]:
    """Return one nvidia-smi snapshot; unavailable metrics stay blank."""
    fields = "index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    result = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            idx = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
            result[idx] = {
                "gpu_util_pct": values[0],
                "gpu_memory_util_pct": values[1],
                "gpu_memory_used_mib": values[2],
                "gpu_memory_total_mib": values[3],
                "gpu_power_w": values[4],
            }
        except ValueError:
            continue
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", default="8100,8101,8200,8201", help="comma-separated vLLM ports")
    ap.add_argument("--interval", type=float, default=0.5, help="poll interval seconds (default 0.5)")
    ap.add_argument("--output", default="/root/vllm-ppd/logs/metrics.csv", help="CSV output path")
    ap.add_argument("--duration", type=float, default=0, help="run for N seconds; 0 = until Ctrl-C")
    ap.add_argument("--run-id", default=os.getenv("PPD_RUN_ID", "unscoped"))
    ap.add_argument(
        "--gpu-map", default="8100:0,8101:1,8200:2,8201:3",
        help="port:GPU-index pairs used to attach one nvidia-smi snapshot",
    )
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    gpu_map = {int(pair.split(":")[0]): int(pair.split(":")[1])
               for pair in args.gpu_map.split(",") if pair.strip()}
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
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
            rows = list(pool.map(lambda port: fetch_metrics(port, args.run_id), ports))
            gpu_metrics = fetch_gpu_metrics()
            for row in rows:
                gpu_index = gpu_map.get(row["port"])
                row["gpu_index"] = gpu_index
                if gpu_index in gpu_metrics:
                    row.update(gpu_metrics[gpu_index])
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
