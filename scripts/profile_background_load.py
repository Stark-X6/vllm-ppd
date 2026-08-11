#!/usr/bin/env python3
"""Background load generator — keeps the pD nodes busy with small concurrent
requests so that a long tool-call pause in the agent session can trigger KV
cache eviction.

Usage:
    nohup python scripts/profile_background_load.py \
        --url http://localhost:10001/v1/chat/completions \
        --model "/root/autodl-tmp/Qwen3-Coder-30B-A3B-Instruct" \
        --concurrency 4 --delay 0.5 \
        > logs/background_load.log 2>&1 &
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import random
import string
import urllib.request

TOPICS = [
    "machine learning optimization techniques",
    "distributed systems design patterns",
    "database indexing strategies",
    "network protocol analysis",
    "compiler optimization passes",
    "operating system schedulers",
    "graph algorithms and their applications",
    "natural language processing pipelines",
    "computer vision architectures",
    "cryptographic hash functions",
    "quantum computing fundamentals",
    "blockchain consensus mechanisms",
    "programming language theory",
    "cloud infrastructure scaling",
    "embedded systems design",
    "bioinformatics sequence alignment",
    "reinforcement learning environments",
    "data compression algorithms",
    "parallel computing paradigms",
    "software testing methodologies",
]


def _one_request(url: str, model: str, idx: int) -> float:
    topic = TOPICS[idx % len(TOPICS)]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                       "content": f"Write a detailed technical essay about {topic}. "
                                  f"Include specific techniques, trade-offs, and real-world examples. "
                                  f"Context seed: {random.randint(0, 999999)}"}],
        "max_tokens": 512,
        "temperature": 0.7,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
    except Exception as e:
        pass  # silently skip transient errors under load
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:10001/v1/chat/completions")
    ap.add_argument("--model", default="/root/autodl-tmp/Qwen3-32B")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between burst rounds")
    args = ap.parse_args()

    print(f"[bg-load] concurrency={args.concurrency} delay={args.delay}s -> {args.url}",
          flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    round_idx = 0
    try:
        while True:
            t0 = time.time()
            futs = [pool.submit(_one_request, args.url, args.model, round_idx * args.concurrency + i)
                    for i in range(args.concurrency)]
            times = [f.result() for f in futs]
            print(f"[bg-load] burst {len(times)} req, "
                  f"max={max(times):.1f}s avg={sum(times)/len(times):.1f}s",
                  flush=True)
            round_idx += 1
            elapsed = time.time() - t0
            sleep = args.delay - elapsed
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pool.shutdown(wait=False)
        print("[bg-load] stopped", flush=True)


if __name__ == "__main__":
    sys.exit(main())
