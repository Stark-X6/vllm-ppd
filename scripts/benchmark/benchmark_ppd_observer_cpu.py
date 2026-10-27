#!/usr/bin/env python3
"""Stdlib-only observer microbenchmark. JSON stdout; no GPU/server side effects.

Uses test fixtures and a SHA256 stand-in for native vLLM block hashing. Timings
exclude native engine integration, GPU synchronization, HTTP, IPC and tokenizing.
The query rate limiter is deliberately bypassed to measure individual queries.
No measured number here is an end-to-end serving slowdown estimate.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace as NS


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "observer_benchmark_fixtures", ROOT / "tests/test_ppd_state_observer.py")
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)


def summary(values):
    ordered = sorted(values)
    return {"median_us": statistics.median(ordered) / 1000,
            "p95_us": ordered[max(0, math.ceil(len(ordered) * .95) - 1)] / 1000,
            "min_us": ordered[0] / 1000, "max_us": ordered[-1] / 1000,
            "samples": len(ordered)}


def paired(baseline, observed, rounds, inner=1):
    for _ in range(3):
        baseline()
        observed()
    values = {"baseline": [], "observer": []}
    deltas = []
    functions = {"baseline": baseline, "observer": observed}
    for repetition in range(rounds):
        order = ("baseline", "observer") if repetition % 2 == 0 else ("observer", "baseline")
        current = {}
        for name in order:
            start = time.perf_counter_ns()
            for _ in range(inner):
                functions[name]()
            current[name] = (time.perf_counter_ns() - start) / inner
            values[name].append(current[name])
        deltas.append(current["observer"] - current["baseline"])
    return {"alternating_order": True, "inner_iterations": inner,
            "baseline": summary(values["baseline"]),
            "observer": summary(values["observer"]),
            "paired_added": summary(deltas)}


def build_fixture(requests, probe_tokens, block_size):
    rig = fixtures.Rig(block_size=block_size)
    rig.pool.num_gpu_blocks = 65536
    rig.pool.free_blocks = 32768
    tokens = [i % 10000 for i in range(probe_tokens)]
    source = rig.request("prefix-source", tokens=tokens)
    for i in range((probe_tokens - 1) // block_size):
        rig.insert(source, i)
    rig.complete(source, probe_tokens - 1)
    del rig.scheduler.requests[source.request_id]
    rig.observer.event("after_update")
    for i in range(requests):
        n = 1025 + (i % 4) * 1024
        req = rig.request("decode-" + str(i), n=n, output=1, status="RUNNING")
        rig.complete(req, n - 1)
    return rig, {"prompt_token_ids": tokens}


def hook_case(requests, block_size, rounds, inner, insert_tokens):
    rigs = [fixtures.Rig(block_size=block_size), fixtures.Rig(block_size=block_size)]
    for rig in rigs:
        for i in range(requests):
            rig.request("r" + str(i), n=insert_tokens + 1, ready=insert_tokens,
                        output=1, status="RUNNING")
    schedules = [NS(num_scheduled_tokens={rid: 1 for rid in r.scheduler.requests}) for r in rigs]
    output = NS(kv_connector_output=NS(invalid_block_ids=()))
    blocks = [NS(block_id=99999 + i, block_hash=("fixture-block-" + str(i), 0),
                 ref_cnt=1, is_null=False) for i in range(insert_tokens // block_size)]

    def cycle(index, with_observer):
        rig = rigs[index]
        # The same synthetic scheduler work runs in both conditions.
        for req in rig.scheduler.requests.values():
            req.num_computed_tokens += 1
            req.num_tokens += 1
        if with_observer:
            observer = rig.observer
            observer.event("begin_execution", schedules[index])
            first = next(iter(rig.scheduler.requests.values()))
            for i, block in enumerate(blocks):
                observer.event("cache_insert", first, block, (i + 1) * block_size)
            observer.event("finish_execution", output)
            observer.event("after_update")
            for block in blocks:
                observer.event("cache_evict", block)

    timing = paired(lambda: cycle(0, False), lambda: cycle(1, True), rounds, inner)
    timing["scope"] = "bookkeeping for one synthetic chunk plus batch lifecycle; no model execution"
    timing["callbacks"] = ["begin_execution", "cache_insert", "finish_execution", "after_update", "cache_evict"]
    timing["requests"] = requests
    timing["insert_tokens"] = insert_tokens
    timing["inserted_then_evicted_blocks"] = len(blocks)
    timing["callbacks_per_batch"] = 3 + 2 * len(blocks)
    timing["observer_error"] = rigs[1].observer.error
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--requests", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--probe-tokens", type=int, nargs="+", default=[4097, 32769, 131073])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--hook-inner", type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 3 or args.hook_inner < 1 or args.block_size < 1:
        parser.error("rounds >= 3, hook-inner >= 1, block-size >= 1 required")
    if len(args.requests) != len(args.probe_tokens):
        parser.error("requests and probe-tokens must have matching lengths")
    if any(not 1 <= n <= 512 for n in args.requests):
        parser.error("requests must be between 1 and 512")
    if any(not 1 <= n <= 262144 for n in args.probe_tokens):
        parser.error("probe-tokens must be between 1 and 262144")
    report = {
        "benchmark": "ppd_observer_fake_cpu_v1",
        "python": platform.python_version(), "platform": platform.platform(),
        "limitations": [
            "fake supported engine objects; not native vLLM integration",
            "SHA256 substitute, not native block hasher; native hash cost is unmeasured",
            "no GPU work, HTTP, IPC, tokenizer, process contention or serving slowdown measured",
            "10 Hz production throttle bypassed solely for per-query microtiming",
            "query baseline is a minimal dispatcher stub, not real serving",
            "hook comparison measures 64-token/8192-token synthetic chunks and batch lifecycle",
        ],
        "block_size": args.block_size, "cases": [],
    }
    for n, token_count in zip(args.requests, args.probe_tokens):
        rig, payload = build_fixture(n, token_count, args.block_size)
        baseline = lambda: len(rig.scheduler.requests)
        state_result = rig.observe()
        probe_result = rig.observe(payload)
        if not state_result["available"] or not probe_result["prefix"]["available"]:
            raise RuntimeError("benchmark fixture unexpectedly unavailable")
        case = {
            "requests": n, "probe_tokens": token_count,
            "ready_cached_tokens": probe_result["prefix"]["c"],
            "state_only": paired(baseline, lambda: rig.observe(), args.rounds),
            "state_plus_probe": paired(baseline, lambda: rig.observe(payload), args.rounds),
            "bookkeeping": [hook_case(n, args.block_size, args.rounds,
                                      min(args.hook_inner, 10) if tokens > 64 else args.hook_inner,
                                      max(args.block_size, tokens // args.block_size * args.block_size))
                            for tokens in (64, 8192)],
        }
        report["cases"].append(case)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
