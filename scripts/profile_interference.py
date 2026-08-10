#!/usr/bin/env python3
"""
Interference Profile — 用 PyTorch Profiler 抓取 GPU kernel 时间线。
只在客户端跑 torch.profiler（可抓到 CUDA event 同步点）。
要抓 vLLM 服务端 kernel，用 nsys 包 serve 进程（见下方注释）。

用法:
  1. 先启动单 GPU 服务器: ./scripts/server/start_single_replica.sh
  2. 跑 profiling:    python scripts/profile_interference.py
  3. 查看:            tensorboard --logdir=./logs/profile --port=6006
"""

import os, sys, time, asyncio, aiohttp, json, argparse, random, string
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
import torch.profiler
import numpy as np
from ppd.config import MODEL_PATH

SERVER_URL = "http://localhost:8300"
OUTPUT_DIR = PROJECT_DIR / "logs" / "profile"

# ── 请求生成 ──────────────────────────────────────────────
def gen_text(tokens: int, seed: str = "x") -> str:
    rng = random.Random(hash(seed))
    words = []
    chars = 0
    while chars < tokens * 5:
        w = ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 7)))
        words.append(w)
        chars += len(w) + 1
    return ' '.join(words)

def make_payload(input_tokens: int, output_tokens: int, context_tokens: int = 0):
    if context_tokens > 0:
        ctx = gen_text(context_tokens, seed=f"ctx_{input_tokens}")
        inp = gen_text(input_tokens, seed=f"inp_{input_tokens}")
        prompt = ctx + "\n" + inp
    else:
        prompt = gen_text(input_tokens)
    return {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
    }

async def send(session, payload):
    t0 = time.time()
    async with session.post(f"{SERVER_URL}/v1/completions", json=payload, timeout=120) as resp:
        data = await resp.json()
    e2e_ms = (time.time() - t0) * 1000
    usage = data.get("usage", {})
    return {
        "ttft_ms": data.get("metrics", {}).get("ttft_ms", 0),
        "tpot_ms": data.get("metrics", {}).get("tpot_ms", 0),
        "e2e_ms": e2e_ms,
        "output_tokens": usage.get("completion_tokens", 0),
        "success": True,
    }

# ── 主逻辑 ────────────────────────────────────────────────
async def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== PPD Interference Profiling ===")
    print(f"Server: {SERVER_URL}")
    print(f"Batch: {args.decode_batch} decode + {args.prefill_count} prefill(s)")
    print(f"Profile: {OUTPUT_DIR}")

    async with aiohttp.ClientSession() as session:
        # Warmup requests
        print("\nWarmup...")
        for i in range(3):
            await send(session, make_payload(16, 32))
            print(f"  warmup {i+1}/3")

        # ── profiling session ──
        print(f"\nProfiling {args.profile_rounds} rounds...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=args.profile_rounds, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(OUTPUT_DIR),
                worker_name=f"bs{args.decode_batch}_pf{args.prefill_count}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            for rnd in range(args.profile_rounds):
                print(f"  Round {rnd+1}/{args.profile_rounds}:")

                # Launch all requests concurrently
                tasks = []
                # Decode requests
                for _ in range(args.decode_batch):
                    tasks.append(send(session, make_payload(4, 64, context_tokens=2048)))
                # Full-prefill request
                if args.prefill_type == "full":
                    tasks.append(send(session, make_payload(1024, 64)))
                else:
                    tasks.append(send(session, make_payload(128, 64, context_tokens=2048)))

                results = await asyncio.gather(*tasks, return_exceptions=True)
                ok = [r for r in results if isinstance(r, dict) and r.get("success")]
                decodes = ok[:-1] if len(ok) > 1 else ok
                prefill = ok[-1] if len(ok) > 1 else None

                decode_tpot = np.mean([d["tpot_ms"] for d in decodes]) if decodes else 0
                print(f"    decode TPOT: {decode_tpot:.1f}ms  "
                      f"prefill TTFT: {prefill['ttft_ms']:.1f}ms" if prefill else "")

                torch.cuda.synchronize()
                prof.step()

        print(f"\nProfile saved to {OUTPUT_DIR}/")
        print(f"Run: tensorboard --logdir={OUTPUT_DIR} --port=6006")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-batch", type=int, default=64)
    parser.add_argument("--prefill-count", type=int, default=1)
    parser.add_argument("--prefill-type", choices=["full", "append"], default="full")
    parser.add_argument("--profile-rounds", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(main(args))
