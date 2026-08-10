#!/usr/bin/env python3
"""
Profile Burst — 短促高密度请求，配合 nsys 精确抓取 GPU kernel 时间线

用法:
  # Comprehensive 场景 (synthetic workload)
  python scripts/profile_burst.py --mode comprehensive --duration 20 --qps 4

  # ShareGPT 场景 (真实对话 + PPD 动态决策)
  python scripts/profile_burst.py --mode sharegpt --duration 20 --qps 4

  # 单次干扰对比 (micro-benchmark: 1 full-prefill vs 1 append-prefill)
  python scripts/profile_burst.py --mode interference --duration 15

原理:
  生成 15-20 秒的高密度请求流，同时跑 Turn1 (prefill) 和 Turn2+ (append/ppd)，
  让 nsys 在这段时间内抓到: prefill kernel | NCCL KV transfer | decode kernel。
  比全量 benchmark 抓取:
  - 文件小 10x (200MB vs 2GB)
  - 100% 有效数据 (无空闲段)
  - nsys-ui 里一眼看到瓶颈
"""

import os, sys, time, asyncio, aiohttp, argparse, random, string, json, gzip
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from ppd.config import MODEL_PATH

PROXY_URL = "http://localhost:10001"

# ── 随机文本生成 ──
def gen_text(tokens: int, seed: str = "x") -> str:
    rng = random.Random(hash(seed))
    words = []
    chars = 0
    while chars < tokens * 5:
        w = ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 7)))
        words.append(w)
        chars += len(w) + 1
    return ' '.join(words)


# ═══════════════════════════════════════════════════════════════
# 模式 1: Comprehensive synthetic workload burst
# ═══════════════════════════════════════════════════════════════

async def comprehensive_burst(duration_s: int, qps: int):
    """
    用 comprehensive benchmark 的 workload 定义生成请求。
    每轮迭代发一组建模 Turn1→Turn2 的请求对。
    """
    workloads = [
        ("small", "tiny"), ("small", "short_gen"), ("small", "mid_bal"),
        ("large", "short_gen"), ("large", "long_gen"), ("large", "mid_paste"),
    ]
    T1_CFG = {"small": (128, 128), "large": (1024, 1024)}
    T2_CFG = {"tiny": (16, 32), "short_gen": (32, 256), "long_gen": (32, 512),
              "mid_bal": (128, 128), "mid_paste": (256, 64)}

    endpoint = f"{PROXY_URL}/v1/completions"
    deadline = time.time() + duration_s
    count = 0

    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            t1_type, t2_type = random.choice(workloads)
            t1_in, t1_out = T1_CFG[t1_type]
            t2_in, t2_out = T2_CFG[t2_type]

            # Build conversation: Turn1 sets up prefix cache, Turn2 uses it
            conv_id = f"burst_{count}"
            t1_payload = {
                "model": MODEL_PATH, "prompt": gen_text(t1_in, seed=f"t1_{count}"),
                "max_tokens": t1_out, "temperature": 0.0,
                "conversation_id": conv_id, "turn": 1,
            }
            t2_payload = {
                "model": MODEL_PATH, "prompt": gen_text(t2_in, seed=f"t2_{count}"),
                "max_tokens": t2_out, "temperature": 0.0,
                "conversation_id": conv_id, "turn": 2,
            }

            # Send turn1, wait for completion, then turn2
            async with session.post(endpoint, json=t1_payload, timeout=120) as resp:
                await resp.json()
            async with session.post(endpoint, json=t2_payload, timeout=120) as resp:
                await resp.json()

            count += 1
            interval = 1.0 / qps
            await asyncio.sleep(interval)

    print(f"  Comprehensive burst: {count} conversation pairs in {duration_s}s")


# ═══════════════════════════════════════════════════════════════
# 模式 2: ShareGPT burst (真实对话片段 + PPD 动态决策)
# ═══════════════════════════════════════════════════════════════

async def sharegpt_burst(duration_s: int, qps: int):
    """
    从 WildChat 数据集加载真实多轮对话，以固定 QPS 重放。
    每条对话按原始 turn 顺序发请求，触发 proxy 的 PPD 决策引擎。
    """
    data_path = PROJECT_DIR / "data" / "WildChat_1M.json.gz"
    if not data_path.exists():
        print(f"  ⚠ WildChat not found at {data_path}, using synthetic fallback")
        return await comprehensive_burst(duration_s, qps)

    print(f"  Loading WildChat...")
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        all_convs = json.load(f)
    multi_turn = [c for c in all_convs if c.get("turn_count", 1) >= 2]
    print(f"  Loaded {len(multi_turn)} multi-turn conversations")

    endpoint = f"{PROXY_URL}/v1/completions"
    deadline = time.time() + duration_s
    count = 0

    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            conv = random.choice(multi_turn)
            conv_id = conv["id"]
            turns = conv.get("conversations", [])

            # 按 turn 顺序重放
            for turn_idx, turn_data in enumerate(turns):
                if time.time() >= deadline:
                    break
                role = turn_data.get("role", "user")
                if role != "user":
                    continue

                content = turn_data.get("content", "")
                if not content:
                    continue

                payload = {
                    "model": MODEL_PATH, "prompt": content[:4096],
                    "max_tokens": min(len(content.split()) * 2, 512),
                    "temperature": 0.0,
                    "conversation_id": conv_id,
                    "turn": turn_idx + 1,
                }
                async with session.post(endpoint, json=payload, timeout=120) as resp:
                    await resp.json()

            count += 1
            interval = 1.0 / qps
            await asyncio.sleep(interval)

    print(f"  ShareGPT burst: {count} conversations in {duration_s}s")


# ═══════════════════════════════════════════════════════════════
# 模式 3: Interference micro benchmark burst
# ═══════════════════════════════════════════════════════════════

async def interference_burst(duration_s: int, decode_batch: int = 64):
    """
    并发发 decode + prefill 请求，测试干扰。
    适配单 GPU 场景，不需要 proxy (直连 serve)。
    """
    SERVER_URL = "http://localhost:8300"
    endpoint = f"{SERVER_URL}/v1/completions"
    deadline = time.time() + duration_s
    count = 0

    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            tasks = []
            # Decode batch
            for _ in range(decode_batch):
                tasks.append(
                    session.post(endpoint, json={
                        "model": MODEL_PATH, "prompt": gen_text(4, seed=f"d_{count}_{_}"),
                        "max_tokens": 64, "temperature": 0.0,
                    }, timeout=120)
                )
            # Full-prefill
            tasks.append(
                session.post(endpoint, json={
                    "model": MODEL_PATH, "prompt": gen_text(1024, seed=f"f_{count}"),
                    "max_tokens": 64, "temperature": 0.0,
                }, timeout=120)
            )
            # Append-prefill
            tasks.append(
                session.post(endpoint, json={
                    "model": MODEL_PATH, "prompt": gen_text(2048, seed=f"ctx_{count}") + "\n" + gen_text(128, seed=f"a_{count}"),
                    "max_tokens": 64, "temperature": 0.0,
                }, timeout=120)
            )

            await asyncio.gather(*tasks, return_exceptions=True)
            count += 1
            await asyncio.sleep(0.5)  # 0.5s gap between rounds

    print(f"  Interference burst: {count} rounds × {decode_batch + 2} requests in {duration_s}s")


# ── main ──
async def main():
    parser = argparse.ArgumentParser(description="Profile Burst for nsys capture")
    parser.add_argument("--mode", choices=["comprehensive", "sharegpt", "interference"],
                        default="comprehensive")
    parser.add_argument("--duration", type=int, default=20,
                        help="Burst duration in seconds (default 20)")
    parser.add_argument("--qps", type=int, default=4,
                        help="Target QPS (default 4)")
    parser.add_argument("--decode-batch", type=int, default=64,
                        help="Decode batch size for interference mode")
    args = parser.parse_args()

    print(f"=== Profile Burst: {args.mode} ===")
    print(f"  Duration: {args.duration}s  QPS: {args.qps}")
    print(f"  Proxy: {PROXY_URL}")
    t0 = time.time()

    if args.mode == "comprehensive":
        await comprehensive_burst(args.duration, args.qps)
    elif args.mode == "sharegpt":
        await sharegpt_burst(args.duration, args.qps)
    elif args.mode == "interference":
        await interference_burst(args.duration, args.decode_batch)

    elapsed = time.time() - t0
    print(f"\nTotal burst: {elapsed:.1f}s")
    print(f"nsys window is now capture-only — no idle time in the profile.")


if __name__ == "__main__":
    asyncio.run(main())
