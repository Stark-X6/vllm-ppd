#!/usr/bin/env python3
"""
PPD Tool Call Profiling — 同时监控 agent 的 tool call 耗时 + vLLM GPU 负载

用法:
  python scripts/profile_tool_calls.py --rounds 10

输出 per-step:
  - LLM 耗时 (GPU 忙)
  - Tool 耗时 (GPU 空闲)
  - GPU 利用率采样 (验证 GPU 在 tool call 期间是否真的空闲)
"""
import sys, os, time, subprocess, json, asyncio, argparse
from pathlib import Path
from datetime import datetime
import threading

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, '/root/vllm-continuum/mini-swe-agent/src')
from minisweagent.models.vllm_model import VllmModel, VllmModelConfig

# ── GPU 监控 ──
class GPUMonitor:
    """后台线程采样 GPU 利用率"""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _loop(self):
        while self._running:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                parts = out.stdout.strip().split(",")
                if len(parts) >= 4:
                    self.samples.append({
                        "ts": time.time(),
                        "gpu_util": int(parts[0].strip()),
                        "mem_used_mb": int(parts[1].strip()),
                        "mem_total_mb": int(parts[2].strip()),
                        "temp": int(parts[3].strip()),
                    })
            except Exception:
                pass
            time.sleep(self.interval)

    def stats(self, start_ts, end_ts):
        """返回一段时间内的 GPU 统计"""
        window = [s for s in self.samples if start_ts <= s["ts"] <= end_ts]
        if not window:
            return {"gpu_util_avg": 0, "gpu_util_max": 0, "mem_avg_mb": 0, "samples": 0}
        return {
            "gpu_util_avg": sum(s["gpu_util"] for s in window) / len(window),
            "gpu_util_max": max(s["gpu_util"] for s in window),
            "mem_avg_mb": sum(s["mem_used_mb"] for s in window) / len(window),
            "samples": len(window),
        }

# ── 安全命令 ──
def run_cmd(command, timeout=15):
    try:
        r = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout)
        return f"<returncode>{r.returncode}</returncode>\n<output>{(r.stdout+r.stderr)[:2000]}</output>"
    except subprocess.TimeoutExpired:
        return "<returncode>124</returncode>\n<output>[timeout after {timeout}s]</output>"

SYSTEM_PROMPT = """You are a code analysis assistant. Analyze the repo step by step.
Format:
THOUGHT: <reasoning>
```bash
<command>
```"""

# ── 探索任务 (逐步深入，产生不同的 tool call) ──
EXPLORATION_STEPS = [
    "List all files and directories in /root/vllm-ppd/",
    "Check the size of the ppd/ directory and its Python files",
    "Read the first 50 lines of ppd/comprehensive_proxy.py",
    "Search for 'def route' or 'def select' in ppd/comprehensive_proxy.py",
    "Count how many lines of code are in ppd/optimizer/ppd_decision_engine.py",
    "Check disk usage of /root/vllm-ppd/ with du",
    "Look at the most recent log files in logs/",
    "Check what Python processes are running with ps aux",
    "Check memory usage with free -h",
    "Read ppd/config.py to understand the project configuration",
    "Search for 'class.*Engine' in ppd/ directory",
    "Count total Python files in the ppd/ directory tree",
    "Check git log for recent changes",
    "Read the interference benchmark results",
    "Check NVIDIA GPU status with nvidia-smi if available",
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--base-url", default="http://localhost:8300/v1")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    model = VllmModel(
        config_class=VllmModelConfig,
        model_name=os.environ.get("MODEL_PATH", "/root/autodl-tmp/Llama-3.1-8B"),
        base_url=args.base_url,
        max_completion_tokens=args.max_tokens,
        timeout=120.0,
        step_limit=args.rounds,
    )

    gpu = GPUMonitor(interval=0.3)
    gpu.start()

    trace = []
    # 每轮一个独立子任务，避免 context 爆炸
    for rnd in range(min(args.rounds, len(EXPLORATION_STEPS))):
        task = EXPLORATION_STEPS[rnd]
        # 只用 system + 当前任务 (1 turn)，context 短
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        # ── LLM call ──
        t0 = time.time()
        resp = model.query(messages)
        t_llm = time.time() - t0
        content = resp["content"]

        cmd = ""
        if "```bash" in content:
            cmd = content.split("```bash")[1].split("```")[0].strip()

        # ── Tool execution (GPU monitor 在后台记录这段时间) ──
        t_tool_start = time.time()
        obs = run_cmd(cmd, timeout=15)
        t_tool = time.time() - t_tool_start

        # GPU 统计
        gpu_llm = gpu.stats(t0, t0 + t_llm)
        gpu_tool = gpu.stats(t_tool_start, t_tool_start + t_tool)

        step = {
            "round": rnd + 1,
            "task": task[:60],
            "llm_s": round(t_llm, 2),
            "tool_s": round(t_tool, 2),
            "cmd": cmd[:80],
            "gpu_llm_util": round(gpu_llm["gpu_util_avg"], 1),
            "gpu_llm_max": round(gpu_llm["gpu_util_max"], 1),
            "gpu_tool_util": round(gpu_tool["gpu_util_avg"], 1),
            "gpu_tool_max": round(gpu_tool["gpu_util_max"], 1),
        }
        trace.append(step)

        tag = "🔴LONG" if t_tool > 3 else "🟢"
        print(f"  R{rnd+1:2d} {tag} LLM={t_llm:.1f}s GPU={gpu_llm['gpu_util_avg']:.0f}% | "
              f"Tool={t_tool:.1f}s GPU={gpu_tool['gpu_util_avg']:.0f}% | "
              f"{cmd[:60]}...")

    gpu.stop()

    # ── 分析 ──
    print(f"\n{'='*70}")
    print(f"=== Tool Call 瓶颈分析 ===")
    llm_times = [s["llm_s"] for s in trace]
    tool_times = [s["tool_s"] for s in trace]
    gpu_llm_utils = [s["gpu_llm_util"] for s in trace]
    gpu_tool_utils = [s["gpu_tool_util"] for s in trace]

    long_tools = [s for s in trace if s["tool_s"] > 3]
    print(f"\n总步数: {len(trace)}")
    print(f"LLM:   avg={sum(llm_times)/len(llm_times):.1f}s  GPU avg={sum(gpu_llm_utils)/len(gpu_llm_utils):.0f}%")
    print(f"Tool:  avg={sum(tool_times)/len(tool_times):.1f}s  GPU avg={sum(gpu_tool_utils)/len(gpu_tool_utils):.0f}%")
    print(f"\n长时间 tool call (>3s): {len(long_tools)}个")
    for s in long_tools:
        print(f"  R{s['round']}: {s['tool_s']:.1f}s | GPU={s['gpu_tool_util']:.0f}% | {s['cmd'][:60]}")
    print(f"\n关键结论:")
    print(f"  LLM 阶段 GPU 利用率: {sum(gpu_llm_utils)/len(gpu_llm_utils):.0f}%  (GPU busy)")
    print(f"  Tool 阶段 GPU 利用率: {sum(gpu_tool_utils)/len(gpu_tool_utils):.0f}%  (GPU idle)")

    tool_total = sum(tool_times)
    total = sum(llm_times) + tool_total
    print(f"  GPU 空闲比例: {tool_total/total*100:.0f}% ({tool_total:.0f}s / {total:.0f}s)")
    print(f"  → PPD TTL 机会: 如果 pin KV cache 越过 tool call, ")
    print(f"    可避免 {len(trace)} 次 full prefill, 节省 ~{sum(llm_times):.0f}s LLM 时间")

    # 保存
    out = PROJECT_DIR / "results" / "tool_profile.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
