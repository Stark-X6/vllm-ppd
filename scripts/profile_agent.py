#!/usr/bin/env python3
"""
PPD Agent Profiling — 真实多轮 think→act→observe 循环
不依赖 Docker/bubblewrap，用 LocalEnvironment 安全执行命令

用法:
  # 单 GPU (当前):
  python scripts/profile_agent.py --rounds 10 --base-url http://localhost:8300/v1

  # 4 GPU + PPD proxy (后续):
  python scripts/profile_agent.py --rounds 10 --base-url http://localhost:10001/v1

生成:
  - 每轮: LLM 调用 (append-prefill) → 命令执行 → observation → 下一轮
  - 真实模拟 SWE agent 行为，但用简单安全的任务
"""

import sys, os, time, subprocess, argparse, json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, '/root/vllm-continuum/mini-swe-agent/src')

from minisweagent.models.vllm_model import VllmModel, VllmModelConfig

# ── 安全命令白名单 ──
SAFE_COMMANDS = {
    "ls":    "ls -la {path}",
    "find":  "find {path} -maxdepth 2 -type f | head -20",
    "cat":   "cat {path}",
    "wc":    "wc -l {path}",
    "du":    "du -sh {path}",
    "df":    "df -h",
    "free":  "free -h",
    "ps":    "ps aux | head -10",
    "echo":  "echo '{msg}'",
    "head":  "head -30 {path}",
    "grep":  "grep -n '{pattern}' {path} | head -10",
    "python": "python3 -c '{code}'",
}

def run_safe_command(command: str, cwd: str = "/root", timeout: int = 15) -> dict:
    """在本地安全执行命令，返回 output + returncode"""
    try:
        result = subprocess.run(
            command, shell=True, text=True, cwd=cwd,
            timeout=timeout, capture_output=True,
            env={**os.environ, "PAGER": "cat", "PIP_PROGRESS_BAR": "off"},
        )
        return {"output": result.stdout + result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"output": "[command timed out]", "returncode": 124}

# ── 多轮 agent 任务模板 ──
AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant that analyzes code repositories.
For each response:
1. Include a THOUGHT section explaining your reasoning
2. Provide exactly ONE bash command to execute

Format:
THOUGHT: <your reasoning>
```bash
<single command>
```"""

TASKS = [
    "Explore the /root/vllm-ppd directory structure. List files, check sizes, and understand the project layout. Continue exploring for {rounds} rounds.",
    "Analyze the Python files in /root/vllm-ppd/ppd/. Read key files, understand the proxy and optimizer structure. Continue for {rounds} rounds.",
    "Check the system resources: disk usage, memory, running processes. Report findings over {rounds} rounds of investigation.",
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Number of agent rounds")
    parser.add_argument("--base-url", default="http://localhost:8300/v1")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results" / "agent_profile.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--task-idx", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = VllmModel(
        config_class=VllmModelConfig,
        model_name=os.environ.get("MODEL_PATH", "/root/autodl-tmp/Llama-3.1-8B"),
        base_url=args.base_url,
        max_completion_tokens=args.max_tokens,
        timeout=120.0,
        step_limit=args.rounds,
    )

    task = TASKS[args.task_idx].format(rounds=args.rounds)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    trace = {
        "config": {"rounds": args.rounds, "base_url": args.base_url, "max_tokens": args.max_tokens},
        "task": task,
        "rounds": [],
        "start_time": datetime.now().isoformat(),
    }

    print(f"=== PPD Agent Profiling ===")
    print(f"Rounds: {args.rounds} | URL: {args.base_url} | Max tokens: {args.max_tokens}")
    print(f"Task: {task[:80]}...\n")

    for rnd in range(args.rounds):
        # ── LLM call (这个就是 profiling 目标) ──
        t_llm_start = time.perf_counter()
        resp = model.query(messages)
        t_llm = time.perf_counter() - t_llm_start
        content = resp["content"]

        # 解析命令
        cmd = ""
        if "```bash" in content:
            cmd = content.split("```bash")[1].split("```")[0].strip()

        # ── 命令执行 (agent pause — 测试 PPD TTL) ──
        obs_text = ""
        t_tool_start = time.perf_counter()
        if cmd:
            obs = run_safe_command(cmd)
            t_tool = time.perf_counter() - t_tool_start
            obs_text = f"<returncode>{obs['returncode']}</returncode>\n<output>{obs['output'][:2000]}</output>"
        else:
            t_tool = 0
            obs_text = "<output>(no command)</output>"

        # ── 记录 ──
        round_data = {
            "round": rnd + 1,
            "llm_time_s": round(t_llm, 3),
            "tool_time_s": round(t_tool, 3),
            "model_calls": model.n_calls,
            "command": cmd[:200],
            "response_len": len(content),
        }
        trace["rounds"].append(round_data)

        toks = content.replace('\n', ' ')[:80]
        print(f"  R{rnd+1:2d} | LLM: {t_llm:.1f}s | Tool: {t_tool:.1f}s | {toks}...")

        # ── 追加到 messages (append-prefill 场景!) ──
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": obs_text})

    # ── 统计 ──
    print(f"\n=== Summary ===")
    llm_times = [r["llm_time_s"] for r in trace["rounds"]]
    tool_times = [r["tool_time_s"] for r in trace["rounds"]]
    total = sum(llm_times) + sum(tool_times)
    print(f"Rounds: {args.rounds}  |  Total LLM calls: {model.n_calls}")
    print(f"LLM:  avg={sum(llm_times)/len(llm_times):.1f}s  total={sum(llm_times):.1f}s  ({sum(llm_times)/total*100:.0f}%)")
    print(f"Tool: avg={sum(tool_times)/len(tool_times):.1f}s  total={sum(tool_times):.1f}s  ({sum(tool_times)/total*100:.0f}%)")

    trace["summary"] = {
        "rounds": args.rounds, "total_model_calls": model.n_calls,
        "total_llm_time_s": sum(llm_times), "total_tool_time_s": sum(tool_times),
        "avg_llm_time_s": sum(llm_times)/len(llm_times),
        "end_time": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"\nTrace saved to {output_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
