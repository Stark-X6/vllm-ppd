#!/usr/bin/env python3
"""Reproduce the pD-node tools-400: send a growing multi-turn tool-calling
conversation through the PPD proxy until the request is rejected.

Hypothesis under test: with a large prompt (close to / over max_model_len),
something in the proxy->vLLM path corrupts the tools field into a truncated
JSON string. We replay the same body both through the proxy and directly to
the pD node to isolate which hop breaks.
"""
import json
import sys
import time
import urllib.request

PROXY = "http://localhost:10001/v1/chat/completions"
PD = "http://172.17.0.6:8201/v1/chat/completions"  # pD node 2 (proxy also routes here)
MODEL = "/root/autodl-tmp/Qwen3-32B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Apply a string replacement edit to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "file path"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in a persistent sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
            },
        },
    },
]

# One big observation (~3K chars) simulating a SWE-bench shell output
BIG_OBS = "task 0 output line\n" + "".join(f"def helper_{i}(x): return x + {i}\n" for i in range(80))


def build_messages(n_turns: int) -> list[dict]:
    msgs = [{"role": "user", "content": "Solve the SWE task: fix the bug in drop_column."}]
    for i in range(n_turns):
        msgs.append({
            "role": "assistant",
            "content": f"Let me check the failing test. Step {i}.",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": f"pytest -q test_{i}.py"})},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "name": "shell",
                     "content": f"{BIG_OBS}\n--- step {i} output ---"})
    msgs.append({"role": "user", "content": "Continue solving."})
    return msgs


def post(url: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, resp.read().decode()[:600]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "proxy"
    url = PROXY if target == "proxy" else PD
    n_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    # simulate the agent's per-turn max_tokens clamp (headroom below 24576)
    max_tokens = 512
    for i in range(1, n_turns + 1):
        msgs = build_messages(i)
        body = {"model": MODEL, "messages": msgs, "tools": TOOLS,
                "tool_choice": "required", "max_tokens": max_tokens,
                "temperature": 0.2}
        t0 = time.time()
        status, text = post(url, body)
        dt = time.time() - t0
        ok = "OK" if "choices" in text else text[:200].replace("\n", " ")
        print(f"[{target}] turns={i} msgs={len(msgs)} body={len(json.dumps(body))}B "
              f"status={status} {dt:.1f}s -> {ok}")
        if status != 200 or "choices" not in text:
            return


if __name__ == "__main__":
    main()
