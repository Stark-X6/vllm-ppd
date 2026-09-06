# Agent Serving Routing and Request-Profile Contract

This document freezes the experiment contract used for matched Static PPD and
PPD-Dynamic live-Agent runs. It changes observability and retry correctness; it
does **not** change the original Dynamic decision policy.

## 1. Identity and deterministic routing

The following identifiers have deliberately different lifetimes:

| Identifier | Meaning | Lifetime/stability |
|---|---|---|
| `run_id` | One profiling run | New for every run |
| `episode_id` / `conversation_id` | One live Agent rollout and its KV affinity | New random value for every rollout |
| `client_request_id` | One logical Agent step (one model query) | Reused by all HTTP and required→auto retries of that query |
| `routing_key` | Experimental sample assignment | Stable across matched runs; `instance_id|rollout=N` |

Conversation state is keyed by `run_id + conversation_id + PPD_PROXY_EPOCH`.
It controls turn count and cache affinity, and therefore cannot leak across
stack epochs. Server assignment is independently keyed by `routing_key`, so a
random episode UUID no longer changes the sample's P/D placement.

In the default `stable_hash` mode:

1. server candidates are sorted by numeric port and full address;
2. each role's slot is derived from SHA-256 of
   `ppd-routing-v1 + seed + role + routing_key`;
3. the seed, key hash, candidate list, selected slot and assignment source are
   saved with every proxy request.

`legacy_hash` is an explicit diagnostic mode that retains both Python's
process-randomized `hash()` and registry insertion order. It must not be used
for matched results.

For the first 16 SWE-bench Verified samples, use the checked-in balanced map:

```text
configs/routing/swebench_verified_first16_balanced_2x2.json
```

It gives exactly 8/8 assignments to both P slots and both D/pD slots. Strict
mode fails the request when a routing key or required role is absent instead
of silently falling back to hashing.

## 2. Logical request and retry contract

The Agent sends the same `X-PPD-Client-Request-ID` for transport retries and
for the one `tool_choice=required` → `tool_choice=auto` recovery request. The
proxy persists the first physical attempt's:

- logical turn;
- P/D/replica addresses;
- routing mode and Dynamic decision;
- decision features, QPS snapshot and calibration cell.

Every later physical attempt with the same run-scoped conversation state key
and client request ID reuses that record. A retry therefore cannot increment
the turn, choose another node, or make a second Dynamic decision. The proxy
reports `logical_requests`, `physical_requests`, `retry_requests`,
`logical_retry` and `logical_physical_attempt` separately.

## 3. Prefill-only request contract

For `pd` and Turn-1 `ppd`, the P request retains the original messages and
complete tools/schema, but is forced to:

```text
tool_choice = none
max_tokens = 1
stream = false
```

The vendored vLLM default `exclude_tools_when_tool_choice_none=false` is kept,
so the P and D prompts still contain the same tools. Changing `tool_choice`
only avoids strict post-generation parsing of a deliberately truncated
required-tool JSON array. The contract is recorded as
`same_prompt_tool_choice_none_max_tokens_1_v1` on each applicable request.

## 4. Schema versions and artifacts

| Producer/artifact | Version | Purpose |
|---|---:|---|
| Agent episode trace | 4 | request/response, reasoning, tools and stable routing key |
| `proxy_timing.jsonl` | 4 | one row per physical proxy request |
| response `ppd_request_metrics` | 1 | vLLM per-request engine phase/tokens |
| `request_profile.jsonl/.csv` | 1 | analyzer join of proxy, Agent and node metrics |

`scripts/run_agent_profile.sh` also records the two Git commits, run command,
proxy status at start/end, and routing map fingerprint. The JSONL request
profile retains every field; CSV provides the stable analysis columns.

## 5. Request-level phase semantics

The non-streaming vLLM Chat Completion response contains:

| Field | Exact meaning |
|---|---|
| `queue_s` | First engine-core `QUEUED` to first `SCHEDULED` event |
| `prefill_elapsed_s` | First schedule to first generated token |
| `decode_elapsed_s` | First generated token to last generated token |
| `inference_elapsed_s` | First schedule to last generated token |
| `ttft_s` | Engine frontend arrival to first generated token |
| `e2e_s` | `ttft_s + decode_elapsed_s` |
| `frontend_overhead_s` | Residual `ttft - queue - prefill`; not proxy HTTP time |
| `num_preemptions` | Scheduler preemptions observed for this request |
| `prompt_tokens` | Full rendered input tokens |
| `cached_prompt_tokens` | Prefix tokens reused by vLLM |
| `uncached_prompt_tokens` | `prompt_tokens - cached_prompt_tokens` |
| `completion_tokens` | Actual generated tokens |
| `requested_max_tokens` | Requested generation cap, not a prediction |

`prefill_elapsed_s` is intentionally called *elapsed*: it includes actual
prompt compute/recompute plus stalls and preemptions between first schedule and
first token. It is not pure GPU kernel time. Similarly,
`uncached_prompt_tokens` contains the legitimate new append as well as any lost
reusable prefix. The analyzer therefore emits separate
`uncached_prompt_tokens_observed`, `previous_reusable_tokens`,
`lost_reusable_tokens` and `cache_retention` fields. It must not label all
uncached tokens as eviction.

For a non-streaming response, proxy `http_first_byte_s` is approximately the
completed-response time, not model TTFT. Use `*_engine_ttft_s` for TTFT and
`*_http_s` for the full proxy-to-server request.

## 6. Server-load semantics

Each request profile records P and generation-node (D/pD) data independently:

- `prefill_load_{enter,max,exit}_{running,waiting,kv_usage}`;
- `decode_load_{enter,max,exit}_{running,waiting,kv_usage}`;
- P/D node preemption counter deltas in the request window;
- response-time P/D ORCA waiting and KV usage;
- exact proxy in-flight request counts before routing;
- other live episodes assigned to the same generation node.

The unprefixed `load_*` and `node_preemptions_delta_during` fields remain
decode/generation-node compatibility aliases.

The two concurrency measures are different by design:

- `proxy_other_inflight_conversations_on_decode` counts other model requests
  that overlap at the proxy at that instant;
- `same_node_other_active_episodes` counts other whole Agent lifetimes whose
  assigned node is the same, even while those Agents are executing tools.

Node metrics are sampled, normally every 0.5 seconds. Entry/exit join deltas
are stored so a missing or distant sample is visible rather than silently
treated as exact.

## 7. Dynamic decision metadata and QPS

The original decision result is preserved. Each request now explains it with:

- `decision_source`, such as `turn1_forced_pd`, `short_append_bypass`,
  `matched_calibration_score` or a named missing-calibration fallback;
- requested context/append/output features and their classification;
- requested QPS, selected discrete QPS and `selected_calibration_cell`;
- calibration availability and metrics when present.

The policy input remains the paper-code lifetime rate:

```text
prior logical requests / max(proxy uptime, 1 second)
```

Retries are excluded. Observational 1/5/10/60-second logical-arrival window
rates are logged separately and do not affect the decision.

## 8. Matched-run startup

Set the routing contract before starting either stack:

```bash
export PPD_ROUTING_MODE=stable_hash
export PPD_ROUTING_SEED=agentic-ppd-v1
export PPD_ROUTING_MAP_PATH=/root/vllm-ppd/configs/routing/swebench_verified_first16_balanced_2x2.json
export PPD_ROUTING_MAP_STRICT=true
```

Then start Static with `scripts/server/start_2P_2pD.sh`, or original Dynamic
with `ENABLE_PPD_MODE=true scripts/server/start_2P_2D.sh`. After startup,
`GET /status` must show the expected map path, fingerprint, 16 assignments and
strict mode before a matched run is accepted.

No result should be compared across configs unless the sample order, routing
key set, map fingerprint, model/Agent commits, model parameters, concurrency
and episode count all match.
