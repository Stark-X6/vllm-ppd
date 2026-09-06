import json
import time

import pytest

from ppd import comprehensive_proxy as proxy
from ppd.routing_contract import RoutingContract


def _reset_proxy_state(monkeypatch):
    for registry in (
        proxy.prefill_instances,
        proxy.decode_instances,
        proxy.ppd_decode_instances,
        proxy.replica_instances,
        proxy.conversation_state,
        proxy.logical_request_state,
    ):
        registry.clear()
    proxy.logical_arrivals.clear()
    with proxy.stats_lock:
        for key in list(proxy.stats):
            proxy.stats[key] = 0
        proxy.stats["start_time"] = time.time() - 10
    monkeypatch.setattr(proxy, "ENABLE_PPD_MODE", False)
    monkeypatch.setattr(
        proxy, "ROUTING_CONTRACT", RoutingContract(seed="request-contract-test")
    )


def test_state_key_is_run_scoped_but_routing_identity_is_not(monkeypatch):
    monkeypatch.setenv("PPD_PROXY_EPOCH", "epoch-1")
    data = {"messages": [{"role": "user", "content": "same task"}]}

    state_a = proxy.get_conversation_hash(data, conversation_id="ep-a", run_id="run-a")
    state_b = proxy.get_conversation_hash(data, conversation_id="ep-b", run_id="run-b")
    routing_a = proxy.get_routing_identity(
        data, explicit_routing_key="astropy__astropy-1|rollout=0"
    )
    routing_b = proxy.get_routing_identity(
        data, explicit_routing_key="astropy__astropy-1|rollout=0"
    )

    assert state_a != state_b
    assert routing_a == routing_b == ("astropy__astropy-1|rollout=0", "header")


def test_select_servers_uses_sorted_frozen_slots(tmp_path, monkeypatch):
    _reset_proxy_state(monkeypatch)
    routing_key = "sample|rollout=0"
    map_path = tmp_path / "routing.json"
    map_path.write_text(json.dumps({
        "schema_version": 1,
        "routing_seed": "matched",
        "assignments": {
            routing_key: {"prefill_slot": 0, "decode_slot": 1},
        },
    }))
    monkeypatch.setattr(
        proxy,
        "ROUTING_CONTRACT",
        RoutingContract(
            seed="matched", map_path=str(map_path), map_strict=True
        ),
    )
    proxy.prefill_instances.update({
        "127.0.0.1:8101": ("p1", time.time() + 60),
        "127.0.0.1:8100": ("p0", time.time() + 60),
    })
    proxy.ppd_decode_instances.update({
        "127.0.0.1:8201": ("d1", time.time() + 60),
        "127.0.0.1:8200": ("d0", time.time() + 60),
    })

    prefill, decode, _, route, metadata = proxy.select_servers(
        "state-key",
        1,
        {},
        routing_key=routing_key,
        include_metadata=True,
    )

    assert prefill == "127.0.0.1:8100"
    assert decode == "127.0.0.1:8201"
    assert route == "ppd"
    assert metadata["decode_candidates"] == [
        "127.0.0.1:8200", "127.0.0.1:8201"
    ]
    assert metadata["decode_assignment_source"] == "frozen_map"


@pytest.mark.asyncio
async def test_retry_reuses_logical_turn_assignment_and_prefill_contract(monkeypatch):
    _reset_proxy_state(monkeypatch)
    proxy.prefill_instances["127.0.0.1:8100"] = ("tcp://p0", time.time() + 60)
    proxy.ppd_decode_instances["127.0.0.1:8200"] = (
        "tcp://d0", time.time() + 60
    )
    calls = []
    timing = []

    async def fake_forward(url, data, request_id, response_meta=None):
        calls.append((url, dict(data), request_id))
        now = time.time()
        response_meta.update({
            "status": 200,
            "http_start_ts": now,
            "http_headers_ts": now + 0.001,
            "http_first_byte_ts": now + 0.002,
            "http_end_ts": now + 0.003,
            "orca": {"num_requests_waiting": 0.0, "kv_cache_usage_perc": 0.25},
        })
        body = {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
            "ppd_request_metrics": {
                "schema_version": 1,
                "queue_s": 0.1,
                "prefill_elapsed_s": 0.2,
                "decode_elapsed_s": 0.0,
                "num_preemptions": 0,
                "prompt_tokens": 100,
                "cached_prompt_tokens": 80,
                "uncached_prompt_tokens": 20,
                "completion_tokens": 1,
                "requested_max_tokens": data.get("max_tokens"),
            },
        }
        yield json.dumps(body).encode()

    monkeypatch.setattr(proxy, "forward_request", fake_forward)
    monkeypatch.setattr(proxy, "_write_timing", lambda row: timing.append(row))
    client = proxy.app.test_client()
    base_headers = {
        "X-PPD-Run-ID": "run-a",
        "X-PPD-Episode-ID": "random-episode",
        "X-PPD-Conversation-ID": "random-episode",
        "X-PPD-Step-ID": "1",
        "X-PPD-Client-Request-ID": "logical-request-1",
        "X-PPD-Routing-Key": "astropy__astropy-1|rollout=0",
    }
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": "fix it"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "view_file",
                "description": "view",
                "parameters": {"type": "object"},
            },
        }],
        "tool_choice": "required",
        "max_tokens": 32,
    }
    first = await client.post("/v1/chat/completions", json=body, headers=base_headers)
    retry_headers = {
        **base_headers,
        "X-PPD-Request-Phase": "tool_choice_auto_retry",
        "X-PPD-Request-Attempt": "1",
    }
    retry = await client.post(
        "/v1/chat/completions",
        json={**body, "tool_choice": "auto"},
        headers=retry_headers,
    )

    assert first.status_code == retry.status_code == 200
    assert len(calls) == 4  # P + D for both physical attempts
    prefill_calls = [call for call in calls if ":8100" in call[0]]
    assert all(call[1]["tool_choice"] == "none" for call in prefill_calls)
    assert all(call[1]["stream"] is False for call in prefill_calls)
    assert all(call[1]["max_tokens"] == 1 for call in prefill_calls)
    assert [row["turn"] for row in timing] == [1, 1]
    assert [row["logical_retry"] for row in timing] == [False, True]
    assert timing[0]["decode_addr"] == timing[1]["decode_addr"]
    assert timing[0]["routing_key_label"] == "astropy__astropy-1|rollout=0"
    assert timing[0]["decode_engine_queue_s"] == 0.1
    assert timing[0]["uncached_prompt_tokens"] == 20
    assert proxy.stats["logical_requests"] == 1
    assert proxy.stats["physical_requests"] == 2
    assert proxy.stats["retry_requests"] == 1
