from scripts.profile_analyze_v2 import build_request_profiles


def test_request_profile_joins_both_nodes_and_live_episodes():
    proxy = [
        {
            "episode_id": "ep-a",
            "client_request_id": "req-a",
            "ts_enter": 10.0,
            "ts_exit": 12.0,
            "prefill_addr": "127.0.0.1:8100",
            "decode_addr": "127.0.0.1:8200",
            "generation_addr": "127.0.0.1:8200",
            "prompt_tokens": 100,
            "cached_tokens": 80,
            "decode_request_metrics": {"uncached_prompt_tokens": 20},
        },
        {
            "episode_id": "ep-b",
            "client_request_id": "req-b",
            "ts_enter": 13.0,
            "ts_exit": 14.0,
            "decode_addr": "127.0.0.1:8200",
        },
    ]
    metrics = [
        {
            "ts": 9.9,
            "port": 8100,
            "status": "ok",
            "running": 3,
            "waiting": 1,
            "cache_usage_pct": 0.2,
            "gpu_util_pct": 40,
            "preemptions": 1,
        },
        {
            "ts": 11.0,
            "port": 8100,
            "status": "ok",
            "running": 4,
            "waiting": 2,
            "cache_usage_pct": 0.3,
            "gpu_util_pct": 50,
            "preemptions": 2,
        },
        {
            "ts": 9.9,
            "port": 8200,
            "status": "ok",
            "running": 1,
            "waiting": 2,
            "cache_usage_pct": 0.4,
            "gpu_util_pct": 30,
            "preemptions": 4,
        },
        {
            "ts": 11.0,
            "port": 8200,
            "status": "ok",
            "running": 2,
            "waiting": 5,
            "cache_usage_pct": 0.7,
            "gpu_util_pct": 90,
            "preemptions": 6,
        },
        {"ts": 11.0, "port": None, "status": "ok"},
    ]
    traces = [
        {"episode_id": "ep-a", "episode_start": 8, "episode_end": 20},
        {"episode_id": "ep-b", "episode_start": 9, "episode_end": 18},
    ]

    profile = build_request_profiles(proxy, metrics, traces)[0]

    assert profile["prefill_load_enter_waiting"] == 1
    assert profile["prefill_load_max_waiting"] == 2
    assert profile["decode_load_enter_waiting"] == 2
    assert profile["decode_load_max_waiting"] == 5
    assert profile["decode_load_max_kv_usage"] == 0.7
    assert profile["load_max_waiting"] == 5  # decode compatibility alias
    assert profile["uncached_prompt_tokens_observed"] == 20
    assert profile["previous_reusable_tokens"] is None  # synthetic rows have no steps
    assert profile["lost_reusable_tokens"] is None
    assert profile["same_node_other_active_episodes"] == 1


def test_request_profile_separates_uncached_append_from_lost_reusable_tokens():
    proxy = [{
        "episode_id": "ep-a",
        "step_id": "2",
        "client_request_id": "request-2",
        "ts_enter": 10.0,
        "ts_exit": 11.0,
        "decode_addr": "127.0.0.1:8200",
        "prompt_tokens": 150,
        "cached_tokens": 80,
        "decode_request_metrics": {"uncached_prompt_tokens": 70},
    }]
    traces = [{
        "episode_id": "ep-a",
        "episode_start": 1.0,
        "episode_end": 20.0,
        "steps_detail": [
            {
                "step": 1,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "reasoning_in_next_prompt": True,
                "assistant_message": {"reasoning": "plan", "tool_calls": [{}]},
            },
            {"step": 2, "prompt_tokens": 150, "completion_tokens": 3},
        ],
    }]

    profile = build_request_profiles(proxy, [], traces)[0]

    assert profile["uncached_prompt_tokens_observed"] == 70
    assert profile["previous_reusable_tokens"] == 120
    assert profile["lost_reusable_tokens"] == 40
    assert profile["cache_retention"] == 80 / 120

    retry = build_request_profiles([{**proxy[0], "logical_retry": True}], [], traces)[0]
    assert retry["lost_reusable_tokens"] is None
    assert retry["cache_retention"] is None
