from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.serving_chat import _build_ppd_request_metrics


def test_request_metrics_expose_engine_phases_and_token_accounting():
    state = SimpleNamespace(
        queued_ts=10.0,
        scheduled_ts=10.4,
        first_token_ts=12.0,
        last_token_ts=15.5,
        first_token_latency=2.25,
        num_preemptions=3,
    )
    output = SimpleNamespace(metrics=state, num_cached_tokens=80)

    metrics = _build_ppd_request_metrics(
        output,
        prompt_tokens=100,
        completion_tokens=20,
        requested_max_tokens=64,
    )

    assert metrics.queue_s == pytest.approx(0.4)
    assert metrics.prefill_elapsed_s == pytest.approx(1.6)
    assert metrics.decode_elapsed_s == pytest.approx(3.5)
    assert metrics.inference_elapsed_s == pytest.approx(5.1)
    assert metrics.ttft_s == pytest.approx(2.25)
    assert metrics.e2e_s == pytest.approx(5.75)
    assert metrics.frontend_overhead_s == pytest.approx(0.25)
    assert metrics.num_preemptions == 3
    assert metrics.prompt_tokens == 100
    assert metrics.cached_prompt_tokens == 80
    assert metrics.uncached_prompt_tokens == 20
    assert metrics.completion_tokens == 20
    assert metrics.requested_max_tokens == 64


def test_request_metrics_remain_available_without_engine_stats():
    output = SimpleNamespace(metrics=None, num_cached_tokens=24)

    metrics = _build_ppd_request_metrics(
        output,
        prompt_tokens=32,
        completion_tokens=2,
        requested_max_tokens=None,
    )

    assert metrics.queue_s is None
    assert metrics.prefill_elapsed_s is None
    assert metrics.uncached_prompt_tokens == 8
    assert metrics.completion_tokens == 2
