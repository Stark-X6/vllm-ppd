"""Tests for live-agent feature extraction in the PPD proxy."""

from ppd import comprehensive_proxy as proxy


class _SuffixTokenizer:
    chat_template = "test-template"

    def __init__(self, token_count: int):
        self.token_count = token_count
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return list(range(self.token_count))


def test_chat_append_uses_suffix_tokenizer_without_retokenizing_context(monkeypatch):
    tokenizer = _SuffixTokenizer(73)
    monkeypatch.setattr(proxy, "_FEATURE_TOKENIZER", tokenizer)
    monkeypatch.setattr(proxy, "_FEATURE_TOKENIZER_PATH", "/models/qwen")
    monkeypatch.setattr(proxy, "_FEATURE_TOKENIZER_ERROR", None)

    data = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "reason",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "shell",
                "content": "Observation:\n/testbed",
            },
        ],
        "max_tokens": 128,
    }

    input_tokens, output_tokens, context_tokens, metadata = (
        proxy._extract_request_features_with_metadata(data)
    )

    assert input_tokens == 73
    assert output_tokens == 128
    assert context_tokens > 0
    assert metadata["feature_token_estimator"] == "hf_suffix_chat_template_v1"
    assert metadata["assistant_reasoning_replayed"] is True
    assert metadata["append_message_roles"] == ["tool"]
    assert len(tokenizer.calls) == 1
    messages, kwargs = tokenizer.calls[0]
    assert [message["role"] for message in messages] == ["tool"]
    assert kwargs["tools"] is None
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is True


def test_chat_append_falls_back_when_tokenizer_is_disabled(monkeypatch):
    monkeypatch.setattr(proxy, "_FEATURE_TOKENIZER", None)
    monkeypatch.setattr(proxy, "_FEATURE_TOKENIZER_PATH", None)
    monkeypatch.setattr(proxy, "_FEATURE_TOKENIZER_ERROR", None)
    data = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "Observation:\nshort"},
        ]
    }

    input_tokens, _, _, metadata = proxy._extract_request_features_with_metadata(data)

    assert input_tokens == proxy.estimate_message_token_count(data["messages"][-1])
    assert metadata["feature_token_estimator"] == "chars_per_4_with_chat_overhead_v1"
    assert metadata["assistant_reasoning_replayed"] is False


def test_message_estimate_counts_reasoning_once():
    reasoning = "private reasoning"
    with_reasoning = proxy.estimate_message_token_count({
        "role": "assistant",
        "content": "",
        "reasoning": reasoning,
        "reasoning_content": reasoning,
    })
    without_reasoning = proxy.estimate_message_token_count({
        "role": "assistant",
        "content": "",
    })

    assert with_reasoning - without_reasoning == len(reasoning) // 4
