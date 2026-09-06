import json

import pytest

from ppd.routing_contract import RoutingContract


def test_stable_hash_is_independent_of_registry_order():
    contract = RoutingContract(mode="stable_hash", seed="matched-v1")
    unordered = ["10.0.0.1:8201", "10.0.0.1:8200"]
    ordered = contract.sorted_addresses(unordered)

    first = contract.choose_slot("sample|rollout=0", "decode", len(ordered))
    second = contract.choose_slot("sample|rollout=0", "decode", len(ordered))

    assert ordered == ["10.0.0.1:8200", "10.0.0.1:8201"]
    assert first == second
    assert first.source == "stable_sha256"


def test_frozen_map_overrides_hash_and_fails_closed(tmp_path):
    path = tmp_path / "routing.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "routing_seed": "matched-v1",
        "assignments": {
            "sample-a|rollout=0": {"prefill_slot": 0, "decode_slot": 1},
        },
    }))
    contract = RoutingContract(
        mode="stable_hash",
        seed="matched-v1",
        map_path=str(path),
        map_strict=True,
    )

    decision = contract.choose_slot("sample-a|rollout=0", "decode", 2)
    assert decision.slot == 1
    assert decision.source == "frozen_map"
    assert contract.descriptor()["map_fingerprint_sha256"]
    with pytest.raises(KeyError):
        contract.choose_slot("sample-b|rollout=0", "decode", 2)


def test_map_rejects_seed_mismatch(tmp_path):
    path = tmp_path / "routing.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "routing_seed": "different",
        "assignments": {"sample": {"decode_slot": 0}},
    }))

    with pytest.raises(ValueError, match="does not match"):
        RoutingContract(seed="matched-v1", map_path=str(path))


def test_legacy_mode_preserves_registry_order():
    contract = RoutingContract(mode="legacy_hash")
    addresses = ["127.0.0.1:8201", "127.0.0.1:8200"]

    assert contract.candidate_addresses(addresses) == addresses
    assert RoutingContract().candidate_addresses(addresses) == list(reversed(addresses))
