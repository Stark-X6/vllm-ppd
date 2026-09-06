"""Deterministic, auditable server-slot assignment for matched PPD runs.

Conversation state and experiment assignment deliberately use different keys:
the former is run/epoch scoped, while the latter is stable across paired
Static/Dynamic runs.  Python's built-in ``hash`` and registry insertion order
must never determine the default experiment assignment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROUTING_CONTRACT_VERSION = "ppd-routing-v1"
ROUTING_MAP_SCHEMA_VERSION = 1
ROUTING_MODES = ("stable_hash", "legacy_hash")


def address_sort_key(address: str) -> tuple[int, str]:
    """Sort node addresses by numeric port and then full address."""
    try:
        port = int(address.rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        port = -1
    return port, str(address)


@dataclass(frozen=True)
class SlotDecision:
    slot: int
    source: str


class RoutingContract:
    """Select stable indices from sorted server registries.

    A map entry has the form::

        "instance-id|rollout=0": {
            "prefill_slot": 0,
            "decode_slot": 1,
            "replica_slot": 0,
            "service_slot": 1
        }

    ``service_slot`` is used only to choose replica-vs-disaggregated capacity in
    hybrid configurations.  With ``map_strict=True``, every requested role must
    be present; this makes a supposedly matched run fail closed instead of
    silently changing its assignment.
    """

    def __init__(
        self,
        *,
        mode: str = "stable_hash",
        seed: str = "agentic-ppd-v1",
        map_path: str | None = None,
        map_strict: bool = False,
    ) -> None:
        if mode not in ROUTING_MODES:
            raise ValueError(f"routing mode must be one of {ROUTING_MODES}, got {mode!r}")
        self.mode = mode
        self.seed = str(seed)
        self.map_path = str(Path(map_path).resolve()) if map_path else None
        self.map_strict = bool(map_strict)
        self.assignments: dict[str, dict[str, int]] = {}
        self.map_metadata: dict[str, Any] = {}
        self.map_fingerprint: str | None = None
        if self.map_path:
            self._load_map(Path(self.map_path))

    def _load_map(self, path: Path) -> None:
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("routing map root must be a JSON object")
        if document.get("schema_version") != ROUTING_MAP_SCHEMA_VERSION:
            raise ValueError(
                "routing map schema_version must be "
                f"{ROUTING_MAP_SCHEMA_VERSION}, got {document.get('schema_version')!r}"
            )
        declared_seed = document.get("routing_seed")
        if declared_seed is not None and str(declared_seed) != self.seed:
            raise ValueError(
                f"routing map seed {declared_seed!r} does not match configured seed {self.seed!r}"
            )
        assignments = document.get("assignments")
        if not isinstance(assignments, dict) or not assignments:
            raise ValueError("routing map assignments must be a non-empty JSON object")
        normalized: dict[str, dict[str, int]] = {}
        allowed = {"prefill_slot", "decode_slot", "replica_slot", "service_slot"}
        for key, value in assignments.items():
            if not isinstance(key, str) or not key:
                raise ValueError("routing map keys must be non-empty strings")
            if not isinstance(value, dict):
                raise ValueError(f"routing assignment for {key!r} must be an object")
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"unknown routing fields for {key!r}: {sorted(unknown)}")
            normalized_value: dict[str, int] = {}
            for field, slot in value.items():
                if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
                    raise ValueError(f"{key!r}.{field} must be a non-negative integer")
                normalized_value[field] = slot
            normalized[key] = normalized_value
        self.assignments = normalized
        self.map_metadata = {
            key: value
            for key, value in document.items()
            if key not in {"assignments"}
        }
        self.map_fingerprint = hashlib.sha256(raw).hexdigest()

    @staticmethod
    def sorted_addresses(addresses: Iterable[str]) -> list[str]:
        return sorted((str(address) for address in addresses), key=address_sort_key)

    def candidate_addresses(self, addresses: Iterable[str]) -> list[str]:
        """Return the server order defined by the selected contract mode.

        ``stable_hash`` sorts addresses; ``legacy_hash`` intentionally retains
        registry insertion order as well as Python's randomized hash so old
        routing behavior remains available only as an explicit diagnostic.
        """
        values = [str(address) for address in addresses]
        return values if self.mode == "legacy_hash" else self.sorted_addresses(values)

    def key_hash(self, routing_key: str) -> str:
        return hashlib.sha256(str(routing_key).encode("utf-8")).hexdigest()

    def _stable_slot(self, routing_key: str, role: str, size: int) -> int:
        payload = (
            f"{ROUTING_CONTRACT_VERSION}\0{self.seed}\0{role}\0{routing_key}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % size

    def choose_slot(
        self,
        routing_key: str,
        role: str,
        size: int,
        *,
        legacy_key: str | None = None,
    ) -> SlotDecision:
        if size <= 0:
            raise ValueError(f"cannot choose {role} slot from an empty registry")
        field = f"{role}_slot"
        assignment = self.assignments.get(routing_key)
        if assignment is not None and field in assignment:
            slot = assignment[field]
            if slot >= size:
                raise ValueError(
                    f"routing map slot {routing_key!r}.{field}={slot} is out of range for {size} nodes"
                )
            return SlotDecision(slot, "frozen_map")
        if self.map_strict:
            if assignment is None:
                raise KeyError(f"routing key {routing_key!r} is absent from strict routing map")
            raise KeyError(f"routing key {routing_key!r} has no {field} in strict routing map")
        if self.mode == "legacy_hash":
            return SlotDecision(hash(legacy_key or routing_key) % size, "legacy_python_hash")
        return SlotDecision(self._stable_slot(routing_key, role, size), "stable_sha256")

    def descriptor(self) -> dict[str, Any]:
        return {
            "contract_version": ROUTING_CONTRACT_VERSION,
            "mode": self.mode,
            "seed": self.seed,
            "map_path": self.map_path,
            "map_strict": self.map_strict,
            "map_fingerprint_sha256": self.map_fingerprint,
            "map_assignment_count": len(self.assignments),
            "map_metadata": self.map_metadata,
        }
