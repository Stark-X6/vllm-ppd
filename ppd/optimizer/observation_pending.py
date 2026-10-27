"""CPU-only, bounded reconciliation. No serving hooks, threads or TTL actions.

Caller obligations: authoritative identity mapping, complete owner-thread
snapshots and timestamps from one proven clock scope. Missing work is not a
terminal ACK. Entry overflow invalidates coverage instead of evicting history.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable

from .state_semantics import PrefillProgress, summarize_backlog


def _text(value, name):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{name} must contain 1..512 characters")


def _time(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("time/duration must be finite and nonnegative")


@dataclass(frozen=True)
class PhysicalWorkId:
    run_id: str
    logical_request_id: str
    attempt_id: int
    destination_node: str
    node_epoch: str

    def __post_init__(self):
        for name in ("run_id", "logical_request_id", "destination_node", "node_epoch"):
            _text(getattr(self, name), name)
        if type(self.attempt_id) is not int or self.attempt_id < 1:
            raise ValueError("attempt_id must be a positive physical ordinal")

    @property
    def token(self):
        return json.dumps([self.run_id, self.logical_request_id, self.attempt_id,
                           self.destination_node, self.node_epoch], separators=(",", ":"))


@dataclass(frozen=True)
class WorkObservation:
    identity: PhysicalWorkId
    progress: PrefillProgress

    def __post_init__(self):
        if self.progress.work_id != self.identity.token:
            raise ValueError("progress work_id must equal the namespaced identity token")


@dataclass(frozen=True)
class LedgerView:
    available: bool
    reason: str
    snapshot_seq: int | None
    engine: tuple[PrefillProgress, ...] = ()
    pending: tuple[PrefillProgress, ...] = ()
    retired_ids: tuple[str, ...] = ()

    def summarize(self):
        if not self.available:
            return None
        return summarize_backlog(self.engine, self.pending, retired_ids=self.retired_ids)


@dataclass(frozen=True)
class RetentionCapability:
    """Metadata only; pin/renew/evict methods intentionally do not exist."""
    interface_version: str = "ppd-retention-observation-v1"
    pin_supported: bool = False
    renewal_supported: bool = False
    eviction_supported: bool = False


class PendingLedger:
    """Single-owner, single-node/epoch ledger; no silent tombstone expiry.

    Mutators return a status string. Accepted work never returns to pending.
    Clock arguments must share clock_scope; receiver receipt time alone cannot
    prove source freshness. Overflow stays unavailable until a proven new epoch.
    """
    def __init__(self, node_id: str, node_epoch: str, *, clock_scope: str,
                 max_entries: int = 4096):
        for name, value in (("node_id", node_id), ("node_epoch", node_epoch),
                            ("clock_scope", clock_scope)):
            _text(value, name)
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.node_id, self.node_epoch, self.clock_scope = node_id, node_epoch, clock_scope
        self.max_entries = max_entries
        self._entries: dict[PhysicalWorkId, tuple[str, PrefillProgress | None]] = {}
        self._engine: dict[PhysicalWorkId, PrefillProgress] = {}
        self._seq = self._sampled_at = None
        self._coverage = "no_snapshot"
        self._overflow = False

    @property
    def entry_count(self):
        return len(self._entries)

    def _scope(self, key):
        return key.destination_node == self.node_id and key.node_epoch == self.node_epoch

    def _reserve(self, keys):
        if len(self._entries) + len(set(keys) - self._entries.keys()) > self.max_entries:
            self._overflow = True
        return not self._overflow

    def dispatch(self, row: WorkObservation):
        if not self._scope(row.identity):
            return "identity_scope_mismatch"
        if row.progress.state not in {"ready", "inflight", "blocked", "remote_kv_pending"}:
            raise ValueError("dispatch cannot describe decode/finished")
        old = self._entries.get(row.identity)
        if old:
            return "duplicate_dispatch" if old == ("dispatched", row.progress) else "identity_already_known"
        if not self._reserve((row.identity,)):
            return "ledger_overflow"
        self._entries[row.identity] = ("dispatched", row.progress)
        return "dispatched"

    def _ack(self, key, state):
        if not self._scope(key):
            return "identity_scope_mismatch"
        if not self._reserve((key,)):
            return "ledger_overflow"
        if self._entries.get(key, (None,))[0] == "terminal":
            return "already_retired"
        self._entries[key] = (state, None)
        if state == "terminal":
            self._engine.pop(key, None)
        return state

    def acknowledge_accepted(self, key: PhysicalWorkId):
        return self._ack(key, "accepted")

    def acknowledge_terminal(self, key: PhysicalWorkId, *, outcome: str):
        """Requires authoritative engine completion/cancel/failure acknowledgement."""
        if outcome not in {"completed", "cancelled", "failed"}:
            raise ValueError("outcome is not an authoritative terminal outcome")
        return self._ack(key, "terminal")

    def mark_uncertain(self, key: PhysicalWorkId):
        """Use for transport timeout/disconnect; never infer engine termination."""
        return self._ack(key, "uncertain")

    def observe_snapshot(self, *, node_id: str, node_epoch: str, snapshot_seq: int,
                         sampled_at_s: float, clock_scope: str,
                         rows: Iterable[WorkObservation], complete: bool, coherent: bool):
        if (node_id, node_epoch) != (self.node_id, self.node_epoch):
            return "snapshot_scope_mismatch"
        if clock_scope != self.clock_scope:
            return "clock_scope_mismatch"
        if type(snapshot_seq) is not int or snapshot_seq < 0:
            raise ValueError("snapshot_seq must be nonnegative")
        _time(sampled_at_s)
        if type(complete) is not bool or type(coherent) is not bool:
            raise ValueError("coverage flags must be boolean")
        if self._seq is not None and snapshot_seq <= self._seq:
            return "out_of_order_snapshot"
        if self._sampled_at is not None and sampled_at_s < self._sampled_at:
            return "clock_regression"
        incoming = {}
        for row in rows:
            if not self._scope(row.identity):
                return "identity_scope_mismatch"
            if row.identity in incoming:
                raise ValueError("duplicate engine work identity")
            if len(incoming) == self.max_entries:
                self._overflow = True
                return "ledger_overflow"
            incoming[row.identity] = row.progress
        self._seq, self._sampled_at = snapshot_seq, sampled_at_s
        if not complete or not coherent:
            self._coverage = "incomplete_snapshot" if not complete else "incoherent_snapshot"
            self._engine = {}
            return self._coverage
        if not self._reserve(incoming):
            return "ledger_overflow"
        for key, progress in incoming.items():
            terminal = self._entries.get(key, (None,))[0] == "terminal" or progress.state == "finished"
            self._entries[key] = ("terminal" if terminal else "accepted", None)
        self._engine = {key: row for key, row in incoming.items()
                        if self._entries[key][0] != "terminal"}
        self._coverage = "ok"
        return "snapshot_applied"

    def view(self, *, now_s: float, max_age_s: float, exclude: PhysicalWorkId | None = None):
        """Pure read. Unknown coverage returns no numeric partial-sum features."""
        _time(now_s)
        _time(max_age_s)
        if exclude is not None and not self._scope(exclude):
            raise ValueError("excluded identity must be in this node/epoch")
        reason = "ledger_overflow" if self._overflow else self._coverage
        if reason == "ok":
            if now_s < self._sampled_at:
                reason = "snapshot_from_future"
            elif now_s - self._sampled_at > max_age_s:
                reason = "stale_snapshot"
            elif any(key != exclude and item[0] == "uncertain" for key, item in self._entries.items()):
                reason = "termination_unknown"
            elif any(key != exclude and item[0] == "accepted" and key not in self._engine
                     for key, item in self._entries.items()):
                reason = "accepted_missing_without_terminal_ack"
        if reason != "ok":
            return LedgerView(False, reason, self._seq)
        return LedgerView(True, "ok", self._seq,
                          tuple(row for key, row in self._engine.items() if key != exclude),
                          tuple(item[1] for key, item in self._entries.items()
                                if key != exclude and item[0] == "dispatched"),
                          tuple(key.token for key, item in self._entries.items() if item[0] == "terminal"))

    def replace_epoch(self, *, expected_epoch: str, new_epoch: str):
        """Only after confirmed engine replacement, never a transport reconnect."""
        _text(new_epoch, "new_epoch")
        if expected_epoch != self.node_epoch or new_epoch == expected_epoch:
            return "epoch_replacement_mismatch"
        self.node_epoch = new_epoch
        self._entries.clear()
        self._engine.clear()
        self._seq = self._sampled_at = None
        self._coverage, self._overflow = "no_snapshot", False
        return "epoch_replaced"
