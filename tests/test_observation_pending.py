"""CPU-only pending/engine reconciliation tests."""
import unittest

from ppd.optimizer.observation_pending import (
    PendingLedger, PhysicalWorkId, RetentionCapability, WorkObservation,
)
from ppd.optimizer.state_semantics import PrefillProgress


class PendingLedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = PendingLedger("D0", "e1", clock_scope="test", max_entries=8)

    def key(self, attempt=1, node="D0", epoch="e1"):
        return PhysicalWorkId("run", "request", attempt, node, epoch)

    def row(self, key=None, ready=20, scheduled=20, state="ready"):
        key = key or self.key()
        return WorkObservation(key, PrefillProgress(key.token, 100, ready, scheduled, state))

    def observe(self, rows=(), seq=1, at=1, **kwargs):
        args = dict(node_id="D0", node_epoch="e1", snapshot_seq=seq, sampled_at_s=at,
                    clock_scope="test", rows=rows, complete=True, coherent=True)
        args.update(kwargs)
        return self.ledger.observe_snapshot(**args)

    def view(self, **kwargs):
        return self.ledger.view(now_s=2, max_age_s=10, **kwargs)

    def test_duplicate_dispatch_once(self):
        self.assertEqual(self.ledger.dispatch(self.row()), "dispatched")
        self.assertEqual(self.ledger.dispatch(self.row()), "duplicate_dispatch")
        self.assertEqual(self.ledger.dispatch(self.row(ready=10)), "identity_already_known")
        self.observe()
        self.assertEqual(self.view().summarize().q_tok, 80)
        self.assertEqual(self.ledger.entry_count, 1)

    def test_engine_wins_and_inflight_not_double_counted(self):
        self.ledger.dispatch(self.row())
        self.observe([self.row(ready=40, scheduled=60, state="inflight")])
        summary = self.view().summarize()
        self.assertEqual((summary.q_tok, summary.inflight_tokens, summary.pending_requests), (60, 20, 0))

    def test_remote_wait_blocks_pending_compute(self):
        self.ledger.dispatch(self.row())
        self.observe([self.row(ready=None, scheduled=None, state="remote_kv_pending")])
        summary = self.view().summarize()
        self.assertEqual((summary.q_tok, summary.blocked_requests), (0, 1))

    def test_accepted_disappears_without_pending_resurrection(self):
        self.ledger.dispatch(self.row())
        self.observe([self.row()])
        self.observe(seq=2, at=2)
        self.assertEqual(self.view().reason, "accepted_missing_without_terminal_ack")
        self.assertIsNone(self.view().summarize())
        self.assertEqual(self.ledger.dispatch(self.row()), "identity_already_known")

    def test_acceptance_ack_requires_snapshot_or_terminal(self):
        self.ledger.dispatch(self.row())
        self.observe()
        self.ledger.acknowledge_accepted(self.key())
        self.assertFalse(self.view().available)
        self.observe([self.row()], seq=2, at=2)
        self.assertTrue(self.view().available)

    def test_terminal_tombstone_suppresses_late_active_row(self):
        self.ledger.dispatch(self.row())
        self.ledger.acknowledge_terminal(self.key(), outcome="completed")
        self.observe([self.row()])
        self.assertEqual(self.view().summarize().q_tok, 0)
        self.assertEqual(self.ledger.dispatch(self.row()), "identity_already_known")
        self.assertEqual(self.ledger.acknowledge_accepted(self.key()), "already_retired")

    def test_finished_snapshot_is_terminal_ack(self):
        self.observe([self.row(state="finished")])
        self.observe(seq=2, at=2)
        self.assertTrue(self.view().available)
        self.assertEqual(self.view().summarize().q_tok, 0)

    def test_retry_has_distinct_identity(self):
        self.ledger.acknowledge_terminal(self.key(), outcome="failed")
        self.ledger.dispatch(self.row(self.key(attempt=2)))
        self.observe()
        self.assertEqual(self.view().summarize().pending_requests, 1)
        self.assertEqual(self.ledger.entry_count, 2)

    def test_transport_timeout_is_not_successful_termination(self):
        self.ledger.dispatch(self.row())
        self.ledger.mark_uncertain(self.key())
        self.observe()
        self.assertEqual(self.view().reason, "termination_unknown")
        self.ledger.acknowledge_terminal(self.key(), outcome="cancelled")
        self.assertTrue(self.view().available)

    def test_new_presence_resolves_uncertain_transport(self):
        self.ledger.mark_uncertain(self.key())
        self.observe([self.row()])
        self.assertTrue(self.view().available)

    def test_out_of_order_snapshot_rejected(self):
        self.observe([self.row(ready=50, scheduled=50)], seq=3)
        self.assertEqual(self.observe(seq=2), "out_of_order_snapshot")
        self.assertEqual(self.observe(seq=3), "out_of_order_snapshot")
        self.assertEqual(self.view().summarize().q_tok, 50)

    def test_incomplete_and_incoherent_snapshots_are_not_zero(self):
        self.observe(seq=3, complete=False)
        self.assertEqual(self.view().reason, "incomplete_snapshot")
        self.assertEqual(self.observe(seq=2), "out_of_order_snapshot")
        self.observe(seq=4, coherent=False)
        self.assertEqual(self.view().reason, "incoherent_snapshot")
        self.assertIsNone(self.view().summarize())

    def test_clock_and_freshness(self):
        self.assertEqual(self.observe(clock_scope="other"), "clock_scope_mismatch")
        self.observe(seq=2, at=2)
        self.assertEqual(self.observe(seq=3, at=1), "clock_regression")
        self.assertEqual(self.ledger.view(now_s=1, max_age_s=10).reason, "snapshot_from_future")
        self.assertEqual(self.ledger.view(now_s=20, max_age_s=1).reason, "stale_snapshot")

    def test_target_self_exclusion_preserves_other_attempt(self):
        self.ledger.dispatch(self.row())
        other = self.key(attempt=2)
        self.ledger.dispatch(self.row(other))
        self.observe()
        self.assertEqual(self.view(exclude=other).summarize().q_tok, 80)

    def test_epoch_replacement_does_not_transfer_old_work(self):
        self.ledger.dispatch(self.row())
        self.assertEqual(self.ledger.replace_epoch(expected_epoch="e1", new_epoch="e2"), "epoch_replaced")
        self.assertEqual(self.ledger.entry_count, 0)
        self.assertEqual(self.ledger.dispatch(self.row()), "identity_scope_mismatch")
        self.assertEqual(self.observe(), "snapshot_scope_mismatch")
        self.assertEqual(self.view().reason, "no_snapshot")
        self.observe(node_epoch="e2")
        self.assertTrue(self.view().available)

    def test_same_epoch_reset_forbidden(self):
        self.assertEqual(self.ledger.replace_epoch(expected_epoch="e1", new_epoch="e1"), "epoch_replacement_mismatch")
        self.assertEqual(self.ledger.replace_epoch(expected_epoch="bad", new_epoch="e2"), "epoch_replacement_mismatch")

    def test_different_nodes_are_separate(self):
        p = PendingLedger("P0", "e1", clock_scope="test")
        prow = self.row(self.key(node="P0"))
        self.assertEqual(p.dispatch(prow), "dispatched")
        self.assertEqual(self.ledger.dispatch(prow), "identity_scope_mismatch")
        self.assertEqual(p.entry_count, 1)
        self.assertEqual(self.ledger.entry_count, 0)

    def test_overflow_keeps_tombstones_and_invalidates_coverage(self):
        self.ledger = PendingLedger("D0", "e1", clock_scope="test", max_entries=1)
        self.ledger.acknowledge_terminal(self.key(), outcome="completed")
        self.assertEqual(self.ledger.dispatch(self.row(self.key(attempt=2))), "ledger_overflow")
        self.observe()
        self.assertEqual(self.view().reason, "ledger_overflow")
        self.assertEqual(self.ledger.entry_count, 1)

    def test_engine_materialization_is_bounded(self):
        self.ledger = PendingLedger("D0", "e1", clock_scope="test", max_entries=1)
        self.assertEqual(self.observe([self.row(), self.row(self.key(attempt=2))]), "ledger_overflow")
        self.assertEqual(self.ledger.entry_count, 0)

    def test_bad_identity_and_duplicate_engine_rows(self):
        with self.assertRaises(ValueError):
            WorkObservation(self.key(attempt=2), self.row().progress)
        with self.assertRaises(ValueError):
            self.observe([self.row(), self.row()])
        for attempt in (0, True, 1.5):
            with self.assertRaises(ValueError):
                self.key(attempt=attempt)

    def test_view_does_not_mutate_ledger(self):
        self.ledger.dispatch(self.row())
        self.observe()
        before = (dict(self.ledger._entries), dict(self.ledger._engine))
        self.view().summarize()
        self.assertEqual(before, (self.ledger._entries, self.ledger._engine))

    def test_ttl_metadata_has_no_actions(self):
        value = RetentionCapability()
        self.assertFalse(value.pin_supported or value.renewal_supported or value.eviction_supported)
        self.assertFalse(hasattr(self.ledger, "pin"))


if __name__ == "__main__":
    unittest.main()
