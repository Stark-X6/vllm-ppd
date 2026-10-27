"""Pure-CPU tests for the frozen state and measurement semantics.

These tests supply observations; they do not establish that an engine observer
can produce them, and do not import vLLM or allocate any GPU resources.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from ppd.optimizer.state_semantics import (
    LEGACY_STREAM_BASIS,
    NODE_ENGINE_BASIS,
    CalibrationObservation,
    DecodeSample,
    PrefillProgress,
    attention_pairs,
    request_tpot_s,
    split_prompt_ids,
    summarize_backlog,
    summarize_calibration,
    summarize_decode,
    uncached_tokens,
)


class PromptSemanticsTests(unittest.TestCase):
    def test_verified_history_is_a_slice_of_the_full_sequence(self) -> None:
        split = split_prompt_ids([11, 12, 13, 14], [11, 12])
        self.assertEqual((split.n, split.h, split.a), (4, 2, 2))
        self.assertEqual(split.boundary_status, "verified")

    def test_mismatch_does_not_substitute_longest_common_prefix(self) -> None:
        split = split_prompt_ids([11, 12, 13, 14], [11, 12, 99])
        self.assertEqual(split.n, 4)
        self.assertIsNone(split.h)
        self.assertIsNone(split.a)
        self.assertEqual(split.boundary_status, "history_not_prefix")

    def test_history_longer_than_prompt_is_not_verified(self) -> None:
        split = split_prompt_ids([11, 12], [11, 12, 13])
        self.assertEqual(split.boundary_status, "history_not_prefix")
        self.assertIsNone(split.h)
        self.assertIsNone(split.a)

    def test_absent_history_is_unknown_not_empty(self) -> None:
        unknown = split_prompt_ids([1, 2], None)
        empty = split_prompt_ids([1, 2], [])
        self.assertEqual(unknown.boundary_status, "unavailable")
        self.assertEqual((unknown.h, unknown.a), (None, None))
        self.assertEqual((empty.h, empty.a), (0, 2))

    def test_complete_history_can_have_zero_logical_append(self) -> None:
        split = split_prompt_ids([1, 2], [1, 2])
        self.assertEqual((split.n, split.h, split.a), (2, 2, 0))
        # Logical append zero does not mean the engine can skip the logits tail.
        self.assertEqual(uncached_tokens(split.n, 1), 1)

    def test_empty_prompt_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            split_prompt_ids([], [])

    def test_token_ids_must_be_nonnegative_integers(self) -> None:
        for invalid in (-1, 1.5, True, "1", float("nan")):
            with self.subTest(token=invalid):
                with self.assertRaises(ValueError):
                    split_prompt_ids([invalid], None)
                with self.assertRaises(ValueError):
                    split_prompt_ids([1], [invalid])

    def test_all_small_verified_splits_preserve_token_count(self) -> None:
        for n in range(1, 9):
            full = list(range(n))
            for h in range(n + 1):
                split = split_prompt_ids(full, full[:h])
                self.assertEqual((split.n, split.h, split.a), (n, h, n - h))


class ComputeWorkSemanticsTests(unittest.TestCase):
    def test_unknown_prefix_is_not_zero_cache_hit(self) -> None:
        self.assertIsNone(uncached_tokens(10, None))
        self.assertEqual(uncached_tokens(10, 0), 10)

    def test_effective_prefix_must_leave_at_least_a_tail(self) -> None:
        for n in range(1, 10):
            for c in range(n):
                self.assertEqual(uncached_tokens(n, c), n - c)
            for c in (n, n + 1):
                with self.assertRaises(ValueError):
                    uncached_tokens(n, c)

    def test_invalid_prompt_and_cache_counts(self) -> None:
        for n in (0, -1, True, 2.0):
            with self.assertRaises(ValueError):
                uncached_tokens(n, None)
        for c in (-1, False, 1.5):
            with self.assertRaises(ValueError):
                uncached_tokens(10, c)

    def test_attention_count_and_chunk_additivity(self) -> None:
        for prefix in range(7):
            for suffix in range(9):
                total = attention_pairs(prefix, suffix)
                self.assertEqual(total, sum(prefix + j for j in range(1, suffix + 1)))
                for first_chunk in range(suffix + 1):
                    self.assertEqual(
                        total,
                        attention_pairs(prefix, first_chunk)
                        + attention_pairs(prefix + first_chunk, suffix - first_chunk),
                    )

    def test_attention_requires_integer_nonnegative_counts(self) -> None:
        for invalid in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                attention_pairs(invalid, 2)
            with self.assertRaises(ValueError):
                attention_pairs(2, invalid)


class DecodeSemanticsTests(unittest.TestCase):
    def test_coherence_requires_a_real_boolean(self) -> None:
        for value in ("false", 0, 1, None):
            with self.assertRaises(ValueError):
                summarize_decode([], coherent=value)

    def test_empty_coherent_set_is_known_idle(self) -> None:
        summary = summarize_decode([], coherent=True)
        self.assertEqual((summary.b_d, summary.k_d, summary.unknown_contexts), (0, 0, 0))

    def test_unavailable_or_incoherent_set_is_not_idle(self) -> None:
        for rows, coherent in ((None, True), ([], False), ([DecodeSample("a", 8)], False)):
            summary = summarize_decode(rows, coherent=coherent)
            self.assertEqual(
                (summary.b_d, summary.k_d, summary.unknown_contexts),
                (None, None, None),
            )

    def test_batch_and_context_sum_describe_the_same_eligible_set(self) -> None:
        summary = summarize_decode([DecodeSample("a", 8), DecodeSample("b", 32)], coherent=True)
        self.assertEqual((summary.b_d, summary.k_d, summary.unknown_contexts), (2, 40, 0))

    def test_known_batch_with_unknown_context_is_partial_not_a_lower_bound(self) -> None:
        summary = summarize_decode([DecodeSample("a", 8), DecodeSample("b", None)], coherent=True)
        self.assertEqual(summary.b_d, 2)
        self.assertIsNone(summary.k_d)
        self.assertEqual(summary.unknown_contexts, 1)

    def test_duplicate_and_empty_work_ids_are_rejected(self) -> None:
        for samples in ([DecodeSample("", 8)], [DecodeSample("a", 8), DecodeSample("a", 9)]):
            with self.assertRaises(ValueError):
                summarize_decode(samples, coherent=True)

    def test_ready_context_is_a_positive_integer(self) -> None:
        for invalid in (0, -1, 2.5, True):
            with self.subTest(context=invalid):
                with self.assertRaises(ValueError):
                    summarize_decode([DecodeSample("a", invalid)], coherent=True)


class BacklogSemanticsTests(unittest.TestCase):
    def test_progress_requires_ordered_frontiers(self) -> None:
        invalid = ((10, 11, 11), (10, 5, 11), (10, 6, 5), (-1, 0, 0))
        for target, ready, scheduled in invalid:
            with self.subTest(target=target, ready=ready, scheduled=scheduled):
                with self.assertRaises(ValueError):
                    PrefillProgress("a", target, ready, scheduled)
        for invalid_count in (True, 1.5, -1):
            with self.assertRaises(ValueError):
                PrefillProgress("a", 10, invalid_count, None)
            with self.assertRaises(ValueError):
                PrefillProgress("a", 10, None, invalid_count)

    def test_progress_requires_an_identity_and_supported_state(self) -> None:
        with self.assertRaises(ValueError):
            PrefillProgress("", 10, 0, 0)
        with self.assertRaises(ValueError):
            PrefillProgress("a", 10, 0, 0, state="imaginary")

    def test_inflight_work_is_included_once_not_added_twice(self) -> None:
        summary = summarize_backlog([PrefillProgress("a", 10, 4, 7, state="inflight")])
        self.assertEqual(summary.q_tok, 6)
        self.assertEqual(summary.q_att, attention_pairs(4, 6))
        self.assertEqual(summary.inflight_tokens, 3)
        self.assertEqual(summary.compute_requests, 1)

    def test_all_small_frontiers_keep_total_work_independent_of_dispatch(self) -> None:
        for target in range(8):
            for ready in range(target + 1):
                for scheduled in range(ready, target + 1):
                    summary = summarize_backlog([PrefillProgress("a", target, ready, scheduled)])
                    self.assertEqual(summary.q_tok, target - ready)
                    self.assertEqual(summary.q_att, attention_pairs(ready, target - ready))
                    self.assertEqual(summary.inflight_tokens, scheduled - ready)

    def test_blocked_remote_decode_finished_are_not_local_prefill(self) -> None:
        rows = [PrefillProgress(state, 100, None, None, state=state) for state in
                ("blocked", "remote_kv_pending", "decode", "finished")]
        summary = summarize_backlog(rows)
        self.assertEqual((summary.q_tok, summary.q_att, summary.inflight_tokens), (0, 0, 0))
        self.assertEqual(summary.compute_requests, 0)
        self.assertEqual(summary.blocked_requests, 2)
        self.assertEqual(summary.unknown_work_requests, 0)

    def test_engine_observation_supersedes_proxy_pending(self) -> None:
        summary = summarize_backlog(
            [PrefillProgress("a", 10, 5, 7)],
            [PrefillProgress("a", 10, 0, 0), PrefillProgress("b", 4, 1, 1)],
        )
        self.assertEqual(summary.q_tok, 8)
        self.assertEqual(summary.q_att, attention_pairs(5, 5) + attention_pairs(1, 3))
        self.assertEqual(summary.inflight_tokens, 2)
        self.assertEqual((summary.compute_requests, summary.pending_requests), (2, 1))

    def test_engine_blocked_or_finished_overrides_pending_ready(self) -> None:
        for state in ("blocked", "remote_kv_pending", "finished"):
            summary = summarize_backlog(
                [PrefillProgress("a", 10, None, None, state=state)],
                [PrefillProgress("a", 10, 0, 0)],
            )
            self.assertEqual(summary.q_tok, 0)
            self.assertEqual(summary.pending_requests, 0)

    def test_terminal_tombstone_prevents_phantom_pending_after_completion(self) -> None:
        summary = summarize_backlog(
            [], [PrefillProgress("done", 10, 0, 0)], retired_ids=["done"],
        )
        self.assertEqual((summary.q_tok, summary.compute_requests, summary.pending_requests), (0, 0, 0))

    def test_target_request_is_not_background_even_when_prefix_unknown(self) -> None:
        summary = summarize_backlog(
            [PrefillProgress("self", 10, None, None), PrefillProgress("other", 4, 2, 2)],
            [PrefillProgress("self", 10, 0, 0)], exclude_work_id="self",
        )
        self.assertEqual(summary.q_tok, 2)
        self.assertEqual(summary.compute_requests, 1)
        self.assertEqual(summary.unknown_work_requests, 0)

    def test_unknown_ready_frontier_invalidates_total_not_zero_fills(self) -> None:
        summary = summarize_backlog([
            PrefillProgress("known", 10, 4, 7), PrefillProgress("unknown", 20, None, 10),
        ])
        self.assertIsNone(summary.q_tok)
        self.assertIsNone(summary.q_att)
        self.assertIsNone(summary.inflight_tokens)
        self.assertEqual(summary.compute_requests, 2)
        self.assertEqual(summary.unknown_work_requests, 1)

    def test_unknown_scheduled_frontier_only_invalidates_inflight_sum(self) -> None:
        summary = summarize_backlog([PrefillProgress("a", 10, 4, None)])
        self.assertEqual(summary.q_tok, 6)
        self.assertEqual(summary.q_att, attention_pairs(4, 6))
        self.assertIsNone(summary.inflight_tokens)
        self.assertEqual(summary.unknown_work_requests, 0)

    def test_duplicate_ids_inside_either_source_are_rejected(self) -> None:
        row = PrefillProgress("a", 10, 0, 0)
        with self.assertRaises(ValueError):
            summarize_backlog([row, row])
        with self.assertRaises(ValueError):
            summarize_backlog([], [row, row])


class ResultSemanticsTests(unittest.TestCase):
    def observation(self, work_id: str = "a", **changes: object) -> CalibrationObservation:
        base = CalibrationObservation(work_id, LEGACY_STREAM_BASIS, True, 3, 1.0, 0.5)
        return replace(base, **changes)

    def test_request_tpot_uses_post_first_intervals(self) -> None:
        self.assertEqual(request_tpot_s(2.0, 5), 0.5)
        self.assertEqual(request_tpot_s(0.0, 2), 0.0)

    def test_success_and_warmup_require_booleans(self) -> None:
        for field in ("success", "warmup"):
            for value in ("false", 0, 1, None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.observation(**{field: value})

    def test_auto_recovery_is_not_a_second_first_attempt(self) -> None:
        # Normalized proxy ordinals, not the resetting client attempt header.
        rows = [
            self.observation("logical-a:required:1", success=False, attempt=1,
                             generated_tokens=None, ttft_s=None, tpot_s=None),
            self.observation("logical-a:auto:2", success=True, attempt=2),
        ]
        summary = summarize_calibration(rows, metric_basis=LEGACY_STREAM_BASIS)
        self.assertEqual((summary.n_total, summary.n_success, summary.n_failed), (1, 0, 1))
        self.assertIsNone(summary.avg_ttft_s)
        self.assertIsNone(summary.avg_tpot_s)

    def test_request_tpot_undefined_is_not_zero(self) -> None:
        for tokens in (None, 0, 1):
            self.assertIsNone(request_tpot_s(3.0, tokens))
        self.assertIsNone(request_tpot_s(None, 4))

    def test_invalid_durations_and_token_counts_are_rejected(self) -> None:
        for duration in (-1, float("nan"), float("inf"), float("-inf"), True, "1"):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    request_tpot_s(duration, 3)
                with self.assertRaises(ValueError):
                    self.observation(ttft_s=duration)
                with self.assertRaises(ValueError):
                    self.observation(tpot_s=duration)
        for tokens in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                request_tpot_s(1.0, tokens)
            with self.assertRaises(ValueError):
                self.observation(generated_tokens=tokens)

    def test_result_consistency_checks(self) -> None:
        for tokens in (None, 0, 1):
            with self.assertRaises(ValueError):
                self.observation(generated_tokens=tokens)
        with self.assertRaises(ValueError):
            self.observation(generated_tokens=0, tpot_s=None)
        for changes in ({"work_id": ""}, {"role": "other"}, {"metric_basis": "other"},
                        {"turn": 0}, {"attempt": 0}, {"turn": True}, {"attempt": 1.5}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.observation(**changes)

    def test_population_is_candidate_turn2_first_attempt_without_warmup(self) -> None:
        rows = [self.observation("kept")]
        for index, changes in enumerate((
            {"role": "background"}, {"role": "prefill_only"}, {"turn": 1},
            {"turn": 3}, {"attempt": 2}, {"warmup": True},
        )):
            rows.append(self.observation(f"excluded-{index}", ttft_s=100, tpot_s=100, **changes))
        summary = summarize_calibration(rows, metric_basis=LEGACY_STREAM_BASIS)
        self.assertEqual((summary.n_total, summary.n_success), (1, 1))
        self.assertEqual(summary.population, "candidate_turn2_first_attempt_v1")
        self.assertEqual((summary.avg_ttft_s, summary.avg_tpot_s), (1.0, 0.5))

    def test_means_are_request_weighted_not_output_token_weighted(self) -> None:
        rows = [self.observation("short", generated_tokens=2, ttft_s=1, tpot_s=2),
                self.observation("long", generated_tokens=101, ttft_s=3, tpot_s=4)]
        summary = summarize_calibration(rows, metric_basis=LEGACY_STREAM_BASIS)
        self.assertEqual((summary.avg_ttft_s, summary.avg_tpot_s), (2.0, 3.0))
        self.assertNotEqual(summary.avg_tpot_s, (2 + 100 * 4) / 101)

    def test_failed_and_missing_results_remain_in_denominator_counters(self) -> None:
        rows = [
            self.observation("ok"),
            self.observation("failed", success=False, ttft_s=99, tpot_s=99),
            self.observation("missing", generated_tokens=None, ttft_s=None, tpot_s=None),
            self.observation("zero", generated_tokens=0, ttft_s=None, tpot_s=None),
            self.observation("one", generated_tokens=1, ttft_s=3, tpot_s=None),
        ]
        summary = summarize_calibration(rows, metric_basis=LEGACY_STREAM_BASIS)
        self.assertEqual((summary.n_total, summary.n_success, summary.n_failed), (5, 4, 1))
        self.assertEqual((summary.n_ttft, summary.n_tpot), (2, 1))
        self.assertEqual((summary.n_missing_usage, summary.n_zero_output), (1, 1))
        self.assertEqual((summary.avg_ttft_s, summary.avg_tpot_s), (2.0, 0.5))

    def test_zero_tpot_is_not_eligible_under_legacy_positive_only_rule(self) -> None:
        summary = summarize_calibration([self.observation(tpot_s=0)], metric_basis=LEGACY_STREAM_BASIS)
        self.assertEqual(summary.n_ttft, 1)
        self.assertEqual(summary.n_tpot, 0)
        self.assertIsNone(summary.avg_tpot_s)

    def test_empty_population_has_unknown_means(self) -> None:
        summary = summarize_calibration([], metric_basis=LEGACY_STREAM_BASIS)
        self.assertEqual((summary.n_total, summary.n_ttft, summary.n_tpot), (0, 0, 0))
        self.assertIsNone(summary.avg_ttft_s)
        self.assertIsNone(summary.avg_tpot_s)

    def test_mixed_timing_bases_are_rejected_not_silently_pooled(self) -> None:
        with self.assertRaises(ValueError):
            summarize_calibration([
                self.observation("legacy"), self.observation("engine", metric_basis=NODE_ENGINE_BASIS),
            ], metric_basis=LEGACY_STREAM_BASIS)
        with self.assertRaises(ValueError):
            summarize_calibration([], metric_basis="unknown")

    def test_excluded_population_cannot_contaminate_timing_basis(self) -> None:
        summary = summarize_calibration([
            self.observation("a"),
            self.observation("a", role="background", metric_basis=NODE_ENGINE_BASIS),
        ], metric_basis=LEGACY_STREAM_BASIS)
        self.assertEqual(summary.n_total, 1)

    def test_duplicate_selected_attempt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_calibration([self.observation(), self.observation()], metric_basis=LEGACY_STREAM_BASIS)


if __name__ == "__main__":
    unittest.main()
