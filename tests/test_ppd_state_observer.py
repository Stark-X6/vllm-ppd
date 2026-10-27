"""CPU-only tests of the opt-in observer, without importing vLLM or torch.

The fake hasher is a deterministic chained SHA256 substitute, NOT a test of
vLLM's native hash construction. GPU readiness is represented by explicit
completion events; these tests do not establish native engine integration.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ppd_observer_cpu_under_test",
    ROOT / "vllm-source/vllm/v1/core/ppd_observer.py",
)
observer_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer_module)
PpdStateObserver = observer_module.PpdStateObserver


class FullAttentionSpec:
    pass


class FullAttentionManager:
    def __init__(self):
        self.req_to_blocks = {}


class FakePool:
    def __init__(self, free_blocks=1024):
        self.num_gpu_blocks = 2048
        self.free_blocks = free_blocks
        self.index = {}
        self.blocks = {}
        self.free_queue = []
        self.stats = {"hits": 0, "queries": 0}

    def get_cached_block(self, block_hash, group_ids):
        assert group_ids == [0]
        return self.index.get(block_hash)

    def get_num_free_blocks(self):
        return self.free_blocks


class P2pNcclConnector:
    def __init__(self, remote_ids=(), producer=False):
        self.is_producer = producer
        self.remote_ids = set(remote_ids)

    def is_pd_request(self, request_id):
        return request_id in self.remote_ids


class Rig:
    """Reusable stdlib-only fake engine for tests and microbenchmark."""

    def __init__(self, block_size=4):
        self.block_size = block_size
        self.pool = FakePool()
        self.manager = FullAttentionManager()
        coordinator = NS(single_type_managers=[self.manager],
                         kv_cache_spec=FullAttentionSpec(), block_size=block_size)
        cache = NS(block_pool=self.pool, coordinator=coordinator, enable_caching=True)
        self.scheduler = NS(kv_cache_manager=cache, requests={}, connector=None,
                            max_num_scheduled_tokens=4096)
        self.config = NS(
            parallel_config=NS(data_parallel_size=1, tensor_parallel_size=1,
                               pipeline_parallel_size=1, decode_context_parallel_size=1,
                               prefill_context_parallel_size=1),
            lora_config=None,
            model_config=NS(is_multimodal_model=False, max_model_len=262144,
                            get_vocab_size=lambda: 10000),
            scheduler_config=NS(long_prefill_token_threshold=0),
        )
        self.core = NS(vllm_config=self.config, scheduler=self.scheduler,
                       async_scheduling=False, batch_queue=None,
                       use_spec_decode=False, request_block_hasher=self.hashes)
        self.observer = PpdStateObserver(self.core)

    def hashes(self, req):
        parent = "root"
        result = []
        for offset in range(0, len(req.all_token_ids) - self.block_size + 1, self.block_size):
            payload = [parent, getattr(req, "cache_salt", None),
                       req.all_token_ids[offset:offset + self.block_size]]
            parent = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
            result.append(parent)
        return result

    def request(self, rid="r", n=9, ready=0, output=0, status="WAITING",
                preemptions=0, salt=None, tokens=None):
        tokens = list(range(n)) if tokens is None else list(tokens)
        req = NS(request_id=rid, status=NS(name=status), num_tokens=len(tokens),
                 num_computed_tokens=ready, num_output_tokens=output,
                 num_preemptions=preemptions, all_token_ids=tokens, cache_salt=salt,
                 block_hashes=[], mm_features=[], lora_request=None,
                 prompt_embeds=None, pooling_params=None,
                 skip_reading_prefix_cache=False)
        req.block_hashes = self.hashes(req)
        self.scheduler.requests[rid] = req
        self.observer.event("request_added", req)
        return req

    def insert(self, req, block_index=0, block_id=None, refs=0):
        block_id = block_index if block_id is None else block_id
        block_hash = req.block_hashes[block_index]
        block = NS(block_id=block_id, block_hash=(block_hash, 0), ref_cnt=refs,
                   is_null=False)
        self.pool.index[block_hash] = [block]
        self.pool.blocks[block_id] = block
        if refs == 0 and block_id not in self.pool.free_queue:
            self.pool.free_queue.append(block_id)
        self.observer.event("cache_insert", req, block,
                            (block_index + 1) * self.block_size)
        return block

    def complete(self, req, frontier=None, invalid=()):
        if frontier is not None:
            req.num_computed_tokens = frontier
        scheduled = NS(num_scheduled_tokens={req.request_id: 1})
        self.observer.event("begin_execution", scheduled)
        self.observer.event("finish_execution",
                            NS(kv_connector_output=NS(invalid_block_ids=invalid)))

    def cached(self, n=9, count=2, salt=None, rid="producer"):
        req = self.request(rid, n=n, salt=salt)
        blocks = [self.insert(req, i) for i in range(count)]
        self.complete(req, count * self.block_size)
        del self.scheduler.requests[rid]
        self.observer.event("after_update")
        return req, blocks

    def observe(self, payload=None):
        # Avoid the production 10 Hz throttle in logical unit tests only.
        self.observer.last_query = float("-inf")
        return self.observer.observe({} if payload is None else payload)

    def probe(self, req):
        payload = {"prompt_token_ids": req.all_token_ids}
        if req.cache_salt is not None:
            payload["cache_salt"] = req.cache_salt
        return self.observe(payload)

    def serving_state(self):
        return copy.deepcopy((self.pool.index, self.pool.blocks, self.pool.free_queue,
                              self.pool.stats, self.pool.free_blocks,
                              self.manager.req_to_blocks, self.scheduler.requests))


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def test_no_match_is_known_zero(self):
        result = self.rig.observe({"prompt_token_ids": list(range(9))})
        self.assertEqual((result["prefix"]["c"], result["prefix"]["u"]), (0, 9))

    def test_insert_before_completion_is_unknown_not_hit_or_miss(self):
        req = self.rig.request()
        self.rig.insert(req)
        prefix = self.rig.probe(req)["prefix"]
        self.assertIsNone(prefix["c"])
        self.assertIsNone(prefix["u"])
        self.assertEqual(prefix["status"], "matched_block_readiness_unknown")

    def test_successful_forward_proves_only_completed_blocks(self):
        req = self.rig.request()
        self.rig.insert(req, 0)
        self.rig.insert(req, 1)
        self.rig.complete(req, 4)
        self.assertEqual(self.rig.observer.marks[0][1], True)
        self.assertEqual(self.rig.observer.marks[1][1], False)
        self.assertIsNone(self.rig.probe(req)["prefix"]["c"])
        self.rig.complete(req, 8)
        self.assertEqual(self.rig.probe(req)["prefix"]["c"], 8)

    def test_insert_after_successful_output_registration_is_ready(self):
        req = self.rig.request()
        self.rig.complete(req, 8)
        self.rig.insert(req, 0)
        self.rig.insert(req, 1)
        self.assertEqual(self.rig.probe(req)["prefix"]["c"], 8)

    def test_reserve_last_token_and_round_to_blocks(self):
        req, _ = self.rig.cached(n=9)
        for n, expected in [(1, 0), (4, 0), (5, 4), (8, 4), (9, 8)]:
            with self.subTest(n=n):
                got = self.rig.observe({"prompt_token_ids": req.all_token_ids[:n]})
                self.assertEqual(got["prefix"]["c"], expected)

    def test_probe_is_readonly_for_refs_lru_index_requests_and_stats(self):
        req, _ = self.rig.cached()
        before = self.rig.serving_state()
        for _ in range(3):
            result = self.rig.probe(req)
            self.assertEqual(result["prefix"]["c"], 8)
        self.assertEqual(before, self.rig.serving_state())
        self.assertTrue(result["readonly"])

    def test_cache_salt_changes_chain_and_digest_is_token_only(self):
        req, _ = self.rig.cached(salt="one")
        good = self.rig.probe(req)["prefix"]
        other = self.rig.observe({"prompt_token_ids": req.all_token_ids,
                                  "cache_salt": "two"})["prefix"]
        self.assertEqual(good["c"], 8)
        self.assertEqual(other["c"], 0)
        self.assertEqual(good["token_digest"], other["token_digest"])
        self.assertFalse(good["canonical_chat_verified"])

    def test_eviction_then_same_hash_same_block_needs_new_proof(self):
        req, blocks = self.rig.cached()
        self.rig.observer.event("cache_evict", blocks[0])
        replacement = self.rig.request("replacement")
        self.rig.insert(replacement, 0, block_id=0)
        self.assertIsNone(self.rig.probe(req)["prefix"]["c"])
        self.rig.complete(replacement, 4)
        self.assertEqual(self.rig.probe(req)["prefix"]["c"], 8)

    def test_reset_invalidates_retained_hash_proof_and_generation(self):
        req, _ = self.rig.cached()
        self.rig.observer.event("cache_reset")
        result = self.rig.probe(req)
        self.assertEqual(result["cache_generation"], 1)
        self.assertIsNone(result["prefix"]["c"])

    def test_invalid_remote_kv_discards_all_ready_evidence(self):
        req, _ = self.rig.cached()
        active = self.rig.request("new")
        self.rig.complete(active, 8, invalid={123})
        self.assertEqual(self.rig.observer.cache_generation, 1)
        self.assertIsNone(self.rig.probe(req)["prefix"]["c"])

    def test_preempted_generation_does_not_borrow_completed_frontier(self):
        req = self.rig.request()
        self.rig.complete(req, 8)
        req.num_preemptions += 1
        req.num_computed_tokens = 0
        self.rig.observer.event("after_update")
        self.rig.insert(req)
        self.assertNotIn((req.request_id, 0), self.rig.observer.frontiers)
        self.assertIsNone(self.rig.probe(req)["prefix"]["c"])

    def test_old_inflight_completion_cannot_prove_replacement_generation(self):
        req = self.rig.request(ready=8)
        self.rig.insert(req)
        self.rig.observer.event("begin_execution", NS(num_scheduled_tokens={req.request_id: 8}))
        req.num_preemptions += 1
        self.rig.insert(req)  # same ID/hash now belongs to a new generation
        self.rig.observer.event("finish_execution", NS(kv_connector_output=None))
        self.assertFalse(self.rig.observer.marks[0][1])
        self.assertIsNone(self.rig.probe(req)["prefix"]["c"])

    def test_canceled_unfinished_owner_does_not_create_readiness(self):
        req = self.rig.request()
        self.rig.insert(req)
        del self.rig.scheduler.requests[req.request_id]
        self.rig.observer.event("after_update")
        self.assertIsNone(self.rig.probe(req)["prefix"]["c"])
        self.assertIn(req.request_id, self.rig.observer.terminals)

    def test_completed_terminal_keeps_valid_idle_cache(self):
        req, _ = self.rig.cached()
        self.assertEqual(self.rig.probe(req)["prefix"]["c"], 8)
        self.assertNotIn((req.request_id, 0), self.rig.observer.frontiers)

    def test_caching_disabled_is_known_zero(self):
        req, _ = self.rig.cached()
        self.rig.scheduler.kv_cache_manager.enable_caching = False
        result = self.rig.probe(req)["prefix"]
        self.assertEqual(result["c"], 0)
        self.assertEqual(result["status"], "prefix_caching_disabled")

    def test_missing_native_hasher_is_unknown(self):
        self.rig.core.request_block_hasher = None
        result = self.rig.observe({"prompt_token_ids": list(range(9))})
        self.assertIsNone(result["prefix"]["c"])
        self.assertIsNone(result["prefix"]["u"])
        self.assertFalse(result["prefix"]["available"])

    def test_incomplete_hash_chain_never_becomes_false_cold_miss(self):
        self.rig.core.request_block_hasher = lambda req: []
        result = self.rig.observe({"prompt_token_ids": list(range(9))})
        self.assertIsNone(result["prefix"]["c"])
        self.assertFalse(result["prefix"]["available"])

    def test_invalid_forward_then_abort_cannot_leave_ready_frontier(self):
        req = self.rig.request()
        self.rig.complete(req, 8)
        self.rig.insert(req)
        self.rig.complete(req, 8, invalid={0})
        del self.rig.scheduler.requests[req.request_id]
        self.rig.observer.event("after_update")
        # Late hash registration cannot reuse readiness from the invalid step.
        self.rig.insert(req)
        result = self.rig.probe(req)
        self.assertIsNone(result["prefix"]["c"])
        self.assertTrue(result["coherent_snapshot"])
        self.assertTrue(result["coverage_complete"])
        self.assertTrue(result["available"])  # no active requests; probe is separate
        self.assertFalse(result["prefix"]["available"])


class StateTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def row(self, req):
        return next(r for r in self.rig.observe()["requests"] if r["request_id"] == req.request_id)

    def test_decode_counts_only_ready_one_token_continuations(self):
        for i, n in enumerate([9, 17]):
            req = self.rig.request(str(i), n=n, output=2, status="RUNNING")
            self.rig.complete(req, n - 1)
        result = self.rig.observe()
        self.assertEqual(result["decode"], {"b_d": 2, "k_d": 24})
        self.assertEqual(result["prefill"], {"q_tok": 0, "q_att": 0})

    def test_single_token_prefill_is_not_decode(self):
        req = self.rig.request(n=1, status="RUNNING")
        self.rig.complete(req, 0)
        row = self.row(req)
        self.assertEqual(row["phase"], "prefill_ready")
        self.assertEqual(row["remaining_tokens"], 1)

    def test_recompute_of_generated_request_is_prefill(self):
        req = self.rig.request(n=17, output=5, status="RUNNING", preemptions=1)
        self.rig.complete(req, 4)
        row = self.row(req)
        self.assertEqual(row["phase"], "prefill_ready")
        self.assertEqual(row["remaining_tokens"], 13)
        self.assertEqual(row["attention_pairs"], 4 * 13 + 13 * 14 // 2)

    def test_fsm_is_excluded_from_compute_sums(self):
        self.rig.request(status="WAITING_FOR_FSM")
        result = self.rig.observe()
        self.assertEqual(result["requests"][0]["phase"], "blocked_fsm")
        self.assertEqual(result["blocked_requests"], 1)
        self.assertEqual(result["decode"]["b_d"], 0)
        self.assertEqual(result["prefill"]["q_tok"], 0)

    def test_remote_wait_and_unproven_pd_consumer_are_excluded(self):
        self.rig.request("waiting", status="WAITING_FOR_REMOTE_KVS")
        self.rig.request("pd", status="WAITING")
        self.rig.scheduler.connector = P2pNcclConnector(["pd"])
        result = self.rig.observe()
        self.assertEqual([r["phase"] for r in result["requests"]],
                         ["remote_kv_pending", "remote_kv_pending"])
        self.assertEqual(result["blocked_requests"], 2)

    def test_proven_pd_consumer_can_decode(self):
        req = self.rig.request("pd", output=1, status="RUNNING")
        self.rig.scheduler.connector = P2pNcclConnector(["pd"])
        self.rig.complete(req, 8)
        self.assertEqual(self.row(req)["phase"], "decode_ready")

    def test_unknown_frontier_makes_aggregate_unknown_not_zero(self):
        self.rig.request("unknown", ready=8, output=1, status="RUNNING")
        self.rig.request("known", n=3)
        result = self.rig.observe()
        self.assertFalse(result["available"])
        self.assertTrue(result["coherent_snapshot"])
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["unknown_requests"], 1)
        self.assertIsNone(result["decode"]["b_d"])
        self.assertIsNone(result["decode"]["k_d"])
        self.assertIsNone(result["prefill"]["q_tok"])
        self.assertIsNone(result["prefill"]["q_att"])

    def test_unknown_cached_prefix_poison_sums(self):
        req = self.rig.request()
        self.rig.insert(req)
        result = self.rig.observe()
        self.assertIsNone(result["prefill"]["q_tok"])

    def test_pending_ready_prefix_normalizes_s_without_double_count(self):
        source, _ = self.rig.cached()
        req = self.rig.request("waiting", tokens=source.all_token_ids)
        result = self.rig.observe()
        row = result["requests"][0]
        self.assertEqual((row["ready_tokens"], row["scheduled_tokens"]), (8, 8))
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(result["prefill"], {"q_tok": 1, "q_att": 9})

    def test_skip_prefix_cache_means_recompute_even_if_cached(self):
        source, _ = self.rig.cached()
        req = self.rig.request("waiting", tokens=source.all_token_ids)
        req.skip_reading_prefix_cache = True
        row = self.row(req)
        self.assertEqual(row["ready_tokens"], 0)
        self.assertEqual(row["remaining_tokens"], 9)

    def test_chunk_capacity_checks_next_chunk_not_entire_remaining(self):
        req = self.rig.request(n=20)
        self.rig.config.scheduler_config.long_prefill_token_threshold = 4
        self.rig.pool.free_blocks = 1
        self.assertEqual(self.row(req)["phase"], "prefill_ready")
        self.rig.config.scheduler_config.long_prefill_token_threshold = 0
        row = self.row(req)
        self.assertEqual(row["phase"], "prefill_ready")
        self.assertEqual(row["capacity_risk"], "chunk_size_dependent")

    def test_smaller_budget_chunk_preserves_known_backlog(self):
        req = self.rig.request(n=8)
        self.rig.scheduler.max_num_scheduled_tokens = 8
        self.rig.pool.free_blocks = 1  # fits 4 tokens, not the maximum 8
        result = self.rig.observe()
        row = next(r for r in result["requests"] if r["request_id"] == req.request_id)
        self.assertEqual(row["phase"], "prefill_ready")
        self.assertEqual(row["capacity_risk"], "chunk_size_dependent")
        self.assertEqual(result["prefill"], {"q_tok": 8, "q_att": 36})
        self.rig.pool.free_blocks = 0
        self.assertEqual(self.row(req)["phase"], "blocked_capacity")

    def test_capacity_counts_reclaimable_hits_acquired_again(self):
        source, _ = self.rig.cached()
        req = self.rig.request("waiting", tokens=source.all_token_ids)
        self.rig.pool.free_blocks = 2  # two idle cached hits + one new block needs three
        self.assertEqual(self.row(req)["phase"], "blocked_capacity")
        self.rig.pool.free_blocks = 3
        self.assertEqual(self.row(req)["phase"], "prefill_ready")

    def test_existing_refs_do_not_count_as_reclaimable_acquisition(self):
        source, blocks = self.rig.cached()
        for block in blocks:
            block.ref_cnt = 1
        req = self.rig.request("waiting", tokens=source.all_token_ids)
        self.rig.pool.free_blocks = 1
        self.assertEqual(self.row(req)["phase"], "prefill_ready")

    def test_decode_without_capacity_is_separately_blocked(self):
        req = self.rig.request(output=1, status="RUNNING")
        self.rig.complete(req, 8)
        self.rig.manager.req_to_blocks[req.request_id] = [object(), object()]
        self.rig.pool.free_blocks = 0
        result = self.rig.observe()
        self.assertEqual(result["requests"][0]["phase"], "blocked_capacity")
        self.assertEqual(result["decode"]["b_d"], 0)

    def test_inflight_observation_never_exports_scheduled_as_ready(self):
        req = self.rig.request(ready=8, status="RUNNING")
        self.rig.observer.event("begin_execution", NS(num_scheduled_tokens={req.request_id: 8}))
        result = self.rig.observe()
        self.assertFalse(result["available"])
        self.assertFalse(result["coherent_snapshot"])
        self.assertEqual(result["reason"], "inflight_execution")
        self.assertIsNone(result["decode"])
        self.assertIsNone(result["prefill"])

    def test_identity_presence_for_pending_reconciliation(self):
        req = self.rig.request("engine-id")
        payload = {"request_ids": [req.request_id, "not-observed"]}
        self.assertEqual(self.rig.observe(payload)["request_presence"],
                         {"engine-id": "accepted", "not-observed": "unknown"})
        del self.rig.scheduler.requests[req.request_id]
        self.rig.observer.event("after_update")
        self.assertEqual(self.rig.observe(payload)["request_presence"]["engine-id"], "terminal")


class BoundaryTests(unittest.TestCase):
    def test_invalid_payloads(self):
        invalid = [None, [], {"other": 1}, {"pin": [1]}, {"ttl": 60}, {"prompt_token_ids": []},
                   {"prompt_token_ids": [True]}, {"prompt_token_ids": [-1]},
                   {"prompt_token_ids": [10000]}, {"prompt_token_ids": [1.0]},
                   {"request_ids": [""]}, {"request_ids": [1]},
                   {"request_ids": ["r"] * 129}, {"cache_salt": 42},
                   {"expected_node_epoch": "x" * 257}]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    Rig().observer.observe(payload)

    def test_prompt_length_limits(self):
        rig = Rig()
        rig.config.model_config.max_model_len = 4
        with self.assertRaises(ValueError):
            rig.observe({"prompt_token_ids": [1] * 5})
        with patch.object(observer_module, "MAX_PROBE_TOKENS", 2):
            with self.assertRaises(ValueError):
                rig.observe({"prompt_token_ids": [1] * 3})

    def test_epoch_mismatch_and_ttl_disabled(self):
        rig = Rig()
        result = rig.observe({"expected_node_epoch": "old-epoch"})
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "node_epoch_mismatch")
        self.assertEqual(result["ttl"], {"supported": False, "enabled": False})

    def test_rate_limit_is_fail_closed(self):
        rig = Rig()
        with patch.object(observer_module.time, "monotonic", return_value=100):
            self.assertTrue(rig.observer.observe({})["available"])
            second = rig.observer.observe({})
        self.assertFalse(second["available"])
        self.assertEqual(second["reason"], "rate_limited")

    def test_request_budget(self):
        rig = Rig()
        rig.request("one")
        rig.request("two")
        with patch.object(observer_module, "MAX_REQUESTS", 1):
            self.assertEqual(rig.observe()["reason"], "request_budget_exceeded")

    def test_lookup_budget_returns_unknown(self):
        rig = Rig()
        req, _ = rig.cached()
        with patch.object(observer_module, "MAX_LOOKUPS", 1):
            result = rig.probe(req)["prefix"]
        self.assertIsNone(result["c"])
        self.assertEqual(result["status"], "lookup_budget_exceeded")

    def test_callback_error_disables_observer_not_raises_into_engine(self):
        rig = Rig()
        before = rig.serving_state()
        rig.observer.event("does_not_exist")
        result = rig.observe()
        self.assertFalse(result["available"])
        self.assertFalse(result["coherent_snapshot"])
        self.assertEqual(result["reason"], "observer_hook_error:AttributeError")
        self.assertEqual(before, rig.serving_state())
        self.assertEqual(rig.observer.hook_calls, 1)

    def test_reused_terminal_id_disables_identity_tracking(self):
        rig = Rig()
        req = rig.request()
        del rig.scheduler.requests[req.request_id]
        rig.observer.event("after_update")
        rig.observer.event("request_added", req)
        self.assertEqual(rig.observe()["reason"], "observer_hook_error:ValueError")

    def test_identity_capacity_exhaustion_fails_closed(self):
        rig = Rig()
        with patch.object(observer_module, "MAX_TERMINALS", 1):
            rig.request("first")
            rig.request("second")
        self.assertFalse(rig.observe()["available"])
        self.assertEqual(rig.observer.error, "observer_hook_error:ValueError")

    def test_unsupported_request_produces_unknown(self):
        rig = Rig()
        req = rig.request()
        req.prompt_embeds = object()
        self.assertFalse(rig.observe()["available"])
        self.assertEqual(rig.observe()["requests"][0]["reason"], "unsupported_request")

    def test_unsupported_execution_layouts(self):
        changes = [
            (lambda r: setattr(r.config.parallel_config, "tensor_parallel_size", 2), "unsupported_parallel_layout"),
            (lambda r: setattr(r.core, "async_scheduling", True), "unsupported_async_or_batch_queue"),
            (lambda r: setattr(r.core, "batch_queue", []), "unsupported_async_or_batch_queue"),
            (lambda r: setattr(r.core, "use_spec_decode", True), "unsupported_speculative_or_lora"),
            (lambda r: setattr(r.config, "lora_config", object()), "unsupported_speculative_or_lora"),
            (lambda r: setattr(r.config.model_config, "is_multimodal_model", True), "unsupported_multimodal"),
            (lambda r: setattr(r.scheduler.kv_cache_manager.coordinator, "single_type_managers", []), "unsupported_cache_groups"),
            (lambda r: setattr(r.scheduler.kv_cache_manager.coordinator, "kv_cache_spec", object()), "unsupported_attention_type"),
            (lambda r: setattr(r.scheduler, "connector", object()), "unsupported_connector"),
        ]
        for change, expected in changes:
            with self.subTest(expected=expected):
                rig = Rig()
                change(rig)
                rig.observer = PpdStateObserver(rig.core)
                self.assertEqual(rig.observe()["reason"], expected)

    def test_empty_supported_snapshot_has_coherent_known_zero(self):
        result = Rig().observe()
        self.assertTrue(result["available"])
        self.assertTrue(result["coherent_snapshot"])
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["decode"], {"b_d": 0, "k_d": 0})
        self.assertEqual(result["prefill"], {"q_tok": 0, "q_att": 0})
        self.assertGreaterEqual(result["observer_elapsed_s"], 0)


if __name__ == "__main__":
    unittest.main()
