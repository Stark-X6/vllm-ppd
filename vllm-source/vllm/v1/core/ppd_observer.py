"""Opt-in, owner-thread PPD observation; no scheduling or KV ownership writes.

This module deliberately depends only on the standard library. Cache lifecycle
hooks update observer-private readiness evidence. Unknown evidence is never a
cache miss. Unsupported execution layouts fail closed without synchronizing GPU.
"""

import hashlib
import json
import os
import socket
import time
import uuid
from types import SimpleNamespace


MAX_REQUESTS = 512
MAX_LOOKUPS = 32768
MAX_TERMINALS = 4096
MAX_PROBE_TOKENS = 262144


def enabled():
    return os.environ.get("VLLM_PPD_OBSERVER_ENABLED") == "1"


def _pairs(prefix, suffix):
    return prefix * suffix + suffix * (suffix + 1) // 2


def _status(request):
    return str(getattr(request.status, "name", request.status))


class PpdStateObserver:
    def __init__(self, core):
        self.core = core
        self.epoch = uuid.uuid4().hex
        self.node_id = f"{socket.gethostname()}:{os.getpid()}"
        self.cache_generation = 0
        self.step = self.seq = 0
        self.last_query = float("-inf")
        self.error = None
        self.hook_calls = self.hook_ns = 0
        self.inflight = False
        self.jobs = {}
        self.frontiers = {}
        self.marks = {}  # block_id -> (hash-with-group, proven ready)
        self.owners = {}  # block_id -> (physical engine request ID, preemptions)
        self.pending = {}  # owner -> {block_id: (hash, full-block end)}
        self.accepted = set()
        self.terminals = set()
        self.unsupported = self._unsupported_reason()

    def _unsupported_reason(self):
        cfg = self.core.vllm_config
        parallel = cfg.parallel_config
        if any(getattr(parallel, k, 1) != 1 for k in (
            "data_parallel_size", "tensor_parallel_size", "pipeline_parallel_size",
            "decode_context_parallel_size", "prefill_context_parallel_size",
        )):
            return "unsupported_parallel_layout"
        if self.core.async_scheduling or self.core.batch_queue is not None:
            return "unsupported_async_or_batch_queue"
        if self.core.use_spec_decode or getattr(cfg, "lora_config", None):
            return "unsupported_speculative_or_lora"
        if getattr(cfg.model_config, "is_multimodal_model", False):
            return "unsupported_multimodal"
        manager = getattr(self.core.scheduler, "kv_cache_manager", None)
        coordinator = getattr(manager, "coordinator", None)
        managers = getattr(coordinator, "single_type_managers", ())
        if len(managers) != 1 or type(managers[0]).__name__ != "FullAttentionManager":
            return "unsupported_cache_groups"
        if type(getattr(coordinator, "kv_cache_spec", None)).__name__ != "FullAttentionSpec":
            return "unsupported_attention_type"
        connector = getattr(self.core.scheduler, "connector", None)
        if connector is not None and type(connector).__name__ != "P2pNcclConnector":
            return "unsupported_connector"
        return None

    @property
    def pool(self):
        return self.core.scheduler.kv_cache_manager.block_pool

    @property
    def block_size(self):
        return self.core.scheduler.kv_cache_manager.coordinator.block_size

    def event(self, name, *args):
        """Failure disables observation, never the actual serving operation."""
        if self.unsupported or self.error:
            return
        begin = time.perf_counter_ns()
        try:
            getattr(self, "_" + name)(*args)
        except Exception as exc:
            self.error = "observer_hook_error:" + type(exc).__name__
        finally:
            self.hook_calls += 1
            self.hook_ns += time.perf_counter_ns() - begin

    def _forget(self, block_id):
        self.marks.pop(block_id, None)
        owner = self.owners.pop(block_id, None)
        entries = self.pending.get(owner)
        if entries is not None:
            entries.pop(block_id, None)
            if not entries:
                self.pending.pop(owner, None)

    def _cache_insert(self, request, block, end_tokens):
        self._forget(block.block_id)
        owner = (request.request_id, request.num_preemptions)
        # This also handles registration during output processing AFTER a
        # successful forward. Preemption generations cannot borrow old proof.
        ready = end_tokens <= self.frontiers.get(owner, -1) and not self.inflight
        self.marks[block.block_id] = (block.block_hash, ready)
        if not ready:
            self.owners[block.block_id] = owner
            self.pending.setdefault(owner, {})[block.block_id] = (block.block_hash, end_tokens)

    def _cache_evict(self, block):
        self._forget(block.block_id)

    def _cache_reset(self):
        self.marks.clear()
        self.owners.clear()
        self.pending.clear()
        self.frontiers.clear()
        self.cache_generation += 1

    def _request_added(self, request):
        if request.request_id in self.terminals:
            raise ValueError("reused physical engine ID in one epoch")
        self.accepted.add(request.request_id)
        if len(self.accepted) + len(self.terminals) > MAX_TERMINALS:
            raise ValueError("identity tracking capacity exhausted")

    def _begin_execution(self, scheduled):
        if self.inflight:
            raise ValueError("overlapping execution is unsupported")
        self.inflight = True
        self.jobs = {}
        for request_id in scheduled.num_scheduled_tokens:
            req = self.core.scheduler.requests[request_id]
            owner = (request_id, req.num_preemptions)
            self.jobs[owner] = req.num_computed_tokens

    def _finish_execution(self, model_output):
        invalid = getattr(getattr(model_output, "kv_connector_output", None), "invalid_block_ids", ())
        if invalid:
            # No false readiness after a load failure or rollback. Retained
            # physical hashes now have unknown proof until recomputed/rebuilt.
            self._cache_reset()
        else:
            for owner, frontier in self.jobs.items():
                self.frontiers[owner] = frontier
                for block_id, (block_hash, end) in list(self.pending.get(owner, {}).items()):
                    if end <= frontier and self.marks.get(block_id) == (block_hash, False):
                        self.marks[block_id] = (block_hash, True)
                        self.owners.pop(block_id, None)
                        self.pending[owner].pop(block_id)
                if not self.pending.get(owner):
                    self.pending.pop(owner, None)
        self.jobs.clear()
        self.inflight = False
        self.step += 1

    def _after_update(self):
        requests = self.core.scheduler.requests
        done = {rid for rid in self.accepted
                if rid not in requests or _status(requests[rid]).startswith("FINISHED")}
        self.accepted.difference_update(done)
        self.terminals.update(done)
        for owner in list(self.frontiers):
            req = requests.get(owner[0])
            if req is None or req.num_preemptions != owner[1] or owner[0] in done:
                self.frontiers.pop(owner, None)
        if len(self.accepted) + len(self.terminals) > MAX_TERMINALS:
            raise ValueError("identity tracking capacity exhausted")

    def _probe_hashes(self, hashes, n):
        if not self.core.scheduler.kv_cache_manager.enable_caching:
            return 0, [], "prefix_caching_disabled"
        if self.inflight:
            return None, [], "inflight_readiness_unknown"
        if len(hashes) < (n - 1) // self.block_size:
            return None, [], "hash_chain_incomplete"
        hits = []
        for block_hash in hashes[:(n - 1) // self.block_size]:
            self.lookups += 1
            if self.lookups > MAX_LOOKUPS:
                return None, [], "lookup_budget_exceeded"
            blocks = self.pool.get_cached_block(block_hash, [0])
            if blocks is None:
                break
            block = blocks[0]
            if self.marks.get(block.block_id) != (block.block_hash, True):
                return None, [], "matched_block_readiness_unknown"
            hits.append(block)
        return len(hits) * self.block_size, hits, "ready"

    def _capacity(self, req, prefix, hits, scheduled_tokens):
        """Pure equivalent of the supported manager's allocation count.

        Do NOT call get_num_blocks_to_allocate(): it indexes a defaultdict
        and would create request state during a read-only observation.
        """
        manager = self.core.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        held = manager.req_to_blocks.get(req.request_id, ())
        new = max(0, (prefix + scheduled_tokens + self.block_size - 1) // self.block_size
                  - len(held) - len(hits))
        acquire = sum(block.ref_cnt == 0 and not block.is_null for block in hits)
        return new + acquire <= self.pool.get_num_free_blocks()

    def _request_row(self, req):
        status = _status(req)
        row = {"request_id": req.request_id, "status": status,
               "preemptions": req.num_preemptions, "target_tokens": req.num_tokens,
               "prompt_tokens": getattr(req, "num_prompt_tokens", None),
               "generated_tokens": req.num_output_tokens,
               "ready_tokens": None, "scheduled_tokens": req.num_computed_tokens,
               "phase": "unknown"}
        if status.startswith("FINISHED"):
            row["phase"] = "finished"
            return row
        if req.mm_features or req.lora_request or req.prompt_embeds is not None or req.pooling_params is not None:
            row["reason"] = "unsupported_request"
            return row
        if status == "WAITING_FOR_FSM":
            row["phase"] = "blocked_fsm"
            return row
        owner = (req.request_id, req.num_preemptions)
        proven = self.frontiers.get(owner)
        connector = self.core.scheduler.connector
        remote = (connector is not None and not connector.is_producer
                  and connector.is_pd_request(req.request_id))
        if status == "WAITING_FOR_REMOTE_KVS" or (remote and proven is None):
            row["phase"] = "remote_kv_pending"
            return row
        hits = []
        if status in {"WAITING", "PREEMPTED"}:
            if getattr(req, "skip_reading_prefix_cache", False):
                proven, hits, reason = 0, [], "request_skips_prefix_cache"
            else:
                proven, hits, reason = self._probe_hashes(req.block_hashes, req.num_tokens)
            row["prefix_status"] = reason
            # Pending prefill can have cached E even though scheduler's raw
            # counter is still zero. Normalized S includes that usable prefix.
            row["scheduled_tokens"] = proven
        elif proven != req.num_computed_tokens:
            proven = None
        row["ready_tokens"] = proven
        if proven is None:
            row["reason"] = "ready_frontier_unknown"
            return row
        if not 0 <= proven <= req.num_tokens:
            row["reason"] = "invalid_frontier"
            return row
        remaining = req.num_tokens - proven
        decode = req.num_output_tokens > 0 and remaining == 1 and status == "RUNNING"
        if decode:
            if self._capacity(req, proven, [], 1):
                row.update(phase="decode_ready", context_tokens=proven)
            else:
                row["phase"] = "blocked_capacity"
            return row
        chunk = min(remaining, self.core.scheduler.max_num_scheduled_tokens)
        threshold = self.core.vllm_config.scheduler_config.long_prefill_token_threshold
        if threshold > 0:
            chunk = min(chunk, threshold)
        if chunk and not self._capacity(req, proven, hits, chunk):
            # The scheduler's REMAINING token budget can admit a smaller
            # chunk. Failure of the maximum chunk is not zero compute demand.
            if not self._capacity(req, proven, hits, 1):
                row["phase"] = "blocked_capacity"
                return row
            row["capacity_risk"] = "chunk_size_dependent"
        row.update(phase="prefill_ready", remaining_tokens=remaining,
                   attention_pairs=_pairs(proven, remaining))
        return row

    def observe(self, payload):
        begin = time.perf_counter_ns()
        cfg = self.core.vllm_config.model_config
        allowed = {"prompt_token_ids", "cache_salt", "expected_node_epoch", "request_ids"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError("invalid observer payload")
        # Rate-limit before walking a large token array or hashing its contents.
        now = time.monotonic()
        self.seq += 1
        base = {"schema_version": 1, "semantics_version": "ppd-state-semantics-v1",
                "node_id": self.node_id, "node_epoch": self.epoch,
                "clock_scope": f"process-monotonic:{self.node_id}:{self.epoch}",
                "snapshot_seq": self.seq, "scheduler_step": self.step,
                "sampled_at_monotonic_s": now, "cache_generation": self.cache_generation,
                "readonly": True, "coherent_snapshot": False,
                "coverage_complete": False,
                "ttl": {"supported": False, "enabled": False}}
        reason = self.unsupported or self.error
        if now - self.last_query < 0.1:
            reason = "rate_limited"
        else:
            self.last_query = now
        if self.inflight or self.core.batch_queue:
            reason = "inflight_execution"
        if reason:
            return dict(base, available=False, reason=reason, prefix=None,
                        decode=None, prefill=None,
                        observer_elapsed_s=(time.perf_counter_ns() - begin) / 1e9)
        ids = payload.get("prompt_token_ids")
        if "prompt_token_ids" in payload and ids is None:
            raise ValueError("explicit null token probe is invalid")
        if ids is not None:
            vocab = cfg.get_vocab_size()
            if (not isinstance(ids, list) or not ids or
                    len(ids) > min(cfg.max_model_len, MAX_PROBE_TOKENS) or
                    any(type(token) is not int or not 0 <= token < vocab for token in ids)):
                raise ValueError("invalid prompt token IDs")
        salt = payload.get("cache_salt")
        expected = payload.get("expected_node_epoch")
        for key, value in (("cache_salt", salt), ("expected_node_epoch", expected)):
            if key in payload and (not isinstance(value, str) or not 1 <= len(value) <= 256):
                raise ValueError("invalid observer string")
        if salt is not None and ids is None:
            raise ValueError("cache_salt requires token IDs")
        tracked = payload.get("request_ids", [])
        if (not isinstance(tracked, list) or len(tracked) > 128
                or any(not isinstance(v, str) or not v or len(v) > 256 for v in tracked)):
            raise ValueError("invalid tracked request IDs")
        if len(tracked) != len(set(tracked)):
            raise ValueError("duplicate tracked request IDs")
        if expected is not None and expected != self.epoch:
            return dict(base, available=False, reason="node_epoch_mismatch", prefix=None,
                        decode=None, prefill=None,
                        observer_elapsed_s=(time.perf_counter_ns() - begin) / 1e9)
        self.lookups = 0
        requests = self.core.scheduler.requests
        if len(requests) > MAX_REQUESTS:
            return dict(base, available=False, reason="request_budget_exceeded",
                        prefix=None, decode=None, prefill=None,
                        observer_elapsed_s=(time.perf_counter_ns() - begin) / 1e9)
        prefix = None
        if ids is not None:
            view = SimpleNamespace(all_token_ids=ids, num_tokens=len(ids),
                                   block_hashes=[], mm_features=[], lora_request=None,
                                   prompt_embeds=None, cache_salt=salt)
            if self.core.request_block_hasher is None and self.core.scheduler.kv_cache_manager.enable_caching:
                c, status = None, "hash_builder_unavailable"
            else:
                hashes = self.core.request_block_hasher(view) if self.core.request_block_hasher else []
                c, _, status = self._probe_hashes(hashes, len(ids))
            prefix = {"n": len(ids), "c": c, "u": None if c is None else len(ids) - c,
                      "available": c is not None,
                      "status": status, "source": "caller_supplied_token_ids",
                      "canonical_chat_verified": False,
                      "prefix_read_policy": "normal_generation_no_prompt_logprobs",
                      "token_digest": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
        rows = [self._request_row(req) for req in requests.values()]
        unknown = sum(r["phase"] == "unknown" for r in rows)
        dec = [r for r in rows if r["phase"] == "decode_ready"]
        pre = [r for r in rows if r["phase"] == "prefill_ready"]
        presence = {}
        for rid in tracked:
            presence[rid] = ("terminal" if rid in self.terminals else
                             "accepted" if rid in requests else "unknown")
        base.update(available=not unknown, coherent_snapshot=True, coverage_complete=not unknown,
                    reason="request_state_unknown" if unknown else None,
                    prefix=prefix, requests=rows, request_presence=presence,
                    decode={"b_d": None if unknown else len(dec),
                            "k_d": None if unknown else sum(r["context_tokens"] for r in dec)},
                    prefill={"q_tok": None if unknown else sum(r["remaining_tokens"] for r in pre),
                             "q_att": None if unknown else sum(r["attention_pairs"] for r in pre)},
                    blocked_requests=sum(r["phase"].startswith("blocked") or r["phase"] == "remote_kv_pending" for r in rows),
                    unknown_requests=unknown, inflight_requests=0,
                    free_blocks=self.pool.get_num_free_blocks(),
                    physical_blocks=self.pool.num_gpu_blocks,
                    block_lookups=self.lookups, hook_calls=self.hook_calls,
                    hook_elapsed_s=self.hook_ns / 1e9)
        base["observer_elapsed_s"] = (time.perf_counter_ns() - begin) / 1e9
        return base
