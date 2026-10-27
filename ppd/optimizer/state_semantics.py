"""CPU reference arithmetic for the stage-5 semantic contract.

No engine adapter, tokenizer, observer, router, or TTL policy is installed here.
Callers must establish the documented token, readiness, identity and clock
provenance before supplying observations. None means unknown, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Iterable, Sequence


SEMANTICS_VERSION = "ppd-state-semantics-v1"
LEGACY_STREAM_BASIS = "legacy_client_stream_text_v1"
NODE_ENGINE_BASIS = "node_engine_generation_v1"


def _count(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")


def _elapsed(value: float | None, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative duration or None")


def _flag(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class PromptSplit:
    n: int
    h: int | None
    a: int | None
    boundary_status: str


def split_prompt_ids(
    full_ids: Sequence[int], history_prefix_ids: Sequence[int] | None,
) -> PromptSplit:
    """Verify a proposed historical token prefix; never substitute token LCP.

    The caller chooses the message boundary and renders with the same effective
    template context. A standalone suffix length is not an input to this helper.
    """
    full = tuple(full_ids)
    if not full:
        raise ValueError("full prompt must contain tokens")
    for token in full:
        _count(token, "token ID")
    if history_prefix_ids is None:
        return PromptSplit(len(full), None, None, "unavailable")
    history = tuple(history_prefix_ids)
    for token in history:
        _count(token, "history token ID")
    if len(history) > len(full) or full[:len(history)] != history:
        return PromptSplit(len(full), None, None, "history_not_prefix")
    return PromptSplit(len(full), len(history), len(full) - len(history), "verified")


def uncached_tokens(n: int, effective_prefix: int | None) -> int | None:
    """U=N-C for a fresh normal generation, after engine tail/block rules.

    Raw resident length is not C. This fork must compute a tail for logits,
    so a valid effective prefix is strictly shorter than the full prompt.
    """
    _count(n, "N", positive=True)
    if effective_prefix is None:
        return None
    _count(effective_prefix, "C")
    if effective_prefix >= n:
        raise ValueError("effective C must be < N; apply tail/block rules first")
    return n - effective_prefix


def attention_pairs(prefix: int, suffix: int) -> int:
    """Structural causal-attention work, not FLOPs, bytes, or milliseconds."""
    _count(prefix, "prefix")
    _count(suffix, "suffix")
    return prefix * suffix + suffix * (suffix + 1) // 2


@dataclass(frozen=True)
class DecodeSample:
    work_id: str
    ready_context_tokens: int | None


@dataclass(frozen=True)
class DecodeSummary:
    b_d: int | None
    k_d: int | None
    unknown_contexts: int | None


def summarize_decode(
    eligible: Sequence[DecodeSample] | None, *, coherent: bool,
) -> DecodeSummary:
    """Summarize an externally identified eligible, not merely scheduled, set.

    v1 requires post-completion/pre-schedule state (no unresolved decode work).
    Async/in-flight state without a proven ready frontier is unavailable.
    Empty coherent observations are different from an unavailable observation.
    """
    _flag(coherent, "coherent")
    if not coherent or eligible is None:
        return DecodeSummary(None, None, None)
    seen: set[str] = set()
    total = missing = 0
    for sample in eligible:
        if not sample.work_id or sample.work_id in seen:
            raise ValueError("decode work IDs must be nonempty and unique")
        seen.add(sample.work_id)
        if sample.ready_context_tokens is None:
            missing += 1
        else:
            _count(sample.ready_context_tokens, "decode context", positive=True)
            total += sample.ready_context_tokens
    return DecodeSummary(len(seen), None if missing else total, missing)


@dataclass(frozen=True)
class PrefillProgress:
    work_id: str
    target_tokens: int
    ready_tokens: int | None
    scheduled_tokens: int | None
    state: str = "ready"

    def __post_init__(self) -> None:
        if not self.work_id:
            raise ValueError("work_id is required")
        if self.state not in {
            "ready", "inflight", "blocked", "remote_kv_pending", "decode", "finished",
        }:
            raise ValueError("unknown progress state")
        _count(self.target_tokens, "T")
        for name, value in (("E", self.ready_tokens), ("S", self.scheduled_tokens)):
            if value is not None:
                _count(value, name)
                if value > self.target_tokens:
                    raise ValueError(f"{name} must not exceed T")
        if (self.ready_tokens is not None and self.scheduled_tokens is not None
                and self.ready_tokens > self.scheduled_tokens):
            raise ValueError("ready E must not exceed scheduled S")


@dataclass(frozen=True)
class BacklogSummary:
    q_tok: int | None
    q_att: int | None
    inflight_tokens: int | None
    compute_requests: int
    blocked_requests: int
    pending_requests: int
    unknown_work_requests: int


def _unique_progress(rows: Iterable[PrefillProgress]) -> dict[str, PrefillProgress]:
    result: dict[str, PrefillProgress] = {}
    for row in rows:
        if row.work_id in result:
            raise ValueError("duplicate physical work ID in one source")
        result[row.work_id] = row
    return result


def summarize_backlog(
    engine: Iterable[PrefillProgress],
    pending: Iterable[PrefillProgress] = (),
    *,
    retired_ids: Iterable[str] = (),
    exclude_work_id: str | None = None,
) -> BacklogSummary:
    """Count local compute once, retaining blocked and incomplete diagnostics.

    Inputs must concern ONE node/epoch with reconciled physical attempt IDs.
    An engine observation wins over pending, even if it is blocked/finished.
    retired_ids are terminal ACK tombstones, not absence from a snapshot.
    Callers with incomplete enumeration must mark the enclosing snapshot
    unavailable; partial sums from such inputs are not valid lookup features.
    """
    active = _unique_progress(engine)
    proxy = _unique_progress(pending)
    retired = set(retired_ids)
    pending_only = set(proxy) - set(active) - retired
    merged = {**proxy, **active}
    q_tok = q_att = inflight = compute = blocked = pending_count = missing = 0
    unknown_inflight = False
    for work_id, row in merged.items():
        if work_id in retired or work_id == exclude_work_id:
            continue
        if row.state in {"blocked", "remote_kv_pending"}:
            blocked += 1
            continue
        if row.state not in {"ready", "inflight"}:
            continue
        compute += 1
        pending_count += work_id in pending_only
        if row.ready_tokens is None:
            missing += 1
            unknown_inflight = True
            continue
        remaining = row.target_tokens - row.ready_tokens
        q_tok += remaining
        q_att += attention_pairs(row.ready_tokens, remaining)
        if row.scheduled_tokens is None:
            unknown_inflight = True
        else:
            inflight += row.scheduled_tokens - row.ready_tokens
    return BacklogSummary(
        None if missing else q_tok, None if missing else q_att,
        None if unknown_inflight else inflight, compute, blocked, pending_count, missing,
    )


def request_tpot_s(
    post_first_elapsed_s: float | None, generated_tokens: int | None,
) -> float | None:
    """Per-request mean; caller supplies a single documented timing basis.

    Engine basis uses first-to-last generation elapsed. Legacy stream basis
    uses first-text-to-stream-end elapsed. Never pool those observations.
    """
    _elapsed(post_first_elapsed_s, "post-first duration")
    if generated_tokens is not None:
        _count(generated_tokens, "G")
    if post_first_elapsed_s is None or generated_tokens is None or generated_tokens <= 1:
        return None
    return post_first_elapsed_s / (generated_tokens - 1)


@dataclass(frozen=True)
class CalibrationObservation:
    """One API attempt; attempt is the proxy's cross-phase physical ordinal.

    success means a complete, successful API response, not tool/task correctness.
    A normal finish_reason=length response remains an API success.
    """

    work_id: str
    metric_basis: str
    success: bool
    generated_tokens: int | None
    ttft_s: float | None
    tpot_s: float | None
    role: str = "candidate"
    turn: int = 2
    attempt: int = 1
    warmup: bool = False

    def __post_init__(self) -> None:
        _flag(self.success, "success")
        _flag(self.warmup, "warmup")
        if not self.work_id:
            raise ValueError("work_id is required")
        if self.role not in {"candidate", "background", "prefill_only"}:
            raise ValueError("unknown measurement role")
        if self.metric_basis not in {LEGACY_STREAM_BASIS, NODE_ENGINE_BASIS}:
            raise ValueError("unknown metric basis")
        _count(self.turn, "turn", positive=True)
        _count(self.attempt, "attempt", positive=True)
        if self.generated_tokens is not None:
            _count(self.generated_tokens, "G")
        _elapsed(self.ttft_s, "TTFT")
        _elapsed(self.tpot_s, "TPOT")
        if (self.generated_tokens is None or self.generated_tokens <= 1) and self.tpot_s is not None:
            raise ValueError("TPOT must be None when G is unknown or <= 1")
        if self.generated_tokens == 0 and self.ttft_s is not None:
            raise ValueError("zero generated tokens cannot have a first-token latency")


@dataclass(frozen=True)
class CalibrationSummary:
    metric_basis: str
    population: str
    n_total: int
    n_success: int
    n_failed: int
    n_ttft: int
    n_tpot: int
    n_missing_usage: int
    n_zero_output: int
    avg_ttft_s: float | None
    avg_tpot_s: float | None


def summarize_calibration(
    observations: Iterable[CalibrationObservation], *, metric_basis: str,
) -> CalibrationSummary:
    """Request-weighted candidate-turn2/first-attempt summary; no routing.

    Failure counts remain visible. G<=1 is not a zero-latency TPOT sample.
    Positive TPOT eligibility matches the legacy scorer's averaging rule.
    Missing observations are not fabricated; planned/attempted counts must
    additionally be tracked by the future calibration runner.
    """
    if metric_basis not in {LEGACY_STREAM_BASIS, NODE_ENGINE_BASIS}:
        raise ValueError("unknown metric basis")
    selected = [o for o in observations
                if o.role == "candidate" and o.turn == 2 and o.attempt == 1 and not o.warmup]
    seen: set[str] = set()
    for row in selected:
        if row.work_id in seen:
            raise ValueError("duplicate candidate attempt")
        seen.add(row.work_id)
        if row.metric_basis != metric_basis:
            raise ValueError("cannot mix timing scopes/bases")
    successes = [row for row in selected if row.success]
    ttfts = [row.ttft_s for row in successes if row.ttft_s is not None]
    tpots = [row.tpot_s for row in successes if row.tpot_s is not None and row.tpot_s > 0]
    return CalibrationSummary(
        metric_basis, "candidate_turn2_first_attempt_v1",
        len(selected), len(successes), len(selected) - len(successes), len(ttfts), len(tpots),
        sum(row.generated_tokens is None for row in selected),
        sum(row.generated_tokens == 0 for row in selected),
        mean(ttfts) if ttfts else None, mean(tpots) if tpots else None,
    )
