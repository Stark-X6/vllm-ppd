#!/usr/bin/env python3
"""
PPD Decision Engine - Determines when to use PPD mode based on benchmark data.

This engine builds a lookup table from comprehensive benchmark results to decide
whether Turn 2+ requests should use PPD mode (direct to pD) or PD mode (through P->D).

Key factors for decision:
1. context_length: Accumulated tokens from previous turns (affects KV cache size)
2. input_tokens: New tokens in current turn
3. output_tokens: Expected output tokens (available from ShareGPT)
4. current_qps: Current system load

The lookup table compares PPD config (e.g., 2P_2pD) vs PD config (e.g., 2P_2D)
and records which mode has better T2 TTFT for each (workload, qps) combination.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from .calibration_contract import calibration_result_filename_candidates

logger = logging.getLogger(__name__)


@dataclass
class DecisionStats:
    """Statistics about PPD decisions made."""
    total_decisions: int = 0
    ppd_decisions: int = 0
    pd_decisions: int = 0
    turn1_skipped: int = 0
    decisions_by_workload: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, workload: str, use_ppd: bool, is_turn1: bool = False):
        self.total_decisions += 1
        if is_turn1:
            self.turn1_skipped += 1
            return

        if use_ppd:
            self.ppd_decisions += 1
        else:
            self.pd_decisions += 1

        if workload not in self.decisions_by_workload:
            self.decisions_by_workload[workload] = {"ppd": 0, "pd": 0}

        key = "ppd" if use_ppd else "pd"
        self.decisions_by_workload[workload][key] += 1

    def to_dict(self) -> dict:
        return {
            "total_decisions": self.total_decisions,
            "ppd_decisions": self.ppd_decisions,
            "pd_decisions": self.pd_decisions,
            "turn1_skipped": self.turn1_skipped,
            "ppd_ratio": self.ppd_decisions / max(1, self.ppd_decisions + self.pd_decisions),
            "decisions_by_workload": self.decisions_by_workload,
        }


# Context length classes based on T1 configs in benchmark
# small: 128 input + 128 output = 256 tokens
# large: 1024 input + 1024 output = 2048 tokens
CONTEXT_THRESHOLDS = {
    "small": (0, 512),      # context <= 512 tokens
    "large": (512, 4096),   # 512 < context <= 4096 tokens
    "huge": (4096, float('inf')),  # context > 4096 tokens
}

# Workload classification based on input/output ratio
# Maps to benchmark T2 configs
T2_WORKLOAD_CONFIGS = {
    "tiny":         {"input": 16, "output": 32},      # ratio 0.5
    "short_gen":    {"input": 32, "output": 256},     # ratio 0.125
    "long_gen":     {"input": 32, "output": 512},     # ratio 0.0625
    "very_long_gen":{"input": 64, "output": 1024},    # ratio 0.0625
    "small_bal":    {"input": 64, "output": 64},      # ratio 1.0
    "mid_bal":      {"input": 128, "output": 128},    # ratio 1.0
    "mid_paste":    {"input": 256, "output": 64},     # ratio 4.0
    "big_paste":    {"input": 512, "output": 64},     # ratio 8.0
    "huge_paste":   {"input": 1024, "output": 32},    # ratio 32.0
}

# PD <-> PPD config pairs
CONFIG_PAIRS = {
    # Pure PD -> Pure PPD
    "1P_3D": "1P_3pD",
    "2P_2D": "2P_2pD",
    "3P_1D": "3P_1pD",
    # Hybrid PD -> Hybrid PPD
    "1R_1P_2D": "1R_1P_2pD",
    "1R_2P_1D": "1R_2P_1pD",
    "2R_1P_1D": "2R_1P_1pD",
}

# Available QPS points from benchmark
QPS_POINTS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]


def classify_context_length(context_tokens: int) -> str:
    """Classify context length into categories.

    Args:
        context_tokens: Total tokens from previous turns

    Returns:
        Context class: "small", "large", or "huge"
    """
    for ctx_class, (low, high) in CONTEXT_THRESHOLDS.items():
        if low < context_tokens <= high:
            return ctx_class
    return "small"


def classify_workload(input_tokens: int, output_tokens: int) -> str:
    """Classify workload based on input/output tokens.

    Maps real workload characteristics to benchmark T2 workload types.

    Args:
        input_tokens: Number of input tokens in current turn
        output_tokens: Number of output tokens (from ShareGPT record)

    Returns:
        Workload type name matching benchmark configs
    """
    if output_tokens == 0:
        output_tokens = 1

    ratio = input_tokens / output_tokens

    # Classify by ratio and absolute values
    if ratio < 0.25:  # Decode-heavy (short input, long output)
        if output_tokens <= 64:
            return "tiny"
        elif output_tokens <= 256:
            return "short_gen"
        elif output_tokens <= 512:
            return "long_gen"
        else:
            return "very_long_gen"
    elif ratio <= 2.0:  # Balanced
        if input_tokens <= 64:
            return "small_bal"
        else:
            return "mid_bal"
    else:  # Prefill-heavy (long input, short output)
        if input_tokens <= 256:
            return "mid_paste"
        elif input_tokens <= 512:
            return "big_paste"
        else:
            return "huge_paste"


def find_nearest_qps(qps: float) -> float:
    """Find the nearest QPS point from benchmark data."""
    if qps <= QPS_POINTS[0]:
        return QPS_POINTS[0]
    if qps >= QPS_POINTS[-1]:
        return QPS_POINTS[-1]

    # Find the two nearest points
    for i in range(len(QPS_POINTS) - 1):
        if QPS_POINTS[i] <= qps <= QPS_POINTS[i + 1]:
            # Return the closer one
            if qps - QPS_POINTS[i] < QPS_POINTS[i + 1] - qps:
                return QPS_POINTS[i]
            else:
                return QPS_POINTS[i + 1]

    return QPS_POINTS[-1]


class PPDDecisionEngine:
    """Engine for making PPD vs PD routing decisions based on benchmark data.

    Uses weighted score calculation:
        score = w_ttft * ttft_improvement - w_tpot * tpot_degradation
        use_ppd = score > 0
    """

    def __init__(
        self,
        benchmark_data_path: str,
        base_config: str = "2P_2D",
        default_use_ppd: bool = True,
        w_ttft: float = 1.0,
        w_tpot: float = 1.0,
    ):
        """Initialize the decision engine.

        Args:
            benchmark_data_path: Path to benchmark results directory
            base_config: The PD config to compare against (e.g., "2P_2D")
            default_use_ppd: Default decision when no benchmark data is available
            w_ttft: Weight for TTFT improvement (default 1.0)
            w_tpot: Weight for TPOT degradation penalty (default 1.0)
        """
        self.benchmark_path = Path(benchmark_data_path)
        self.base_config = base_config
        self.ppd_config = CONFIG_PAIRS.get(base_config, base_config.replace("D", "pD"))
        self.default_use_ppd = default_use_ppd
        self.w_ttft = w_ttft
        self.w_tpot = w_tpot

        # Lookup table: (context_class, workload, qps) -> use_ppd
        self.lookup_table: Dict[Tuple[str, str, float], bool] = {}

        # Performance data: (context_class, workload, qps) -> {pd_ttft, ppd_ttft, pd_tpot, ppd_tpot, score}
        self.performance_data: Dict[Tuple[str, str, float], dict] = {}

        # Decision statistics
        self.stats = DecisionStats()

        # Build the lookup table
        self._build_lookup_table()

        logger.info(f"PPD Decision Engine initialized")
        logger.info(f"  Base config: {self.base_config}")
        logger.info(f"  PPD config: {self.ppd_config}")
        logger.info(f"  Weights: w_ttft={self.w_ttft}, w_tpot={self.w_tpot}")
        logger.info(f"  Lookup table entries: {len(self.lookup_table)}")

    def _load_benchmark_result(self, config: str, workload: str, qps: float) -> Optional[dict]:
        """Load one result using the shared canonical name or a legacy alias."""
        candidates = [
            self.benchmark_path / config / filename
            for filename in calibration_result_filename_candidates(
                config, workload, qps
            )
        ]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            return None
        filepath = existing[0]
        if len(existing) > 1:
            logger.warning(
                "Multiple calibration files represent %s/%s/QPS=%s; using %s",
                config,                
                workload,
                qps,
                filepath,
            )

        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None

        expected = (config, workload, float(qps))
        try:
            actual = (
                result.get("config"),
                result.get("workload"),
                float(result.get("qps")),
            )
        except (TypeError, ValueError):
            logger.warning("Invalid calibration identity in %s", filepath)
            return None
        if actual != expected:
            logger.warning(
                "Calibration identity mismatch in %s: expected=%s actual=%s",
                filepath,
                expected,
                actual,
            )

            return None
        return result

    def _calculate_score(
        self,
        pd_ttft: float,
        ppd_ttft: float,
        pd_tpot: float,
        ppd_tpot: float,
    ) -> float:
        """Calculate weighted score for PPD decision.

        score = w_ttft * ttft_improvement - w_tpot * tpot_degradation
        Positive score means PPD is better overall.

        Args:
            pd_ttft: PD Turn 2 TTFT (ms)
            ppd_ttft: PPD Turn 2 TTFT (ms)
            pd_tpot: PD Turn 2 TPOT (ms)
            ppd_tpot: PPD Turn 2 TPOT (ms)

        Returns:
            Weighted score (positive = use PPD, negative = use PD)
        """
        # TTFT improvement (positive = PPD is better)
        ttft_improvement = (pd_ttft - ppd_ttft) / pd_ttft if pd_ttft > 0 else 0

        # TPOT degradation (positive = PPD is worse)
        tpot_degradation = (ppd_tpot - pd_tpot) / pd_tpot if pd_tpot > 0 else 0

        score = self.w_ttft * ttft_improvement - self.w_tpot * tpot_degradation
        return score

    def _build_lookup_table(self):
        """Build lookup table from benchmark data.

        Uses weighted score calculation:
            score = w_ttft * ttft_improvement - w_tpot * tpot_degradation
            use_ppd = score > 0
        """
        logger.info(f"Building lookup table from {self.benchmark_path}")

        # Context classes from T1 configs
        context_classes = ["small", "large"]

        for ctx_class in context_classes:
            for workload_type in T2_WORKLOAD_CONFIGS.keys():
                full_workload = f"{ctx_class}_{workload_type}"

                for qps in QPS_POINTS:
                    # Load PD config result
                    pd_result = self._load_benchmark_result(
                        self.base_config, full_workload, qps
                    )

                    # Load PPD config result
                    ppd_result = self._load_benchmark_result(
                        self.ppd_config, full_workload, qps
                    )

                    if pd_result is None or ppd_result is None:
                        # Use default if data not available
                        self.lookup_table[(ctx_class, workload_type, qps)] = self.default_use_ppd
                        continue

                    # Extract T2 metrics
                    pd_t2_ttft = pd_result.get("turn2", {}).get("avg_ttft_ms", float('inf'))
                    ppd_t2_ttft = ppd_result.get("turn2", {}).get("avg_ttft_ms", float('inf'))
                    pd_t2_tpot = pd_result.get("turn2", {}).get("avg_tpot_ms", 0)
                    ppd_t2_tpot = ppd_result.get("turn2", {}).get("avg_tpot_ms", 0)

                    # If PD benchmark failed (SR=0 or all-zero metrics),
                    # PD is not viable — always use PPD (local processing)
                    pd_sr = pd_result.get("success_rate", 0)
                    if pd_sr == 0 or (pd_t2_ttft == 0 and pd_t2_tpot == 0):
                        self.lookup_table[(ctx_class, workload_type, qps)] = True
                        self.performance_data[(ctx_class, workload_type, qps)] = {
                            "pd_ttft": pd_t2_ttft, "ppd_ttft": ppd_t2_ttft,
                            "pd_tpot": pd_t2_tpot, "ppd_tpot": ppd_t2_tpot,
                            "score": float('inf'),
                        }
                        logger.debug(
                            f"  {ctx_class}_{workload_type} @ QPS={qps}: "
                            f"PD benchmark failed (SR={pd_sr}), forcing PPD"
                        )
                        continue

                    # Calculate weighted score
                    score = self._calculate_score(pd_t2_ttft, ppd_t2_ttft, pd_t2_tpot, ppd_t2_tpot)

                    # Store performance data for analysis
                    self.performance_data[(ctx_class, workload_type, qps)] = {
                        "pd_ttft": pd_t2_ttft,
                        "ppd_ttft": ppd_t2_ttft,
                        "pd_tpot": pd_t2_tpot,
                        "ppd_tpot": ppd_t2_tpot,
                        "score": score,
                    }

                    # Decision: use PPD if score > 0
                    use_ppd = score > 0
                    self.lookup_table[(ctx_class, workload_type, qps)] = use_ppd

                    ttft_imp = (pd_t2_ttft - ppd_t2_ttft) / pd_t2_ttft * 100 if pd_t2_ttft > 0 else 0
                    tpot_deg = (ppd_t2_tpot - pd_t2_tpot) / pd_t2_tpot * 100 if pd_t2_tpot > 0 else 0
                    logger.debug(
                        f"  {ctx_class}_{workload_type} @ QPS={qps}: "
                        f"TTFT={ttft_imp:+.1f}%, TPOT={tpot_deg:+.1f}%, "
                        f"score={score:.3f}, use_ppd={use_ppd}"
                    )

        # Log summary
        ppd_wins = sum(1 for v in self.lookup_table.values() if v)
        total = len(self.lookup_table)
        loaded = len(self.performance_data)
        logger.info(
            "Lookup table built: loaded matched calibration pairs=%d/%d; "
            "PPD decisions=%d/%d (%.1f%%, including defaults)",
            loaded, total, ppd_wins, total, ppd_wins / total * 100,
        )

    def _extrapolate_for_huge_context(
        self,
        workload_type: str,
        qps: float,
    ) -> bool:
        """Extrapolate decision for huge context based on large context trend.

        For huge context (>4096 tokens), we don't have benchmark data.
        We extrapolate based on large context data trends.

        Args:
            workload_type: Workload type classification
            qps: QPS value

        Returns:
            Extrapolated decision (True = use PPD)
        """
        # Try to use large context data for extrapolation
        large_key = ("large", workload_type, qps)

        if large_key in self.performance_data:
            # Use large context's decision as baseline for huge
            # Huge context tends to favor PPD even more (larger KV cache = more transfer savings)
            # But also more sensitive to memory pressure, so keep the same decision
            data = self.performance_data[large_key]
            return data["score"] > 0

        # Fallback to default
        return self.default_use_ppd

    def decide_with_metadata(
        self,
        turn: int,
        input_tokens: int,
        output_tokens: int,
        current_qps: float,
        context_tokens: int = 0,
    ) -> Tuple[bool, dict]:
        """Return the original decision plus an auditable explanation.

        This method deliberately preserves ``should_use_ppd`` behavior.  The
        metadata only exposes which bypass/default/calibration path supplied
        the answer and which offline cell was selected.
        """
        ctx_class = classify_context_length(context_tokens)
        workload_type = classify_workload(input_tokens, output_tokens)
        nearest_qps = find_nearest_qps(current_qps)
        metadata = {
            "decision_source": None,
            "context_class": ctx_class,
            "workload_type": workload_type,
            "requested_qps": current_qps,
            "selected_qps": nearest_qps,
            "selected_calibration_cell": None,
            "calibration_pair_available": False,
            "bypass_threshold": int(os.environ.get("PPD_BYPASS_THRESHOLD", "512")),
            "base_config": self.base_config,
            "ppd_config": self.ppd_config,
        }

        if turn == 1:
            self.stats.record("turn1", False, is_turn1=True)
            metadata["decision_source"] = "turn1_forced_pd"
            return False, metadata

        bypass_threshold = metadata["bypass_threshold"]
        if input_tokens < bypass_threshold:
            self.stats.record("short_input_bypass", True)
            metadata["decision_source"] = "short_append_bypass"
            logger.debug(
                "PPD decision: turn=%s, input_tokens=%s < %s, bypass to PPD",
                turn,
                input_tokens,
                bypass_threshold,
            )
            return True, metadata

        selected_context = ctx_class
        key = (ctx_class, workload_type, nearest_qps)
        performance = None
        if key in self.lookup_table:
            use_ppd = self.lookup_table[key]
            performance = self.performance_data.get(key)
            if performance is None:
                source = "default_missing_calibration_pair"
            elif performance.get("score") == float("inf"):
                source = "pd_calibration_failed_forced_ppd"
            else:
                source = "matched_calibration_score"
        elif ctx_class == "huge":
            selected_context = "large"
            selected_key = (selected_context, workload_type, nearest_qps)
            performance = self.performance_data.get(selected_key)
            use_ppd = self._extrapolate_for_huge_context(workload_type, nearest_qps)
            source = (
                "huge_context_extrapolated_from_large"
                if performance is not None
                else "huge_context_default_missing_calibration"
            )
            logger.debug("Extrapolated decision for huge context: %s", use_ppd)
        else:
            use_ppd = self.default_use_ppd
            source = "default_missing_lookup_key"

        metadata["decision_source"] = source
        metadata["selected_calibration_cell"] = {
            "context_class": selected_context,
            "workload_type": workload_type,
            "qps": nearest_qps,
        }
        metadata["calibration_pair_available"] = performance is not None
        if performance is not None:
            metadata["calibration_metrics"] = {
                key: (None if value == float("inf") else value)
                for key, value in performance.items()
            }

        self.stats.record(workload_type, use_ppd)
        logger.debug(
            "PPD decision: turn=%s, ctx=%s, workload=%s, qps=%s->%s, "
            "source=%s, decision=%s",
            turn,
            ctx_class,
            workload_type,
            current_qps,
            nearest_qps,
            source,
            "PPD" if use_ppd else "PD",
        )
        return use_ppd, metadata

    def should_use_ppd(
        self,
        turn: int,
        input_tokens: int,
        output_tokens: int,
        current_qps: float,
        context_tokens: int = 0,
    ) -> bool:
        """Backward-compatible boolean API for benchmark and legacy callers."""
        decision, _ = self.decide_with_metadata(
            turn=turn,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            current_qps=current_qps,
            context_tokens=context_tokens,
        )
        return decision

    def get_decision_stats(self) -> dict:
        """Get statistics about decisions made."""
        return self.stats.to_dict()

    def get_performance_comparison(self) -> Dict[str, dict]:
        """Get performance comparison between PD and PPD.

        Returns a summary of how PPD compares to PD across different workloads,
        including both TTFT and TPOT metrics.
        """
        results = {}

        for (ctx_class, workload_type, qps), data in self.performance_data.items():
            key = f"{ctx_class}_{workload_type}"
            if key not in results:
                results[key] = {
                    "pd_ttft": [],
                    "ppd_ttft": [],
                    "pd_tpot": [],
                    "ppd_tpot": [],
                    "scores": [],
                    "ttft_improvement_pct": [],
                    "tpot_degradation_pct": [],
                    "ppd_wins": 0,
                    "total": 0,
                }

            pd_ttft = data["pd_ttft"]
            ppd_ttft = data["ppd_ttft"]
            pd_tpot = data["pd_tpot"]
            ppd_tpot = data["ppd_tpot"]
            score = data["score"]

            results[key]["pd_ttft"].append(pd_ttft)
            results[key]["ppd_ttft"].append(ppd_ttft)
            results[key]["pd_tpot"].append(pd_tpot)
            results[key]["ppd_tpot"].append(ppd_tpot)
            results[key]["scores"].append(score)

            ttft_imp = (pd_ttft - ppd_ttft) / pd_ttft * 100 if pd_ttft > 0 else 0
            tpot_deg = (ppd_tpot - pd_tpot) / pd_tpot * 100 if pd_tpot > 0 else 0
            results[key]["ttft_improvement_pct"].append(ttft_imp)
            results[key]["tpot_degradation_pct"].append(tpot_deg)

            results[key]["total"] += 1
            if score > 0:
                results[key]["ppd_wins"] += 1

        # Compute averages
        summary = {}
        for key, data in results.items():
            summary[key] = {
                "pd_avg_ttft_ms": sum(data["pd_ttft"]) / len(data["pd_ttft"]),
                "ppd_avg_ttft_ms": sum(data["ppd_ttft"]) / len(data["ppd_ttft"]),
                "pd_avg_tpot_ms": sum(data["pd_tpot"]) / len(data["pd_tpot"]),
                "ppd_avg_tpot_ms": sum(data["ppd_tpot"]) / len(data["ppd_tpot"]),
                "avg_ttft_improvement_pct": sum(data["ttft_improvement_pct"]) / len(data["ttft_improvement_pct"]),
                "avg_tpot_degradation_pct": sum(data["tpot_degradation_pct"]) / len(data["tpot_degradation_pct"]),
                "avg_score": sum(data["scores"]) / len(data["scores"]),
                "ppd_win_rate": data["ppd_wins"] / data["total"],
            }

        return summary

    def export_lookup_table(self, output_path: str):
        """Export lookup table to JSON for inspection."""
        export_data = {
            "base_config": self.base_config,
            "ppd_config": self.ppd_config,
            "weights": {"w_ttft": self.w_ttft, "w_tpot": self.w_tpot},
            "entries": [
                {
                    "context_class": k[0],
                    "workload_type": k[1],
                    "qps": k[2],
                    "use_ppd": v,
                    "pd_ttft_ms": self.performance_data.get(k, {}).get("pd_ttft"),
                    "ppd_ttft_ms": self.performance_data.get(k, {}).get("ppd_ttft"),
                    "pd_tpot_ms": self.performance_data.get(k, {}).get("pd_tpot"),
                    "ppd_tpot_ms": self.performance_data.get(k, {}).get("ppd_tpot"),
                    "score": self.performance_data.get(k, {}).get("score"),
                }
                for k, v in sorted(self.lookup_table.items())
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Lookup table exported to {output_path}")


def main():
    """Test the PPD decision engine."""
    import sys

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Default benchmark path
    benchmark_path = Path(__file__).parent.parent / "results" / "comprehensive"

    if not benchmark_path.exists():
        print(f"Benchmark path not found: {benchmark_path}")
        sys.exit(1)

    # Create engine
    engine = PPDDecisionEngine(
        benchmark_data_path=str(benchmark_path),
        base_config="2P_2D",
    )

    # Test some decisions
    print("\n" + "="*60)
    print("Testing PPD decisions:")
    print("="*60)

    test_cases = [
        # (turn, input_tokens, output_tokens, qps, context_tokens)
        (1, 128, 128, 4.0, 0),        # Turn 1 - always PD
        (2, 16, 32, 4.0, 256),        # Turn 2, tiny workload, small context
        (2, 32, 256, 4.0, 256),       # Turn 2, short_gen workload
        (2, 512, 64, 4.0, 256),       # Turn 2, big_paste workload
        (2, 1024, 32, 8.0, 2048),     # Turn 2, huge_paste, large context
        (3, 64, 64, 2.0, 512),        # Turn 3, small_bal
    ]

    for turn, inp, out, qps, ctx in test_cases:
        decision = engine.should_use_ppd(turn, inp, out, qps, ctx)
        workload = classify_workload(inp, out)
        ctx_class = classify_context_length(ctx)
        print(f"Turn {turn}: {inp}→{out} tokens, QPS={qps}, ctx={ctx} ({ctx_class})")
        print(f"  Workload: {workload}, Decision: {'PPD' if decision else 'PD'}")

    # Show performance comparison
    print("\n" + "="*60)
    print("Performance Comparison (PD vs PPD):")
    print("="*60)

    comparison = engine.get_performance_comparison()
    for workload, data in sorted(comparison.items()):
        print(f"\n{workload}:")
        print(f"  PD:  TTFT={data['pd_avg_ttft_ms']:.1f}ms, TPOT={data['pd_avg_tpot_ms']:.2f}ms")
        print(f"  PPD: TTFT={data['ppd_avg_ttft_ms']:.1f}ms, TPOT={data['ppd_avg_tpot_ms']:.2f}ms")
        print(f"  TTFT improvement: {data['avg_ttft_improvement_pct']:+.1f}%")
        print(f"  TPOT degradation: {data['avg_tpot_degradation_pct']:+.1f}%")
        print(f"  Avg score: {data['avg_score']:.3f}, PPD win rate: {data['ppd_win_rate']*100:.0f}%")

    # Export lookup table
    export_path = benchmark_path.parent / "ppd_lookup_table.json"
    engine.export_lookup_table(str(export_path))

    print("\n" + "="*60)
    print(f"Decision stats: {engine.get_decision_stats()}")


if __name__ == "__main__":
    main()
