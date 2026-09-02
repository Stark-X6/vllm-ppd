"""Regression tests for benchmark/loader calibration filename compatibility."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ppd.optimizer.calibration_contract import (
    calibration_result_filename,
    calibration_result_filename_candidates,
    canonical_qps_label,
)
from ppd.optimizer.ppd_decision_engine import PPDDecisionEngine


def _write_result(
    root: Path,
    *,
    config: str,
    filename: str,
    qps: float,
    ttft_ms: float,
    tpot_ms: float,
) -> None:
    path = root / config / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "config": config,
        "workload": "small_tiny",
        "qps": qps,
        "success_rate": 100.0,
        "turn2": {
            "avg_ttft_ms": ttft_ms,
            "avg_tpot_ms": tpot_ms,
        },
    }))


class CalibrationContractTests(unittest.TestCase):
    def test_qps_label_is_stable_across_numeric_types(self) -> None:
        self.assertEqual(canonical_qps_label(1), "1.0")
        self.assertEqual(canonical_qps_label(1.0), "1.0")
        self.assertEqual(canonical_qps_label("1"), "1.0")
        self.assertEqual(canonical_qps_label(0.5), "0.5")

    def test_candidates_prefer_canonical_and_include_legacy_integer(self) -> None:
        names = calibration_result_filename_candidates(
            "2P_2D", "small_tiny", 1
        )
        self.assertEqual(names[0], "2P_2D_small_tiny_1.0.json")
        self.assertIn("2P_2D_small_tiny_1.json", names)

    def _assert_loader_reads(self, *, legacy: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config, ttft, tpot in (
                ("2P_2D", 100.0, 10.0),
                ("2P_2pD", 200.0, 20.0),
            ):
                filename = (
                    f"{config}_small_tiny_1.json"
                    if legacy
                    else calibration_result_filename(
                        config, "small_tiny", 1
                    )
                )
                _write_result(
                    root,
                    config=config,
                    filename=filename,
                    qps=1.0,
                    ttft_ms=ttft,
                    tpot_ms=tpot,
                )

            engine = PPDDecisionEngine(str(root))
            key = ("small", "tiny", 1.0)
            self.assertIn(key, engine.performance_data)
            self.assertFalse(engine.lookup_table[key])

    def test_loader_reads_canonical_float_filename(self) -> None:
        self._assert_loader_reads(legacy=False)

    def test_loader_remains_compatible_with_legacy_integer_filename(self) -> None:
        self._assert_loader_reads(legacy=True)


if __name__ == "__main__":
    unittest.main()
