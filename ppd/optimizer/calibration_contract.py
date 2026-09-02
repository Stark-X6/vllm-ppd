"""Shared filename contract for comprehensive PPD calibration results.

QPS values arrive as both int (the built-in benchmark matrix) and float
(CLI arguments). Relying on Python's default interpolation therefore produced
both names ending in _1.json and _1.0.json for the same logical point. New
files use the canonical decimal form while readers retain legacy aliases.
"""

from __future__ import annotations

import math
from typing import Union

QPSValue = Union[int, float, str]


def canonical_qps_label(qps: QPSValue) -> str:
    """Return one stable filename label, for example 1 and 1.0 become "1.0"."""
    value = float(qps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"QPS must be a finite positive number, got {qps!r}")
    return str(value)


def calibration_result_filename(config: str, workload: str, qps: QPSValue) -> str:
    """Build the canonical JSON filename for one calibration point."""
    return f"{config}_{workload}_{canonical_qps_label(qps)}.json"


def calibration_result_filename_candidates(
    config: str,
    workload: str,
    qps: QPSValue,
) -> list[str]:
    """Return canonical then legacy filename spellings, without duplicates."""
    value = float(qps)
    canonical = calibration_result_filename(config, workload, value)
    candidates = [canonical]

    # Older runs interpolated an integer QPS directly and therefore omitted
    # the .0 suffix. Preserve read compatibility; new writes are canonical.
    compact = f"{value:g}"
    legacy = f"{config}_{workload}_{compact}.json"
    if legacy not in candidates:
        candidates.append(legacy)

    raw = f"{config}_{workload}_{qps}.json"
    if raw not in candidates:
        candidates.append(raw)
    return candidates
