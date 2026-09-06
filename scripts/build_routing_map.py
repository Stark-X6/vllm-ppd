#!/usr/bin/env python3
"""Build a balanced, explicit PPD routing map from ordered sample identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _instance_id(row):
    if not isinstance(row, dict):
        return row
    if row.get("instance_id"):
        return row["instance_id"]
    try:
        return row["extra_info"]["tools_kwargs"]["task"]["metadata"]["instance_id"]
    except (KeyError, TypeError):
        return None


def load_instance_ids(path: Path, limit: int) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            document = json.loads(path.read_text())
            rows = document if isinstance(document, list) else document.get("data", [])
        values = [_instance_id(row) for row in rows]
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("pyarrow is required to read parquet input") from exc
        # Support both the raw SWE-bench schema and uni-agent's preprocessed
        # nested task-config schema.
        table = pq.read_table(path)
        values = [_instance_id(row) for row in table.to_pylist()]
    else:
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    instance_ids = [str(value) for value in values if value]
    if limit > 0:
        instance_ids = instance_ids[:limit]
    if not instance_ids:
        raise SystemExit("input contains no instance IDs")
    if len(instance_ids) != len(set(instance_ids)):
        raise SystemExit("instance IDs must be unique; use rollout indices for repeated samples")
    return instance_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="parquet, JSON/JSONL, or one ID per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--prefill-slots", type=int, default=2)
    parser.add_argument("--decode-slots", type=int, default=2)
    parser.add_argument("--seed", default="agentic-ppd-v1")
    args = parser.parse_args()
    if min(args.rollouts, args.prefill_slots, args.decode_slots) <= 0:
        parser.error("rollouts and slot counts must be positive")

    input_path = Path(args.input).resolve()
    instance_ids = load_instance_ids(input_path, args.limit)
    assignments = {}
    ordinal = 0
    for instance_id in instance_ids:
        for rollout in range(args.rollouts):
            assignments[f"{instance_id}|rollout={rollout}"] = {
                "prefill_slot": ordinal % args.prefill_slots,
                "decode_slot": ordinal % args.decode_slots,
            }
            ordinal += 1
    document = {
        "schema_version": 1,
        "routing_seed": args.seed,
        "strategy": "ordered_round_robin_v1",
        "source": str(input_path),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "limit": args.limit,
        "rollouts": args.rollouts,
        "prefill_slots": args.prefill_slots,
        "decode_slots": args.decode_slots,
        "assignments": assignments,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(assignments)} assignments to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
