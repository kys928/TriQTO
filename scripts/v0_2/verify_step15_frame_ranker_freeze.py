#!/usr/bin/env python3
"""Verify the immutable Step-15 FIT-only rank-adapter freeze before Step 7."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import benchmark_step6_cheap_baselines as baseline

DEFAULT_PARENT = Path("/workspace/triqto-data/step15_frame_ranking")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-parent", type=Path, default=DEFAULT_PARENT)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    pointer_path = args.output_parent / "current_step15_frame_ranker.json"
    pointer = read_json(pointer_path)
    method_dir = Path(str(pointer["method_dir"])).resolve()
    if not method_dir.is_dir():
        raise RuntimeError(f"Step-15 method directory missing: {method_dir}")

    expected = {
        "frame_rank_adapter.json": str(pointer["adapter_sha256"]),
        "fit_development_result.json": str(pointer["fit_development_result_sha256"]),
        "adapter_freeze.json": str(pointer["adapter_freeze_sha256"]),
        "step15_complete.json": str(pointer["step15_complete_sha256"]),
    }
    actual: dict[str, str] = {}
    for name, want in expected.items():
        path = method_dir / name
        got = baseline.sha256_file(path)
        actual[name] = got
        if got != want:
            raise RuntimeError(f"Step-15 hash mismatch for {name}: {got} != {want}")

    adapter = read_json(method_dir / "frame_rank_adapter.json")
    result = read_json(method_dir / "fit_development_result.json")
    freeze = read_json(method_dir / "adapter_freeze.json")
    complete = read_json(method_dir / "step15_complete.json")

    method_id = str(pointer["method_id"])
    for name, payload in (
        ("adapter", adapter),
        ("result", result),
        ("freeze", freeze),
        ("complete", complete),
    ):
        if str(payload.get("method_id")) != method_id:
            raise RuntimeError(f"Step-15 {name} method_id drift")

    if result.get("status") != "COMPLETE_FIT_ONLY_STEP15_FRAME_RANKING":
        raise RuntimeError("Step-15 development result is not complete")
    if freeze.get("status") != "IMMUTABLY_FROZEN_BEFORE_SPENT_HOLDOUT_EVALUATION":
        raise RuntimeError("Step-15 adapter is not frozen before holdout evaluation")
    if complete.get("status") != "HASH_VERIFIED_READY_FOR_SPENT_HOLDOUT_AUDIT":
        raise RuntimeError("Step-15 completion status is not holdout-ready")
    if not bool(result.get("zero_adapter_regression", {}).get("exact_array_equal")):
        raise RuntimeError("Step-15 zero-adapter exact-recovery regression is not satisfied")
    for payload_name, payload in (("adapter", adapter), ("result", result), ("freeze", freeze), ("complete", complete), ("pointer", pointer)):
        if bool(payload.get("spent_holdout_accessed", False)):
            raise RuntimeError(f"Step-15 {payload_name} indicates premature spent-holdout access")

    frozen_copy = dict(freeze)
    recorded_payload_sha = str(frozen_copy.pop("freeze_payload_sha256"))
    recomputed_payload_sha = canonical_sha(frozen_copy)
    if recorded_payload_sha != recomputed_payload_sha:
        raise RuntimeError(
            f"Step-15 freeze payload hash mismatch: {recomputed_payload_sha} != {recorded_payload_sha}"
        )
    if str(freeze["adapter_sha256"]) != actual["frame_rank_adapter.json"]:
        raise RuntimeError("Step-15 freeze does not bind the adapter bytes")
    if str(freeze["fit_development_result_sha256"]) != actual["fit_development_result.json"]:
        raise RuntimeError("Step-15 freeze does not bind the development-result bytes")
    if str(complete["adapter_freeze_sha256"]) != actual["adapter_freeze.json"]:
        raise RuntimeError("Step-15 complete manifest does not bind the freeze bytes")

    print(json.dumps({
        "status": "VERIFIED_STEP15_FROZEN_BEFORE_SPENT_HOLDOUT",
        "method_id": method_id,
        "method_dir": str(method_dir),
        "file_sha256": actual,
        "freeze_payload_sha256": recorded_payload_sha,
        "zero_adapter_exact_recovery": True,
        "spent_holdout_accessed": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
