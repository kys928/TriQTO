#!/usr/bin/env python3
"""Materialize and spend the frozen Step-14 simulator + legacy outer cohorts once.

This runner is deliberately post-selection only. It verifies the exact frozen
training run/selection hash, materializes the simulator outer and fresh legacy
retention outer cohorts, evaluates the already-frozen Step-14 ensemble against
the frozen Step-10C ensemble, and writes a read-only-friendly current pointer.
It never retrains, changes checkpoints, touches future-hardware reserve, or uses
QPU access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
TRAINING_PARENT = Path("/workspace/triqto-data/step14_cross_motif_training")
CROSS_PARENT = Path("/workspace/triqto-data/step14_cross_motif_dataset")
LEGACY_PARENT = Path("/workspace/triqto-data/step14_fresh_legacy_retention_outer")
EVAL_PARENT = Path("/workspace/triqto-data/step14_outer_evaluation")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify_freeze(expected_run_id: str, expected_sha: str) -> Path:
    pointer = TRAINING_PARENT / "current_training_run.json"
    if not pointer.is_file():
        raise RuntimeError("Step-14 current training pointer is missing")
    current = read_json(pointer)
    if current.get("schema") != "triqto.v0_2.step14_current_training_run.v1":
        raise RuntimeError("unexpected Step-14 current training pointer schema")
    if str(current.get("run_id")) != expected_run_id:
        raise RuntimeError("current Step-14 training run does not match frozen requested run")
    if str(current.get("selection_freeze_sha256")) != expected_sha:
        raise RuntimeError("current Step-14 selection-freeze hash does not match requested hash")
    run_dir = Path(str(current.get("run_dir", ""))).expanduser().resolve()
    root = TRAINING_PARENT.resolve()
    if run_dir.parent != root or run_dir.name != expected_run_id:
        raise RuntimeError("unsafe or unexpected Step-14 training run directory")
    freeze = run_dir / "selection_freeze.json"
    if not freeze.is_file() or sha256_file(freeze) != expected_sha:
        raise RuntimeError("frozen Step-14 selection marker missing or hash mismatch")
    payload = read_json(freeze)
    if payload.get("status") != "FROZEN_BEFORE_ANY_STEP14_OUTER_MATERIALIZATION":
        raise RuntimeError("Step-14 selection marker is not frozen before outer")
    if not bool(payload.get("all_three_seed_checkpoints_frozen")):
        raise RuntimeError("Step-14 selection marker does not freeze all three seeds")
    return freeze


def resolve_product(pointer: Path, expected_schema: str, expected_status: str) -> Path:
    value = read_json(pointer)
    product = Path(str(value.get("product_dir", ""))).expanduser().resolve()
    complete = read_json(product / "dataset_complete.json")
    if complete.get("schema") != expected_schema or complete.get("status") != expected_status:
        raise RuntimeError(f"unexpected frozen product at {product}")
    return product


def resolve_evaluation(selection_sha: str) -> Path:
    matches: list[Path] = []
    for path in EVAL_PARENT.glob("evaluation_*/outer_result.json"):
        try:
            value = read_json(path)
        except Exception:
            continue
        if value.get("status") != "COMPLETE_OUTER_SPENT":
            continue
        if value.get("identity", {}).get("selection_freeze_sha256") == selection_sha:
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one completed outer evaluation for frozen selection, found {len(matches)}")
    return matches[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--progress-every", type=int, default=5000)
    args = p.parse_args()
    if not args.selection_freeze_sha256.startswith("sha256:") or len(args.selection_freeze_sha256) != 71:
        raise ValueError("selection freeze must be a sha256:<64-hex> digest")
    if args.progress_every < 1 or args.progress_every > 100000:
        raise ValueError("progress_every must be between 1 and 100000")

    freeze = verify_freeze(args.training_run_id, args.selection_freeze_sha256)
    print("TRIQTO STEP 14 FROZEN OUTER PIPELINE — NO RETRAINING / NO QPU", flush=True)
    print("Selection freeze:", freeze, flush=True)

    run([
        sys.executable,
        str(ROOT / "scripts/v0_2/generate_step14_cross_motif_dataset.py"),
        "--config", str(CONFIG),
        "--mode", "simulator_outer",
        "--selection-freeze", str(freeze),
        "--progress-every", "25",
    ])
    cross = resolve_product(
        CROSS_PARENT / "current_simulator_outer_product.json",
        "triqto.v0_2.step14_cross_motif_dataset.v1",
        "COMPLETE_FROZEN_SIMULATOR_OUTER",
    )

    run([
        sys.executable,
        str(ROOT / "scripts/v0_2/generate_step14_fresh_legacy_retention_outer.py"),
        "--config", str(CONFIG),
        "--selection-freeze", str(freeze),
        "--progress-every", "50",
        "--eda-progress-every", "2000",
    ])
    legacy = resolve_product(
        LEGACY_PARENT / "current_product.json",
        "triqto.v0_2.step14_fresh_legacy_retention_outer.v1",
        "COMPLETE_FROZEN_OUTER_VALIDATION",
    )

    run([
        sys.executable,
        str(ROOT / "scripts/v0_2/evaluate_step14_outer.py"),
        "--config", str(CONFIG),
        "--selection-freeze", str(freeze),
        "--cross-motif-outer-dir", str(cross),
        "--legacy-retention-outer-dir", str(legacy),
        "--output-parent", str(EVAL_PARENT),
        "--device", "cuda",
        "--progress-every", str(args.progress_every),
    ])

    evaluation = resolve_evaluation(args.selection_freeze_sha256)
    result = evaluation / "outer_result.json"
    complete = evaluation / "evaluation_complete.json"
    result_payload = read_json(result)
    complete_payload = read_json(complete)
    pointer = {
        "schema": "triqto.v0_2.step14_current_outer_evaluation.v1",
        "status": "COMPLETE_OUTER_SPENT",
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "evaluation_id": result_payload["evaluation_id"],
        "evaluation_dir": str(evaluation),
        "outer_result_sha256": sha256_file(result),
        "evaluation_complete_sha256": sha256_file(complete),
        "support_gate_passed": bool(complete_payload["support_gate_passed"]),
        "outer_selects_nothing": True,
        "qpu_executed": False,
        "future_hardware_reserve_accessed": False,
    }
    atomic_json(EVAL_PARENT / "current_evaluation.json", pointer)
    print("CURRENT OUTER POINTER:", EVAL_PARENT / "current_evaluation.json", flush=True)


if __name__ == "__main__":
    main()
