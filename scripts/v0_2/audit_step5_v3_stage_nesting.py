#!/usr/bin/env python3
"""Verify that a larger Step 5 v3 cohort exactly contains the accepted prior stage.

The staged data contract counts independent clean-circuit roots and declares the
500 -> 1000 -> 2000 -> 5000 sequence nested. A larger product must therefore
preserve every prior root and every derivative example byte-for-byte (as bound
by artifact SHA-256), rather than silently regenerating a different development
universe under the same root indices.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PRODUCT_SCHEMA = "triqto.v0_2.step5_matched_diagnostic_training_dataset.v3"
EVIDENCE_SCHEMA = "triqto.v0_2.step5_v3_stage_nesting.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-product-dir", type=Path, required=True)
    parser.add_argument("--current-product-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    previous = args.previous_product_dir.expanduser().resolve()
    current = args.current_product_dir.expanduser().resolve()

    previous_complete = read_json(previous / "dataset_complete.json")
    current_complete = read_json(current / "dataset_complete.json")
    for label, value in (("previous", previous_complete), ("current", current_complete)):
        if value.get("schema") != PRODUCT_SCHEMA or value.get("status") != "COMPLETE":
            raise RuntimeError(f"{label} product is not a complete Step 5 v3 product")

    previous_count = int(previous_complete["clean_circuit_root_count"])
    current_count = int(current_complete["clean_circuit_root_count"])
    if current_count <= previous_count:
        raise RuntimeError("current product must contain more clean roots than previous product")

    prev_roots = read_csv(previous / "manifests/clean_circuit_manifest.csv")
    curr_roots = read_csv(current / "manifests/clean_circuit_manifest.csv")
    prev_examples = read_csv(previous / "manifests/example_manifest.csv")
    curr_examples = read_csv(current / "manifests/example_manifest.csv")

    prev_root_by_index = {int(row["root_index"]): row for row in prev_roots}
    curr_root_by_index = {int(row["root_index"]): row for row in curr_roots}
    if len(prev_root_by_index) != previous_count:
        raise RuntimeError("previous clean-root manifest count/identity mismatch")
    if len(curr_root_by_index) != current_count:
        raise RuntimeError("current clean-root manifest count/identity mismatch")

    root_mismatches: list[int] = []
    for root_index in range(previous_count):
        if root_index not in curr_root_by_index:
            root_mismatches.append(root_index)
            continue
        if prev_root_by_index[root_index] != curr_root_by_index[root_index]:
            root_mismatches.append(root_index)

    prev_example_by_id = {str(row["example_id"]): row for row in prev_examples}
    curr_example_by_id = {str(row["example_id"]): row for row in curr_examples}
    if len(prev_example_by_id) != len(prev_examples):
        raise RuntimeError("duplicate example id in previous product")
    if len(curr_example_by_id) != len(curr_examples):
        raise RuntimeError("duplicate example id in current product")

    missing_examples: list[str] = []
    row_mismatch_examples: list[str] = []
    artifact_hash_mismatch_examples: list[str] = []
    for example_id, previous_row in prev_example_by_id.items():
        current_row = curr_example_by_id.get(example_id)
        if current_row is None:
            missing_examples.append(example_id)
            continue
        if previous_row != current_row:
            row_mismatch_examples.append(example_id)
        if str(previous_row["artifact_sha256"]) != str(current_row["artifact_sha256"]):
            artifact_hash_mismatch_examples.append(example_id)

    passed = not (
        root_mismatches
        or missing_examples
        or row_mismatch_examples
        or artifact_hash_mismatch_examples
    )
    decision = "NESTING_VALID" if passed else "BLOCKED"
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "decision": decision,
        "previous_product_id": previous_complete.get("product_id"),
        "current_product_id": current_complete.get("product_id"),
        "previous_clean_root_count": previous_count,
        "current_clean_root_count": current_count,
        "previous_example_count_checked": len(prev_examples),
        "root_row_mismatch_count": len(root_mismatches),
        "missing_previous_example_count": len(missing_examples),
        "example_row_mismatch_count": len(row_mismatch_examples),
        "artifact_hash_mismatch_count": len(artifact_hash_mismatch_examples),
        "root_mismatches_first20": root_mismatches[:20],
        "missing_examples_first20": missing_examples[:20],
        "row_mismatch_examples_first20": row_mismatch_examples[:20],
        "artifact_hash_mismatch_examples_first20": artifact_hash_mismatch_examples[:20],
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("TRIQTO STEP 5 V3 STAGE-NESTING AUDIT")
    print(f"Previous roots: {previous_count}")
    print(f"Current roots: {current_count}")
    print(f"Previous examples checked: {len(prev_examples)}")
    print(f"Root-row mismatches: {len(root_mismatches)}")
    print(f"Missing previous examples: {len(missing_examples)}")
    print(f"Example-row mismatches: {len(row_mismatch_examples)}")
    print(f"Artifact-hash mismatches: {len(artifact_hash_mismatch_examples)}")
    print(f"Decision: {decision}")
    if args.output is not None:
        print(f"Evidence: {args.output.expanduser().resolve()}")

    if not passed:
        print(json.dumps(evidence, indent=2, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
