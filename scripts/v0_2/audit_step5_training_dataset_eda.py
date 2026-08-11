#!/usr/bin/env python3
"""Full-artifact EDA and promotion audit for Step 5 v2 products.

This audit is intentionally independent of model training.  It scans every NPZ
artifact, verifies manifest hashes/targets/schema, quantifies the repaired
factorial design, and reports finite-shot diagnostic noise relative to the exact
audit-only correlation deltas.  Shot-noise signal ratios are descriptive: the
audit never promotes exact simulator quantities into model inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "triqto.v0_2.step5_training_dataset_eda.v1"
PRODUCT_SCHEMA = "triqto.v0_2.step5_matched_diagnostic_training_dataset.v2"
DEFAULT_POINTER = Path(
    "/workspace/triqto-data/step5_matched_diagnostic_training_v2/current_product.json"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/step5_matched_diagnostic_training_v2_eda"
)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
)
MECHANISM_CODES = {-1: "clean_control", 0: "rz_drift", 1: "rx_overrotation", 2: "ry_overrotation"}
PHENOTYPE_CODES = {
    0: "phase_dominant",
    1: "mixed",
    2: "population_dominant",
    3: "negligible",
    4: "clean_control",
}
FORBIDDEN_X_TOKENS = (
    "mechanism",
    "phenomenology",
    "effect_present",
    "clean_control",
    "population_component",
    "phase_component",
    "overlap",
    "affected_qubit",
    "insertion",
    "strength",
    "statevector",
    "reference_window_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def resolve_product(args: argparse.Namespace) -> Path:
    if args.product_dir is not None:
        root = args.product_dir.expanduser().resolve()
    else:
        pointer = read_json(args.product_pointer.expanduser().resolve())
        root = Path(str(pointer["product_dir"])).expanduser().resolve()
        completion = root / "dataset_complete.json"
        if sha256_file(completion) != str(pointer["dataset_complete_sha256"]):
            raise RuntimeError("current product pointer completion hash mismatch")
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def cramers_v(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    left_values = sorted({str(row[left]) for row in rows})
    right_values = sorted({str(row[right]) for row in rows})
    if len(left_values) < 2 or len(right_values) < 2:
        return 0.0
    li = {value: index for index, value in enumerate(left_values)}
    ri = {value: index for index, value in enumerate(right_values)}
    table = np.zeros((len(left_values), len(right_values)), dtype=np.float64)
    for row in rows:
        table[li[str(row[left])], ri[str(row[right])]] += 1.0
    total = float(np.sum(table))
    row_sums = np.sum(table, axis=1, keepdims=True)
    col_sums = np.sum(table, axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    valid = expected > 0.0
    chi2 = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt((chi2 / total) / denominator)) if denominator > 0 else 0.0


def rms(value: np.ndarray) -> float:
    numeric = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(numeric * numeric))) if numeric.size else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def total_variation_share(
    rows: Sequence[Mapping[str, Any]], category: str, split_key: str = "split"
) -> float:
    splits = sorted({str(row[split_key]) for row in rows})
    if splits != ["train", "validation"]:
        return float("nan")
    categories = sorted({str(row[category]) for row in rows})
    distributions: dict[str, dict[str, float]] = {}
    for split in splits:
        subset = [row for row in rows if str(row[split_key]) == split]
        denominator = float(len(subset))
        distributions[split] = {
            value: sum(str(row[category]) == value for row in subset) / denominator
            for value in categories
        }
    return 0.5 * sum(
        abs(distributions["train"][value] - distributions["validation"][value])
        for value in categories
    )


def summarize_cross(
    rows: Sequence[Mapping[str, Any]], first: str, second: str, count_name: str
) -> list[dict[str, Any]]:
    first_values = sorted({str(row[first]) for row in rows})
    second_values = sorted({str(row[second]) for row in rows})
    output: list[dict[str, Any]] = []
    for first_value in first_values:
        for second_value in second_values:
            count = sum(
                str(row[first]) == first_value and str(row[second]) == second_value
                for row in rows
            )
            output.append(
                {first: first_value, second: second_value, count_name: count}
            )
    return output


def main() -> None:
    args = parse_args()
    product = resolve_product(args)
    config = read_json(args.config.expanduser().resolve())
    completion = read_json(product / "dataset_complete.json")
    validation = read_json(product / "stage_validation.json")
    if completion.get("schema") != PRODUCT_SCHEMA:
        raise RuntimeError("EDA requires a Step 5 v2 product")
    if completion.get("status") != "COMPLETE" or validation.get("status") != "PASS":
        raise RuntimeError("Step 5 v2 product is not complete/pass")

    manifests = product / "manifests"
    for name, expected in completion["manifest_hashes"].items():
        observed = sha256_file(manifests / name)
        if observed != str(expected):
            raise RuntimeError(f"manifest hash mismatch for {name}")
    expected_stage_hash = completion.get("stage_validation_sha256")
    if expected_stage_hash and sha256_file(product / "stage_validation.json") != expected_stage_hash:
        raise RuntimeError("stage validation hash mismatch")

    roots = read_csv(manifests / "clean_circuit_manifest.csv")
    examples = read_csv(manifests / "example_manifest.csv")
    if len(roots) != int(completion["clean_circuit_root_count"]):
        raise RuntimeError("clean root manifest count mismatch")
    if len(examples) != int(completion["example_count"]):
        raise RuntimeError("example manifest count mismatch")

    identity = {
        "product_id": completion["product_id"],
        "dataset_complete_sha256": sha256_file(product / "dataset_complete.json"),
        "config_sha256": sha256_file(args.config.expanduser().resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    audit_id = "audit_" + hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output_root = output_parent / audit_id
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite EDA audit {output_root}")
    staging = output_parent / f".{audit_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    manifest_by_path = {str(row["artifact_path"]): row for row in examples}
    if len(manifest_by_path) != len(examples):
        raise RuntimeError("duplicate artifact path in manifest")

    artifact_issues: list[dict[str, Any]] = []
    schema_counts: Counter[tuple[str, ...]] = Counter()
    numeric_rows: list[dict[str, Any]] = []
    max_abs_empirical = {"local": 0.0, "pairwise": 0.0, "parity": 0.0}
    max_abs_exact = {"local": 0.0, "pairwise": 0.0, "parity": 0.0}

    for index, row in enumerate(examples, start=1):
        rel = str(row["artifact_path"])
        path = product / rel
        if not path.is_file():
            artifact_issues.append({"artifact_path": rel, "issue": "missing_file"})
            continue
        observed_sha = sha256_file(path)
        if observed_sha != str(row["artifact_sha256"]):
            artifact_issues.append({"artifact_path": rel, "issue": "sha_mismatch"})
            continue
        try:
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        except Exception as exc:
            artifact_issues.append(
                {"artifact_path": rel, "issue": "npz_load_failure", "detail": str(exc)}
            )
            continue

        schema_counts[tuple(arrays)] += 1
        x_keys = [key for key in arrays if key.startswith("x__")]
        leaking = [
            key
            for key in x_keys
            if any(token in key.lower() for token in FORBIDDEN_X_TOKENS)
        ]
        if leaking:
            artifact_issues.append(
                {
                    "artifact_path": rel,
                    "issue": "privileged_x_key",
                    "detail": "|".join(leaking),
                }
            )

        try:
            example_id = str(arrays["meta__example_id"][0])
            group_id = str(arrays["meta__clean_circuit_group_id"][0])
            clean_control = bool(arrays["y__clean_control_target"][0])
            effect_present = bool(arrays["y__effect_present_target"][0])
            mechanism_loss_mask = bool(arrays["y__mechanism_loss_mask"][0])
            mechanism = MECHANISM_CODES[int(arrays["y__mechanism_target"][0])]
            phenotype = PHENOTYPE_CODES[int(arrays["y__phenomenology_target"][0])]
            affected_qubit = int(arrays["audit__affected_qubit"][0])
            boundary = int(arrays["audit__insertion_boundary_rank"][0])
            strength = float(arrays["audit__strength"][0])
            if example_id != str(row["example_id"]):
                raise ValueError("example_id")
            if group_id != str(row["clean_circuit_group_id"]):
                raise ValueError("clean_group_id")
            if clean_control != as_bool(row["clean_control"]):
                raise ValueError("clean_control")
            if effect_present != as_bool(row["effect_present"]):
                raise ValueError("effect_present")
            if mechanism_loss_mask != as_bool(row["mechanism_loss_mask"]):
                raise ValueError("mechanism_loss_mask")
            if mechanism != str(row["mechanism"]):
                raise ValueError("mechanism")
            if phenotype != str(row["phenomenology"]):
                raise ValueError("phenomenology")
            if affected_qubit != int(row["affected_qubit"]):
                raise ValueError("affected_qubit")
            if boundary != int(row["insertion_boundary_rank"]):
                raise ValueError("insertion_boundary_rank")
            if not math.isclose(strength, float(row["strength"]), abs_tol=1e-12):
                raise ValueError("strength")
        except Exception as exc:
            artifact_issues.append(
                {
                    "artifact_path": rel,
                    "issue": "manifest_target_mismatch",
                    "detail": str(exc),
                }
            )
            continue

        local = np.asarray(arrays["x__delta_local_expectations"], dtype=np.float64)
        pairwise = np.asarray(arrays["x__delta_pairwise_correlations"], dtype=np.float64)
        parity = np.asarray(arrays["x__delta_global_parity"], dtype=np.float64)
        exact_local = np.asarray(
            arrays["audit__exact_delta_local_expectations"], dtype=np.float64
        )
        exact_pairwise = np.asarray(
            arrays["audit__exact_delta_pairwise_correlations"], dtype=np.float64
        )
        exact_parity = np.asarray(
            arrays["audit__exact_delta_global_parity"], dtype=np.float64
        )
        for name, value, exact in (
            ("local", local, exact_local),
            ("pairwise", pairwise, exact_pairwise),
            ("parity", parity, exact_parity),
        ):
            if not np.all(np.isfinite(value)) or not np.all(np.isfinite(exact)):
                artifact_issues.append(
                    {"artifact_path": rel, "issue": f"nonfinite_{name}"}
                )
            if value.size:
                max_abs_empirical[name] = max(
                    max_abs_empirical[name], float(np.max(np.abs(value)))
                )
            if exact.size:
                max_abs_exact[name] = max(
                    max_abs_exact[name], float(np.max(np.abs(exact)))
                )

        n_qubits = int(row["n_qubits"])
        pair_count = n_qubits * (n_qubits - 1) // 2
        expected_shapes = {
            "local": (3, n_qubits),
            "pairwise": (3, pair_count),
            "parity": (3,),
        }
        if local.shape != expected_shapes["local"]:
            artifact_issues.append({"artifact_path": rel, "issue": "local_shape"})
        if pairwise.shape != expected_shapes["pairwise"]:
            artifact_issues.append({"artifact_path": rel, "issue": "pairwise_shape"})
        if parity.shape != expected_shapes["parity"]:
            artifact_issues.append({"artifact_path": rel, "issue": "parity_shape"})
        if tuple(arrays["x__pair_indices"].shape) != (pair_count, 2):
            artifact_issues.append({"artifact_path": rel, "issue": "pair_index_shape"})
        shots = int(row["shots"])
        if not np.all(arrays["x__observed_shots"] == shots):
            artifact_issues.append({"artifact_path": rel, "issue": "observed_shots"})
        if not np.all(arrays["x__reference_shots"] == shots):
            artifact_issues.append({"artifact_path": rel, "issue": "reference_shots"})
        if not np.all(arrays["x__reference_available_mask"]):
            artifact_issues.append({"artifact_path": rel, "issue": "reference_mask"})
        if not np.array_equal(
            arrays["x__diagnostic_basis_codes"], np.asarray([0, 1, 2], dtype=np.int8)
        ):
            artifact_issues.append({"artifact_path": rel, "issue": "basis_codes"})
        if not np.array_equal(
            arrays["x__layout_logical_to_physical"], np.arange(n_qubits, dtype=np.int16)
        ):
            artifact_issues.append({"artifact_path": rel, "issue": "layout"})

        total_size = local.size + pairwise.size + parity.size
        exact_total_sq = (
            float(np.sum(exact_local * exact_local))
            + float(np.sum(exact_pairwise * exact_pairwise))
            + float(np.sum(exact_parity * exact_parity))
        )
        error_total_sq = (
            float(np.sum((local - exact_local) ** 2))
            + float(np.sum((pairwise - exact_pairwise) ** 2))
            + float(np.sum((parity - exact_parity) ** 2))
        )
        empirical_total_sq = (
            float(np.sum(local * local))
            + float(np.sum(pairwise * pairwise))
            + float(np.sum(parity * parity))
        )
        exact_core_rms = math.sqrt(exact_total_sq / total_size)
        error_core_rms = math.sqrt(error_total_sq / total_size)
        empirical_core_rms = math.sqrt(empirical_total_sq / total_size)
        numeric_rows.append(
            {
                "artifact_path": rel,
                "split": str(row["split"]),
                "family": str(row["family"]),
                "n_qubits": n_qubits,
                "mechanism": mechanism,
                "clean_control": clean_control,
                "effect_present": effect_present,
                "phenomenology": phenotype,
                "depth": str(row["insertion_depth_bin"]),
                "strength": float(row["strength"]),
                "shots": shots,
                "empirical_core_rms": empirical_core_rms,
                "exact_core_rms": exact_core_rms,
                "shot_error_core_rms": error_core_rms,
                "noise_gt_exact_signal": bool(error_core_rms > exact_core_rms),
            }
        )
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"EDA scanned {index}/{len(examples)} artifacts", flush=True)

    write_csv(staging / "artifact_issues.csv", artifact_issues)

    root_rows_typed: list[dict[str, Any]] = []
    for row in roots:
        root_rows_typed.append(
            {
                **row,
                "n_qubits": int(row["n_qubits"]),
                "root_index": int(row["root_index"]),
            }
        )
    example_rows_typed: list[dict[str, Any]] = []
    for row in examples:
        example_rows_typed.append(
            {
                **row,
                "clean_control": as_bool(row["clean_control"]),
                "effect_present": as_bool(row["effect_present"]),
                "mechanism_loss_mask": as_bool(row["mechanism_loss_mask"]),
                "n_qubits": int(row["n_qubits"]),
                "strength": float(row["strength"]),
                "shots": int(row["shots"]),
            }
        )

    family_split_v = cramers_v(root_rows_typed, "family", "split")
    injected = [row for row in example_rows_typed if not row["clean_control"]]
    depth_strength_v = cramers_v(injected, "insertion_depth_bin", "strength")
    family_tv = total_variation_share(root_rows_typed, "family")
    phenotype_tv = total_variation_share(injected, "phenomenology")

    family_split_rows = summarize_cross(
        root_rows_typed, "family", "split", "clean_root_count"
    )
    qubit_split_rows = summarize_cross(
        root_rows_typed, "n_qubits", "split", "clean_root_count"
    )
    depth_strength_rows = summarize_cross(
        injected, "insertion_depth_bin", "strength", "example_count"
    )
    write_csv(staging / "family_split.csv", family_split_rows)
    write_csv(staging / "qubit_split.csv", qubit_split_rows)
    write_csv(staging / "depth_strength.csv", depth_strength_rows)

    mechanism_rows: list[dict[str, Any]] = []
    for mechanism in sorted({str(row["mechanism"]) for row in example_rows_typed}):
        subset = [row for row in example_rows_typed if str(row["mechanism"]) == mechanism]
        injected_subset = [row for row in subset if not row["clean_control"]]
        mechanism_rows.append(
            {
                "mechanism": mechanism,
                "example_count": len(subset),
                "effectful_count": sum(row["effect_present"] for row in subset),
                "negligible_injected_count": sum(
                    (not row["clean_control"]) and (not row["effect_present"])
                    for row in subset
                ),
                "effectful_fraction_among_injected": (
                    sum(row["effect_present"] for row in injected_subset)
                    / len(injected_subset)
                    if injected_subset
                    else 0.0
                ),
            }
        )
    write_csv(staging / "mechanism_effect_summary.csv", mechanism_rows)

    phenotype_rows: list[dict[str, Any]] = []
    for mechanism in sorted({str(row["mechanism"]) for row in injected}):
        subset = [row for row in injected if str(row["mechanism"]) == mechanism]
        for phenotype in sorted({str(row["phenomenology"]) for row in injected}):
            count = sum(str(row["phenomenology"]) == phenotype for row in subset)
            phenotype_rows.append(
                {
                    "mechanism": mechanism,
                    "phenomenology": phenotype,
                    "example_count": count,
                    "fraction_within_mechanism": count / len(subset),
                }
            )
    write_csv(staging / "phenomenology_summary.csv", phenotype_rows)

    shot_noise_rows: list[dict[str, Any]] = []
    for shots in sorted({int(row["shots"]) for row in numeric_rows}):
        subset = [row for row in numeric_rows if int(row["shots"]) == shots]
        error_values = [float(row["shot_error_core_rms"]) for row in subset]
        empirical_values = [float(row["empirical_core_rms"]) for row in subset]
        exact_values = [float(row["exact_core_rms"]) for row in subset]
        shot_noise_rows.append(
            {
                "shots": shots,
                "example_count": len(subset),
                "median_empirical_core_rms": percentile(empirical_values, 0.5),
                "median_exact_core_rms": percentile(exact_values, 0.5),
                "median_shot_error_core_rms": percentile(error_values, 0.5),
                "p95_shot_error_core_rms": percentile(error_values, 0.95),
                "noise_gt_exact_signal_fraction": sum(
                    bool(row["noise_gt_exact_signal"]) for row in subset
                )
                / len(subset),
            }
        )
    write_csv(staging / "shot_noise_summary.csv", shot_noise_rows)

    strength_shot_rows: list[dict[str, Any]] = []
    effectful_numeric = [
        row
        for row in numeric_rows
        if bool(row["effect_present"]) and not bool(row["clean_control"])
    ]
    for strength in sorted({float(row["strength"]) for row in effectful_numeric}):
        for shots in sorted({int(row["shots"]) for row in effectful_numeric}):
            subset = [
                row
                for row in effectful_numeric
                if math.isclose(float(row["strength"]), strength, abs_tol=1e-12)
                and int(row["shots"]) == shots
            ]
            if not subset:
                continue
            strength_shot_rows.append(
                {
                    "strength": strength,
                    "shots": shots,
                    "effectful_example_count": len(subset),
                    "median_exact_core_rms": percentile(
                        [float(row["exact_core_rms"]) for row in subset], 0.5
                    ),
                    "median_shot_error_core_rms": percentile(
                        [float(row["shot_error_core_rms"]) for row in subset], 0.5
                    ),
                    "noise_gt_exact_signal_fraction": sum(
                        bool(row["noise_gt_exact_signal"]) for row in subset
                    )
                    / len(subset),
                }
            )
    write_csv(staging / "strength_shot_noise_summary.csv", strength_shot_rows)

    schema_count = len(schema_counts)
    all_hash_schema_target_valid = len(artifact_issues) == 0 and schema_count == 1
    family_threshold = float(config["stage_validation"]["maximum_family_split_cramers_v"])
    depth_threshold = float(config["stage_validation"]["maximum_depth_strength_cramers_v"])
    family_cells_ok = all(
        int(row["clean_root_count"]) > 0 for row in family_split_rows
    )
    threeq_cells = [
        row for row in qubit_split_rows if str(row["n_qubits"]) == "3"
    ]
    threeq_both_ok = len(threeq_cells) == 2 and all(
        int(row["clean_root_count"]) > 0 for row in threeq_cells
    )
    depth_strength_cells_ok = all(
        int(row["example_count"]) > 0 for row in depth_strength_rows
    )
    bound = float(config["stage_validation"]["diagnostic_delta_absolute_bound"])
    numeric_bounds_ok = all(value <= bound for value in max_abs_empirical.values())

    gates = {
        "artifact_hash_schema_target_integrity": all_hash_schema_target_valid,
        "family_split_cramers_v": family_split_v <= family_threshold,
        "every_family_in_both_splits": family_cells_ok,
        "three_qubit_in_both_splits": threeq_both_ok,
        "depth_strength_cramers_v": depth_strength_v <= depth_threshold,
        "every_depth_strength_cell_present": depth_strength_cells_ok,
        "diagnostics_finite_and_bounded": numeric_bounds_ok,
        "stage_validation_passed": validation.get("status") == "PASS",
    }
    decision = "PROMOTION_READY" if all(gates.values()) else "BLOCKED"

    complete = {
        "schema": SCHEMA,
        "status": "AUDIT_COMPLETE",
        "decision": decision,
        "audit_id": audit_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_product": str(product),
        "identity": identity,
        "artifact_count": len(examples),
        "artifact_issue_count": len(artifact_issues),
        "distinct_array_schema_count": schema_count,
        "family_split_cramers_v": family_split_v,
        "family_split_total_variation": family_tv,
        "depth_strength_cramers_v": depth_strength_v,
        "phenomenology_train_validation_total_variation": phenotype_tv,
        "max_abs_empirical_diagnostics": max_abs_empirical,
        "max_abs_exact_audit_diagnostics": max_abs_exact,
        "three_qubit_train_root_count": sum(
            int(row["n_qubits"]) == 3 and str(row["split"]) == "train"
            for row in root_rows_typed
        ),
        "three_qubit_validation_root_count": sum(
            int(row["n_qubits"]) == 3 and str(row["split"]) == "validation"
            for row in root_rows_typed
        ),
        "gates": gates,
        "shot_noise_is_report_only": True,
        "historical_v0_1_test_accessed": False,
        "spent_confirmatory_cohort_accessed": False,
        "classifier_trained": False,
        "scientific_boundary": (
            "Full-artifact Step 5 dataset-quality EDA only. Exact audit deltas are used "
            "to quantify finite-shot noise and never promoted into deployable model inputs."
        ),
    }
    atomic_json(staging / "eda_complete.json", complete)
    os.replace(staging, output_root)

    print("\nTRIQTO STEP 5 FULL-ARTIFACT EDA COMPLETE\n")
    print(f"Decision: {decision}")
    print(f"Artifacts verified: {len(examples) - len(artifact_issues)}/{len(examples)}")
    print(f"Distinct NPZ schemas: {schema_count}")
    print(f"Family/split Cramer's V: {family_split_v:.6f}")
    print(f"Depth/strength Cramer's V: {depth_strength_v:.6f}")
    print(
        "3q train/validation roots: "
        f"{complete['three_qubit_train_root_count']}/"
        f"{complete['three_qubit_validation_root_count']}"
    )
    print(f"Family split TV: {family_tv:.6f}")
    print(f"Phenomenology train/validation TV: {phenotype_tv:.6f}")
    print("Shot-noise SNR: REPORT ONLY; see shot_noise_summary.csv")
    print("Historical v0.1 test accessed: NO")
    print("Spent confirmatory cohort accessed: NO")
    print("Classifier trained: NO")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
