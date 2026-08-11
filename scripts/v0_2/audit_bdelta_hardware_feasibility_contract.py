#!/usr/bin/env python3
"""Step 4: B_delta hardware-feasibility contract audit.

No QPU is used. The audit has two jobs:

1. validate the frozen acquisition/reference/privileged-input contract for B_delta;
2. re-score the completed Step 3.5 matched pairs using only the proposed
   hardware-scalable local Pauli expectation-delta core.

The Step 3.5 full evidence remains immutable. This audit does not train a
classifier, change TriQTO architecture, access the historical v0.1 test, or
reuse the spent confirmatory cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "triqto.v0_2.bdelta_hardware_feasibility_contract_audit_result.v1"
DEFAULT_STEP3_5 = Path(
    "/workspace/triqto-data/step3_5_generalized_bdelta_identifiability/"
    "audit_eec59162a060a951d97b80fe"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/step4_bdelta_hardware_feasibility_contract"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step3-5-audit-dir", type=Path, default=DEFAULT_STEP3_5)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "configs/v0_2/bdelta_hardware_feasibility_contract.json"
        ),
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def expectation_signal(row: Mapping[str, Any]) -> float:
    return float(
        np.mean(
            [
                0.5 * float(row["expectation_rms_x"]),
                0.5 * float(row["expectation_rms_y"]),
                0.5 * float(row["expectation_rms_z"]),
            ]
        )
    )


def expectation_pair_score(row: Mapping[str, Any]) -> float:
    return float(
        np.mean(
            [
                0.5 * float(row["pair_expectation_rms_x"]),
                0.5 * float(row["pair_expectation_rms_y"]),
                0.5 * float(row["pair_expectation_rms_z"]),
            ]
        )
    )


def build_scalable_pair_rows(
    counterfactual_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in counterfactual_rows:
        key = (str(row["matched_context_id"]), str(row["mechanism"]))
        if key in by_key:
            raise RuntimeError(f"duplicate counterfactual key {key}")
        by_key[key] = row

    ablation = config["scalable_core_ablation"]
    epsilon = 1e-12
    raw_min = float(ablation["minimum_raw_separation"])
    relative_min = float(ablation["minimum_relative_separation"])
    collision_max = float(ablation["numerical_collision_score_max"])
    output: list[dict[str, Any]] = []

    for row in pair_rows:
        matched = str(row["matched_context_id"])
        left_mechanism = str(row["left_mechanism"])
        right_mechanism = str(row["right_mechanism"])
        left = by_key[(matched, left_mechanism)]
        right = by_key[(matched, right_mechanism)]
        left_signal = expectation_signal(left)
        right_signal = expectation_signal(right)
        score = expectation_pair_score(row)
        relative = 2.0 * score / (left_signal + right_signal + epsilon)
        left_negligible = str(left["phenotype"]) == "negligible"
        right_negligible = str(right["phenotype"]) == "negligible"
        output.append(
            {
                "clean_context_id": str(row["clean_context_id"]),
                "matched_context_id": matched,
                "pair_type": str(row["pair_type"]),
                "family": str(row["family"]),
                "n_qubits": int(row["n_qubits"]),
                "strength_key": str(row["strength_key"]),
                "affected_qubit": int(row["affected_qubit"]),
                "qubit_position_class": str(row["qubit_position_class"]),
                "insertion_depth_bin": str(row["insertion_depth_bin"]),
                "terminal_insertion": as_bool(row["terminal_insertion"]),
                "left_mechanism": left_mechanism,
                "right_mechanism": right_mechanism,
                "left_phenotype": str(left["phenotype"]),
                "right_phenotype": str(right["phenotype"]),
                "effectful_pair": not left_negligible and not right_negligible,
                "left_expectation_signal_score": left_signal,
                "right_expectation_signal_score": right_signal,
                "expectation_only_pair_separation_score": score,
                "expectation_only_relative_separation": relative,
                "expectation_only_numerical_collision": score <= collision_max,
                "expectation_only_strong_pair": bool(
                    score >= raw_min and relative >= relative_min
                ),
            }
        )
    return output


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": len(rows),
        "strong_pair_fraction": finite_mean(
            float(bool(row["expectation_only_strong_pair"])) for row in rows
        ),
        "numerical_collision_fraction": finite_mean(
            float(bool(row["expectation_only_numerical_collision"])) for row in rows
        ),
        "median_pair_separation_score": (
            float(
                np.median(
                    [float(row["expectation_only_pair_separation_score"]) for row in rows]
                )
            )
            if rows
            else float("nan")
        ),
        "median_relative_separation": (
            float(
                np.median(
                    [float(row["expectation_only_relative_separation"]) for row in rows]
                )
            )
            if rows
            else float("nan")
        ),
    }


def grouped_bootstrap_fraction(
    rows: Sequence[Mapping[str, Any]], *, repeats: int, seed: int
) -> tuple[float, float]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["clean_context_id"])].append(row)
    keys = sorted(groups)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample: list[Mapping[str, Any]] = []
        for key in chosen:
            sample.extend(groups[str(key)])
        values[index] = np.mean(
            [float(bool(row["expectation_only_strong_pair"])) for row in sample]
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def stratify_effectful(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    min_pairs = int(config["scalable_core_ablation"]["minimum_eligible_stratum_pairs"])
    dimensions = (
        "pair_type",
        "family",
        "n_qubits",
        "strength_key",
        "affected_qubit",
        "qubit_position_class",
        "insertion_depth_bin",
        "terminal_insertion",
    )
    output: list[dict[str, Any]] = []
    for dimension in dimensions:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[dimension])].append(row)
        for value in sorted(grouped):
            subset = grouped[value]
            stats = summarize(subset)
            output.append(
                {
                    "stratum_type": dimension,
                    "stratum_value": value,
                    "eligible": len(subset) >= min_pairs,
                    **stats,
                }
            )
    return output


def validate_hardware_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    references = config["reference_contract"]
    primary_name = str(references["primary_step5_reference_kind"])
    primary_reference = references["reference_kinds"][primary_name]
    primary_reference_valid = primary_reference["deployability"] == "HARDWARE_VALID"

    primary_features = [
        row
        for row in config["feature_contract"]
        if row["step5_role"] == "PRIMARY_HARDWARE_SCALABLE_CORE"
    ]
    allowed = {"FINITE_SHOT_SAMPLER_DERIVED", "FINITE_SHOT_ESTIMATOR_DERIVED"}
    invalid_primary_features = [
        str(row["name"])
        for row in primary_features
        if str(row["hardware_obtainability"]) not in allowed
    ]
    simulator_inputs = [
        str(row["name"])
        for row in config["feature_contract"]
        if "PRIVILEGED_SIMULATOR" in str(row["hardware_obtainability"])
        and str(row["step5_role"]) not in {"SUPERVISION_ONLY", "TARGET_AUDIT_ONLY", "SUPERVISION_AND_MECHANISM_LOSS_MASK"}
    ]
    return {
        "primary_reference_kind": primary_name,
        "primary_reference_hardware_valid": primary_reference_valid,
        "primary_core_feature_count": len(primary_features),
        "invalid_primary_core_features": invalid_primary_features,
        "privileged_features_leaking_into_deployable_inputs": simulator_inputs,
        "hardware_contract_valid": bool(
            primary_reference_valid
            and primary_features
            and not invalid_primary_features
            and not simulator_inputs
            and not bool(config["hardware_measurement_contract"]["exact_probabilities_available_on_hardware"])
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    if config.get("schema") != "triqto.v0_2.bdelta_hardware_feasibility_contract.v1":
        raise RuntimeError("unexpected Step 4 config schema")

    source = args.step3_5_audit_dir.expanduser().resolve()
    required = {
        "audit_complete.json",
        "decision.json",
        "counterfactual_metrics.csv",
        "pairwise_metrics.csv",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"Step 3.5 audit missing required files: {missing}")
    complete = read_json(source / "audit_complete.json")
    source_cfg = config["source_step3_5"]
    if str(complete.get("audit_id")) != str(source_cfg["audit_id"]):
        raise RuntimeError("Step 3.5 audit id mismatch")
    if int(complete.get("pair_count", -1)) != int(source_cfg["expected_pair_count"]):
        raise RuntimeError("Step 3.5 pair count mismatch")
    if int(complete.get("counterfactual_count", -1)) != int(
        source_cfg["expected_counterfactual_count"]
    ):
        raise RuntimeError("Step 3.5 counterfactual count mismatch")

    counterfactual_rows = read_csv(source / "counterfactual_metrics.csv")
    pair_rows = read_csv(source / "pairwise_metrics.csv")
    reduced = build_scalable_pair_rows(counterfactual_rows, pair_rows, config)
    effectful = [row for row in reduced if bool(row["effectful_pair"])]
    nonterminal_non_q0 = [
        row
        for row in effectful
        if not bool(row["terminal_insertion"]) and int(row["affected_qubit"]) != 0
    ]

    ablation_cfg = config["scalable_core_ablation"]
    repeats = int(ablation_cfg["bootstrap_repeats"])
    seed = int(ablation_cfg["bootstrap_seed"])
    effectful_stats = summarize(effectful)
    effectful_stats["strong_pair_fraction_ci95"] = list(
        grouped_bootstrap_fraction(effectful, repeats=repeats, seed=seed)
    )
    subset_stats = summarize(nonterminal_non_q0)
    subset_stats["strong_pair_fraction_ci95"] = list(
        grouped_bootstrap_fraction(nonterminal_non_q0, repeats=repeats, seed=seed + 1)
    )
    all_stats = summarize(reduced)

    strata = stratify_effectful(effectful, config)
    severe_cut = float(ablation_cfg["severe_stratum_strong_fraction_below"])
    severe = [
        row
        for row in strata
        if bool(row["eligible"])
        and float(row["strong_pair_fraction"]) < severe_cut
    ]

    core_pass = bool(
        float(effectful_stats["strong_pair_fraction"])
        >= float(ablation_cfg["minimum_effectful_pair_strong_fraction"])
        and float(subset_stats["strong_pair_fraction"])
        >= float(
            ablation_cfg["minimum_effectful_nonterminal_non_q0_strong_fraction"]
        )
        and not severe
    )
    hardware_contract = validate_hardware_contract(config)
    if not hardware_contract["hardware_contract_valid"]:
        decision = "NOT_DEPLOYABLE_AS_DEFINED"
    elif core_pass:
        decision = "DEPLOYABLE_WITH_PAIRED_REFERENCE_CORE"
    else:
        decision = "DEPLOYABLE_CONTRACT_CORE_IDENTIFIABILITY_UNPROVEN"

    feature_rows = [dict(row) for row in config["feature_contract"]]
    source_hashes = {
        name: sha256_file(source / name)
        for name in sorted(required)
    }
    identity = {
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "step3_5_source_hashes": source_hashes,
    }
    audit_id = "audit_" + hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_root = output_parent / audit_id
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing Step 4 audit: {output_root}")
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{audit_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    try:
        atomic_csv(staging / "feature_contract.csv", feature_rows)
        atomic_csv(staging / "scalable_core_pair_metrics.csv", reduced)
        atomic_csv(staging / "scalable_core_stratified_metrics.csv", strata)
        atomic_json(
            staging / "reference_contract.json",
            dict(config["reference_contract"]),
        )
        ablation_summary = {
            "schema": SCHEMA,
            "variant": ablation_cfg["variant"],
            "all_pairs": all_stats,
            "effectful_pairs": effectful_stats,
            "effectful_nonterminal_non_q0_pairs": subset_stats,
            "effectful_pair_count": len(effectful),
            "effectful_nonterminal_non_q0_pair_count": len(nonterminal_non_q0),
            "severe_eligible_strata": severe,
            "pass": core_pass,
        }
        atomic_json(staging / "scalable_core_ablation.json", ablation_summary)
        decision_payload = {
            "schema": SCHEMA,
            "decision": decision,
            "hardware_contract": hardware_contract,
            "scalable_core_ablation": ablation_summary,
            "full_register_distribution_policy": (
                "Hardware-sampleable but optional small-n evidence; not a mandatory scalable "
                "TriQTO hardware input because support grows as 2^n."
            ),
            "step5_reference_policy": config["reference_contract"][
                "primary_step5_reference_kind"
            ],
            "step5_required_fields": list(config["step5_required_fields"]),
            "classifier_trained": False,
            "hardware_executed": False,
            "historical_v0_1_test_accessed": False,
            "spent_confirmatory_cohort_accessed": False,
        }
        atomic_json(staging / "decision.json", decision_payload)
        complete_payload = {
            "schema": SCHEMA,
            "status": "AUDIT_COMPLETE",
            "audit_id": audit_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "identity": identity,
            "source_step3_5_audit": str(source),
            "hardware_executed": False,
            "classifier_trained": False,
            "historical_v0_1_test_accessed": False,
            "spent_confirmatory_cohort_accessed": False,
        }
        atomic_json(staging / "audit_complete.json", complete_payload)
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise

    print("\nTRIQTO STEP 4 B_DELTA HARDWARE-FEASIBILITY CONTRACT AUDIT COMPLETE\n")
    print(f"Decision: {decision}")
    print(
        "Hardware acquisition contract valid: "
        f"{hardware_contract['hardware_contract_valid']}"
    )
    print(
        "Local-Pauli core effectful strong-pair fraction: "
        f"{effectful_stats['strong_pair_fraction']:.4f} "
        f"(95% clean-circuit bootstrap CI "
        f"{effectful_stats['strong_pair_fraction_ci95'][0]:.4f}.."
        f"{effectful_stats['strong_pair_fraction_ci95'][1]:.4f})"
    )
    print(
        "Local-Pauli core effectful nonterminal+non-q0 strong fraction: "
        f"{subset_stats['strong_pair_fraction']:.4f}"
    )
    print(f"Severe eligible core strata: {len(severe)}")
    for row in severe[:10]:
        print(
            f"  {row['stratum_type']}={row['stratum_value']}: "
            f"strong={float(row['strong_pair_fraction']):.4f}, "
            f"pairs={int(row['pair_count'])}"
        )
    print("Hardware executed: NO")
    print("Classifier trained: NO")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
