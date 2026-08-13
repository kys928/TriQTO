#!/usr/bin/env python3
"""Generate the repaired Step 5 v2 TriQTO diagnostic training cohorts.

Step 5 v1 was rejected after full-artifact EDA found two deterministic design
confounds: global root_index % 5 split assignment aliased with the family cycle,
and a fixed depth-ordered strength schedule made depth and strength perfectly
confounded.  V2 preserves the Step 4.1 hardware-valid correlation core while:

* splitting by within-family occurrence index for exact nested 80/20 coverage;
* alternating weak/strong schedules by within-family occurrence parity;
* adding 3-qubit clean circuits to the non-Bell variable-size universe;
* keeping raw reference-window identifiers meta/audit-only;
* enforcing the repaired factorial/split design before publication.

No classifier is trained.  Statevectors remain transient simulator generation
machinery and are never persisted as deployable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "generate_step5_matched_diagnostic_training_dataset.py"
BASE_MODULE_NAME = "triqto_step5_v1_generator_base"
SPEC = importlib.util.spec_from_file_location(BASE_MODULE_NAME, BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load Step 5 v1 generator utilities from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[BASE_MODULE_NAME] = BASE
SPEC.loader.exec_module(BASE)


SCHEMA = "triqto.v0_2.step5_matched_diagnostic_training_dataset.v2"
CURRENT_POINTER_SCHEMA = "triqto.v0_2.step5_current_product.v2"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/step5_matched_diagnostic_training_v2"
)
MECHANISMS = tuple(BASE.MECHANISMS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--clean-circuit-roots", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def family_occurrence_index(
    root_index: int, family: str, config: Mapping[str, Any]
) -> int:
    """Return a zero-based occurrence index of `family` at this root index."""
    cycle = [str(value) for value in config["clean_circuit_generation"]["family_cycle"]]
    if not cycle:
        raise ValueError("family cycle is empty")
    if str(cycle[root_index % len(cycle)]) != family:
        raise ValueError("family does not match deterministic family cycle")
    per_cycle = cycle.count(family)
    if per_cycle <= 0:
        raise ValueError(f"family {family!r} is absent from family cycle")
    completed_cycles = root_index // len(cycle)
    offset = root_index % len(cycle)
    within_cycle_zero_based = (
        sum(1 for value in cycle[: offset + 1] if value == family) - 1
    )
    return completed_cycles * per_cycle + within_cycle_zero_based


def split_for_root_v2(
    root_index: int, family: str, config: Mapping[str, Any]
) -> tuple[str, int]:
    policy = config["independence_and_splits"]
    occurrence = family_occurrence_index(root_index, family, config)
    modulus = int(policy["modulus"])
    residue = int(policy["validation_residue"])
    split = "validation" if occurrence % modulus == residue else "train"
    return split, occurrence


def strength_schedule_for_occurrence(
    family_occurrence: int, config: Mapping[str, Any]
) -> list[float]:
    design = config["matched_intervention_design"]
    key = (
        "even_family_occurrence_schedule"
        if family_occurrence % 2 == 0
        else "odd_family_occurrence_schedule"
    )
    schedule = [float(value) for value in design[key]]
    depth_count = len(design["depth_target_fractions"])
    if len(schedule) != depth_count:
        raise RuntimeError("strength schedule length differs from depth target count")
    allowed = {float(value) for value in design["strength_values"]}
    if set(schedule) != allowed:
        raise RuntimeError("strength schedule does not contain the frozen strength values")
    return schedule


def root_plan(root_count: int, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for root_index in range(root_count):
        family = BASE.family_for_root(root_index, config)
        split, occurrence = split_for_root_v2(root_index, family, config)
        n_qubits = BASE.choose_n_qubits(root_index, family, config)
        output.append(
            {
                "root_index": root_index,
                "family": family,
                "family_occurrence_index": occurrence,
                "split": split,
                "n_qubits": n_qubits,
                "strength_schedule": strength_schedule_for_occurrence(
                    occurrence, config
                ),
            }
        )
    return output


def cramers_v(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    left_values = sorted({str(row[left]) for row in rows})
    right_values = sorted({str(row[right]) for row in rows})
    if len(left_values) < 2 or len(right_values) < 2:
        return 0.0
    left_index = {value: index for index, value in enumerate(left_values)}
    right_index = {value: index for index, value in enumerate(right_values)}
    table = np.zeros((len(left_values), len(right_values)), dtype=np.float64)
    for row in rows:
        table[left_index[str(row[left])], right_index[str(row[right])]] += 1.0
    total = float(np.sum(table))
    if total <= 0.0:
        return 0.0
    row_sums = np.sum(table, axis=1, keepdims=True)
    col_sums = np.sum(table, axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    valid = expected > 0.0
    chi2 = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt((chi2 / total) / denominator)) if denominator > 0 else 0.0


def nested_counts(
    rows: Sequence[Mapping[str, Any]], first: str, second: str
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for row in rows:
        first_value = str(row[first])
        second_value = str(row[second])
        output.setdefault(first_value, {})
        output[first_value][second_value] = output[first_value].get(second_value, 0) + 1
    return {
        key: dict(sorted(value.items())) for key, value in sorted(output.items())
    }


def validate_stage_v2(
    roots: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    root_count: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["stage_validation"]
    expected_per_root = int(policy["expected_examples_per_clean_root"])
    expected_examples = root_count * expected_per_root
    if len(roots) != root_count:
        raise RuntimeError("clean root count mismatch")
    if len(examples) != expected_examples:
        raise RuntimeError(
            f"example count mismatch: {len(examples)} != {expected_examples}"
        )

    group_to_splits: dict[str, set[str]] = defaultdict(set)
    group_to_graphs: dict[str, set[str]] = defaultdict(set)
    group_to_counts: Counter[str] = Counter()
    for row in examples:
        group = str(row["clean_circuit_group_id"])
        group_to_splits[group].add(str(row["split"]))
        group_to_graphs[group].add(str(row["graph_sha256"]))
        group_to_counts[group] += 1
        if bool(row["clean_control"]) and bool(row["mechanism_loss_mask"]):
            raise RuntimeError("clean control has mechanism loss enabled")
        if (not bool(row["effect_present"])) and bool(row["mechanism_loss_mask"]):
            raise RuntimeError("negligible example has mechanism loss enabled")
    if any(len(values) != 1 for values in group_to_splits.values()):
        raise RuntimeError("clean circuit group crosses train/validation split")
    if any(len(values) != 1 for values in group_to_graphs.values()):
        raise RuntimeError("root derivatives do not share exactly one graph hash")
    if any(count != expected_per_root for count in group_to_counts.values()):
        raise RuntimeError("root derivative count mismatch")

    root_hashes = [str(row["graph_sha256"]) for row in roots]
    if len(set(root_hashes)) != len(root_hashes):
        raise RuntimeError("duplicate clean circuit graph detected")

    mechanism_counts = Counter(str(row["mechanism"]) for row in examples)
    expected_mechanism = root_count * int(
        config["matched_intervention_design"]["contexts_per_clean_root"]
    )
    for mechanism in MECHANISMS:
        if mechanism_counts[mechanism] != expected_mechanism:
            raise RuntimeError(f"mechanism count mismatch for {mechanism}")
    if mechanism_counts["clean_control"] != root_count:
        raise RuntimeError("clean control count mismatch")

    train_roots = sum(str(row["split"]) == "train" for row in roots)
    validation_roots = sum(str(row["split"]) == "validation" for row in roots)
    modulus = int(config["independence_and_splits"]["modulus"])
    if validation_roots != root_count // modulus:
        raise RuntimeError("unexpected validation root count")
    if train_roots != root_count - validation_roots:
        raise RuntimeError("unexpected train root count")

    families = sorted({str(row["family"]) for row in roots})
    family_split_counts: dict[str, dict[str, int]] = {}
    for family in families:
        subset = [row for row in roots if str(row["family"]) == family]
        train = sum(str(row["split"]) == "train" for row in subset)
        validation = sum(str(row["split"]) == "validation" for row in subset)
        if bool(policy["require_each_family_in_each_split"]) and (
            train == 0 or validation == 0
        ):
            raise RuntimeError(f"family {family} missing from one split")
        expected_validation = len(subset) // modulus
        if validation != expected_validation:
            raise RuntimeError(
                f"family {family} validation count {validation} != {expected_validation}"
            )
        family_split_counts[family] = {
            "train": train,
            "validation": validation,
            "total": len(subset),
        }

    family_split_v = cramers_v(roots, "family", "split")
    if family_split_v > float(policy["maximum_family_split_cramers_v"]):
        raise RuntimeError(
            f"family/split association too large: Cramer's V={family_split_v:.6f}"
        )

    injected = [row for row in examples if not bool(row["clean_control"])]
    strength_values = [
        float(value) for value in config["matched_intervention_design"]["strength_values"]
    ]
    depth_values = ("early", "middle", "late", "terminal")
    min_share = float(policy["minimum_strength_share_within_each_depth_split_cell"])

    depth_strength_counts: dict[str, dict[str, int]] = {}
    for depth in depth_values:
        depth_subset = [
            row for row in injected if str(row["insertion_depth_bin"]) == depth
        ]
        if not depth_subset:
            raise RuntimeError(f"missing depth bin {depth}")
        depth_strength_counts[depth] = {}
        for strength in strength_values:
            count = sum(
                abs(float(row["strength"]) - strength) < 1e-12
                for row in depth_subset
            )
            depth_strength_counts[depth][f"{strength:.12g}"] = count
            if count == 0:
                raise RuntimeError(f"missing depth/strength cell {depth}/{strength}")

    for split in ("train", "validation"):
        for depth in depth_values:
            subset = [
                row
                for row in injected
                if str(row["split"]) == split
                and str(row["insertion_depth_bin"]) == depth
            ]
            if not subset:
                raise RuntimeError(f"missing split/depth cell {split}/{depth}")
            for strength in strength_values:
                count = sum(
                    abs(float(row["strength"]) - strength) < 1e-12
                    for row in subset
                )
                share = float(count) / float(len(subset))
                if count == 0 or share < min_share:
                    raise RuntimeError(
                        f"insufficient strength share in {split}/{depth}/{strength}: {share:.4f}"
                    )

    if bool(policy["require_each_depth_strength_cell_in_each_family"]):
        for family in families:
            family_subset = [
                row for row in injected if str(row["family"]) == family
            ]
            for depth in depth_values:
                for strength in strength_values:
                    if not any(
                        str(row["insertion_depth_bin"]) == depth
                        and abs(float(row["strength"]) - strength) < 1e-12
                        for row in family_subset
                    ):
                        raise RuntimeError(
                            f"family {family} missing depth/strength {depth}/{strength}"
                        )

    depth_strength_v = cramers_v(injected, "insertion_depth_bin", "strength")
    if depth_strength_v > float(policy["maximum_depth_strength_cramers_v"]):
        raise RuntimeError(
            f"depth/strength association too large: Cramer's V={depth_strength_v:.6f}"
        )

    if root_count == 500:
        family_counts = Counter(str(row["family"]) for row in roots)
        if min(family_counts.values()) < int(policy["minimum_family_root_count_at_500_stage"]):
            raise RuntimeError("500-root family coverage below frozen minimum")
        non_bell_n = Counter(
            int(row["n_qubits"])
            for row in roots
            if str(row["family"]) != "bell_like"
        )
        for n_qubits in config["clean_circuit_generation"]["qubit_choices"]:
            if non_bell_n[int(n_qubits)] < int(
                policy["minimum_each_non_bell_qubit_count_at_500_stage"]
            ):
                raise RuntimeError(
                    f"500-root qubit-count coverage below minimum for {n_qubits}q"
                )
        three_train = sum(
            int(row["n_qubits"]) == 3 and str(row["split"]) == "train"
            for row in roots
        )
        three_validation = sum(
            int(row["n_qubits"]) == 3 and str(row["split"]) == "validation"
            for row in roots
        )
        if three_train < int(policy["minimum_three_qubit_train_roots_at_500_stage"]):
            raise RuntimeError("3-qubit train coverage below frozen minimum")
        if three_validation < int(
            policy["minimum_three_qubit_validation_roots_at_500_stage"]
        ):
            raise RuntimeError("3-qubit validation coverage below frozen minimum")
        for mechanism in MECHANISMS:
            if mechanism_counts[mechanism] < int(
                policy["minimum_each_mechanism_count_at_500_stage"]
            ):
                raise RuntimeError("500-root mechanism coverage below frozen minimum")

    return {
        "status": "PASS",
        "schema": SCHEMA,
        "clean_root_count": root_count,
        "example_count": len(examples),
        "train_clean_root_count": train_roots,
        "validation_clean_root_count": validation_roots,
        "clean_group_cross_split_count": 0,
        "duplicate_clean_graph_count": 0,
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "mechanism_supervised_example_count": sum(
            bool(row["mechanism_loss_mask"]) for row in examples
        ),
        "negligible_injected_example_count": sum(
            (not bool(row["clean_control"])) and (not bool(row["effect_present"]))
            for row in examples
        ),
        "clean_control_count": mechanism_counts["clean_control"],
        "family_split_counts": family_split_counts,
        "family_split_cramers_v": family_split_v,
        "depth_strength_counts": depth_strength_counts,
        "depth_strength_cramers_v": depth_strength_v,
        "qubit_split_counts": nested_counts(roots, "n_qubits", "split"),
        "three_qubit_train_root_count": sum(
            int(row["n_qubits"]) == 3 and str(row["split"]) == "train"
            for row in roots
        ),
        "three_qubit_validation_root_count": sum(
            int(row["n_qubits"]) == 3 and str(row["split"]) == "validation"
            for row in roots
        ),
        "raw_reference_window_identifier_role": config["finite_shot_acquisition"][
            "raw_reference_window_identifier_role"
        ],
    }


def write_additional_summaries(
    manifests: Path,
    roots: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> None:
    family_split_rows: list[dict[str, Any]] = []
    for family in sorted({str(row["family"]) for row in roots}):
        for split in ("train", "validation"):
            subset = [
                row
                for row in roots
                if str(row["family"]) == family and str(row["split"]) == split
            ]
            family_split_rows.append(
                {"family": family, "split": split, "clean_root_count": len(subset)}
            )
    BASE.write_csv(manifests / "family_split_summary.csv", family_split_rows)

    qubit_split_rows: list[dict[str, Any]] = []
    for n_qubits in sorted({int(row["n_qubits"]) for row in roots}):
        for split in ("train", "validation"):
            subset = [
                row
                for row in roots
                if int(row["n_qubits"]) == n_qubits and str(row["split"]) == split
            ]
            qubit_split_rows.append(
                {
                    "n_qubits": n_qubits,
                    "split": split,
                    "clean_root_count": len(subset),
                }
            )
    BASE.write_csv(manifests / "qubit_split_summary.csv", qubit_split_rows)

    injected = [row for row in examples if not bool(row["clean_control"])]
    depth_strength_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for depth in ("early", "middle", "late", "terminal"):
            for strength in (0.05, 0.15):
                subset = [
                    row
                    for row in injected
                    if str(row["split"]) == split
                    and str(row["insertion_depth_bin"]) == depth
                    and abs(float(row["strength"]) - strength) < 1e-12
                ]
                depth_strength_rows.append(
                    {
                        "split": split,
                        "insertion_depth_bin": depth,
                        "strength": strength,
                        "example_count": len(subset),
                        "matched_context_count": len(subset) // len(MECHANISMS),
                    }
                )
    BASE.write_csv(manifests / "depth_strength_summary.csv", depth_strength_rows)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = BASE.read_json(config_path)
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected Step 5 v2 config schema")

    root_count = int(args.clean_circuit_roots)
    allowed = [int(value) for value in config["stage_progression"]["allowed_root_counts"]]
    if root_count not in allowed:
        raise ValueError(f"--clean-circuit-roots must be one of {allowed}")

    runner_path = Path(__file__).resolve()
    identity = {
        "schema": SCHEMA,
        "config_sha256": BASE.sha256_file(config_path),
        "runner_sha256": BASE.sha256_file(runner_path),
        "base_v1_runner_sha256": BASE.sha256_file(BASE_PATH),
        "clean_circuit_root_count": root_count,
        "rejected_v1_product_id": config["rejected_v1"]["product_id"],
    }
    product_id = "product_" + hashlib.sha256(
        BASE.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    product_root = output_parent / product_id
    if product_root.exists():
        raise RuntimeError(f"refusing to overwrite existing Step 5 v2 product: {product_root}")
    staging = output_parent / f".{product_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    root_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    seen_graph_hashes: set[str] = set()
    bound = float(config["stage_validation"]["diagnostic_delta_absolute_bound"])
    phenomenology_cfg = config["privileged_supervision"]
    epsilon = float(phenomenology_cfg["epsilon"])
    negligible_floor = float(phenomenology_cfg["negligible_overlap_loss_floor"])
    dominance_ratio = float(phenomenology_cfg["phenomenology_strong_dominance_ratio"])
    depths = [
        float(value)
        for value in config["matched_intervention_design"]["depth_target_fractions"]
    ]
    shots_cycle = [int(value) for value in config["finite_shot_acquisition"]["shots_cycle"]]
    plan = root_plan(root_count, config)

    try:
        for planned in plan:
            root_index = int(planned["root_index"])
            family = str(planned["family"])
            occurrence = int(planned["family_occurrence_index"])
            split = str(planned["split"])
            n_qubits = int(planned["n_qubits"])
            strengths = [float(value) for value in planned["strength_schedule"]]

            clean = BASE.build_clean_circuit(root_index, family, n_qubits, config)
            graph = BASE.serialize_graph(clean)
            ghash = BASE.graph_hash(graph)
            if ghash in seen_graph_hashes:
                raise RuntimeError(f"duplicate generated clean graph at root {root_index}")
            seen_graph_hashes.add(ghash)
            clean_group_id = BASE.sha256_bytes(
                BASE.canonical_json(
                    {
                        "root_index": root_index,
                        "family": family,
                        "family_occurrence_index": occurrence,
                        "n_qubits": n_qubits,
                        "graph_sha256": ghash,
                    }
                ).encode("utf-8")
            )
            clean_state = BASE.normalized_state(clean)
            pairs = BASE.pair_indices(n_qubits)
            clean_basis_probs = {
                basis: BASE.basis_probabilities(clean_state, n_qubits, basis)
                for basis in BASE.BASIS_ORDER
            }
            layout_identity = "identity:" + ",".join(str(i) for i in range(n_qubits))

            root_rows.append(
                {
                    "root_index": root_index,
                    "family_occurrence_index": occurrence,
                    "clean_circuit_group_id": clean_group_id,
                    "split": split,
                    "family": family,
                    "n_qubits": n_qubits,
                    "unitary_event_count": len(clean.data),
                    "graph_sha256": ghash,
                    "physical_layout_identity": layout_identity,
                }
            )

            clean_shots = shots_cycle[root_index % len(shots_cycle)]
            clean_context_key = "clean_control"
            reference_bundle, pairs = BASE.make_reference_bundle(
                clean_basis_probs,
                n_qubits,
                clean_shots,
                root_index,
                clean_context_key,
            )
            diagnostic, audit_diag = BASE.diagnostic_arrays(
                clean_state,
                clean_basis_probs,
                reference_bundle,
                n_qubits,
                pairs,
                clean_shots,
                root_index,
                clean_context_key,
                "clean_control",
            )
            example_id = BASE.sha256_bytes(
                BASE.canonical_json([clean_group_id, "clean_control"]).encode("utf-8")
            )
            zero_targets = {
                "population_component": 0.0,
                "phase_component": 0.0,
                "dominance_log_ratio": 0.0,
                "total_overlap_loss": 0.0,
            }
            artifact_rel = Path("artifacts") / split / f"{example_id.split(':',1)[1]}.npz"
            artifact_path = staging / artifact_rel
            BASE.save_example(
                artifact_path,
                graph=graph,
                n_qubits=n_qubits,
                diagnostic=diagnostic,
                audit_diagnostic=audit_diag,
                example_id=example_id,
                clean_group_id=clean_group_id,
                clean_control=True,
                effect_present=False,
                mechanism_code=-1,
                mechanism_loss_mask=False,
                phenotype="clean_control",
                continuous=zero_targets,
                affected_qubit=-1,
                boundary=-1,
                strength=0.0,
            )
            with np.load(artifact_path, allow_pickle=False) as loaded:
                BASE.validate_array_contract(dict(loaded), bound)
            example_rows.append(
                {
                    "example_id": example_id,
                    "root_index": root_index,
                    "family_occurrence_index": occurrence,
                    "clean_circuit_group_id": clean_group_id,
                    "split": split,
                    "family": family,
                    "n_qubits": n_qubits,
                    "clean_control": True,
                    "mechanism": "clean_control",
                    "effect_present": False,
                    "mechanism_loss_mask": False,
                    "phenomenology": "clean_control",
                    "affected_qubit": -1,
                    "insertion_boundary_rank": -1,
                    "insertion_depth_bin": "clean_control",
                    "strength": 0.0,
                    "shots": clean_shots,
                    "reference_kind": config["finite_shot_acquisition"]["reference_kind"],
                    "backend_identity": config["finite_shot_acquisition"]["simulation_backend_identity"],
                    "physical_layout_identity": layout_identity,
                    "meta_reference_window_id": f"root{root_index}:clean",
                    "graph_sha256": ghash,
                    "artifact_path": artifact_rel.as_posix(),
                    "artifact_sha256": BASE.sha256_file(artifact_path),
                }
            )

            for context_index, (fraction, strength) in enumerate(zip(depths, strengths)):
                boundary = BASE.depth_boundary(len(clean.data), fraction)
                affected_qubit = int((root_index + context_index) % n_qubits)
                shots = shots_cycle[(root_index + context_index) % len(shots_cycle)]
                context_key = (
                    f"ctx{context_index}:q{affected_qubit}:b{boundary}:s{strength:.12g}"
                )
                reference_bundle, pairs = BASE.make_reference_bundle(
                    clean_basis_probs,
                    n_qubits,
                    shots,
                    root_index,
                    context_key,
                )

                for mechanism in MECHANISMS:
                    observed = BASE.inject_hidden_rotation(
                        clean,
                        boundary,
                        affected_qubit,
                        mechanism,
                        strength,
                    )
                    observed_state = BASE.normalized_state(observed)
                    truth = BASE.state_diagnostics(
                        clean_state,
                        observed_state,
                        epsilon=epsilon,
                        negligible_floor=negligible_floor,
                        dominance_ratio=dominance_ratio,
                    )
                    diagnostic, audit_diag = BASE.diagnostic_arrays(
                        observed_state,
                        clean_basis_probs,
                        reference_bundle,
                        n_qubits,
                        pairs,
                        shots,
                        root_index,
                        context_key,
                        mechanism,
                    )
                    mechanism_loss_mask = bool(truth["effect_present"])
                    example_id = BASE.sha256_bytes(
                        BASE.canonical_json(
                            [
                                clean_group_id,
                                context_index,
                                affected_qubit,
                                boundary,
                                strength,
                                mechanism,
                            ]
                        ).encode("utf-8")
                    )
                    artifact_rel = (
                        Path("artifacts") / split / f"{example_id.split(':',1)[1]}.npz"
                    )
                    artifact_path = staging / artifact_rel
                    BASE.save_example(
                        artifact_path,
                        graph=graph,
                        n_qubits=n_qubits,
                        diagnostic=diagnostic,
                        audit_diagnostic=audit_diag,
                        example_id=example_id,
                        clean_group_id=clean_group_id,
                        clean_control=False,
                        effect_present=bool(truth["effect_present"]),
                        mechanism_code=BASE.MECHANISM_CODES[mechanism],
                        mechanism_loss_mask=mechanism_loss_mask,
                        phenotype=str(truth["phenotype"]),
                        continuous=truth,
                        affected_qubit=affected_qubit,
                        boundary=boundary,
                        strength=strength,
                    )
                    with np.load(artifact_path, allow_pickle=False) as loaded:
                        BASE.validate_array_contract(dict(loaded), bound)
                    example_rows.append(
                        {
                            "example_id": example_id,
                            "root_index": root_index,
                            "family_occurrence_index": occurrence,
                            "clean_circuit_group_id": clean_group_id,
                            "split": split,
                            "family": family,
                            "n_qubits": n_qubits,
                            "clean_control": False,
                            "mechanism": mechanism,
                            "effect_present": bool(truth["effect_present"]),
                            "mechanism_loss_mask": mechanism_loss_mask,
                            "phenomenology": str(truth["phenotype"]),
                            "affected_qubit": affected_qubit,
                            "insertion_boundary_rank": boundary,
                            "insertion_depth_bin": BASE.depth_bin(boundary, len(clean.data)),
                            "strength": strength,
                            "shots": shots,
                            "reference_kind": config["finite_shot_acquisition"]["reference_kind"],
                            "backend_identity": config["finite_shot_acquisition"]["simulation_backend_identity"],
                            "physical_layout_identity": layout_identity,
                            "meta_reference_window_id": f"root{root_index}:ctx{context_index}",
                            "graph_sha256": ghash,
                            "artifact_path": artifact_rel.as_posix(),
                            "artifact_sha256": BASE.sha256_file(artifact_path),
                        }
                    )

            if args.progress_every > 0 and (root_index + 1) % args.progress_every == 0:
                print(
                    f"Generated {root_index + 1}/{root_count} clean roots "
                    f"({len(example_rows)} examples)",
                    flush=True,
                )

        manifests = staging / "manifests"
        BASE.write_csv(manifests / "clean_circuit_manifest.csv", root_rows)
        BASE.write_csv(manifests / "example_manifest.csv", example_rows)
        family_rows, split_rows, mechanism_rows = BASE.summarize_rows(
            root_rows, example_rows
        )
        BASE.write_csv(manifests / "family_summary.csv", family_rows)
        BASE.write_csv(manifests / "split_summary.csv", split_rows)
        BASE.write_csv(manifests / "mechanism_summary.csv", mechanism_rows)
        write_additional_summaries(manifests, root_rows, example_rows)

        validation = validate_stage_v2(root_rows, example_rows, root_count, config)
        BASE.atomic_json(staging / "stage_validation.json", validation)

        manifest_names = [
            "clean_circuit_manifest.csv",
            "example_manifest.csv",
            "family_summary.csv",
            "split_summary.csv",
            "mechanism_summary.csv",
            "family_split_summary.csv",
            "qubit_split_summary.csv",
            "depth_strength_summary.csv",
        ]
        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "product_id": product_id,
            "identity": identity,
            "clean_circuit_root_count": root_count,
            "example_count": len(example_rows),
            "train_clean_root_count": validation["train_clean_root_count"],
            "validation_clean_root_count": validation["validation_clean_root_count"],
            "train_example_count": next(
                int(row["example_count"])
                for row in split_rows
                if row["split"] == "train"
            ),
            "validation_example_count": next(
                int(row["example_count"])
                for row in split_rows
                if row["split"] == "validation"
            ),
            "mechanism_counts": validation["mechanism_counts"],
            "mechanism_supervised_example_count": validation[
                "mechanism_supervised_example_count"
            ],
            "negligible_injected_example_count": validation[
                "negligible_injected_example_count"
            ],
            "clean_control_count": validation["clean_control_count"],
            "family_split_cramers_v": validation["family_split_cramers_v"],
            "depth_strength_cramers_v": validation["depth_strength_cramers_v"],
            "three_qubit_train_root_count": validation[
                "three_qubit_train_root_count"
            ],
            "three_qubit_validation_root_count": validation[
                "three_qubit_validation_root_count"
            ],
            "selected_diagnostic_variant": config["deployable_diagnostic_input"][
                "selected_step4_1_variant"
            ],
            "primary_input_is_empirical_finite_shot": True,
            "statevectors_persisted_in_example_artifacts": False,
            "raw_reference_window_identifier_persisted_as_model_input": False,
            "historical_v0_1_test_accessed": False,
            "spent_confirmatory_cohort_accessed": False,
            "classifier_trained": False,
            "model_architecture_changed": False,
            "manifest_hashes": {
                name: BASE.sha256_file(manifests / name) for name in manifest_names
            },
            "stage_validation_sha256": BASE.sha256_file(
                staging / "stage_validation.json"
            ),
            "scientific_boundary": (
                "Step 5 v2 development dataset generation only. The hidden intervention "
                "is absent from deployable graph inputs; finite-shot diagnostics are "
                "simulator-sampled; no model-quality or hardware-robustness claim is made. "
                "Promotion requires the separate full-artifact EDA audit."
            ),
        }
        BASE.atomic_json(staging / "dataset_complete.json", completion)
        os.replace(staging, product_root)
        pointer = {
            "schema": CURRENT_POINTER_SCHEMA,
            "product_dir": str(product_root),
            "product_id": product_id,
            "clean_circuit_root_count": root_count,
            "dataset_complete_sha256": BASE.sha256_file(
                product_root / "dataset_complete.json"
            ),
        }
        BASE.atomic_json(output_parent / "current_product.json", pointer)

        print("\nTRIQTO STEP 5 V2 MATCHED DIAGNOSTIC TRAINING DATASET COMPLETE\n")
        print(f"Stage clean roots: {root_count}")
        print(f"Examples: {len(example_rows)}")
        print(
            f"Train/validation clean roots: {validation['train_clean_root_count']}/"
            f"{validation['validation_clean_root_count']}"
        )
        print(f"Mechanism counts: {validation['mechanism_counts']}")
        print(
            "Mechanism-supervised examples: "
            f"{validation['mechanism_supervised_example_count']}"
        )
        print(
            "Injected-but-negligible examples: "
            f"{validation['negligible_injected_example_count']}"
        )
        print(f"Clean controls: {validation['clean_control_count']}")
        print(f"Family/split Cramer's V: {validation['family_split_cramers_v']:.6f}")
        print(
            "Depth/strength Cramer's V: "
            f"{validation['depth_strength_cramers_v']:.6f}"
        )
        print(
            "3q train/validation clean roots: "
            f"{validation['three_qubit_train_root_count']}/"
            f"{validation['three_qubit_validation_root_count']}"
        )
        print(
            "Selected diagnostic core: "
            f"{config['deployable_diagnostic_input']['selected_step4_1_variant']}"
        )
        print("Primary diagnostics: empirical finite-shot")
        print("Hidden intervention present in graph input: NO")
        print("Raw reference-window ID present in model input: NO")
        print("Statevectors persisted in example artifacts: NO")
        print("Historical v0.1 test accessed: NO")
        print("Spent confirmatory cohort accessed: NO")
        print("Classifier trained: NO")
        print(f"Product: {product_root}")
        print(
            "Next required gate: audit_step5_training_dataset_eda.py "
            "must return PROMOTION_READY before 1000 roots."
        )

    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
