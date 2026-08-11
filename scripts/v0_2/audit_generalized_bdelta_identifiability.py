#!/usr/bin/env python3
"""Step 3.5 generalized matched B_delta identifiability audit.

This development-only audit asks whether the Step 3 exact-simulator identifiability
result survives removal of two shortcuts in the pilot cohort:

1. the affected qubit was always q0;
2. the audited perturbation was the final unitary operation before measurement.

Every source example is first replay-validated with the merged Step 3 v3 semantics.
The original audited distortion is then removed, clean unitary circuits are
deduplicated, and a deterministic generalized factorial cohort is generated over:

* every qubit in the clean circuit;
* distinct insertion boundaries near 25%, 50%, 75% and 100% unitary depth;
* strengths 0.05 and 0.15;
* matched RZ, RX and RY perturbations.

No classifier is trained. Exact statevectors are privileged audit-only quantities.
The evidence being tested remains clean-relative exact Z/X/Y probabilities and
single-qubit Pauli expectation changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "audit_matched_bdelta_replay_identifiability_v3.py"
V3_MODULE_NAME = "triqto_v0_2_step3_replay_v3"
SCHEMA = "triqto.v0_2.generalized_bdelta_identifiability_audit_result.v1"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/v0_2/generalized_bdelta_identifiability_audit.json"
)
DEFAULT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_pilot_v2/data/"
    "v0_2_phase_amplitude_identifiability_pilot"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/step3_5_generalized_bdelta_identifiability"
)
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
PAIR_TYPES = (
    ("rz_drift", "rx_overrotation", "rz_vs_rx"),
    ("rz_drift", "ry_overrotation", "rz_vs_ry"),
    ("rx_overrotation", "ry_overrotation", "rx_vs_ry"),
)


def load_v3():
    spec = importlib.util.spec_from_file_location(V3_MODULE_NAME, V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Step 3 v3 runner from {V3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[V3_MODULE_NAME] = module
    spec.loader.exec_module(module)
    module.V2.simulate_records = module.simulate_records
    module.V2.replay_source = module.replay_source
    return module


V3 = load_v3()
V2 = V3.V2
BASE = V2.BASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def resolve_product(arguments: argparse.Namespace) -> Path:
    if arguments.product_dir is not None:
        root = arguments.product_dir.expanduser().resolve()
    else:
        pointer = arguments.product_parent.expanduser().resolve() / "current_product.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"missing product pointer: {pointer}")
        root = Path(str(BASE.read_json(pointer)["product_dir"])).expanduser().resolve()
    marker = root / "generation_complete.json"
    manifest = root / "manifests" / "item_manifest.parquet"
    if not marker.is_file() or not manifest.is_file():
        raise RuntimeError(f"incomplete source product: {root}")
    return root


def canonical_unitary_records(
    records: Sequence[Mapping[str, Any]], removed_index: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        index = int(record["index"])
        name = str(record["name"]).lower()
        if index == removed_index:
            continue
        if name in V3.NON_EVOLUTION_GATE_NAMES:
            continue
        output.append(
            {
                "name": name,
                "qubits": [int(value) for value in record["qubits"]],
                "params": [float(value) for value in record["params"]],
            }
        )
    if not output:
        raise ValueError("clean circuit has no unitary events after source distortion removal")
    return output


def clean_unitary_hash(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "name": str(record["name"]),
            "qubits": list(record["qubits"]),
            "params": [round(float(value), 12) for value in record["params"]],
        }
        for record in records
    ]
    return BASE.sha256_bytes(BASE.canonical_json(payload).encode("utf-8"))


def depth_boundaries(
    unitary_count: int, target_fractions: Sequence[float]
) -> list[dict[str, Any]]:
    """Map target fractions to distinct physical insertion boundaries."""
    if unitary_count <= 0:
        raise ValueError("unitary_count must be positive")
    boundary_to_targets: dict[int, list[float]] = defaultdict(list)
    for value in target_fractions:
        fraction = float(value)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"invalid target depth fraction {value!r}")
        boundary = int(math.ceil(fraction * unitary_count))
        boundary = max(1, min(unitary_count, boundary))
        boundary_to_targets[boundary].append(fraction)

    output: list[dict[str, Any]] = []
    for boundary in sorted(boundary_to_targets):
        actual = float(boundary) / float(unitary_count)
        if boundary == unitary_count:
            label = "terminal"
        elif actual <= 0.34:
            label = "early"
        elif actual <= 0.67:
            label = "middle"
        else:
            label = "late"
        output.append(
            {
                "boundary_rank": boundary,
                "actual_depth_fraction": actual,
                "depth_bin": label,
                "target_fractions": "|".join(
                    f"{target:.2f}" for target in sorted(boundary_to_targets[boundary])
                ),
                "terminal_insertion": boundary == unitary_count,
            }
        )
    return output


def qubit_position_class(qubit: int, n_qubits: int) -> str:
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError("qubit outside circuit")
    if n_qubits == 1:
        return "only_qubit"
    if qubit == 0:
        return "first_q0"
    if qubit == n_qubits - 1:
        return "last_qN_minus_1"
    return "interior"


def build_circuit(
    records: Sequence[Mapping[str, Any]], n_qubits: int
) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    for index, record in enumerate(records):
        normalized = {
            "index": index,
            "name": str(record["name"]),
            "qubits": list(record["qubits"]),
            "params": list(record["params"]),
        }
        V2.append_gate(circuit, normalized)
    return circuit


def prefix_state(
    records: Sequence[Mapping[str, Any]], n_qubits: int, boundary_rank: int
) -> np.ndarray:
    if boundary_rank < 0 or boundary_rank > len(records):
        raise ValueError("boundary_rank outside clean unitary sequence")
    if boundary_rank == 0:
        state = np.zeros(1 << n_qubits, dtype=np.complex128)
        state[0] = 1.0
        return state
    circuit = build_circuit(records[:boundary_rank], n_qubits)
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    state /= np.linalg.norm(state)
    return state


def suffix_operator(
    records: Sequence[Mapping[str, Any]], n_qubits: int, boundary_rank: int
) -> np.ndarray | None:
    suffix = records[boundary_rank:]
    if not suffix:
        return None
    circuit = build_circuit(suffix, n_qubits)
    return np.asarray(Operator(circuit).data, dtype=np.complex128)


def propagate_suffix(state: np.ndarray, operator: np.ndarray | None) -> np.ndarray:
    output = np.asarray(state, dtype=np.complex128)
    if operator is not None:
        output = operator @ output
    norm = float(np.linalg.norm(output))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid propagated state norm")
    return output / norm


def generated_context_id(
    clean_context_id: str,
    affected_qubit: int,
    boundary_rank: int,
    strength: float,
) -> str:
    payload = {
        "clean_context_id": clean_context_id,
        "affected_qubit": int(affected_qubit),
        "boundary_rank": int(boundary_rank),
        "strength": round(float(strength), 12),
    }
    return BASE.sha256_bytes(BASE.canonical_json(payload).encode("utf-8"))


def add_custom_strata(
    pairs: Sequence[Mapping[str, Any]], strata: list[dict[str, Any]]
) -> None:
    dimensions = (
        "qubit_position_class",
        "insertion_depth_bin",
        "terminal_insertion",
        "pair_type_x_depth",
    )
    for dimension in dimensions:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pairs:
            grouped[str(row[dimension])].append(row)
        for value in sorted(grouped):
            strata.append(
                BASE.summarize_pairs(
                    grouped[value], stratum_type=dimension, stratum_value=value
                )
            )


def fraction(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(bool(row[key])) for row in rows]))


def shortcut_removal_summary(
    pairs: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    policy = config["shortcut_removal_gate"]
    minimum_pairs = int(policy["minimum_subset_pairs"])
    subsets = {
        "nonterminal": [row for row in pairs if not bool(row["terminal_insertion"])],
        "non_q0": [row for row in pairs if int(row["affected_qubit"]) != 0],
        "nonterminal_non_q0": [
            row
            for row in pairs
            if (not bool(row["terminal_insertion"]))
            and int(row["affected_qubit"]) != 0
        ],
    }
    threshold_keys = {
        "nonterminal": "nonterminal_strong_pair_fraction_min",
        "non_q0": "non_q0_strong_pair_fraction_min",
        "nonterminal_non_q0": "nonterminal_non_q0_strong_pair_fraction_min",
    }
    details: dict[str, Any] = {}
    overall_pass = True
    for name, subset in subsets.items():
        observed = fraction(subset, "strong_pair")
        threshold = float(policy[threshold_keys[name]])
        eligible = len(subset) >= minimum_pairs
        passed = bool(eligible and np.isfinite(observed) and observed >= threshold)
        details[name] = {
            "pair_count": len(subset),
            "strong_pair_fraction": observed,
            "minimum_required": threshold,
            "eligible": eligible,
            "pass": passed,
        }
        overall_pass = overall_pass and passed
    return {
        "pass": overall_pass,
        "minimum_subset_pairs": minimum_pairs,
        "subsets": details,
    }


def apply_shortcut_gate(
    base_decision: Mapping[str, Any], shortcut: Mapping[str, Any]
) -> tuple[str, str]:
    status = str(base_decision["status"])
    if status == "IDENTIFIABLE" and not bool(shortcut["pass"]):
        return (
            "CONTEXT_DEPENDENT",
            "Base decision was IDENTIFIABLE but at least one frozen shortcut-removal subset failed.",
        )
    return status, "Base decision retained after the frozen shortcut-removal gate."


def validate_clean_reconstruction(
    info: Mapping[str, Any],
    unitary_records: Sequence[Mapping[str, Any]],
    n_qubits: int,
    overlap_loss_max: float,
    amplitude_error_max: float,
) -> dict[str, float]:
    replay = np.asarray(
        Statevector.from_instruction(build_circuit(unitary_records, n_qubits)).data,
        dtype=np.complex128,
    )
    replay /= np.linalg.norm(replay)
    error = V2.replay_error(np.asarray(info["clean_state"]), replay)
    if (
        error["overlap_loss"] > overlap_loss_max
        or error["aligned_max_abs_error"] > amplitude_error_max
    ):
        raise ValueError(
            "clean-unitary reconstruction does not match replay-validated clean state"
        )
    return error


def main() -> None:
    arguments = parse_args()
    config_path = arguments.config.expanduser().resolve()
    config = BASE.read_json(config_path)
    if config.get("schema") != "triqto.v0_2.generalized_bdelta_identifiability_audit.v1":
        raise RuntimeError("unexpected generalized identifiability config schema")

    root = resolve_product(arguments)
    rows = V2.load_rows(root, int(config["source"]["expected_source_examples"]))
    manifest_path = root / "manifests" / "item_manifest.parquet"

    identity = {
        "config_sha256": BASE.sha256_file(config_path),
        "source_manifest_sha256": BASE.sha256_file(manifest_path),
        "source_generation_complete_sha256": BASE.sha256_file(
            root / "generation_complete.json"
        ),
        "runner_sha256": BASE.sha256_file(Path(__file__).resolve()),
        "step3_v3_runner_sha256": BASE.sha256_file(V3_PATH),
    }
    audit_id = "audit_" + hashlib.sha256(
        BASE.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = arguments.output_parent.expanduser().resolve()
    output_root = output_parent / audit_id
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing audit: {output_root}")
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{audit_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    replay_contract = config["clean_circuit_reconstruction"]
    overlap_loss_max = float(replay_contract["state_overlap_loss_max"])
    amplitude_error_max = float(
        replay_contract["aligned_amplitude_max_abs_error_max"]
    )

    preflight_rows: list[dict[str, Any]] = []
    replay_infos: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    try:
        for index, row in enumerate(rows, start=1):
            try:
                info = V3.replay_source(
                    root,
                    row,
                    overlap_loss_max=overlap_loss_max,
                    amplitude_error_max=amplitude_error_max,
                )
                clean_records = canonical_unitary_records(
                    info["records"], int(info["removed_index"])
                )
                clean_error = validate_clean_reconstruction(
                    info,
                    clean_records,
                    int(info["n_qubits"]),
                    overlap_loss_max,
                    amplitude_error_max,
                )
                public = {
                    "valid": bool(info["valid"]),
                    "entity_id": info["entity_id"],
                    "split_group_id": info["split_group_id"],
                    "family": info["family"],
                    "phase_sensitive_family": info["phase_sensitive_family"],
                    "n_qubits": info["n_qubits"],
                    "source_raw_label": info["raw_label"],
                    "source_affected_qubit": info["affected_qubit"],
                    "source_removed_index": info["removed_index"],
                    "source_terminal_with_respect_to_unitary_evolution": info[
                        "terminal_with_respect_to_unitary_evolution"
                    ],
                    "clean_unitary_event_count": len(clean_records),
                    "clean_unitary_hash": clean_unitary_hash(clean_records),
                    "clean_reconstruction_overlap_loss": clean_error["overlap_loss"],
                    "clean_reconstruction_aligned_max_abs_error": clean_error[
                        "aligned_max_abs_error"
                    ],
                }
                preflight_rows.append(public)
                if not bool(info["valid"]):
                    failure_rows.append({**public, "reason": "step3_v3_replay_invalid"})
                else:
                    replay_infos.append(
                        {
                            **info,
                            "clean_unitary_records": clean_records,
                            "generalized_clean_unitary_hash": public[
                                "clean_unitary_hash"
                            ],
                        }
                    )
            except Exception as exc:
                failure = {
                    "valid": False,
                    "entity_id": str(row.get("entity_id", "")),
                    "family": str(row.get("family", "")),
                    "n_qubits": int(row.get("n_qubits", 0)),
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
                preflight_rows.append(failure)
                failure_rows.append(failure)
            if arguments.progress_every > 0 and index % min(arguments.progress_every, 25) == 0:
                print(f"Source replay validated {index}/{len(rows)} examples", flush=True)

        BASE.atomic_csv(staging / "source_replay_preflight.csv", preflight_rows)
        preflight = {
            "schema": SCHEMA,
            "status": "PASS" if not failure_rows else "REFUSED",
            "source_example_count": len(rows),
            "replay_valid_count": len(replay_infos),
            "failure_count": len(failure_rows),
            "failure_reasons": dict(
                sorted(Counter(str(row.get("reason", "")) for row in failure_rows).items())
            ),
            "scientific_gate": (
                "All 280 development source examples must replay exactly under the "
                "merged Step 3 v3 measurement-tail semantics before generalized "
                "counterfactual generation."
            ),
        }
        BASE.atomic_json(staging / "source_replay_preflight.json", preflight)
        if failure_rows:
            complete = {
                "schema": SCHEMA,
                "status": "REFUSED_SOURCE_REPLAY",
                "audit_id": audit_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(root),
                "identity": identity,
                "preflight": preflight,
                "classifier_trained": False,
                "historical_v0_1_test_accessed": False,
                "spent_confirmatory_cohort_accessed": False,
            }
            BASE.atomic_json(staging / "audit_complete.json", complete)
            os.replace(staging, output_root)
            raise RuntimeError(
                "generalized identifiability audit refused because source replay "
                f"validation failed; see {output_root}"
            )

        clean_contexts: dict[str, dict[str, Any]] = {}
        for info in replay_infos:
            payload = {
                "clean_unitary_hash": info["generalized_clean_unitary_hash"],
                "family": info["family"],
                "n_qubits": int(info["n_qubits"]),
                "phase_sensitive_family": bool(info["phase_sensitive_family"]),
            }
            clean_context_id = BASE.sha256_bytes(
                BASE.canonical_json(payload).encode("utf-8")
            )
            if clean_context_id not in clean_contexts:
                clean_contexts[clean_context_id] = {
                    "clean_context_id": clean_context_id,
                    "clean_unitary_hash": info["generalized_clean_unitary_hash"],
                    "clean_state": np.asarray(info["clean_state"], dtype=np.complex128),
                    "records": info["clean_unitary_records"],
                    "family": info["family"],
                    "phase_sensitive_family": info["phase_sensitive_family"],
                    "n_qubits": int(info["n_qubits"]),
                    "source_entity_ids": [info["entity_id"]],
                    "source_split_group_ids": [info["split_group_id"]],
                }
            else:
                context = clean_contexts[clean_context_id]
                state_error = V2.replay_error(
                    np.asarray(context["clean_state"]), np.asarray(info["clean_state"])
                )
                if (
                    state_error["overlap_loss"] > overlap_loss_max
                    or state_error["aligned_max_abs_error"] > amplitude_error_max
                ):
                    raise RuntimeError(
                        "identical clean-unitary hash produced inconsistent clean states"
                    )
                context["source_entity_ids"].append(info["entity_id"])
                context["source_split_group_ids"].append(info["split_group_id"])

        factor = config["generalized_factorial"]
        target_fractions = [float(value) for value in factor["depth_target_fractions"]]
        strengths = [float(value) for value in factor["strengths"]]
        if tuple(str(value) for value in factor["mechanisms"]) != MECHANISMS:
            raise RuntimeError("generalized mechanism order differs from frozen Step 3 set")
        if str(factor["affected_qubit_policy"]) != "ALL_QUBITS":
            raise RuntimeError("this runner requires the frozen ALL_QUBITS policy")

        clean_manifest_rows: list[dict[str, Any]] = []
        generalized_plan: list[dict[str, Any]] = []
        for clean_context_id, context in sorted(clean_contexts.items()):
            n_qubits = int(context["n_qubits"])
            boundaries = depth_boundaries(len(context["records"]), target_fractions)
            if len(boundaries) < int(
                factor["minimum_distinct_depth_boundaries_per_clean_circuit"]
            ):
                raise RuntimeError("clean circuit has too few distinct depth boundaries")
            clean_manifest_rows.append(
                {
                    "clean_context_id": clean_context_id,
                    "clean_unitary_hash": context["clean_unitary_hash"],
                    "family": context["family"],
                    "phase_sensitive_family": context["phase_sensitive_family"],
                    "n_qubits": n_qubits,
                    "clean_unitary_event_count": len(context["records"]),
                    "distinct_depth_boundary_count": len(boundaries),
                    "source_entity_count": len(set(context["source_entity_ids"])),
                    "source_split_group_count": len(
                        set(context["source_split_group_ids"])
                    ),
                    "source_entity_ids": "|".join(
                        sorted(set(context["source_entity_ids"]))
                    ),
                    "source_split_group_ids": "|".join(
                        sorted(set(context["source_split_group_ids"]))
                    ),
                }
            )
            for qubit in range(n_qubits):
                for boundary in boundaries:
                    for strength in strengths:
                        matched_id = generated_context_id(
                            clean_context_id,
                            qubit,
                            int(boundary["boundary_rank"]),
                            strength,
                        )
                        generalized_plan.append(
                            {
                                "matched_context_id": matched_id,
                                "clean_context_id": clean_context_id,
                                "family": context["family"],
                                "phase_sensitive_family": context[
                                    "phase_sensitive_family"
                                ],
                                "n_qubits": n_qubits,
                                "affected_qubit": qubit,
                                "affected_qubit_signature": f"q{qubit}",
                                "qubit_position_class": qubit_position_class(
                                    qubit, n_qubits
                                ),
                                "insertion_boundary_rank": int(
                                    boundary["boundary_rank"]
                                ),
                                "insertion_depth_fraction": float(
                                    boundary["actual_depth_fraction"]
                                ),
                                "insertion_depth_bin": boundary["depth_bin"],
                                "target_depth_fractions": boundary[
                                    "target_fractions"
                                ],
                                "terminal_insertion": bool(
                                    boundary["terminal_insertion"]
                                ),
                                "strength": strength,
                                "strength_key": f"{strength:.12g}",
                            }
                        )

        BASE.atomic_csv(staging / "clean_circuit_manifest.csv", clean_manifest_rows)
        BASE.atomic_csv(staging / "generalized_contexts.csv", generalized_plan)

        print(
            "Generalized cohort plan: "
            f"{len(clean_contexts)} clean circuits, "
            f"{len(generalized_plan)} matched qubit/depth/strength contexts, "
            f"{len(generalized_plan) * 3} counterfactuals, "
            f"{len(generalized_plan) * 3} mechanism pairs",
            flush=True,
        )

        epsilon = float(config["phenomenology"]["epsilon"])
        negligible_floor = float(
            config["phenomenology"]["negligible_overlap_loss_floor"]
        )
        dominance_ratio = float(config["phenomenology"]["strong_dominance_ratio"])
        pair_config = config["pairwise_identifiability"]
        raw_minimum = float(pair_config["minimum_raw_separation"])
        relative_minimum = float(pair_config["minimum_relative_separation"])
        collision_maximum = float(pair_config["numerical_collision_score_max"])

        counterfactual_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        plan_by_clean: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in generalized_plan:
            plan_by_clean[str(row["clean_context_id"])].append(row)

        processed_contexts = 0
        for clean_context_id, context in sorted(clean_contexts.items()):
            records = context["records"]
            n_qubits = int(context["n_qubits"])
            clean = np.asarray(context["clean_state"], dtype=np.complex128)
            local_plan = sorted(
                plan_by_clean[clean_context_id],
                key=lambda row: (
                    int(row["insertion_boundary_rank"]),
                    int(row["affected_qubit"]),
                    float(row["strength"]),
                ),
            )
            cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
            for boundary in sorted(
                {int(row["insertion_boundary_rank"]) for row in local_plan}
            ):
                prefix = prefix_state(records, n_qubits, boundary)
                suffix = suffix_operator(records, n_qubits, boundary)
                reconstructed_clean = propagate_suffix(prefix, suffix)
                error = V2.replay_error(clean, reconstructed_clean)
                if (
                    error["overlap_loss"] > overlap_loss_max
                    or error["aligned_max_abs_error"] > amplitude_error_max
                ):
                    raise RuntimeError(
                        f"prefix/suffix decomposition failed for {clean_context_id} "
                        f"boundary {boundary}"
                    )
                cache[boundary] = (prefix, suffix)

            for plan_row in local_plan:
                boundary = int(plan_row["insertion_boundary_rank"])
                qubit = int(plan_row["affected_qubit"])
                strength = float(plan_row["strength"])
                prefix, suffix = cache[boundary]
                generated: dict[str, dict[str, Any]] = {}

                for mechanism in MECHANISMS:
                    axis = BASE.axis_for_mechanism(mechanism)
                    rotated_prefix = BASE.apply_single_qubit_matrix(
                        prefix,
                        qubit,
                        BASE.rotation_matrix(axis, strength),
                    )
                    distorted = propagate_suffix(rotated_prefix, suffix)
                    decomposition = BASE.overlap_decomposition(
                        clean, distorted, epsilon=epsilon
                    )
                    label = BASE.phenotype(
                        decomposition,
                        negligible_floor=negligible_floor,
                        dominance_ratio=dominance_ratio,
                    )
                    evidence = BASE.evidence_for_state(
                        clean, distorted, n_qubits=n_qubits
                    )
                    generated[mechanism] = {
                        "phenotype": label,
                        "decomposition": decomposition,
                        "evidence": evidence,
                    }
                    counterfactual_rows.append(
                        {
                            "context_id": clean_context_id,
                            **plan_row,
                            "mechanism": mechanism,
                            "axis": axis,
                            "phenotype": label,
                            "signal_score": evidence["signal_score"],
                            "tv_x": evidence["tv"]["X"],
                            "tv_y": evidence["tv"]["Y"],
                            "tv_z": evidence["tv"]["Z"],
                            "expectation_rms_x": evidence["exp_rms"]["X"],
                            "expectation_rms_y": evidence["exp_rms"]["Y"],
                            "expectation_rms_z": evidence["exp_rms"]["Z"],
                            **decomposition,
                        }
                    )

                for left_name, right_name, pair_type in PAIR_TYPES:
                    left = generated[left_name]
                    right = generated[right_name]
                    separation = BASE.pair_separation(
                        left["evidence"],
                        right["evidence"],
                        epsilon=epsilon,
                        raw_minimum=raw_minimum,
                        relative_minimum=relative_minimum,
                        collision_maximum=collision_maximum,
                    )
                    pair_rows.append(
                        {
                            "context_id": clean_context_id,
                            **plan_row,
                            "pair_type": pair_type,
                            "pair_type_x_depth": (
                                f"{pair_type}|{plan_row['insertion_depth_bin']}"
                            ),
                            "left_mechanism": left_name,
                            "right_mechanism": right_name,
                            "left_phenotype": left["phenotype"],
                            "right_phenotype": right["phenotype"],
                            "phenotype_differs": (
                                left["phenotype"] != right["phenotype"]
                            ),
                            "left_signal_score": left["evidence"]["signal_score"],
                            "right_signal_score": right["evidence"]["signal_score"],
                            "left_dominance_log_ratio": left["decomposition"][
                                "dominance_log_ratio"
                            ],
                            "right_dominance_log_ratio": right["decomposition"][
                                "dominance_log_ratio"
                            ],
                            **separation,
                        }
                    )

                processed_contexts += 1
                if (
                    arguments.progress_every > 0
                    and processed_contexts % arguments.progress_every == 0
                ):
                    print(
                        f"Audited {processed_contexts}/{len(generalized_plan)} "
                        "generalized matched contexts",
                        flush=True,
                    )

        strata = BASE.stratified_pair_summaries(pair_rows)
        add_custom_strata(pair_rows, strata)
        base_decision = BASE.decide(pair_rows, counterfactual_rows, strata, config)
        shortcut = shortcut_removal_summary(pair_rows, config)
        final_status, final_reason = apply_shortcut_gate(base_decision, shortcut)

        pair_type_summary: dict[str, Any] = {}
        for pair_type in ("rz_vs_rx", "rz_vs_ry", "rx_vs_ry"):
            subset = [row for row in pair_rows if row["pair_type"] == pair_type]
            pair_type_summary[pair_type] = {
                "pair_count": len(subset),
                "strong_pair_fraction": fraction(subset, "strong_pair"),
                "numerical_collision_fraction": fraction(
                    subset, "numerical_collision"
                ),
                "median_pair_separation_score": float(
                    np.median(
                        [float(row["pair_separation_score"]) for row in subset]
                    )
                ),
                "median_relative_separation": float(
                    np.median(
                        [float(row["relative_separation"]) for row in subset]
                    )
                ),
            }

        phenotype_counts: dict[str, dict[str, int]] = {}
        for mechanism in MECHANISMS:
            counter = Counter(
                str(row["phenotype"])
                for row in counterfactual_rows
                if row["mechanism"] == mechanism
            )
            phenotype_counts[mechanism] = dict(sorted(counter.items()))

        decision = {
            **base_decision,
            "status": final_status,
            "base_status_before_shortcut_gate": base_decision["status"],
            "shortcut_removal_gate": shortcut,
            "shortcut_gate_interpretation": final_reason,
            "schema": SCHEMA,
            "audit_id": audit_id,
            "source_example_count": len(rows),
            "deduplicated_clean_circuit_count": len(clean_contexts),
            "generalized_matched_context_count": len(generalized_plan),
            "counterfactual_count": len(counterfactual_rows),
            "pair_count": len(pair_rows),
            "pair_type_summary": pair_type_summary,
            "phenotype_counts": phenotype_counts,
            "affected_qubit_values": sorted(
                {int(row["affected_qubit"]) for row in pair_rows}
            ),
            "insertion_depth_bins": sorted(
                {str(row["insertion_depth_bin"]) for row in pair_rows}
            ),
            "classifier_trained": False,
            "historical_v0_1_test_accessed": False,
            "spent_confirmatory_cohort_accessed": False,
        }

        BASE.atomic_csv(staging / "counterfactual_metrics.csv", counterfactual_rows)
        BASE.atomic_csv(staging / "pairwise_metrics.csv", pair_rows)
        BASE.atomic_csv(staging / "stratified_metrics.csv", strata)
        BASE.atomic_json(staging / "decision.json", decision)

        output_names = (
            "source_replay_preflight.csv",
            "source_replay_preflight.json",
            "clean_circuit_manifest.csv",
            "generalized_contexts.csv",
            "counterfactual_metrics.csv",
            "pairwise_metrics.csv",
            "stratified_metrics.csv",
            "decision.json",
        )
        file_hashes = {
            name: BASE.sha256_file(staging / name) for name in output_names
        }
        complete = {
            "schema": SCHEMA,
            "status": "AUDIT_COMPLETE",
            "audit_id": audit_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(root),
            "identity": identity,
            "file_hashes": file_hashes,
            "decision_status": final_status,
            "source_example_count": len(rows),
            "deduplicated_clean_circuit_count": len(clean_contexts),
            "generalized_matched_context_count": len(generalized_plan),
            "counterfactual_count": len(counterfactual_rows),
            "pair_count": len(pair_rows),
            "classifier_trained": False,
            "historical_v0_1_test_accessed": False,
            "spent_confirmatory_cohort_accessed": False,
            "scientific_boundary": (
                "Exact noiseless simulator generalized identifiability only. "
                "Finite-shot and hardware-feasibility questions remain later gates."
            ),
        }
        BASE.atomic_json(staging / "audit_complete.json", complete)
        os.replace(staging, output_root)

    except Exception:
        if staging.exists():
            if not output_root.exists() and (staging / "audit_complete.json").is_file():
                os.replace(staging, output_root)
            else:
                for path in sorted(staging.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                staging.rmdir()
        raise

    print()
    print("TRIQTO STEP 3.5 GENERALIZED B_DELTA IDENTIFIABILITY AUDIT COMPLETE")
    print()
    print(f"Decision: {decision['status']}")
    print(
        f"Replay-valid source examples: {preflight['replay_valid_count']}/{len(rows)}"
    )
    print(f"Deduplicated clean circuits: {len(clean_contexts)}")
    print(f"Generalized matched contexts: {len(generalized_plan)}")
    print(f"Mechanism pairs: {len(pair_rows)}")
    print(
        "Strong mechanism-pair fraction: "
        f"{decision['overall_strong_pair_fraction']:.4f} "
        f"(95% clean-circuit bootstrap CI "
        f"{decision['overall_strong_pair_fraction_ci95'][0]:.4f}.."
        f"{decision['overall_strong_pair_fraction_ci95'][1]:.4f})"
    )
    print(
        "Different-phenotype pair strong fraction: "
        f"{decision['different_phenotype_strong_fraction']:.4f}"
    )
    print(f"Numerical collision fraction: {decision['numerical_collision_fraction']:.4f}")
    print(
        "Negligible counterfactual fraction: "
        f"{decision['negligible_counterfactual_fraction']:.4f}"
    )
    for name, subset in decision["shortcut_removal_gate"]["subsets"].items():
        print(
            f"{name}: strong={subset['strong_pair_fraction']:.4f} "
            f"pairs={subset['pair_count']} pass={subset['pass']}"
        )
    print(f"Bad eligible context strata: {len(decision['bad_eligible_strata'])}")
    print(
        "Historical v0.1 test accessed: NO\n"
        "Spent confirmatory cohort accessed: NO\n"
        "Classifier trained: NO"
    )
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
