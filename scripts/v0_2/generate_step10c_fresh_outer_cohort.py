#!/usr/bin/env python3
"""Generate the frozen fresh Step-10C outer-validation cohort.

This stage is deliberately validation-only.  It creates two fresh simulator
cohorts in the existing Step-5/Step-10 NPZ contract:

* original domain: new Step-5-v3-family roots using previously unused global
  root indices; and
* bridge domain: new Step-10 bridge parents from an independent frozen seed.

The spent Step-10B outer rows are never copied, reshuffled, or reused.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import benchmark_step6_cheap_baselines as baseline
import generate_step5_matched_diagnostic_training_dataset_v3 as step5v3
import generate_step10_leakage_safe_training_mixture as step10a

BASE = step5v3.BASE
V2 = step5v3.V2

SCHEMA = "triqto.v0_2.step10c_fresh_outer_cohort.v1"
CURRENT_POINTER_SCHEMA = "triqto.v0_2.step10c_fresh_outer_current_product.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10c_fresh_outer_cohort.json"
DEFAULT_V2_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
DEFAULT_V3_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step5_matched_diagnostic_training_dataset_v3.json"
DEFAULT_STEP10_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_STEP10_PRODUCT = Path("/workspace/triqto-data/step10_training_mixture/product_0f7112597501f7ea5fbe123b")
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step10c_fresh_outer_cohort")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    parser.add_argument("--v3-config", type=Path, default=DEFAULT_V3_CONFIG)
    parser.add_argument("--step10-config", type=Path, default=DEFAULT_STEP10_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--step10-product-dir", type=Path, default=DEFAULT_STEP10_PRODUCT)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fresh_group_id(domain: str, payload: Mapping[str, Any]) -> str:
    return BASE.sha256_bytes(
        BASE.canonical_json({"step": "step10c", "domain": domain, **dict(payload)}).encode("utf-8")
    )


def _artifact_rel(domain: str, example_id: str) -> Path:
    return Path(f"{domain}_artifacts") / "outer_validation" / f"{example_id.split(':', 1)[1]}.npz"


def _save_clean_example(
    *,
    staging: Path,
    domain: str,
    root_index: int,
    group_id: str,
    graph: Mapping[str, np.ndarray],
    graph_hash: str,
    family: str,
    family_occurrence: int,
    n_qubits: int,
    clean_state: np.ndarray,
    clean_probs: Mapping[str, np.ndarray],
    shots: int,
    reference_kind: str,
    backend_identity: str,
    extra: Mapping[str, Any],
    diagnostic_bound: float,
) -> dict[str, Any]:
    ref, pairs = BASE.make_reference_bundle(clean_probs, n_qubits, shots, root_index, "clean_control")
    diagnostic, audit = BASE.diagnostic_arrays(
        clean_state,
        clean_probs,
        ref,
        n_qubits,
        pairs,
        shots,
        root_index,
        "clean_control",
        "clean_control",
    )
    example_id = BASE.sha256_bytes(BASE.canonical_json([group_id, "clean_control"]).encode("utf-8"))
    rel = _artifact_rel(domain, example_id)
    path = staging / rel
    zero = {
        "population_component": 0.0,
        "phase_component": 0.0,
        "dominance_log_ratio": 0.0,
        "total_overlap_loss": 0.0,
    }
    BASE.save_example(
        path,
        graph=graph,
        n_qubits=n_qubits,
        diagnostic=diagnostic,
        audit_diagnostic=audit,
        example_id=example_id,
        clean_group_id=group_id,
        clean_control=True,
        effect_present=False,
        mechanism_code=-1,
        mechanism_loss_mask=False,
        phenotype="clean_control",
        continuous=zero,
        affected_qubit=-1,
        boundary=-1,
        strength=0.0,
    )
    with np.load(path, allow_pickle=False) as loaded:
        BASE.validate_array_contract(dict(loaded), diagnostic_bound)
    row = {
        "example_id": example_id,
        "root_index": root_index,
        "family_occurrence_index": family_occurrence,
        "clean_circuit_group_id": group_id,
        "split": "validation",
        "step10c_partition": "fresh_outer_validation",
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
        "shots": shots,
        "reference_kind": reference_kind,
        "backend_identity": backend_identity,
        "physical_layout_identity": "identity:" + ",".join(str(i) for i in range(n_qubits)),
        "meta_reference_window_id": f"step10c:{domain}:root{root_index}:clean",
        "graph_sha256": graph_hash,
        "artifact_path": rel.as_posix(),
        "artifact_sha256": BASE.sha256_file(path),
    }
    row.update(extra)
    return row


def _save_injected_example(
    *,
    staging: Path,
    domain: str,
    root_index: int,
    group_id: str,
    graph: Mapping[str, np.ndarray],
    graph_hash: str,
    family: str,
    family_occurrence: int,
    clean_circuit: Any,
    clean_state: np.ndarray,
    clean_probs: Mapping[str, np.ndarray],
    n_qubits: int,
    context_index: int,
    affected_qubit: int,
    boundary: int,
    strength: float,
    shots: int,
    mechanism: str,
    reference_kind: str,
    backend_identity: str,
    v2cfg: Mapping[str, Any],
    extra: Mapping[str, Any],
    diagnostic_bound: float,
) -> dict[str, Any]:
    context = f"ctx{context_index}:q{affected_qubit}:b{boundary}:s{strength:.12g}:shots{shots}"
    ref, pairs = BASE.make_reference_bundle(clean_probs, n_qubits, shots, root_index, context)
    observed = BASE.inject_hidden_rotation(clean_circuit, boundary, affected_qubit, mechanism, strength)
    observed_state = BASE.normalized_state(observed)
    privileged = v2cfg["privileged_supervision"]
    truth = BASE.state_diagnostics(
        clean_state,
        observed_state,
        epsilon=float(privileged["epsilon"]),
        negligible_floor=float(privileged["negligible_overlap_loss_floor"]),
        dominance_ratio=float(privileged["phenomenology_strong_dominance_ratio"]),
    )
    diagnostic, audit = BASE.diagnostic_arrays(
        observed_state,
        clean_probs,
        ref,
        n_qubits,
        pairs,
        shots,
        root_index,
        context,
        mechanism,
    )
    example_id = BASE.sha256_bytes(
        BASE.canonical_json(
            [group_id, context_index, affected_qubit, boundary, strength, shots, mechanism]
        ).encode("utf-8")
    )
    rel = _artifact_rel(domain, example_id)
    path = staging / rel
    BASE.save_example(
        path,
        graph=graph,
        n_qubits=n_qubits,
        diagnostic=diagnostic,
        audit_diagnostic=audit,
        example_id=example_id,
        clean_group_id=group_id,
        clean_control=False,
        effect_present=bool(truth["effect_present"]),
        mechanism_code=BASE.MECHANISM_CODES[mechanism],
        mechanism_loss_mask=bool(truth["effect_present"]),
        phenotype=str(truth["phenotype"]),
        continuous=truth,
        affected_qubit=affected_qubit,
        boundary=boundary,
        strength=strength,
    )
    with np.load(path, allow_pickle=False) as loaded:
        BASE.validate_array_contract(dict(loaded), diagnostic_bound)
    row = {
        "example_id": example_id,
        "root_index": root_index,
        "family_occurrence_index": family_occurrence,
        "clean_circuit_group_id": group_id,
        "split": "validation",
        "step10c_partition": "fresh_outer_validation",
        "family": family,
        "n_qubits": n_qubits,
        "clean_control": False,
        "mechanism": mechanism,
        "effect_present": bool(truth["effect_present"]),
        "mechanism_loss_mask": bool(truth["effect_present"]),
        "phenomenology": str(truth["phenotype"]),
        "affected_qubit": affected_qubit,
        "insertion_boundary_rank": boundary,
        "insertion_depth_bin": BASE.depth_bin(boundary, len(clean_circuit.data)),
        "strength": strength,
        "shots": shots,
        "reference_kind": reference_kind,
        "backend_identity": backend_identity,
        "physical_layout_identity": "identity:" + ",".join(str(i) for i in range(n_qubits)),
        "meta_reference_window_id": f"step10c:{domain}:root{root_index}:{context}",
        "graph_sha256": graph_hash,
        "artifact_path": rel.as_posix(),
        "artifact_sha256": BASE.sha256_file(path),
    }
    row.update(extra)
    return row


def _old_graph_hashes(
    step10_product: Path,
    step10_cfg: Mapping[str, Any],
    step7_cfg: Mapping[str, Any],
) -> tuple[set[str], set[str], dict[str, Any]]:
    original_product, original_complete, original_rows = step10a._verify_original_product(step10_cfg, step7_cfg)
    original_hashes = {str(row["graph_sha256"]) for row in original_rows}
    bridge_roots = BASE.read_json(step10_product / "dataset_complete.json")
    if bridge_roots.get("product_id") != "product_0f7112597501f7ea5fbe123b":
        raise RuntimeError("unexpected Step-10 source product for Step-10C outer generation")
    root_rows = baseline.read_csv(step10_product / "manifests" / "bridge_root_manifest.csv")
    bridge_hashes = {str(row["graph_sha256"]) for row in root_rows}
    return original_hashes, bridge_hashes, {
        "original_product": str(original_product),
        "original_product_id": original_complete["product_id"],
        "step10_product_id": bridge_roots["product_id"],
    }


def _generate_original(
    *,
    staging: Path,
    cfg: Mapping[str, Any],
    v2cfg: Mapping[str, Any],
    v3cfg: Mapping[str, Any],
    old_original_hashes: set[str],
    old_bridge_hashes: set[str],
    diagnostic_bound: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = cfg["original_domain"]
    start = int(spec["global_root_index_start"])
    count = int(spec["clean_root_count"])
    roots: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    fresh_hashes: set[str] = set()
    reference_kind = str(v2cfg["finite_shot_acquisition"]["reference_kind"])
    depths = [float(v) for v in v2cfg["matched_intervention_design"]["depth_target_fractions"]]

    for local in range(count):
        root_index = start + local
        family = BASE.family_for_root(root_index, v2cfg)
        occurrence = V2.family_occurrence_index(root_index, family, v2cfg)
        n_qubits = BASE.choose_n_qubits(root_index, family, v2cfg)
        strengths = V2.strength_schedule_for_occurrence(occurrence, v2cfg)
        q_schedule = step5v3.affected_qubit_schedule(root_index, nqubits, v3cfg)
        shot_schedule = step5v3.intervention_shot_schedule(root_index, v3cfg)
        circuit = BASE.build_clean_circuit(root_index, family, n_qubits, v2cfg)
        graph = BASE.serialize_graph(circuit)
        graph_hash = BASE.graph_hash(graph)
        if graph_hash in old_original_hashes or graph_hash in old_bridge_hashes:
            raise RuntimeError(f"fresh original graph overlaps Step-10 history at root {root_index}")
        if graph_hash in fresh_hashes:
            raise RuntimeError(f"duplicate fresh original graph at root {root_index}")
        fresh_hashes.add(graph_hash)
        group_id = _fresh_group_id(
            "original",
            {
                "root_index": root_index,
                "family": family,
                "family_occurrence_index": occurrence,
                "n_qubits": n_qubits,
                "graph_sha256": graph_hash,
            },
        )
        clean_state = BASE.normalized_state(circuit)
        clean_probs = {
            basis: BASE.basis_probabilities(clean_state, n_qubits, basis)
            for basis in BASE.BASIS_ORDER
        }
        roots.append(
            {
                "domain": "original",
                "root_index": root_index,
                "global_generator_root_index": root_index,
                "family_occurrence_index": occurrence,
                "clean_circuit_group_id": group_id,
                "split": "validation",
                "step10c_partition": "fresh_outer_validation",
                "family": family,
                "n_qubits": n_qubits,
                "unitary_event_count": len(circuit.data),
                "graph_sha256": graph_hash,
                "physical_layout_identity": "identity:" + ",".join(str(i) for i in range(n_qubits)),
            }
        )
        clean_shots = step5v3.clean_control_shots(occurrence, v3cfg)
        examples.append(
            _save_clean_example(
                staging=staging,
                domain="original",
                root_index=root_index,
                group_id=group_id,
                graph=graph,
                graph_hash=graph_hash,
                family=family,
                family_occurrence=occurrence,
                n_qubits=n_qubits,
                clean_state=clean_state,
                clean_probs=clean_probs,
                shots=clean_shots,
                reference_kind=reference_kind,
                backend_identity="step10c_fresh_original_paired_reference_simulator",
                extra={"domain": "original", "global_generator_root_index": root_index},
                diagnostic_bound=diagnostic_bound,
            )
        )
        for ci, (fraction, strength) in enumerate(zip(depths, strengths)):
            boundary = BASE.depth_boundary(len(circuit.data), fraction)
            affected_qubit = int(q_schedule[ci])
            shots = int(shot_schedule[ci])
            for mechanism in BASE.MECHANISMS:
                examples.append(
                    _save_injected_example(
                        staging=staging,
                        domain="original",
                        root_index=root_index,
                        group_id=group_id,
                        graph=graph,
                        graph_hash=graph_hash,
                        family=family,
                        family_occurrence=occurrence,
                        clean_circuit=circuit,
                        clean_state=clean_state,
                        clean_probs=clean_probs,
                        n_qubits=n_qubits,
                        context_index=ci,
                        affected_qubit=affected_qubit,
                        boundary=boundary,
                        strength=float(strength),
                        shots=shots,
                        mechanism=mechanism,
                        reference_kind=reference_kind,
                        backend_identity="step10c_fresh_original_paired_reference_simulator",
                        v2cfg=v2cfg,
                        extra={"domain": "original", "global_generator_root_index": root_index},
                        diagnostic_bound=diagnostic_bound,
                    )
                )
        if progress_every and (local + 1) % progress_every == 0:
            print(f"Generated Step-10C original outer {local + 1}/{count} roots", flush=True)
    return roots, examples


def _generate_bridge(
    *,
    staging: Path,
    cfg: Mapping[str, Any],
    v2cfg: Mapping[str, Any],
    old_original_hashes: set[str],
    old_bridge_hashes: set[str],
    diagnostic_bound: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = cfg["bridge_domain"]
    bridge_cfg = {"bridge": spec}
    parent_count = int(spec["parent_groups"])
    variants = int(spec["variants_per_parent"])
    roots: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    fresh_hashes: set[str] = set()
    pilot_hash = step10a._pilot_graph_hash()
    reference_kind = str(v2cfg["finite_shot_acquisition"]["reference_kind"])
    family_occurrence: Counter[str] = Counter()
    local_root = 0

    for parent_index in range(parent_count):
        parent_id = _fresh_group_id(
            "bridge_parent",
            {"base_seed": spec["base_seed"], "parent_group_index": parent_index},
        )
        for variant_index in range(variants):
            circuit, anchor_boundary, motif, meta = step10a._build_bridge_circuit(
                parent_index, variant_index, bridge_cfg
            )
            family = f"step10c_bridge_{motif}"
            occurrence = int(family_occurrence[family])
            family_occurrence[family] += 1
            graph = BASE.serialize_graph(circuit)
            graph_hash = BASE.graph_hash(graph)
            if graph_hash == pilot_hash:
                raise RuntimeError("fresh Step-10C bridge generated exact Step-9D pilot graph")
            if graph_hash in old_original_hashes or graph_hash in old_bridge_hashes:
                raise RuntimeError(f"fresh bridge graph overlaps Step-10 history parent={parent_index} variant={variant_index}")
            if graph_hash in fresh_hashes:
                raise RuntimeError("duplicate fresh bridge graph")
            fresh_hashes.add(graph_hash)
            generation_root_index = 10_000_000 + local_root
            group_id = _fresh_group_id(
                "bridge",
                {
                    "parent_group_id": parent_id,
                    "variant_index": variant_index,
                    "graph_sha256": graph_hash,
                },
            )
            n_qubits = int(circuit.num_qubits)
            clean_state = BASE.normalized_state(circuit)
            clean_probs = {
                basis: BASE.basis_probabilities(clean_state, n_qubits, basis)
                for basis in BASE.BASIS_ORDER
            }
            extra = {
                "domain": "bridge",
                "parent_group_id": parent_id,
                "parent_group_index": parent_index,
                "variant_index": variant_index,
                "motif": motif,
                "phi": float(meta["phi"]),
                "beta": float(meta["beta"]),
            }
            roots.append(
                {
                    **extra,
                    "root_index": generation_root_index,
                    "family_occurrence_index": occurrence,
                    "clean_circuit_group_id": group_id,
                    "split": "validation",
                    "step10c_partition": "fresh_outer_validation",
                    "family": family,
                    "n_qubits": n_qubits,
                    "unitary_event_count": len(circuit.data),
                    "anchor_boundary": int(anchor_boundary),
                    "graph_sha256": graph_hash,
                    "physical_layout_identity": "identity:" + ",".join(str(i) for i in range(n_qubits)),
                }
            )
            clean_shots = int(spec["shot_levels"](parent_index + variant_index) % 4])
            examples.append(
                _save_clean_example(
                    staging=staging,
                    domain="bridge",
                    root_index=generation_root_index,
                    group_id=group_id,
                    graph=graph,
                    graph_hash=graph_hash,
                    family=family,
                    family_occurrence=occurrence,
                    n_qubits=n_qubits,
                    clean_state=clean_state,
                    clean_probs=clean_probs,
                    shots=clean_shots,
                    reference_kind=reference_kind,
                    backend_identity="step10c_fresh_bridge_paired_reference_simulator",
                    extra=extra,
                    diagnostic_bound=diagnostic_bound,
                )
            )
            for context in step10a._context_schedule(
                root_index=generation_root_index,
                circuit=circuit,
                anchor_boundary=anchor_boundary,
                cfg=bridge_cfg,
            ):
                ci = int(context["context_index"])
                affected_qubit = int(context["affected_qubit"])
                boundary = int(context["boundary"])
                strength = float(context["strength"])
                shots = int(context["shots"])
                for mechanism in BASE.MECHANISMS:
                    examples.append(
                        _save_injected_example(
                            staging=staging,
                            domain="bridge",
                            root_index=generation_root_index,
                          group_id=group_id,
                          graph=graph,
                          graph_hash=graph_hash,
                            family=family,
                            family_occurrence=occurrence,
                          clean_circuit=circuit,
                            clean_state=clean_state,
                          clean_probs=clean_probs,
                          n_qubits=n_qubits,
                          context_index=ci,
                            affected_qubit=affected_qubit,
                            boundary=boundary,
                          strength=strength,
                          shots=shots,
                          mechanism=mechanism,
                            reference_kind=reference_kind,
                            backend_identity="step10c_fresh_bridge_paired_reference_simulator",
                            v2cfg=v2cfg,
                          extra=extra,
                          diagnostic_bound=diagnostic_bound,
                        )
                    )
            local_root += 1
            if progress_every and local_root % progress_every == 0:
                print(
                    f"Generated Step-10C bridge outer {local_root}/{parent_count * variants} roots",
                    flush=True,
                )
    return roots, examples


def _effect_rates(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mechanism in BASE.MECHANISMS:
        subset = [row for row in examples if str(row["mechanism"]) == mechanism]
        effect = sum(_bool(row["effect_present"]) for row in subset)
        output[mechanism] = {
            "count": len(subset),
            "effect_positive": effect,
            "effect_positive_rate": effect / max(1, len(subset)),
        }
    return output


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        retur {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.1)),
        "p90": float(np.quantile(arr, 0.9)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }



def _scan_artifacts(
    product: Path,
    examples: Sequence[Mapping[str, Any]],
    diagnostic_bound: float,
) -> dict[str, Any]:
    hash_mismatch = 0
    nonfinite = 0
    bound_violations = 0
    manifest_target_mismatch = 0
    graph_hash_mismatch = 0
    forbidden_statevector_key_count = 0
    max_abs = 0.0
    schema_sets: Counter[tuple[str, ...]] = Counter()
    norm_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    exact_error_by_shots: dict[str, list[float]] = defaultdict(list)
    exact_error_by_domain_shots: dict[str, list[float]] = defaultdict(list)
    exact_error_by_domain_mechanism: dict[str, list[float]] = defaultdict(list)
    diagnostic_component_abs: dict[str, list[float]] = defaultdict(list)

    for row in examples:
        path = product / str(row["artifact_path"])
        if BASE.sha256_file(path) != str(row["artifact_sha256"]):
            hash_mismatch += 1
        with np.load(path, allow_pickle=False) as loaded:
            arrays = dict(loaded)
            schema_sets[tuple(sorted(arrays))] += 1
            BASE.validate_array_contract(arrays, diagnostic_bound)
            forbidden_statevector_key_count += sum("statevector" in key.lower() for key in arrays)

            # Manifest/target/identity cross-checks are done from bytes on disk.
            def scalar(name: str) -> Any:
                return np.asarray(arraysname].reshape(-1)[0]
            if str(scalar("meta__example_id")) != str(row["example_id"]):
                manifest_target_mismatch += 1
            if str(scalar("meta__clean_circuit_group_id")) != str(row["clean_circuit_group_id"]):
                manifest_target_mismatch += 1
            if bool(scalar