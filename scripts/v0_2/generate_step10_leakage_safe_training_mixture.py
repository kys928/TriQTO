#!/usr/bin/env python3
"""Generate Step 10 leakage-safe original+deployment-domain training mixture.

The original Step-5 v3 product is referenced immutably and never rewritten.
This script materializes only the new simulator bridge product plus manifests
that freeze how the two domains are combined for Step 10B.
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
from typing import Any, Mapping

import numpy as np
from qiskit import QuantumCircuit

import benchmark_step6_cheap_baselines as baseline
import generate_step5_matched_diagnostic_training_dataset_v3 as step5v3

BASE = step5v3.BASE

SCHEMA = "triqto.v0_2.step10_training_mixture.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10_training_mixture.json"
DEFAULT_V2_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step10_training_mixture")

PARTITION_BY_FOLD = {
    0: "outer_validation",
    1: "fit",
    2: "fit",
    3: "fit",
    4: "selection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _stable_seed(*parts: Any) -> int:
    payload = BASE.canonical_json([str(part) for part in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _wrap_delta(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _shift_from_pilot(phi: float, pilot: float, half_width: float) -> float:
    if abs(_wrap_delta(phi, pilot)) >= half_width:
        return phi
    shifted = phi + (2.5 * half_width if _wrap_delta(phi, pilot) >= 0.0 else -2.5 * half_width)
    return float((shifted + math.pi) % (2.0 * math.pi) - math.pi)


def _parent_parameters(parent_index: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    bridge = cfg["bridge"]
    rng = np.random.default_rng(_stable_seed(bridge["base_seed"], "parent", parent_index))
    phi_lo, phi_hi = [float(v) for v in bridge["parent_phi_range"]]
    beta_lo, beta_hi = [float(v) for v in bridge["parent_beta_range"]]
    phi = float(rng.uniform(phi_lo, phi_hi))
    phi = _shift_from_pilot(phi, float(bridge["pilot_phi"]), float(bridge["pilot_exclusion_half_width"]))
    beta = float(rng.uniform(beta_lo, beta_hi))
    gamma = float(rng.uniform(-1.0, 1.0))
    delta = float(rng.uniform(-0.8, 0.8))
    qubits = [int(v) for v in bridge["qubit_choices"]]
    n_qubits = qubits[parent_index % len(qubits)]
    return {
        "phi_center": phi,
        "beta_center": beta,
        "gamma": gamma,
        "delta": delta,
        "n_qubits": n_qubits,
    }


def _build_bridge_circuit(
    parent_index: int,
    variant_index: int,
    cfg: Mapping[str, Any],
) -> tuple[QuantumCircuit, int, str, dict[str, Any]]:
    bridge = cfg["bridge"]
    parent = _parent_parameters(parent_index, cfg)
    offsets = [float(v) for v in bridge["variant_phi_offsets"]]
    motifs = [str(v) for v in bridge["motifs"]]
    if variant_index < 0 or variant_index >= len(offsets):
        raise ValueError("variant index outside frozen offset schedule")

    phi = _shift_from_pilot(
        float(parent["phi_center"]) + offsets[variant_index],
        float(bridge["pilot_phi"]),
        float(bridge["pilot_exclusion_half_width"]),
    )
    beta = float(parent["beta_center"]) + 0.025 * float((variant_index % 4) - 1.5)
    gamma = float(parent["gamma"]) + 0.02 * float((variant_index % 3) - 1)
    delta = float(parent["delta"]) - 0.015 * float((variant_index % 5) - 2)
    n_qubits = int(parent["n_qubits"])
    motif = motifs[variant_index % len(motifs)]

    qc = QuantumCircuit(n_qubits, name=f"step10_bridge_p{parent_index}_v{variant_index}_{motif}")

    if n_qubits >= 3 and variant_index % 2 == 1:
        qc.ry(gamma, 2)
    if n_qubits >= 4 and variant_index % 3 == 0:
        qc.rz(delta, 3)

    if motif == "pilot_core_variant":
        qc.h(0)
        qc.rz(phi, 0)
        anchor_boundary = len(qc.data)
        qc.h(0)
        qc.cx(0, 1)
    elif motif == "spectator_pre":
        qc.ry(beta, 1)
        qc.h(0)
        qc.rz(phi, 0)
        anchor_boundary = len(qc.data)
        qc.h(0)
        qc.cx(0, 1)
    elif motif == "spectator_mid":
        qc.h(0)
        qc.rz(phi, 0)
        anchor_boundary = len(qc.data)
        qc.rz(beta, 1)
        qc.h(0)
        qc.cx(0, 1)
    elif motif == "spectator_tail":
        qc.h(0)
        qc.rz(phi, 0)
        anchor_boundary = len(qc.data)
        qc.h(0)
        qc.ry(beta, 1)
        qc.cx(0, 1)
    else:
        raise RuntimeError(f"unsupported bridge motif {motif!r}")

    if n_qubits >= 3:
        if variant_index % 2 == 0:
            qc.cx(1, 2)
        else:
            qc.cz(1, 2)
        qc.rz(0.5 * gamma, 2)
    if n_qubits >= 4:
        if variant_index % 2 == 0:
            qc.cz(2, 3)
        else:
            qc.cx(2, 3)
        qc.ry(0.5 * delta, 3)

    meta = {
        "phi": phi,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "n_qubits": n_qubits,
    }
    return qc, int(anchor_boundary), motif, meta


def _context_schedule(
    *,
    root_index: int,
    circuit: QuantumCircuit,
    anchor_boundary: int,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bridge = cfg["bridge"]
    strengths = [float(v) for v in bridge["context_strengths"]]
    shots = [int(v) for v in bridge["shot_levels"]]
    if len(strengths) != 4 or len(shots) != 4:
        raise RuntimeError("Step 10 frozen bridge requires four strengths and four shot levels")

    rng = np.random.default_rng(_stable_seed(bridge["base_seed"], "context_schedule", root_index))
    shots_perm = [int(v) for v in rng.permutation(shots).tolist()]

    nq = int(circuit.num_qubits)
    spectator = 2 if nq >= 3 else 1
    qubits = [0, 0, 1, spectator]
    boundaries = [
        anchor_boundary,
        max(1, anchor_boundary - 1),
        min(len(circuit.data), anchor_boundary + 1),
        anchor_boundary,
    ]
    return [
        {
            "context_index": i,
            "affected_qubit": int(qubits[i]),
            "boundary": int(boundaries[i]),
            "strength": float(strengths[i]),
            "shots": int(shots_perm[i]),
        }
        for i in range(4)
    ]


def _pilot_graph_hash() -> str:
    qc = QuantumCircuit(2, name="frozen_step9d_pilot_reference")
    qc.h(0)
    qc.rz(0.7, 0)
    qc.h(0)
    qc.cx(0, 1)
    return BASE.graph_hash(BASE.serialize_graph(qc))


def _verify_original_product(
    config: Mapping[str, Any],
    step7_config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    original = config["original_domain"]
    product = Path(original["default_product_dir"]).expanduser().resolve()
    complete, rows = baseline.verify_source_product(product, step7_config)
    if complete["product_id"] != original["product_id"]:
        raise RuntimeError("Step 10 original-domain product identity mismatch")
    if int(complete["clean_circuit_root_count"]) != int(original["clean_root_count"]):
        raise RuntimeError("Step 10 original-domain root count mismatch")
    if int(complete["example_count"]) != int(original["example_count"]):
        raise RuntimeError("Step 10 original-domain example count mismatch")
    return product, complete, rows


def _validate_bridge(
    roots: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    pilot_graph_hash: str,
) -> dict[str, Any]:
    bridge = cfg["bridge"]
    expected_roots = int(bridge["expected_clean_roots"])
    expected_examples = int(bridge["expected_examples"])
    variants = int(bridge["variants_per_parent"])
    parents = int(bridge["parent_groups"])
    if len(roots) != expected_roots or len(examples) != expected_examples:
        raise RuntimeError(
            f"bridge count mismatch roots/examples={len(roots)}/{len(examples)} "
            f"expected={expected_roots}/{expected_examples}"
        )

    roots_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_root: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        roots_by_parent[str(root["parent_group_id"])].append(root)
        if root["graph_sha256"] == pilot_graph_hash:
            raise RuntimeError("exact frozen Step-9D pilot graph leaked into Step-10 bridge")
    for row in examples:
        examples_by_root[int(row["root_index"])].append(row)

    if len(roots_by_parent) != parents:
        raise RuntimeError("parent-group count mismatch")
    for parent_id, local in roots_by_parent.items():
        if len(local) != variants:
            raise RuntimeError(f"parent {parent_id} has {len(local)} variants instead of {variants}")
        partitions = {str(row["step10_partition"]) for row in local}
        if len(partitions) != 1:
            raise RuntimeError(f"parent {parent_id} crosses Step-10 partitions")
        splits = {str(row["split"]) for row in local}
        if len(splits) != 1:
            raise RuntimeError(f"parent {parent_id} crosses source train/validation split")

    mechanisms = tuple(str(v) for v in bridge["mechanisms"])
    for root_index, local in examples_by_root.items():
        if len(local) != int(bridge["examples_per_root"]):
            raise RuntimeError(f"root {root_index} has {len(local)} examples")
        partitions = {str(row["step10_partition"]) for row in local}
        if len(partitions) != 1:
            raise RuntimeError(f"root {root_index} crosses Step-10 partitions")
        clean = [row for row in local if str(row["mechanism"]) == "clean_control"]
        injected = [row for row in local if str(row["mechanism"]) != "clean_control"]
        if len(clean) != 1 or len(injected) != 12:
            raise RuntimeError(f"root {root_index} does not have 1 clean + 12 injected examples")
        counts = Counter(str(row["mechanism"]) for row in injected)
        if any(counts.get(mech, 0) != 4 for mech in mechanisms):
            raise RuntimeError(f"root {root_index} mechanism balance failed: {dict(counts)}")

    partition_counts = Counter(str(row["step10_partition"]) for row in roots)
    expected_partition_counts = {
        "fit": int(bridge["parent_split_rule"]["expected_fit_roots"]),
        "selection": int(bridge["parent_split_rule"]["expected_selection_roots"]),
        "outer_validation": int(bridge["parent_split_rule"]["expected_outer_validation_roots"]),
    }
    if dict(partition_counts) != expected_partition_counts:
        raise RuntimeError(
            f"bridge partition counts {dict(partition_counts)} != {expected_partition_counts}"
        )

    graph_hashes = [str(row["graph_sha256"]) for row in roots]
    if len(set(graph_hashes)) != len(graph_hashes):
        raise RuntimeError("duplicate bridge clean graph detected")

    family_partition: dict[str, Counter[str]] = defaultdict(Counter)
    for root in roots:
        family_partition[str(root["family"])][str(root["step10_partition"])] += 1

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "bridge_clean_root_count": len(roots),
        "bridge_example_count": len(examples),
        "partition_root_counts": dict(sorted(partition_counts.items())),
        "parent_group_count": len(roots_by_parent),
        "variants_per_parent": variants,
        "all_parent_groups_single_partition": True,
        "all_root_derivatives_single_partition": True,
        "exact_step9d_pilot_graph_present": False,
        "duplicate_bridge_graph_count": 0,
        "family_partition_root_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_partition.items())
        },
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    v2_path = args.base_v2_config.expanduser().resolve()
    step7_path = args.step7_config.expanduser().resolve()
    cfg = BASE.read_json(config_path)
    v2cfg = BASE.read_json(v2_path)
    step7cfg = BASE.read_json(step7_path)
    if cfg.get("schema") != SCHEMA or cfg.get("status") != "FROZEN_BEFORE_STEP10_DATASET_OUTCOME":
        raise RuntimeError("unexpected Step-10 mixture config schema/status")

    original_product, original_complete, _original_rows = _verify_original_product(cfg, step7cfg)
    bridge = cfg["bridge"]
    parent_count = int(bridge["parent_groups"])
    variants_per_parent = int(bridge["variants_per_parent"])
    root_count = parent_count * variants_per_parent
    if root_count != int(bridge["expected_clean_roots"]):
        raise RuntimeError("frozen Step-10 parent/variant count does not match expected roots")

    identity = {
        "schema": SCHEMA,
        "config_sha256": BASE.sha256_file(config_path),
        "base_v2_config_sha256": BASE.sha256_file(v2_path),
        "step7_config_sha256": BASE.sha256_file(step7_path),
        "runner_sha256": BASE.sha256_file(Path(__file__).resolve()),
        "original_product_id": original_complete["product_id"],
        "original_dataset_complete_sha256": BASE.sha256_file(original_product / "dataset_complete.json"),
        "bridge_parent_groups": parent_count,
        "bridge_variants_per_parent": variants_per_parent,
    }
    product_id = "product_" + hashlib.sha256(
        BASE.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]

    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    product_root = output_parent / product_id
    if product_root.exists():
        raise RuntimeError(f"refusing to overwrite existing Step-10 product: {product_root}")
    staging = output_parent / f".{product_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    roots: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    seen_graph_hashes: set[str] = set()
    pilot_hash = _pilot_graph_hash()
    p = v2cfg["privileged_supervision"]
    bound = float(v2cfg["stage_validation"]["diagnostic_delta_absolute_bound"])
    family_occurrence: Counter[str] = Counter()

    try:
        root_index = 0
        for parent_index in range(parent_count):
            fold = parent_index % 5
            partition = PARTITION_BY_FOLD[fold]
            source_split = "validation" if partition == "outer_validation" else "train"
            parent_id = BASE.sha256_bytes(
                BASE.canonical_json(
                    {
                        "step": "step10",
                        "parent_group_index": parent_index,
                        "fold": fold,
                        "seed": bridge["base_seed"],
                    }
                ).encode("utf-8")
            )
            for variant_index in range(variants_per_parent):
                circuit, anchor_boundary, motif, meta = _build_bridge_circuit(
                    parent_index, variant_index, cfg
                )
                family = f"step10_bridge_{motif}"
                occurrence = int(family_occurrence[family])
                family_occurrence[family] += 1
                graph = BASE.serialize_graph(circuit)
                ghash = BASE.graph_hash(graph)
                if ghash == pilot_hash:
                    raise RuntimeError("exact Step-9D pilot graph generated")
                if ghash in seen_graph_hashes:
                    raise RuntimeError(
                        f"duplicate bridge graph generated at root {root_index}"
                    )
                seen_graph_hashes.add(ghash)
                group_id = BASE.sha256_bytes(
                    BASE.canonical_json(
                        {
                            "root_index": root_index,
                            "parent_group_id": parent_id,
                            "variant_index": variant_index,
                            "family": family,
                            "graph_sha256": ghash,
                        }
                    ).encode("utf-8")
                )
                nq = int(circuit.num_qubits)
                clean_state = BASE.normalized_state(circuit)
                clean_probs = {
                    basis: BASE.basis_probabilities(clean_state, nq, basis)
                    for basis in BASE.BASIS_ORDER
                }
                layout = "identity:" + ",".join(str(i) for i in range(nq))
                roots.append(
                    {
                        "root_index": root_index,
                        "family_occurrence_index": occurrence,
                        "clean_circuit_group_id": group_id,
                        "parent_group_id": parent_id,
                        "parent_group_index": parent_index,
                        "variant_index": variant_index,
                        "parent_fold": fold,
                        "step10_partition": partition,
                        "split": source_split,
                        "family": family,
                        "motif": motif,
                        "n_qubits": nq,
                        "unitary_event_count": len(circuit.data),
                        "anchor_boundary": anchor_boundary,
                        "phi": float(meta["phi"]),
                        "beta": float(meta["beta"]),
                        "graph_sha256": ghash,
                        "physical_layout_identity": layout,
                    }
                )

                context_rows = _context_schedule(
                    root_index=root_index,
                    circuit=circuit,
                    anchor_boundary=anchor_boundary,
                    cfg=cfg,
                )
                clean_shots = int(bridge["shot_levels"][(parent_index + variant_index) % 4])
                ref, pairs = BASE.make_reference_bundle(
                    clean_probs, nq, clean_shots, root_index, "clean_control"
                )
                diagnostic, audit = BASE.diagnostic_arrays(
                    clean_state,
                    clean_probs,
                    ref,
                    nq,
                    pairs,
                    clean_shots,
                    root_index,
                    "clean_control",
                    "clean_control",
                )
                example_id = BASE.sha256_bytes(
                    BASE.canonical_json([group_id, "clean_control"]).encode("utf-8")
                )
                rel = (
                    Path("bridge_artifacts")
                    / partition
                    / f"{example_id.split(':', 1)[1]}.npz"
                )
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
                    n_qubits=nq,
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
                    BASE.validate_array_contract(dict(loaded), bound)
                examples.append(
                    {
                        "example_id": example_id,
                        "root_index": root_index,
                        "family_occurrence_index": occurrence,
                        "clean_circuit_group_id": group_id,
                        "parent_group_id": parent_id,
                        "parent_group_index": parent_index,
                        "variant_index": variant_index,
                        "parent_fold": fold,
                        "step10_partition": partition,
                        "split": source_split,
                        "family": family,
                        "n_qubits": nq,
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
                        "reference_kind": v2cfg["finite_shot_acquisition"]["reference_kind"],
                        "backend_identity": "step10_bridge_paired_reference_simulator",
                        "physical_layout_identity": layout,
                        "meta_reference_window_id": f"step10:root{root_index}:clean",
                        "graph_sha256": ghash,
                        "artifact_path": rel.as_posix(),
                        "artifact_sha256": BASE.sha256_file(path),
                    }
                )

                for context in context_rows:
                    ci = int(context["context_index"])
                    boundary = int(context["boundary"])
                    aq = int(context["affected_qubit"])
                    strength = float(context["strength"])
                    shots = int(context["shots"])
                    context_name = (
                        f"ctx{ci}:q{aq}:b{boundary}:s{strength:.12g}:shots{shots}"
                    )
                    ref, pairs = BASE.make_reference_bundle(
                        clean_probs, nq, shots, root_index, context_name
                    )
                    for mechanism in BASE.MECHANISMS:
                        observed = BASE.inject_hidden_rotation(
                            circuit, boundary, aq, mechanism, strength
                        )
                        observed_state = BASE.normalized_state(observed)
                        truth = BASE.state_diagnostics(
                            clean_state,
                            observed_state,
                            epsilon=float(p["epsilon"]),
                            negligible_floor=float(
                                p["negligible_overlap_loss_floor"]
                            ),
                            dominance_ratio=float(
                                p["phenomenology_strong_dominance_ratio"]
                            ),
                        )
                        diagnostic, audit = BASE.diagnostic_arrays(
                            observed_state,
                            clean_probs,
                            ref,
                            nq,
                            pairs,
                            shots,
                            root_index,
                            context_name,
                            mechanism,
                        )
                        example_id = BASE.sha256_bytes(
                            BASE.canonical_json(
                                [
                                    group_id,
                                    ci,
                                    aq,
                                    boundary,
                                    strength,
                                    shots,
                                    mechanism,
                                ]
                            ).encode("utf-8")
                        )
                        rel = (
                            Path("bridge_artifacts")
                            / partition
                            / f"{example_id.split(':', 1)[1]}.npz"
                        )
                        path = staging / rel
                        BASE.save_example(
                            path,
                            graph=graph,
                            n_qubits=nq,
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
                            affected_qubit=aq,
                            boundary=boundary,
                            strength=strength,
                        )
                        with np.load(path, allow_pickle=False) as loaded:
                            BASE.validate_array_contract(dict(loaded), bound)
                        examples.append(
                            {
                                "example_id": example_id,
                                "root_index": root_index,
                                "family_occurrence_index": occurrence,
                                "clean_circuit_group_id": group_id,
                                "parent_group_id": parent_id,
                                "parent_group_index": parent_index,
                                "variant_index": variant_index,
                                "parent_fold": fold,
                                "step10_partition": partition,
                                "split": source_split,
                                "family": family,
                                "n_qubits": nq,
                                "clean_control": False,
                                "mechanism": mechanism,
                                "effect_present": bool(truth["effect_present"]),
                                "mechanism_loss_mask": bool(
                                    truth["effect_present"]
                                ),
                                "phenomenology": str(truth["phenotype"]),
                                "affected_qubit": aq,
                                "insertion_boundary_rank": boundary,
                                "insertion_depth_bin": BASE.depth_bin(
                                    boundary, len(circuit.data)
                                ),
                                "strength": strength,
                                "shots": shots,
                                "reference_kind": v2cfg[
                                    "finite_shot_acquisition"
                                ]["reference_kind"],
                                "backend_identity": "step10_bridge_paired_reference_simulator",
                                "physical_layout_identity": layout,
                                "meta_reference_window_id": (
                                    f"step10:root{root_index}:ctx{ci}"
                                ),
                                "graph_sha256": ghash,
                                "artifact_path": rel.as_posix(),
                                "artifact_sha256": BASE.sha256_file(path),
                            }
                        )

                root_index += 1
                if (
                    args.progress_every
                    and root_index % int(args.progress_every) == 0
                ):
                    print(
                        f"Generated Step-10 bridge {root_index}/{root_count} roots "
                        f"({len(examples)} examples)",
                        flush=True,
                    )

        validation = _validate_bridge(roots, examples, cfg, pilot_hash)
        manifests = staging / "manifests"
        BASE.write_csv(manifests / "bridge_root_manifest.csv", roots)
        BASE.write_csv(manifests / "bridge_example_manifest.csv", examples)
        BASE.atomic_json(staging / "stage_validation.json", validation)
        original_ref = {
            "schema": "triqto.v0_2.step10_original_domain_reference.v1",
            "product_id": original_complete["product_id"],
            "product_dir": str(original_product),
            "dataset_complete_sha256": BASE.sha256_file(
                original_product / "dataset_complete.json"
            ),
            "manifest_hashes": original_complete.get("manifest_hashes", {}),
            "regenerated": False,
            "relabeled": False,
        }
        BASE.atomic_json(staging / "original_domain_reference.json", original_ref)

        manifest_names = sorted(
            p.name for p in manifests.iterdir() if p.is_file()
        )
        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "product_id": product_id,
            "identity": identity,
            "original_domain": {
                "product_id": original_complete["product_id"],
                "clean_root_count": int(
                    original_complete["clean_circuit_root_count"]
                ),
                "example_count": int(original_complete["example_count"]),
            },
            "bridge_clean_root_count": len(roots),
            "bridge_example_count": len(examples),
            "bridge_fit_root_count": validation["partition_root_counts"]["fit"],
            "bridge_selection_root_count": validation[
                "partition_root_counts"
            ]["selection"],
            "bridge_outer_validation_root_count": validation[
                "partition_root_counts"
            ]["outer_validation"],
            "combined_clean_root_count": int(
                original_complete["clean_circuit_root_count"]
            )
            + len(roots),
            "combined_example_count": int(original_complete["example_count"])
            + len(examples),
            "exact_step9d_pilot_graph_present": False,
            "parent_group_split_unit": True,
            "all_parent_variants_same_partition": True,
            "all_root_derivatives_same_partition": True,
            "statevectors_persisted_in_bridge_artifacts": False,
            "classifier_trained": False,
            "qpu_executed": False,
            "architecture_changed": False,
            "manifest_hashes": {
                name: BASE.sha256_file(manifests / name)
                for name in manifest_names
            },
            "stage_validation_sha256": BASE.sha256_file(
                staging / "stage_validation.json"
            ),
            "original_domain_reference_sha256": BASE.sha256_file(
                staging / "original_domain_reference.json"
            ),
        }
        BASE.atomic_json(staging / "dataset_complete.json", completion)
        os.replace(staging, product_root)
        BASE.atomic_json(
            output_parent / "current_product.json",
            {
                "schema": "triqto.v0_2.step10_current_product.v1",
                "product_id": product_id,
                "product_dir": str(product_root),
                "dataset_complete_sha256": BASE.sha256_file(
                    product_root / "dataset_complete.json"
                ),
            },
        )

        print("\nTRIQTO STEP 10A LEAKAGE-SAFE TRAINING MIXTURE COMPLETE\n")
        print(f"Product: {product_id}")
        print(
            "Original domain (referenced unchanged): "
            f"{original_complete['clean_circuit_root_count']} roots / "
            f"{original_complete['example_count']} examples"
        )
        print(
            "Bridge domain: "
            f"{len(roots)} roots / {len(examples)} examples"
        )
        print(
            "Bridge fit/selection/outer roots: "
            f"{validation['partition_root_counts']['fit']}/"
            f"{validation['partition_root_counts']['selection']}/"
            f"{validation['partition_root_counts']['outer_validation']}"
        )
        print("Parent-group leakage gate: PASS")
        print("Exact frozen Step-9D pilot graph present: NO")
        print("QPU executed: NO")
        print(f"Output: {product_root}")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
