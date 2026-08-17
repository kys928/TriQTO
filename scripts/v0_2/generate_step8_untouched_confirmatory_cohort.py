#!/usr/bin/env python3
"""Generate and seal the untouched Step 8 confirmatory cohort.

The generator intentionally exposes no mechanism/effect/phenomenology summaries.
It reuses the accepted Step-5 v3 physics/acquisition recipe but shifts every
root-specific RNG stream into a new namespace and rejects any clean-graph
collision with the 5,000-root development product.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "generate_step5_matched_diagnostic_training_dataset_v3.py"
SPEC = importlib.util.spec_from_file_location("triqto_step8_step5v3", V3_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Step-5 v3 generator utilities from {V3_PATH}")
V3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V3
SPEC.loader.exec_module(V3)
V2 = V3.V2
BASE = V3.BASE

SCHEMA = "triqto.v0_2.step8_untouched_confirmatory.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step8_untouched_confirmatory.json"
DEFAULT_V3_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step5_matched_diagnostic_training_dataset_v3.json"
DEFAULT_V2_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--step5-v3-config", type=Path, default=DEFAULT_V3_CONFIG)
    parser.add_argument("--step5-v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    parser.add_argument("--development-product-dir", type=Path)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def confirmatory_plan(root_count: int, offset: int, v2cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    cycle_len = len(v2cfg["clean_circuit_generation"]["family_cycle"])
    if offset % cycle_len != 0:
        raise RuntimeError("confirmatory generation offset must preserve the frozen family cycle")
    output: list[dict[str, Any]] = []
    for local_index in range(root_count):
        generation_index = offset + local_index
        family = BASE.family_for_root(local_index, v2cfg)
        generation_family = BASE.family_for_root(generation_index, v2cfg)
        if family != generation_family:
            raise RuntimeError("generation offset changed the frozen family cycle")
        occurrence = V2.family_occurrence_index(local_index, family, v2cfg)
        output.append({
            "root_index": local_index,
            "generation_root_index": generation_index,
            "family": family,
            "family_occurrence_index": occurrence,
            "n_qubits": BASE.choose_n_qubits(generation_index, family, v2cfg),
            "strength_schedule": V2.strength_schedule_for_occurrence(occurrence, v2cfg),
        })
    return output


def verify_development_source(product: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], set[str], set[str]]:
    marker = product / "dataset_complete.json"
    if not marker.is_file():
        raise RuntimeError(f"missing development dataset marker: {marker}")
    complete = BASE.read_json(marker)
    expected = config["development_source"]
    if complete.get("product_id") != expected["product_id"]:
        raise RuntimeError("development product identity mismatch")
    roots = read_csv(product / "manifests" / "clean_circuit_manifest.csv")
    if len(roots) != int(expected["clean_roots"]):
        raise RuntimeError("development clean-root count mismatch")
    graph_hashes = {str(row["graph_sha256"]) for row in roots}
    group_ids = {str(row["clean_circuit_group_id"]) for row in roots}
    if len(graph_hashes) != len(roots) or len(group_ids) != len(roots):
        raise RuntimeError("development root identity is not unique")
    return complete, graph_hashes, group_ids


def public_example_row(*, example_id: str, root_index: int, generation_root_index: int, group_id: str, family: str, n_qubits: int, clean_control: bool, context_index: int, affected_qubit: int, boundary: int, depth_bin: str, strength: float, shots: int, graph_sha256: str, artifact_path: str, artifact_sha256: str, layout: str) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "root_index": root_index,
        "generation_root_index": generation_root_index,
        "clean_circuit_group_id": group_id,
        "split": "confirmatory",
        "family": family,
        "n_qubits": n_qubits,
        "clean_control": clean_control,
        "context_index": context_index,
        "affected_qubit": affected_qubit,
        "insertion_boundary_rank": boundary,
        "insertion_depth_bin": depth_bin,
        "strength": strength,
        "shots": shots,
        "graph_sha256": graph_sha256,
        "physical_layout_identity": layout,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = BASE.read_json(config_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_CONFIRMATORY_COHORT_GENERATION":
        raise RuntimeError("unexpected Step-8 config schema/status")
    v3_path = args.step5_v3_config.expanduser().resolve(); v3cfg = BASE.read_json(v3_path)
    v2_path = args.step5_v2_config.expanduser().resolve(); v2cfg = BASE.read_json(v2_path)
    if v3cfg.get("schema") != V3.SCHEMA or v2cfg.get("schema") != V2.SCHEMA:
        raise RuntimeError("unexpected frozen Step-5 source configuration")

    cohort_cfg = config["confirmatory_cohort"]
    root_count = int(cohort_cfg["clean_circuit_roots"])
    expected_examples = int(cohort_cfg["expected_examples"])
    offset = int(cohort_cfg["generation_root_index_offset"])
    if expected_examples != root_count * int(cohort_cfg["examples_per_root"]):
        raise RuntimeError("frozen Step-8 example count is inconsistent")

    development = args.development_product_dir.expanduser().resolve() if args.development_product_dir else Path(config["development_source"]["default_product_dir"]).expanduser().resolve()
    development_complete, development_graphs, development_groups = verify_development_source(development, config)
    output_parent = args.output_parent.expanduser().resolve() if args.output_parent else Path(cohort_cfg["default_output_parent"]).expanduser().resolve()

    identity = {
        "schema": SCHEMA,
        "config_sha256": BASE.sha256_file(config_path),
        "step5_v3_config_sha256": BASE.sha256_file(v3_path),
        "step5_v2_config_sha256": BASE.sha256_file(v2_path),
        "runner_sha256": BASE.sha256_file(Path(__file__).resolve()),
        "development_product_id": development_complete["product_id"],
        "clean_circuit_roots": root_count,
        "generation_root_index_offset": offset,
    }
    cohort_id = "confirm_" + hashlib.sha256(BASE.canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    output_parent.mkdir(parents=True, exist_ok=True)
    product_root = output_parent / cohort_id
    if product_root.exists():
        raise RuntimeError(f"refusing to overwrite existing confirmatory cohort: {product_root}")
    staging = output_parent / f".{cohort_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    roots: list[dict[str, Any]] = []
    public_examples: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    seen_graphs: set[str] = set(); seen_groups: set[str] = set()
    depths = [float(value) for value in v2cfg["matched_intervention_design"]["depth_target_fractions"]]
    privileged = v2cfg["privileged_supervision"]
    bound = float(v2cfg["stage_validation"]["diagnostic_delta_absolute_bound"])

    try:
        for planned in confirmatory_plan(root_count, offset, v2cfg):
            ri = int(planned["root_index"]); gi = int(planned["generation_root_index"])
            family = str(planned["family"]); occ = int(planned["family_occurrence_index"]); nq = int(planned["n_qubits"])
            strengths = [float(v) for v in planned["strength_schedule"]]
            q_schedule = V3.affected_qubit_schedule(gi, nq, v3cfg)
            shot_schedule = V3.intervention_shot_schedule(gi, v3cfg)
            circuit = BASE.build_clean_circuit(gi, family, nq, v2cfg)
            graph = BASE.serialize_graph(circuit); ghash = BASE.graph_hash(graph)
            if ghash in development_graphs:
                raise RuntimeError(f"confirmatory clean graph collides with development root {ri}")
            if ghash in seen_graphs:
                raise RuntimeError(f"duplicate confirmatory clean graph at root {ri}")
            seen_graphs.add(ghash)
            group_id = BASE.sha256_bytes(BASE.canonical_json({"confirmatory_root_index": ri, "generation_root_index": gi, "family": family, "family_occurrence_index": occ, "n_qubits": nq, "graph_sha256": ghash}).encode())
            if group_id in development_groups or group_id in seen_groups:
                raise RuntimeError("confirmatory clean-group identity collision")
            seen_groups.add(group_id)
            layout = "identity:" + ",".join(str(i) for i in range(nq))
            roots.append({"root_index": ri, "generation_root_index": gi, "family_occurrence_index": occ, "clean_circuit_group_id": group_id, "split": "confirmatory", "family": family, "n_qubits": nq, "unitary_event_count": len(circuit.data), "graph_sha256": ghash, "physical_layout_identity": layout})

            clean_state = BASE.normalized_state(circuit)
            clean_probs = {basis: BASE.basis_probabilities(clean_state, nq, basis) for basis in BASE.BASIS_ORDER}
            clean_shots = V3.clean_control_shots(occ, v3cfg)
            reference, pairs = BASE.make_reference_bundle(clean_probs, nq, clean_shots, gi, "step8_clean_control")
            diagnostic, audit = BASE.diagnostic_arrays(clean_state, clean_probs, reference, nq, pairs, clean_shots, gi, "step8_clean_control", "clean_control")
            example_id = BASE.sha256_bytes(BASE.canonical_json([group_id, "clean_control"]).encode())
            rel = Path("artifacts") / "confirmatory" / f"{example_id.split(':', 1)[1]}.npz"; path = staging / rel
            zero = {"population_component": 0.0, "phase_component": 0.0, "dominance_log_ratio": 0.0, "total_overlap_loss": 0.0}
            BASE.save_example(path, graph=graph, n_qubits=nq, diagnostic=diagnostic, audit_diagnostic=audit, example_id=example_id, clean_group_id=group_id, clean_control=True, effect_present=False, mechanism_code=-1, mechanism_loss_mask=False, phenotype="clean_control", continuous=zero, affected_qubit=-1, boundary=-1, strength=0.0)
            with np.load(path, allow_pickle=False) as loaded:
                BASE.validate_array_contract(dict(loaded), bound)
            public_examples.append(public_example_row(example_id=example_id, root_index=ri, generation_root_index=gi, group_id=group_id, family=family, n_qubits=nq, clean_control=True, context_index=-1, affected_qubit=-1, boundary=-1, depth_bin="clean_control", strength=0.0, shots=clean_shots, graph_sha256=ghash, artifact_path=rel.as_posix(), artifact_sha256=BASE.sha256_file(path), layout=layout))
            design_rows.append({"clean_control": True, "root_index": ri, "family": family, "n_qubits": nq, "split": "confirmatory", "shots": clean_shots, "strength": 0.0, "insertion_depth_bin": "clean_control", "affected_qubit": -1, "clean_circuit_group_id": group_id})

            for ci, (fraction, strength) in enumerate(zip(depths, strengths)):
                boundary = BASE.depth_boundary(len(circuit.data), fraction); aq = int(q_schedule[ci]); shots = int(shot_schedule[ci])
                depth_bin = BASE.depth_bin(boundary, len(circuit.data)); context = f"step8:root{ri}:ctx{ci}:q{aq}:b{boundary}:s{strength:.12g}:shots{shots}"
                reference, pairs = BASE.make_reference_bundle(clean_probs, nq, shots, gi, context)
                for mechanism in BASE.MECHANISMS:
                    observed = BASE.inject_hidden_rotation(circuit, boundary, aq, mechanism, strength)
                    observed_state = BASE.normalized_state(observed)
                    truth = BASE.state_diagnostics(clean_state, observed_state, epsilon=float(privileged["epsilon"]), negligible_floor=float(privileged["negligible_overlap_loss_floor"]), dominance_ratio=float(privileged["phenomenology_strong_dominance_ratio"]))
                    diagnostic, audit = BASE.diagnostic_arrays(observed_state, clean_probs, reference, nq, pairs, shots, gi, context, mechanism)
                    example_id = BASE.sha256_bytes(BASE.canonical_json([group_id, ci, aq, boundary, strength, shots, mechanism]).encode())
                    rel = Path("artifacts") / "confirmatory" / f"{example_id.split(':', 1)[1]}.npz"; path = staging / rel
                    BASE.save_example(path, graph=graph, n_qubits=nq, diagnostic=diagnostic, audit_diagnostic=audit, example_id=example_id, clean_group_id=group_id, clean_control=False, effect_present=bool(truth["effect_present"]), mechanism_code=BASE.MECHANISM_CODES[mechanism], mechanism_loss_mask=bool(truth["effect_present"]), phenotype=str(truth["phenotype"]), continuous=truth, affected_qubit=aq, boundary=boundary, strength=strength)
                    with np.load(path, allow_pickle=False) as loaded:
                        BASE.validate_array_contract(dict(loaded), bound)
                    public_examples.append(public_example_row(example_id=example_id, root_index=ri, generation_root_index=gi, group_id=group_id, family=family, n_qubits=nq, clean_control=False, context_index=ci, affected_qubit=aq, boundary=boundary, depth_bin=depth_bin, strength=strength, shots=shots, graph_sha256=ghash, artifact_path=rel.as_posix(), artifact_sha256=BASE.sha256_file(path), layout=layout))
                    design_rows.append({"clean_control": False, "root_index": ri, "family": family, "n_qubits": nq, "split": "confirmatory", "shots": shots, "strength": strength, "insertion_depth_bin": depth_bin, "affected_qubit": aq, "clean_circuit_group_id": group_id})
            if args.progress_every and (ri + 1) % args.progress_every == 0:
                print(f"Generated and verified {ri + 1}/{root_count} confirmatory roots", flush=True)

        if len(roots) != root_count or len(public_examples) != expected_examples:
            raise RuntimeError("confirmatory cohort count mismatch")
        by_root = Counter(int(row["root_index"]) for row in public_examples)
        if set(by_root.values()) != {int(cohort_cfg["examples_per_root"])}:
            raise RuntimeError("confirmatory derivative count mismatch")
        design_metrics = V3.validate_v3_associations(roots, design_rows, v3cfg)
        design_metrics["depth_strength_cramers_v"] = V2.cramers_v([row for row in design_rows if not bool(row["clean_control"])], "insertion_depth_bin", "strength")
        design_metrics["family_root_counts"] = dict(Counter(str(row["family"]) for row in roots))
        design_metrics["qubit_root_counts"] = dict(Counter(str(row["n_qubits"]) for row in roots))
        design_metrics["development_graph_overlap_count"] = 0
        design_metrics["development_group_overlap_count"] = 0
        design_metrics["confirmatory_unique_graph_count"] = len(seen_graphs)
        design_metrics["target_summaries_exposed"] = False

        # Shuffle the public manifest so row order cannot reveal the generator's
        # deterministic RZ/RX/RY loop ordering.
        rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
        order = rng.permutation(len(public_examples))
        public_examples = [public_examples[int(i)] for i in order]
        manifests = staging / "manifests"
        BASE.write_csv(manifests / "clean_circuit_manifest.csv", roots)
        BASE.write_csv(manifests / "example_manifest.csv", public_examples)
        BASE.atomic_json(staging / "design_validation.json", design_metrics)
        manifest_hashes = {path.name: BASE.sha256_file(path) for path in manifests.iterdir() if path.is_file()}
        sealed = {
            "schema": SCHEMA,
            "status": "SEALED_UNEVALUATED",
            "cohort_id": cohort_id,
            "identity": identity,
            "development_product_id": development_complete["product_id"],
            "clean_circuit_root_count": len(roots),
            "example_count": len(public_examples),
            "generation_root_index_offset": offset,
            "development_graph_overlap_count": 0,
            "development_group_overlap_count": 0,
            "target_summaries_exposed": False,
            "confirmatory_labels_inspected_by_evaluator": False,
            "model_evaluated": False,
            "historical_v0_1_test_accessed": False,
            "spent_phase15_6_confirmatory_accessed": False,
            "manifest_hashes": manifest_hashes,
            "design_validation_sha256": BASE.sha256_file(staging / "design_validation.json"),
        }
        BASE.atomic_json(staging / "sealed_complete.json", sealed)
        os.replace(staging, product_root)
        BASE.atomic_json(output_parent / "current_sealed_cohort.json", {"schema": SCHEMA, "cohort_id": cohort_id, "cohort_dir": str(product_root), "sealed_complete_sha256": BASE.sha256_file(product_root / "sealed_complete.json")})
        print("\nTRIQTO STEP 8 CONFIRMATORY COHORT SEALED\n")
        print(f"Status: SEALED_UNEVALUATED\nCohort: {cohort_id}\nClean roots: {len(roots)}\nExamples: {len(public_examples)}")
        print("Development graph overlap: 0\nTarget summaries exposed: NO\nModel evaluated: NO")
        print(f"Sealed cohort: {product_root}")
        print("Do not inspect y__ arrays or derive target summaries before the one-shot evaluator.")
    except Exception:
        import shutil
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
