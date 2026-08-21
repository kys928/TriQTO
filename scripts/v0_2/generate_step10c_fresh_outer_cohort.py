#!/usr/bin/env python3
"""Generate and exhaustively EDA-audit the frozen fresh Step-10C outer cohort.

This is validation-only data.  It creates:
  * 1,000 fresh Step-5-v3-family original-domain roots using global generator
    root indices 5000..5999; and
  * 60 fresh bridge parent groups x 8 variants from an independent frozen seed.

The spent Step-10B outer rows are never copied, reshuffled, materialized, or
used as model inputs here.  Historical Step-10 manifests are read only to prove
zero clean-graph overlap.
"""
from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--eda-progress-every", type=int, default=2000)
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"cannot parse boolean {value!r}")


def _fresh_group_id(domain: str, payload: Mapping[str, Any]) -> str:
    return BASE.sha256_bytes(
        BASE.canonical_json({"step": "step10c", "domain": domain, **dict(payload)}).encode("utf-8")
    )


def _artifact_rel(domain: str, example_id: str) -> Path:
    return Path(f"{domain}_artifacts") / "outer_validation" / f"{example_id.split(':', 1)[1]}.npz"


def _save_clean_example(
    *, staging: Path, domain: str, root_index: int, group_id: str,
    graph: Mapping[str, np.ndarray], graph_hash: str, family: str,
    family_occurrence: int, n_qubits: int, clean_state: np.ndarray,
    clean_probs: Mapping[str, np.ndarray], shots: int, reference_kind: str,
    backend_identity: str, extra: Mapping[str, Any], diagnostic_bound: float,
) -> dict[str, Any]:
    ref, pairs = BASE.make_reference_bundle(clean_probs, n_qubits, shots, root_index, "clean_control")
    diagnostic, audit = BASE.diagnostic_arrays(
        clean_state, clean_probs, ref, n_qubits, pairs, shots,
        root_index, "clean_control", "clean_control",
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
        path, graph=graph, n_qubits=n_qubits, diagnostic=diagnostic,
        audit_diagnostic=audit, example_id=example_id, clean_group_id=group_id,
        clean_control=True, effect_present=False, mechanism_code=-1,
        mechanism_loss_mask=False, phenotype="clean_control", continuous=zero,
        affected_qubit=-1, boundary=-1, strength=0.0,
    )
    with np.load(path, allow_pickle=False) as loaded:
        BASE.validate_array_contract(dict(loaded), diagnostic_bound)
    row: dict[str, Any] = {
        "domain": domain,
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
    *, staging: Path, domain: str, root_index: int, group_id: str,
    graph: Mapping[str, np.ndarray], graph_hash: str, family: str,
    family_occurrence: int, clean_circuit: Any, clean_state: np.ndarray,
    clean_probs: Mapping[str, np.ndarray], n_qubits: int, context_index: int,
    affected_qubit: int, boundary: int, strength: float, shots: int,
    mechanism: str, reference_kind: str, backend_identity: str,
    v2cfg: Mapping[str, Any], extra: Mapping[str, Any], diagnostic_bound: float,
) -> dict[str, Any]:
    context = f"ctx{context_index}:q{affected_qubit}:b{boundary}:s{strength:.12g}:shots{shots}"
    ref, pairs = BASE.make_reference_bundle(clean_probs, n_qubits, shots, root_index, context)
    observed = BASE.inject_hidden_rotation(clean_circuit, boundary, affected_qubit, mechanism, strength)
    observed_state = BASE.normalized_state(observed)
    privileged = v2cfg["privileged_supervision"]
    truth = BASE.state_diagnostics(
        clean_state, observed_state,
        epsilon=float(privileged["epsilon"]),
        negligible_floor=float(privileged["negligible_overlap_loss_floor"]),
        dominance_ratio=float(privileged["phenomenology_strong_dominance_ratio"]),
    )
    diagnostic, audit = BASE.diagnostic_arrays(
        observed_state, clean_probs, ref, n_qubits, pairs, shots,
        root_index, context, mechanism,
    )
    example_id = BASE.sha256_bytes(
        BASE.canonical_json(
            [group_id, context_index, affected_qubit, boundary, strength, shots, mechanism]
        ).encode("utf-8")
    )
    rel = _artifact_rel(domain, example_id)
    path = staging / rel
    BASE.save_example(
        path, graph=graph, n_qubits=n_qubits, diagnostic=diagnostic,
        audit_diagnostic=audit, example_id=example_id, clean_group_id=group_id,
        clean_control=False, effect_present=bool(truth["effect_present"]),
        mechanism_code=BASE.MECHANISM_CODES[mechanism],
        mechanism_loss_mask=bool(truth["effect_present"]),
        phenotype=str(truth["phenotype"]), continuous=truth,
        affected_qubit=affected_qubit, boundary=boundary, strength=strength,
    )
    with np.load(path, allow_pickle=False) as loaded:
        BASE.validate_array_contract(dict(loaded), diagnostic_bound)
    row: dict[str, Any] = {
        "domain": domain,
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


def _verify_historical_products(
    step10_product: Path,
    step10_cfg: Mapping[str, Any],
    step7_cfg: Mapping[str, Any],
) -> tuple[set[str], set[str], dict[str, Any]]:
    original_product, original_complete, original_rows = step10a._verify_original_product(step10_cfg, step7_cfg)
    original_hashes = {str(row["graph_sha256"]) for row in original_rows}
    complete_path = step10_product / "dataset_complete.json"
    complete = BASE.read_json(complete_path)
    if complete.get("product_id") != "product_0f7112597501f7ea5fbe123b":
        raise RuntimeError("unexpected Step-10 source product for Step-10C outer generation")
    if complete.get("schema") != step10a.SCHEMA or complete.get("status") != "COMPLETE":
        raise RuntimeError("Step-10 source product is incomplete or wrong schema")
    manifests = step10_product / "manifests"
    for name, expected_hash in complete.get("manifest_hashes", {}).items():
        if BASE.sha256_file(manifests / name) != expected_hash:
            raise RuntimeError(f"Step-10 source manifest hash mismatch: {name}")
    if BASE.sha256_file(step10_product / "stage_validation.json") != complete.get("stage_validation_sha256"):
        raise RuntimeError("Step-10 source stage validation hash mismatch")
    bridge_root_rows = baseline.read_csv(manifests / "bridge_root_manifest.csv")
    bridge_hashes = {str(row["graph_sha256"]) for row in bridge_root_rows}
    if len(bridge_hashes) != int(complete["bridge_clean_root_count"]):
        raise RuntimeError("Step-10 source bridge graph uniqueness mismatch")
    return original_hashes, bridge_hashes, {
        "original_product": str(original_product),
        "original_product_id": original_complete["product_id"],
        "original_dataset_complete_sha256": BASE.sha256_file(original_product / "dataset_complete.json"),
        "step10_product": str(step10_product),
        "step10_product_id": complete["product_id"],
        "step10_dataset_complete_sha256": BASE.sha256_file(complete_path),
    }


def _generate_original(
    *, staging: Path, cfg: Mapping[str, Any], v2cfg: Mapping[str, Any],
    v3cfg: Mapping[str, Any], old_original_hashes: set[str],
    old_bridge_hashes: set[str], diagnostic_bound: float, progress_every: int,
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
        q_schedule = step5v3.affected_qubit_schedule(root_index, n_qubits, v3cfg)
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
        roots.append({
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
        })
        clean_shots = step5v3.clean_control_shots(occurrence, v3cfg)
        examples.append(_save_clean_example(
            staging=staging, domain="original", root_index=root_index,
            group_id=group_id, graph=graph, graph_hash=graph_hash, family=family,
            family_occurrence=occurrence, n_qubits=n_qubits, clean_state=clean_state,
            clean_probs=clean_probs, shots=clean_shots, reference_kind=reference_kind,
            backend_identity="step10c_fresh_original_paired_reference_simulator",
            extra={"global_generator_root_index": root_index},
            diagnostic_bound=diagnostic_bound,
        ))
        for ci, (fraction, strength) in enumerate(zip(depths, strengths)):
            boundary = BASE.depth_boundary(len(circuit.data), fraction)
            affected_qubit = int(q_schedule[ci])
            shots = int(shot_schedule[ci])
            for mechanism in BASE.MECHANISMS:
                examples.append(_save_injected_example(
                    staging=staging, domain="original", root_index=root_index,
                    group_id=group_id, graph=graph, graph_hash=graph_hash,
                    family=family, family_occurrence=occurrence,
                    clean_circuit=circuit, clean_state=clean_state,
                    clean_probs=clean_probs, n_qubits=n_qubits, context_index=ci,
                    affected_qubit=affected_qubit, boundary=boundary,
                    strength=float(strength), shots=shots, mechanism=mechanism,
                    reference_kind=reference_kind,
                    backend_identity="step10c_fresh_original_paired_reference_simulator",
                    v2cfg=v2cfg, extra={"global_generator_root_index": root_index},
                    diagnostic_bound=diagnostic_bound,
                ))
        if progress_every and (local + 1) % progress_every == 0:
            print(f"Generated Step-10C original outer {local + 1}/{count} roots", flush=True)
    return roots, examples


def _generate_bridge(
    *, staging: Path, cfg: Mapping[str, Any], v2cfg: Mapping[str, Any],
    old_original_hashes: set[str], old_bridge_hashes: set[str],
    diagnostic_bound: float, progress_every: int,
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
            "bridge_parent", {"base_seed": spec["base_seed"], "parent_group_index": parent_index}
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
                raise RuntimeError(
                    f"fresh bridge graph overlaps Step-10 history parent={parent_index} variant={variant_index}"
                )
            if graph_hash in fresh_hashes:
                raise RuntimeError("duplicate fresh bridge graph")
            fresh_hashes.add(graph_hash)
            generation_root_index = 10_000_000 + local_root
            group_id = _fresh_group_id(
                "bridge",
                {"parent_group_id": parent_id, "variant_index": variant_index, "graph_sha256": graph_hash},
            )
            n_qubits = int(circuit.num_qubits)
            clean_state = BASE.normalized_state(circuit)
            clean_probs = {
                basis: BASE.basis_probabilities(clean_state, n_qubits, basis)
                for basis in BASE.BASIS_ORDER
            }
            extra = {
                "parent_group_id": parent_id,
                "parent_group_index": parent_index,
                "variant_index": variant_index,
                "motif": motif,
                "phi": float(meta["phi"]),
                "beta": float(meta["beta"]),
            }
            roots.append({
                "domain": "bridge",
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
            })
            shot_levels = [int(v) for v in spec["shot_levels"]]
            clean_shots = shot_levels[(parent_index + variant_index) % len(shot_levels)]
            examples.append(_save_clean_example(
                staging=staging, domain="bridge", root_index=generation_root_index,
                group_id=group_id, graph=graph, graph_hash=graph_hash,
                family=family, family_occurrence=occurrence, n_qubits=n_qubits,
                clean_state=clean_state, clean_probs=clean_probs, shots=clean_shots,
                reference_kind=reference_kind,
                backend_identity="step10c_fresh_bridge_paired_reference_simulator",
                extra=extra, diagnostic_bound=diagnostic_bound,
            ))
            for context in step10a._context_schedule(
                root_index=generation_root_index, circuit=circuit,
                anchor_boundary=anchor_boundary, cfg=bridge_cfg,
            ):
                ci = int(context["context_index"])
                affected_qubit = int(context["affected_qubit"])
                boundary = int(context["boundary"])
                strength = float(context["strength"])
                shots = int(context["shots"])
                for mechanism in BASE.MECHANISMS:
                    examples.append(_save_injected_example(
                        staging=staging, domain="bridge", root_index=generation_root_index,
                        group_id=group_id, graph=graph, graph_hash=graph_hash,
                        family=family, family_occurrence=occurrence,
                        clean_circuit=circuit, clean_state=clean_state,
                        clean_probs=clean_probs, n_qubits=n_qubits, context_index=ci,
                        affected_qubit=affected_qubit, boundary=boundary,
                        strength=strength, shots=shots, mechanism=mechanism,
                        reference_kind=reference_kind,
                        backend_identity="step10c_fresh_bridge_paired_reference_simulator",
                        v2cfg=v2cfg, extra=extra, diagnostic_bound=diagnostic_bound,
                    ))
            local_root += 1
            if progress_every and local_root % progress_every == 0:
                print(
                    f"Generated Step-10C bridge outer {local_root}/{parent_count * variants} roots",
                    flush=True,
                )
    return roots, examples


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
                "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _counter(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _effect_rates(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for domain in ("original", "bridge"):
        output[domain] = {}
        local = [row for row in examples if str(row["domain"]) == domain]
        for mechanism in BASE.MECHANISMS:
            subset = [row for row in local if str(row["mechanism"]) == mechanism]
            effect = sum(_as_bool(row["effect_present"]) for row in subset)
            output[domain][mechanism] = {
                "count": len(subset),
                "effect_positive": effect,
                "effect_positive_rate": effect / max(1, len(subset)),
            }
    return output


def _phenomenology(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for domain in ("original", "bridge"):
        output[domain] = {}
        for mechanism in ("clean_control",) + tuple(BASE.MECHANISMS):
            subset = [
                row for row in examples
                if str(row["domain"]) == domain and str(row["mechanism"]) == mechanism
            ]
            output[domain][mechanism] = _counter(subset, "phenomenology") if subset else {}
    return output


def _validate_structure(
    *, cfg: Mapping[str, Any], original_roots: Sequence[Mapping[str, Any]],
    original_examples: Sequence[Mapping[str, Any]], bridge_roots: Sequence[Mapping[str, Any]],
    bridge_examples: Sequence[Mapping[str, Any]], old_original_hashes: set[str],
    old_bridge_hashes: set[str],
) -> dict[str, Any]:
    ospec = cfg["original_domain"]
    bspec = cfg["bridge_domain"]
    if len(original_roots) != int(ospec["clean_root_count"]):
        raise RuntimeError("fresh original root count mismatch")
    if len(original_examples) != int(ospec["expected_examples"]):
        raise RuntimeError("fresh original example count mismatch")
    if len(bridge_roots) != int(bspec["expected_clean_roots"]):
        raise RuntimeError("fresh bridge root count mismatch")
    if len(bridge_examples) != int(bspec["expected_examples"]):
        raise RuntimeError("fresh bridge example count mismatch")

    all_roots = list(original_roots) + list(bridge_roots)
    all_examples = list(original_examples) + list(bridge_examples)
    graph_hashes = [str(row["graph_sha256"]) for row in all_roots]
    if len(set(graph_hashes)) != len(graph_hashes):
        raise RuntimeError("duplicate clean graph within fresh Step-10C cohort")
    overlap_original = set(graph_hashes) & old_original_hashes
    overlap_bridge = set(graph_hashes) & old_bridge_hashes
    if overlap_original or overlap_bridge:
        raise RuntimeError("fresh Step-10C graph overlaps historical Step-10 graph")

    roots_by_id = {int(row["root_index"]): row for row in all_roots}
    examples_by_root: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_examples:
        examples_by_root[int(row["root_index"])].append(row)
    if set(examples_by_root) != set(roots_by_id):
        raise RuntimeError("fresh root/example root-index sets differ")
    expected_shots = {512, 1024, 2048, 4096}
    for root, rows in examples_by_root.items():
        if len(rows) != 13:
            raise RuntimeError(f"fresh root {root} has {len(rows)} examples instead of 13")
        clean = [row for row in rows if _as_bool(row["clean_control"])]
        injected = [row for row in rows if not _as_bool(row["clean_control"])]
        if len(clean) != 1 or len(injected) != 12:
            raise RuntimeError(f"fresh root {root} does not have 1 clean + 12 injected")
        counts = Counter(str(row["mechanism"]) for row in injected)
        if any(counts.get(mech, 0) != 4 for mech in BASE.MECHANISMS):
            raise RuntimeError(f"fresh root {root} mechanism schedule mismatch: {dict(counts)}")
        if {int(row["shots"]) for row in injected} != expected_shots:
            raise RuntimeError(f"fresh root {root} does not contain all four injected shot levels")
        graph_set = {str(row["graph_sha256"]) for row in rows}
        group_set = {str(row["clean_circuit_group_id"]) for row in rows}
        if len(graph_set) != 1 or len(group_set) != 1:
            raise RuntimeError(f"fresh root {root} derivative identity mismatch")
        if any(str(row["step10c_partition"]) != "fresh_outer_validation" for row in rows):
            raise RuntimeError("non-outer row in fresh Step-10C cohort")

    parents: dict[int, set[int]] = defaultdict(set)
    for row in bridge_roots:
        parents[int(row["parent_group_index"])].add(int(row["root_index"]))
    if len(parents) != int(bspec["parent_groups"]):
        raise RuntimeError("fresh bridge parent-group count mismatch")
    if any(len(roots) != int(bspec["variants_per_parent"]) for roots in parents.values()):
        raise RuntimeError("fresh bridge parent does not contain exactly eight variants")

    pilot_hash = step10a._pilot_graph_hash()
    if any(str(row["graph_sha256"]) == pilot_hash for row in bridge_roots):
        raise RuntimeError("exact Step-9D pilot graph present in fresh bridge")
    pilot = float(bspec["pilot_phi"])
    half_width = float(bspec["pilot_exclusion_half_width"])
    minimum_pilot_distance = min(
        abs(step10a._wrap_delta(float(row["phi"]), pilot)) for row in bridge_roots
    )
    if minimum_pilot_distance < half_width - 1e-12:
        raise RuntimeError("fresh bridge phi violates frozen Step-9D pilot exclusion")

    return {
        "original_root_count": len(original_roots),
        "original_example_count": len(original_examples),
        "bridge_parent_group_count": len(parents),
        "bridge_root_count": len(bridge_roots),
        "bridge_example_count": len(bridge_examples),
        "total_root_count": len(all_roots),
        "total_example_count": len(all_examples),
        "duplicate_fresh_graph_count": len(graph_hashes) - len(set(graph_hashes)),
        "overlap_with_step10_original_graphs": len(overlap_original),
        "overlap_with_step10_bridge_graphs": len(overlap_bridge),
        "exact_step9d_pilot_graph_present": False,
        "minimum_wrapped_phi_distance_from_pilot": float(minimum_pilot_distance),
        "all_roots_have_13_examples": True,
        "all_roots_have_1_clean_plus_12_injected": True,
        "all_roots_have_4_each_mechanism": True,
        "all_roots_have_all_four_injected_shot_levels": True,
        "bridge_parent_groups_are_bootstrap_units": True,
        "original_roots_are_bootstrap_units": True,
    }


def _scan_artifacts(
    product: Path,
    examples: Sequence[Mapping[str, Any]],
    diagnostic_bound: float,
    progress_every: int,
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
    component_abs: dict[str, list[float]] = defaultdict(list)

    phenotype_codes = {str(k): int(v) for k, v in BASE.PHENOTYPE_CODES.items()}
    mechanism_codes = {str(k): int(v) for k, v in BASE.MECHANISM_CODES.items()}

    for index, row in enumerate(examples, start=1):
        path = product / str(row["artifact_path"])
        if not path.is_file():
            raise RuntimeError(f"missing fresh outer artifact {path}")
        if BASE.sha256_file(path) != str(row["artifact_sha256"]):
            hash_mismatch += 1
        try:
            with np.load(path, allow_pickle=False) as loaded:
                arrays = dict(loaded)
        except Exception as exc:
            raise RuntimeError(f"pickle-free NPZ load failed for {path}: {exc}") from exc

        schema_sets[tuple(sorted(arrays))] += 1
        forbidden_statevector_key_count += sum("statevector" in key.lower() for key in arrays)
        try:
            BASE.validate_array_contract(arrays, diagnostic_bound)
        except RuntimeError as exc:
            text = str(exc)
            if "non-finite" in text:
                nonfinite += 1
            elif "bound exceeded" in text:
                bound_violations += 1
            else:
                raise

        def scalar(name: str) -> Any:
            return np.asarray(arrays[name]).reshape(-1)[0]

        expected_clean = _as_bool(row["clean_control"])
        expected_effect = _as_bool(row["effect_present"])
        expected_mask = _as_bool(row["mechanism_loss_mask"])
        expected_mech_code = -1 if expected_clean else mechanism_codes[str(row["mechanism"])]
        expected_pheno = phenotype_codes[str(row["phenomenology"])]
        checks = [
            str(scalar("meta__example_id")) == str(row["example_id"]),
            str(scalar("meta__clean_circuit_group_id")) == str(row["clean_circuit_group_id"]),
            bool(scalar("y__clean_control_target")) == expected_clean,
            bool(scalar("y__effect_present_target")) == expected_effect,
            bool(scalar("y__mechanism_loss_mask")) == expected_mask,
            int(scalar("y__mechanism_target")) == expected_mech_code,
            int(scalar("y__phenomenology_target")) == expected_pheno,
            int(scalar("audit__affected_qubit")) == int(row["affected_qubit"]),
            int(scalar("audit__insertion_boundary_rank")) == int(row["insertion_boundary_rank"]),
            abs(float(scalar("audit__strength")) - float(row["strength"])) <= 1e-12,
        ]
        if not all(checks):
            manifest_target_mismatch += 1

        graph = {
            key: arrays[key]
            for key in (
                "x__graph_gate_names", "x__graph_gate_qubit_ptr",
                "x__graph_gate_qubit_indices", "x__graph_gate_parameter_ptr",
                "x__graph_gate_parameter_sin", "x__graph_gate_parameter_cos",
            )
        }
        if BASE.graph_hash(graph) != str(row["graph_sha256"]):
            graph_hash_mismatch += 1

        primary_parts = [
            np.asarray(arrays["x__delta_local_expectations"], dtype=np.float64).reshape(-1),
            np.asarray(arrays["x__delta_pairwise_correlations"], dtype=np.float64).reshape(-1),
            np.asarray(arrays["x__delta_global_parity"], dtype=np.float64).reshape(-1),
        ]
        exact_parts = [
            np.asarray(arrays["audit__exact_delta_local_expectations"], dtype=np.float64).reshape(-1),
            np.asarray(arrays["audit__exact_delta_pairwise_correlations"], dtype=np.float64).reshape(-1),
            np.asarray(arrays["audit__exact_delta_global_parity"], dtype=np.float64).reshape(-1),
        ]
        primary = np.concatenate(primary_parts)
        exact = np.concatenate(exact_parts)
        if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(exact)):
            nonfinite += 1
        if primary.size:
            max_abs = max(max_abs, float(np.max(np.abs(primary))))
            if float(np.max(np.abs(primary))) > diagnostic_bound:
                bound_violations += 1
        rms = float(np.sqrt(np.mean(np.square(primary)))) if primary.size else 0.0
        error = float(np.sqrt(np.mean(np.square(primary - exact)))) if primary.size else 0.0
        domain = str(row["domain"])
        mech = str(row["mechanism"])
        family = str(row["family"])
        nq = str(row["n_qubits"])
        shots = str(row["shots"])
        strength = f"{float(row['strength']):.12g}"
        depth = str(row["insertion_depth_bin"])
        norm_groups["domain"][domain].append(rms)
        norm_groups["domain_family"][f"{domain}|{family}"].append(rms)
        norm_groups["domain_n_qubits"][f"{domain}|{nq}"].append(rms)
        norm_groups["domain_mechanism"][f"{domain}|{mech}"].append(rms)
        norm_groups["domain_shots"][f"{domain}|{shots}"].append(rms)
        norm_groups["domain_strength"][f"{domain}|{strength}"].append(rms)
        norm_groups["domain_depth"][f"{domain}|{depth}"].append(rms)
        exact_error_by_shots[shots].append(error)
        exact_error_by_domain_shots[f"{domain}|{shots}"].append(error)
        exact_error_by_domain_mechanism[f"{domain}|{mech}"].append(error)
        for label, values in (
            ("local", primary_parts[0]), ("pairwise", primary_parts[1]), ("parity", primary_parts[2])
        ):
            component_abs[label].extend(np.abs(values).tolist())

        if progress_every and index % progress_every == 0:
            print(f"EDA scanned {index}/{len(examples)} fresh outer artifacts", flush=True)

    if hash_mismatch or nonfinite or bound_violations or manifest_target_mismatch or graph_hash_mismatch or forbidden_statevector_key_count:
        raise RuntimeError(
            "fresh outer full-artifact EDA failed: "
            f"hash={hash_mismatch} nonfinite={nonfinite} bound={bound_violations} "
            f"target={manifest_target_mismatch} graph={graph_hash_mismatch} "
            f"statevector_keys={forbidden_statevector_key_count}"
        )
    return {
        "artifacts_scanned": len(examples),
        "artifact_hash_mismatch_count": hash_mismatch,
        "pickle_free_npz_load_pass": True,
        "nonfinite_diagnostic_count": nonfinite,
        "diagnostic_bound_violation_count": bound_violations,
        "maximum_absolute_primary_diagnostic": max_abs,
        "manifest_target_identity_mismatch_count": manifest_target_mismatch,
        "recomputed_graph_hash_mismatch_count": graph_hash_mismatch,
        "forbidden_statevector_key_count": forbidden_statevector_key_count,
        "distinct_npz_key_schemas": len(schema_sets),
        "npz_key_schema_counts": {
            hashlib.sha256("\n".join(schema).encode()).hexdigest()[:16]: count
            for schema, count in sorted(schema_sets.items(), key=lambda item: repr(item[0]))
        },
        "diagnostic_rms": {
            group: {key: _numeric_summary(values) for key, values in sorted(mapping.items())}
            for group, mapping in sorted(norm_groups.items())
        },
        "empirical_vs_exact_rms_error": {
            "by_shots": {key: _numeric_summary(values) for key, values in sorted(exact_error_by_shots.items())},
            "by_domain_shots": {key: _numeric_summary(values) for key, values in sorted(exact_error_by_domain_shots.items())},
            "by_domain_mechanism": {key: _numeric_summary(values) for key, values in sorted(exact_error_by_domain_mechanism.items())},
        },
        "primary_component_absolute_value": {
            key: _numeric_summary(values) for key, values in sorted(component_abs.items())
        },
    }


def _association_report(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for domain in ("original", "bridge"):
        injected = [
            row for row in examples
            if str(row["domain"]) == domain and not _as_bool(row["clean_control"])
        ]
        metrics = {}
        for left, right in (
            ("shots", "strength"),
            ("shots", "insertion_depth_bin"),
            ("shots", "family"),
            ("strength", "insertion_depth_bin"),
            ("strength", "family"),
            ("insertion_depth_bin", "family"),
            ("shots", "affected_qubit"),
        ):
            metrics[f"{left}__{right}__cramers_v"] = float(V2.cramers_v(injected, left, right))
        output[domain] = metrics
    return output


def _support_report(roots: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    injected = [row for row in examples if not _as_bool(row["clean_control"])]
    return {
        "root_counts": {
            "domain": _counter(roots, "domain"),
            "domain_family": dict(sorted(Counter(f"{row['domain']}|{row['family']}" for row in roots).items())),
            "domain_n_qubits": dict(sorted(Counter(f"{row['domain']}|{row['n_qubits']}" for row in roots).items())),
        },
        "example_counts": {
            "domain": _counter(examples, "domain"),
            "mechanism": _counter(examples, "mechanism"),
            "domain_mechanism": dict(sorted(Counter(f"{row['domain']}|{row['mechanism']}" for row in examples).items())),
            "domain_shots": dict(sorted(Counter(f"{row['domain']}|{row['shots']}" for row in examples).items())),
            "domain_depth": dict(sorted(Counter(f"{row['domain']}|{row['insertion_depth_bin']}" for row in injected).items())),
            "domain_strength": dict(sorted(Counter(f"{row['domain']}|{float(row['strength']):.12g}" for row in injected).items())),
        },
    }


def _eda_markdown(eda: Mapping[str, Any], product_id: str) -> str:
    freshness = eda["freshness"]
    lines = [
        "# Step 10C fresh outer cohort — frozen EDA",
        "",
        f"Product: `{product_id}`",
        "",
        f"Status: **{eda['status']}**",
        "",
        "This is data-quality EDA only. No Step-10C model was evaluated on this cohort.",
        "",
        "## Freshness and independence",
        "",
        f"- Original roots/examples: {freshness['original_root_count']} / {freshness['original_example_count']}",
        f"- Bridge parents/roots/examples: {freshness['bridge_parent_group_count']} / {freshness['bridge_root_count']} / {freshness['bridge_example_count']}",
        f"- Historical original graph overlap: {freshness['overlap_with_step10_original_graphs']}",
        f"- Historical bridge graph overlap: {freshness['overlap_with_step10_bridge_graphs']}",
        f"- Duplicate fresh clean graphs: {freshness['duplicate_fresh_graph_count']}",
        f"- Exact Step-9D pilot graph present: {freshness['exact_step9d_pilot_graph_present']}",
        f"- Minimum wrapped phi distance from pilot: {freshness['minimum_wrapped_phi_distance_from_pilot']:.8f}",
        "",
        "## Full-artifact audit",
        "",
        f"- NPZ artifacts scanned: {eda['artifact_scan']['artifacts_scanned']}",
        f"- Artifact hash mismatches: {eda['artifact_scan']['artifact_hash_mismatch_count']}",
        f"- Manifest/target mismatches: {eda['artifact_scan']['manifest_target_identity_mismatch_count']}",
        f"- Recomputed graph-hash mismatches: {eda['artifact_scan']['recomputed_graph_hash_mismatch_count']}",
        f"- Non-finite diagnostics: {eda['artifact_scan']['nonfinite_diagnostic_count']}",
        f"- Diagnostic bound violations: {eda['artifact_scan']['diagnostic_bound_violation_count']}",
        f"- Persisted statevector keys: {eda['artifact_scan']['forbidden_statevector_key_count']}",
        f"- Maximum absolute primary diagnostic: {eda['artifact_scan']['maximum_absolute_primary_diagnostic']:.8f}",
        "",
        "Effect-positive rates, phenomenology, support tables, diagnostic norm strata, empirical-vs-exact finite-shot error, and design-association statistics are frozen in `eda.json`.",
        "",
        "## Boundary",
        "",
        "This cohort is outer-validation only. It may not select epochs, thresholds, architecture, or hyperparameters.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    v2_path = args.base_v2_config.expanduser().resolve()
    v3_path = args.v3_config.expanduser().resolve()
    step10_config_path = args.step10_config.expanduser().resolve()
    step7_path = args.step7_config.expanduser().resolve()
    step10_product = args.step10_product_dir.expanduser().resolve()

    cfg = BASE.read_json(config_path)
    v2cfg = BASE.read_json(v2_path)
    v3cfg = BASE.read_json(v3_path)
    step10_cfg = BASE.read_json(step10_config_path)
    step7_cfg = BASE.read_json(step7_path)
    if cfg.get("schema") != SCHEMA or cfg.get("status") != "FROZEN_BEFORE_STEP10C_OUTER_OUTCOME":
        raise RuntimeError("unexpected Step-10C fresh-outer config schema/status")
    if v3cfg.get("schema") != step5v3.SCHEMA:
        raise RuntimeError("unexpected Step-5-v3 overlay schema")
    if step10_cfg.get("schema") != step10a.SCHEMA:
        raise RuntimeError("unexpected Step-10 mixture config schema")

    old_original_hashes, old_bridge_hashes, source_identity = _verify_historical_products(
        step10_product, step10_cfg, step7_cfg
    )
    diagnostic_bound = float(cfg["eda_gates"]["diagnostic_absolute_bound"])
    identity = {
        "schema": SCHEMA,
        "config_sha256": BASE.sha256_file(config_path),
        "base_v2_config_sha256": BASE.sha256_file(v2_path),
        "v3_config_sha256": BASE.sha256_file(v3_path),
        "step10_config_sha256": BASE.sha256_file(step10_config_path),
        "step7_config_sha256": BASE.sha256_file(step7_path),
        "runner_sha256": BASE.sha256_file(Path(__file__).resolve()),
        **source_identity,
        "original_global_root_range": [
            int(cfg["original_domain"]["global_root_index_start"]),
            int(cfg["original_domain"]["global_root_index_start"]) + int(cfg["original_domain"]["clean_root_count"]) - 1,
        ],
        "bridge_base_seed": int(cfg["bridge_domain"]["base_seed"]),
    }
    product_id = "product_" + hashlib.sha256(
        BASE.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    product = output_parent / product_id
    if product.exists():
        raise RuntimeError(f"refusing to overwrite existing Step-10C fresh outer product: {product}")
    staging = output_parent / f".{product_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    try:
        print("STEP 10C FRESH OUTER GENERATION — NO MODEL / NO QPU", flush=True)
        original_roots, original_examples = _generate_original(
            staging=staging, cfg=cfg, v2cfg=v2cfg, v3cfg=v3cfg,
            old_original_hashes=old_original_hashes,
            old_bridge_hashes=old_bridge_hashes,
            diagnostic_bound=diagnostic_bound, progress_every=int(args.progress_every),
        )
        bridge_roots, bridge_examples = _generate_bridge(
            staging=staging, cfg=cfg, v2cfg=v2cfg,
            old_original_hashes=old_original_hashes,
            old_bridge_hashes=old_bridge_hashes,
            diagnostic_bound=diagnostic_bound, progress_every=int(args.progress_every),
        )
        all_roots = list(original_roots) + list(bridge_roots)
        all_examples = list(original_examples) + list(bridge_examples)

        freshness = _validate_structure(
            cfg=cfg, original_roots=original_roots, original_examples=original_examples,
            bridge_roots=bridge_roots, bridge_examples=bridge_examples,
            old_original_hashes=old_original_hashes, old_bridge_hashes=old_bridge_hashes,
        )

        manifests = staging / "manifests"
        BASE.write_csv(manifests / "original_root_manifest.csv", original_roots)
        BASE.write_csv(manifests / "original_example_manifest.csv", original_examples)
        BASE.write_csv(manifests / "bridge_root_manifest.csv", bridge_roots)
        BASE.write_csv(manifests / "bridge_example_manifest.csv", bridge_examples)

        artifact_scan = _scan_artifacts(
            staging, all_examples, diagnostic_bound, int(args.eda_progress_every)
        )
        eda = {
            "schema": "triqto.v0_2.step10c_fresh_outer_eda.v1",
            "status": "PASS",
            "product_id": product_id,
            "model_evaluated": False,
            "qpu_executed": False,
            "step10b_outer_rows_reused": False,
            "freshness": freshness,
            "support": _support_report(all_roots, all_examples),
            "effect_positive_rates": _effect_rates(all_examples),
            "phenomenology": _phenomenology(all_examples),
            "design_associations": _association_report(all_examples),
            "artifact_scan": artifact_scan,
        }
        BASE.atomic_json(staging / "eda.json", eda)
        (staging / "EDA_SUMMARY.md").write_text(_eda_markdown(eda, product_id), encoding="utf-8")

        manifest_names = [
            "original_root_manifest.csv", "original_example_manifest.csv",
            "bridge_root_manifest.csv", "bridge_example_manifest.csv",
        ]
        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE_FROZEN_OUTER_VALIDATION",
            "product_id": product_id,
            "identity": identity,
            "outer_validation_only": True,
            "step10b_outer_reused": False,
            "model_evaluated_before_freeze": False,
            "qpu_executed": False,
            "original_clean_root_count": len(original_roots),
            "original_example_count": len(original_examples),
            "bridge_parent_group_count": int(cfg["bridge_domain"]["parent_groups"]),
            "bridge_clean_root_count": len(bridge_roots),
            "bridge_example_count": len(bridge_examples),
            "total_clean_root_count": len(all_roots),
            "total_example_count": len(all_examples),
            "manifest_hashes": {
                name: BASE.sha256_file(manifests / name) for name in manifest_names
            },
            "eda_sha256": BASE.sha256_file(staging / "eda.json"),
            "eda_summary_sha256": BASE.sha256_file(staging / "EDA_SUMMARY.md"),
        }
        BASE.atomic_json(staging / "dataset_complete.json", completion)
        os.replace(staging, product)
        BASE.atomic_json(
            output_parent / "current_product.json",
            {
                "schema": CURRENT_POINTER_SCHEMA,
                "product_id": product_id,
                "product_dir": str(product),
                "dataset_complete_sha256": BASE.sha256_file(product / "dataset_complete.json"),
                "eda_sha256": BASE.sha256_file(product / "eda.json"),
            },
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 10C FRESH OUTER COHORT COMPLETE\n")
    print(f"Product: {product_id}")
    print(f"Original: {len(original_roots)} roots / {len(original_examples)} examples")
    print(
        f"Bridge: {cfg['bridge_domain']['parent_groups']} parents / "
        f"{len(bridge_roots)} roots / {len(bridge_examples)} examples"
    )
    print("Historical graph overlap: 0")
    print("Exact Step-9D pilot graph present: NO")
    print("Full-artifact EDA: PASS")
    print("Model evaluated: NO")
    print("QPU executed: NO")
    print(f"Output: {product}")


if __name__ == "__main__":
    main()
