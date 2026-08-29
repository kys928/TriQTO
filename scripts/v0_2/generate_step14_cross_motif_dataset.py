#!/usr/bin/env python3
"""Generate the frozen Step-14 cross-motif simulator dataset.

Default mode materializes only fit+selection. Simulator outer requires the
frozen Step-14 selection marker. The future-hardware reserve is never
materialized by this Step-14 runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

import benchmark_step6_cheap_baselines as baseline
import generate_step5_matched_diagnostic_training_dataset_v3 as step5v3
import generate_step10c_fresh_outer_cohort as step10c_outer

BASE = step5v3.BASE
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "triqto.v0_2.step14_cross_motif_dataset.v1"
DEFAULT_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
DEFAULT_V2_CONFIG = ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
DEFAULT_STEP12_CONFIG = ROOT / "configs/v0_2/step12_independent_phase_generalization.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step14_cross_motif_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--base-v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    p.add_argument("--step12-config", type=Path, default=DEFAULT_STEP12_CONFIG)
    p.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    p.add_argument("--mode", choices=("development", "simulator_outer"), default="development")
    p.add_argument("--selection-freeze", type=Path)
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def assert_contract(cfg: Mapping[str, Any]) -> None:
    if cfg.get("schema") != "triqto.v0_2.step14_cross_motif_generalization_training.v1":
        raise RuntimeError("unexpected Step-14 protocol schema")
    if cfg.get("status") != "FROZEN_BEFORE_STEP14_DATASET_GENERATION":
        raise RuntimeError("Step-14 protocol is not frozen before dataset generation")
    d = cfg["cross_motif_dataset"]
    if d.get("schema") != SCHEMA or int(d["family_count_total"]) != 1050 or int(d["variants_per_family"]) != 4:
        raise RuntimeError("Step-14 cross-motif family contract drift")
    if list(d["mechanisms"]) != ["rz_drift", "rx_overrotation", "ry_overrotation"]:
        raise RuntimeError("Step-14 mechanism order drift")
    if list(d["materialization_policy"]["before_training"]) != ["fit", "selection"]:
        raise RuntimeError("Step-14 pretraining materialization boundary drift")
    if bool(d["materialization_policy"]["future_hardware_reserve_materialized_in_step14"]):
        raise RuntimeError("future-hardware reserve may not be materialized in Step 14")


def partition_for_family(family_index: int, cfg: Mapping[str, Any]) -> str:
    split = cfg["cross_motif_dataset"]["family_split"]
    fold = int(family_index) % 7
    for name, key in (("fit", "fit_folds"), ("selection", "selection_folds"),
                      ("simulator_outer", "simulator_outer_folds"),
                      ("future_hardware_reserve", "future_hardware_reserve_folds")):
        if fold in {int(v) for v in split[key]}:
            return name
    raise RuntimeError(f"unassigned family fold {fold}")


def family_indices_for_mode(mode: str, cfg: Mapping[str, Any]) -> list[int]:
    wanted = {"fit", "selection"} if mode == "development" else {"simulator_outer"}
    return [i for i in range(1050) if partition_for_family(i, cfg) in wanted]


def verify_selection_freeze(path: Path | None, config_path: Path) -> None:
    if path is None:
        raise RuntimeError("simulator_outer materialization requires --selection-freeze")
    payload = read_json(path.expanduser().resolve())
    if payload.get("schema") != "triqto.v0_2.step14_selection_freeze.v1":
        raise RuntimeError("unexpected Step-14 selection-freeze schema")
    if payload.get("status") != "FROZEN_BEFORE_ANY_STEP14_OUTER_MATERIALIZATION":
        raise RuntimeError("Step-14 selection is not frozen before outer materialization")
    if payload.get("protocol_config_sha256") != baseline.sha256_file(config_path):
        raise RuntimeError("Step-14 selection-freeze/config hash mismatch")
    if not bool(payload.get("all_three_seed_checkpoints_frozen")):
        raise RuntimeError("Step-14 selection freeze is incomplete")


def rng_for(base: int, *parts: int) -> np.random.Generator:
    seed = int(base) & ((1 << 64) - 1)
    for part in parts:
        seed ^= (int(part) + 0x9E3779B97F4A7C15 + (seed << 6) + (seed >> 2)) & ((1 << 64) - 1)
    return np.random.default_rng(seed)


def topology_edges(name: str, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    path = [(i, i + 1) for i in range(n - 1)]
    if n == 2 or name == "path": return path
    if name == "hub_spoke":
        h = int(rng.integers(0, n)); return [(h, q) for q in range(n) if q != h]
    if name == "two_branch":
        c = n // 2; return [(c, q) for q in range(n) if q != c]
    if name == "staggered_path": return path[::2] + path[1::2]
    if name == "directed_fan":
        a = int(rng.integers(0, n)); return [(a, q) for q in range(n) if q != a]
    if name == "mixed_sparse":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]; rng.shuffle(pairs)
        out = list(path)
        for e in pairs:
            if e not in out: out.append(e)
            if len(out) >= max(2, n): break
        return out
    raise ValueError(name)


def family_blueprint(family_index: int, candidate_offset: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    d = cfg["cross_motif_dataset"]; g = d["circuit_grammar"]
    rng = rng_for(int(d["base_seed"]), family_index, candidate_offset, 17)
    n = int(rng.choice(np.asarray(g["qubit_choices"], dtype=np.int64)))
    topology = str(rng.choice(np.asarray(g["topology_classes"], dtype=object)))
    context = str(rng.choice(np.asarray(g["injection_context_classes"], dtype=object)))
    layers = int(rng.integers(int(g["reference_layer_count_range_inclusive"][0]), int(g["reference_layer_count_range_inclusive"][1]) + 1))
    affected = int(rng.integers(0, n)); edges = topology_edges(topology, n, rng)
    snames = np.asarray(["h", "rx", "ry", "rz"], dtype=object)
    sp = np.asarray([float(g["single_qubit_gate_probabilities"][k]) for k in snames]); sp /= sp.sum()
    tnames = np.asarray(["cx", "cz"], dtype=object)
    tp = np.asarray([float(g["two_qubit_gate_probabilities"][k]) for k in tnames]); tp /= tp.sum()
    events: list[tuple[str, tuple[int, ...]]] = [("ry", (affected,))]
    entangling = 0
    for layer in range(layers):
        q = int((affected + layer + int(rng.integers(0, n))) % n)
        events.append((str(rng.choice(snames, p=sp)), (q,)))
        if layer % 2 == 0 or entangling < int(g["minimum_entangling_operations"]):
            events.append((str(rng.choice(tnames, p=tp)), tuple(int(v) for v in edges[layer % len(edges)])))
            entangling += 1
    while entangling < int(g["minimum_entangling_operations"]):
        events.append(("cz", tuple(int(v) for v in edges[entangling % len(edges)]))); entangling += 1
    events.append(("h", (affected,)))
    two = [i for i, (name, _) in enumerate(events) if name in {"cx", "cz"}]
    if context == "pre_entangling": boundary = max(1, two[0])
    elif context == "post_entangling_recombination": boundary = min(len(events) - 1, two[-1] + 1)
    else: boundary = min(len(events) - 1, max(1, two[len(two)//2] + 1))
    return {"candidate_seed_offset": candidate_offset, "n_qubits": n, "topology_class": topology,
            "injection_context_class": context, "affected_qubit": affected, "events": events,
            "boundary": int(boundary)}


def build_variant(bp: Mapping[str, Any], family_index: int, variant_index: int, cfg: Mapping[str, Any]) -> QuantumCircuit:
    d = cfg["cross_motif_dataset"]; g = d["circuit_grammar"]
    rng = rng_for(int(d["base_seed"]), family_index, int(bp["candidate_seed_offset"]), variant_index, 71)
    lo, hi = [float(v) for v in g["parameterized_rotation_range"]]
    qc = QuantumCircuit(int(bp["n_qubits"]), name=f"step14_f{family_index:04d}_v{variant_index}")
    for name, qubits in bp["events"]:
        if len(qubits) == 1:
            q = int(qubits[0])
            if name == "h": qc.h(q)
            elif name == "rx": qc.rx(float(rng.uniform(lo, hi)), q)
            elif name == "ry": qc.ry(float(rng.uniform(lo, hi)), q)
            elif name == "rz": qc.rz(float(rng.uniform(lo, hi)), q)
            else: raise ValueError(name)
        else:
            a, b = map(int, qubits)
            if name == "cx": qc.cx(a, b)
            elif name == "cz": qc.cz(a, b)
            else: raise ValueError(name)
    return qc


def op_signature(qc: QuantumCircuit) -> list[str]:
    out = []
    for item in qc.data:
        q = [int(qc.find_bit(bit).index) for bit in item.qubits]
        out.append(str(item.operation.name) + ":" + "-".join(f"q{v}" for v in q))
    return out


def expectation(state: Statevector, basis: str, qubits: tuple[int, ...], n: int) -> float:
    label = ["I"] * n
    for q in qubits: label[n - 1 - q] = basis
    return float(np.real(state.expectation_value(Pauli("".join(label)))))


def ideal_vector(reference: QuantumCircuit, observed: QuantumCircuit) -> np.ndarray:
    ref = Statevector.from_instruction(reference); obs = Statevector.from_instruction(observed)
    n = reference.num_qubits; pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    values = []
    for basis in ("Z", "X", "Y"):
        for q in range(n): values.append(expectation(obs, basis, (q,), n) - expectation(ref, basis, (q,), n))
        for i, j in pairs: values.append(expectation(obs, basis, (i, j), n) - expectation(ref, basis, (i, j), n))
        values.append(expectation(obs, basis, tuple(range(n)), n) - expectation(ref, basis, tuple(range(n)), n))
    return np.asarray(values, dtype=np.float64)


def identifiability(bp: Mapping[str, Any], variants: Sequence[QuantumCircuit], cfg: Mapping[str, Any]) -> dict[str, Any]:
    a = cfg["cross_motif_dataset"]["model_blind_identifiability_admission"]
    probe = float(a["probe_strength"]); mechs = list(cfg["cross_motif_dataset"]["mechanisms"])
    norms, distances = [], []
    for circuit in variants:
        vec = {}
        for mech in mechs:
            obs = BASE.inject_hidden_rotation(circuit, int(bp["boundary"]), int(bp["affected_qubit"]), mech, probe)
            vec[mech] = ideal_vector(circuit, obs); norms.append(float(np.linalg.norm(vec[mech])))
        for i in range(3):
            for j in range(i + 1, 3): distances.append(float(np.linalg.norm(vec[mechs[i]] - vec[mechs[j]])))
    mn, md = min(norms), min(distances)
    return {"status": "PASS" if mn >= float(a["minimum_true_mechanism_delta_norm"]) and md >= float(a["minimum_pairwise_mechanism_distance"]) else "FAIL",
            "uses_statevector_only": True, "uses_model_predictions": False,
            "minimum_observed_true_mechanism_delta_norm": mn,
            "minimum_observed_pairwise_mechanism_distance": md}


def strengths_and_shots(family_index: int, variant_index: int, cfg: Mapping[str, Any]) -> tuple[list[float], list[int], int]:
    d = cfg["cross_motif_dataset"]; rng = rng_for(int(d["base_seed"]), family_index, variant_index, 131)
    strengths = [float(rng.uniform(float(lo), float(hi))) for lo, hi in d["strength_design"]["bins"]]
    levels = [int(v) for v in d["shot_design"]["shot_levels"]]
    return strengths, [int(v) for v in rng.permutation(levels).tolist()], int(rng.choice(levels))


def materialize_family(staging: Path, family_index: int, partition: str, bp: Mapping[str, Any],
                       variants: Sequence[QuantumCircuit], cfg: Mapping[str, Any], v2cfg: Mapping[str, Any]):
    family_id = stable_hash({"family_index": family_index, "events": [(n, list(q)) for n, q in bp["events"]],
                             "context": bp["injection_context_class"]})
    family_signature = stable_hash({"events": [(n, list(q)) for n, q in bp["events"]], "context": bp["injection_context_class"]})
    roots, examples = [], []; reference_kind = str(v2cfg["finite_shot_acquisition"]["reference_kind"])
    for variant_index, circuit in enumerate(variants):
        root_index = family_index * 4 + variant_index; graph = BASE.serialize_graph(circuit); graph_hash = BASE.graph_hash(graph)
        n = circuit.num_qubits; clean_state = BASE.normalized_state(circuit)
        clean_probs = {basis: BASE.basis_probabilities(clean_state, n, basis) for basis in BASE.BASIS_ORDER}
        strengths, shots, clean_shots = strengths_and_shots(family_index, variant_index, cfg)
        group_id = BASE.sha256_bytes(BASE.canonical_json([family_id, variant_index, graph_hash]).encode())
        extra = {"step14_partition": partition, "family_id": family_id, "family_index": family_index,
                 "variant_index": variant_index, "topology_class": bp["topology_class"],
                 "injection_context_class": bp["injection_context_class"], "family_signature_sha256": family_signature}
        roots.append({**extra, "root_index": root_index, "clean_circuit_group_id": group_id, "n_qubits": n,
                      "affected_qubit": int(bp["affected_qubit"]), "injection_boundary_rank": int(bp["boundary"]),
                      "graph_sha256": graph_hash, "operation_signature": json.dumps(op_signature(circuit), separators=(",", ":"))})
        clean = step10c_outer._save_clean_example(staging=staging, domain="cross_motif", root_index=root_index,
            group_id=group_id, graph=graph, graph_hash=graph_hash, family=f"step14_{bp['topology_class']}",
            family_occurrence=root_index, n_qubits=n, clean_state=clean_state, clean_probs=clean_probs,
            shots=clean_shots, reference_kind=reference_kind, backend_identity="step14_cross_motif_finite_shot_simulator",
            extra=extra, diagnostic_bound=2.000001)
        clean["split"] = partition; clean["step14_partition"] = partition; examples.append(clean)
        for context_index, strength in enumerate(strengths):
            for mech in cfg["cross_motif_dataset"]["mechanisms"]:
                row = step10c_outer._save_injected_example(staging=staging, domain="cross_motif", root_index=root_index,
                    group_id=group_id, graph=graph, graph_hash=graph_hash, family=f"step14_{bp['topology_class']}",
                    family_occurrence=root_index, clean_circuit=circuit, clean_state=clean_state, clean_probs=clean_probs,
                    n_qubits=n, context_index=context_index, affected_qubit=int(bp["affected_qubit"]),
                    boundary=int(bp["boundary"]), strength=float(strength), shots=int(shots[context_index]), mechanism=mech,
                    reference_kind=reference_kind, backend_identity="step14_cross_motif_finite_shot_simulator",
                    v2cfg=v2cfg, extra=extra, diagnostic_bound=2.000001)
                row["split"] = partition; row["step14_partition"] = partition; examples.append(row)
    return family_id, family_signature, roots, examples


def validate_counts(mode: str, roots: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    e = cfg["cross_motif_dataset"]["expected_counts"]
    er = int(e["fit_roots"]) + int(e["selection_roots"]) if mode == "development" else int(e["simulator_outer_roots"])
    ee = int(e["fit_examples"]) + int(e["selection_examples"]) if mode == "development" else int(e["simulator_outer_examples"])
    if len(roots) != er or len(examples) != ee: raise RuntimeError("Step-14 materialized count mismatch")
    by_root: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in examples: by_root[int(row["root_index"])].append(row)
    for rows in by_root.values():
        if len(rows) != 13: raise RuntimeError("Step-14 root does not contain 13 examples")
        distorted = [r for r in rows if str(r["mechanism"]) != "clean_control"]
        counts = Counter(str(r["mechanism"]) for r in distorted)
        if len(distorted) != 12 or any(counts.get(m, 0) != 4 for m in cfg["cross_motif_dataset"]["mechanisms"]):
            raise RuntimeError("Step-14 mechanism/strength crossing failed")
    return {"status": "PASS", "root_count": len(roots), "example_count": len(examples),
            "partitions": dict(sorted(Counter(str(r["step14_partition"]) for r in roots).items())),
            "future_hardware_reserve_materialized": False}


def main() -> None:
    args = parse_args(); config_path = args.config.expanduser().resolve(); cfg = read_json(config_path); assert_contract(cfg)
    v2cfg = read_json(args.base_v2_config.expanduser().resolve()); step12 = read_json(args.step12_config.expanduser().resolve())
    if args.mode == "simulator_outer": verify_selection_freeze(args.selection_freeze, config_path)
    elif args.selection_freeze is not None: raise RuntimeError("development generation must not consume selection freeze")
    indices = family_indices_for_mode(args.mode, cfg)
    if len(indices) != (750 if args.mode == "development" else 150): raise RuntimeError("family-fold arithmetic drift")
    step12_signatures = {tuple(str(v) for v in raw["reference_operation_signature"]) for raw in step12["generalization_design"]["motifs"]}
    identity = {"schema": SCHEMA, "mode": args.mode, "protocol_config_sha256": baseline.sha256_file(config_path),
                "family_indices_sha256": stable_hash(indices), "model_inference": False, "qpu_access": False}
    product_id = ("development_" if args.mode == "development" else "outer_") + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    parent = args.output_parent.expanduser().resolve(); product = parent / product_id
    if product.exists():
        if read_json(product / "dataset_complete.json").get("identity") != identity: raise RuntimeError("existing product identity mismatch")
        print("Step-14 product already complete:", product); return
    staging = parent / f".{product_id}.staging-{uuid.uuid4().hex}"; staging.mkdir(parents=True, exist_ok=False)
    roots, examples, families = [], [], []; seen_graphs, seen_family_signatures = set(), set(); rejected = 0
    try:
        for pos, family_index in enumerate(indices, start=1):
            offset = 0
            while True:
                bp = family_blueprint(family_index, offset, cfg); variants = [build_variant(bp, family_index, v, cfg) for v in range(4)]
                sigs = [op_signature(v) for v in variants]; core = ["h:q0", "rz:q0", "h:q0", "cx:q0-q1"]
                forbidden = any(tuple(sig) in step12_signatures for sig in sigs) or any(any(sig[i:i+4] == core for i in range(max(0, len(sig)-3))) for sig in sigs)
                audit = identifiability(bp, variants, cfg) if not forbidden else {"status": "FAIL"}
                if not forbidden and audit["status"] == "PASS": break
                offset += 1; rejected += 1
            partition = partition_for_family(family_index, cfg)
            family_id, family_signature, local_roots, local_examples = materialize_family(staging, family_index, partition, bp, variants, cfg, v2cfg)
            if family_signature in seen_family_signatures: raise RuntimeError("duplicate Step-14 family signature")
            seen_family_signatures.add(family_signature)
            for row in local_roots:
                if row["graph_sha256"] in seen_graphs: raise RuntimeError("duplicate Step-14 root graph")
                seen_graphs.add(row["graph_sha256"])
            roots.extend(local_roots); examples.extend(local_examples)
            families.append({"family_index": family_index, "family_id": family_id, "step14_partition": partition,
                             "candidate_seed_offset": offset, "n_qubits": bp["n_qubits"], "topology_class": bp["topology_class"],
                             "injection_context_class": bp["injection_context_class"], "family_signature_sha256": family_signature,
                             "identifiability_status": audit["status"],
                             "identifiability_min_delta_norm": audit["minimum_observed_true_mechanism_delta_norm"],
                             "identifiability_min_pairwise_distance": audit["minimum_observed_pairwise_mechanism_distance"]})
            if args.progress_every and pos % args.progress_every == 0:
                print(f"Generated {pos}/{len(indices)} Step-14 families (rejected candidates={rejected})", flush=True)
        structure = validate_counts(args.mode, roots, examples, cfg); manifests = staging / "manifests"
        BASE.write_csv(manifests / "family_manifest.csv", families); BASE.write_csv(manifests / "root_manifest.csv", roots); BASE.write_csv(manifests / "example_manifest.csv", examples)
        for row in examples:
            path = staging / row["artifact_path"]
            if BASE.sha256_file(path) != row["artifact_sha256"]: raise RuntimeError("Step-14 artifact hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                if any("statevector" in key.lower() for key in z.files): raise RuntimeError("statevector persisted in Step-14 model artifact")
        eda = {"schema": "triqto.v0_2.step14_cross_motif_eda.v1", "status": "PASS", "mode": args.mode,
               "model_evaluated": False, "qpu_executed": False, "structural": structure,
               "family_rejection_count_before_admission": rejected,
               "support": {"n_qubits": dict(Counter(str(r["n_qubits"]) for r in families)),
                           "topology_class": dict(Counter(str(r["topology_class"]) for r in families)),
                           "injection_context_class": dict(Counter(str(r["injection_context_class"]) for r in families))}}
        BASE.atomic_json(staging / "eda.json", eda)
        names = ["family_manifest.csv", "root_manifest.csv", "example_manifest.csv"]
        completion = {"schema": SCHEMA, "status": "COMPLETE_FROZEN_DEVELOPMENT" if args.mode == "development" else "COMPLETE_FROZEN_SIMULATOR_OUTER",
                      "product_id": product_id, "identity": identity, "mode": args.mode, "family_count": len(families),
                      "root_count": len(roots), "example_count": len(examples), "partitions": structure["partitions"],
                      "model_evaluated_before_freeze": False, "qpu_executed": False, "future_hardware_reserve_materialized": False,
                      "manifest_hashes": {name: BASE.sha256_file(manifests / name) for name in names}, "eda_sha256": BASE.sha256_file(staging / "eda.json")}
        BASE.atomic_json(staging / "dataset_complete.json", completion); os.replace(staging, product)
        pointer = "current_development_product.json" if args.mode == "development" else "current_simulator_outer_product.json"
        BASE.atomic_json(parent / pointer, {"schema": "triqto.v0_2.step14_cross_motif_current_product.v1", "mode": args.mode,
                                            "product_id": product_id, "product_dir": str(product),
                                            "dataset_complete_sha256": BASE.sha256_file(product / "dataset_complete.json")})
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print("\nTRIQTO STEP 14 CROSS-MOTIF DATASET COMPLETE")
    print("Mode:", args.mode); print("Families:", len(families)); print("Roots:", len(roots)); print("Examples:", len(examples))
    print("Partitions:", structure["partitions"]); print("Identifiability admission: PASS for every materialized family")
    print("Future hardware reserve materialized: NO"); print("Model evaluated: NO"); print("QPU executed: NO"); print("Output:", product)


if __name__ == "__main__":
    main()
