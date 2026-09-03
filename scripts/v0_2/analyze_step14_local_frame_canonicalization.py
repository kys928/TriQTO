#!/usr/bin/env python3
"""Step-14 local-Pauli-frame canonicalization diagnostic.

Post-hoc diagnostic only. The frozen TriQTO model is never updated. The study
uses only the already-materialized Step-14 FIT + selection development cohort;
simulator outer, future-hardware reserve, and QPU access are forbidden.

The affected qubit and true injection boundary are privileged generator
metadata and remain analysis-only. For each clean circuit/root, the diagnostic
computes the exact first-order Born-response Jacobian of the model-visible
local/pair/parity Pauli evidence with respect to infinitesimal Z/X/Y rotations
at the true injection point. This is the measured-observable representation of
propagating the local Pauli generators through the suffix circuit. Finite-shot
evidence is then mapped into canonical Z/X/Y coordinates with a fixed ridge
pseudoinverse.

Frozen comparison:
  raw diagnostics
  -> analysis-only affected-qubit oracle
  -> analysis-only affected-qubit + raw insertion neighborhood
  -> mathematically canonicalized local Pauli frame

A deterministic argmax(|canonical coordinate|) rule is reported alongside the
same small probe used by the preceding oracle decomposition. A high-capacity
probe on the canonical coordinates is an approximate transferable ceiling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

import analyze_step14_oracle_raw_evidence_ceiling as oracle
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import run_step14_cross_motif_training as step14

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_local_frame_canonicalization")
PREVIOUS_ORACLE_PARENT = Path("/workspace/triqto-data/step14_oracle_raw_evidence_ceiling")
PREVIOUS_ORACLE_ID = "oracle_raw_8c74105460a3c6ae9daf03ed"
PREVIOUS_ORACLE_RESULT_SHA256 = "sha256:8605df5497d8f452c083666a04362df6b3fe027dea2247f27db7fab4415f93cd"
SCHEMA = "triqto.v0_2.step14_local_frame_canonicalization.v1"
MAX_GATES = 17
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
AXES = ("Z", "X", "Y")
TARGET = {name: index for index, name in enumerate(MECHANISMS)}
PROBE_SEEDS = tuple(int(v) for v in oracle.PROBE_SEEDS)
PROBE_SPEC = {
    "small_probe": "same one-hidden-layer MLP as frozen oracle decomposition",
    "high_capacity_probe": "same 512-256-128 MLP as frozen oracle decomposition",
    "probe_seeds": list(PROBE_SEEDS),
    "selection_used_for_training_or_early_stopping": False,
    "main_model_weights_updated": False,
    "canonicalization": "shot-whitened normalized first-order Born Jacobian ridge pseudoinverse",
    "ridge": 1.0e-3,
    "meaningful_canonicalization_delta_minimum": 0.05,
    "canonical_transferable_ceiling_low_ba": 0.60,
    "canonical_transferable_ceiling_high_ba": 0.70,
    "bootstrap_replicates": int(oracle.PROBE_SPEC["bootstrap_replicates"]),
    "bootstrap_unit": "cross_motif_family_id",
    "bootstrap_seed": 2026090301,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def verify_previous_oracle() -> dict[str, Any]:
    pointer_path = PREVIOUS_ORACLE_PARENT / "current_oracle_raw_evidence_diagnostic.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if str(pointer.get("diagnostic_id")) != PREVIOUS_ORACLE_ID:
        raise RuntimeError("local-frame diagnostic is pinned to the completed oracle decomposition")
    if str(pointer.get("diagnostic_result_sha256")) != PREVIOUS_ORACLE_RESULT_SHA256:
        raise RuntimeError("previous oracle diagnostic result hash drift")
    result_path = PREVIOUS_ORACLE_PARENT / PREVIOUS_ORACLE_ID / "diagnostic_result.json"
    if baseline.sha256_file(result_path) != PREVIOUS_ORACLE_RESULT_SHA256:
        raise RuntimeError("previous oracle diagnostic result object hash mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE_FROZEN_ORACLE_RAW_EVIDENCE_CEILING":
        raise RuntimeError("previous oracle decomposition is not complete")
    boundaries = result.get("scientific_boundaries", {})
    if bool(boundaries.get("simulator_outer_accessed")) or bool(boundaries.get("future_hardware_reserve_accessed")) or bool(boundaries.get("qpu_executed")):
        raise RuntimeError("previous oracle diagnostic violated the required scientific boundary")
    return result


def circuit_from_serialized(loaded: Mapping[str, np.ndarray]) -> QuantumCircuit:
    n_qubits = int(np.asarray(loaded["x__layout_logical_to_physical"]).size)
    names = [str(v).lower() for v in np.asarray(loaded["x__graph_gate_names"]).tolist()]
    qptr = np.asarray(loaded["x__graph_gate_qubit_ptr"], dtype=np.int64).reshape(-1)
    qidx = np.asarray(loaded["x__graph_gate_qubit_indices"], dtype=np.int64).reshape(-1)
    pptr = np.asarray(loaded["x__graph_gate_parameter_ptr"], dtype=np.int64).reshape(-1)
    psin = np.asarray(loaded["x__graph_gate_parameter_sin"], dtype=np.float64).reshape(-1)
    pcos = np.asarray(loaded["x__graph_gate_parameter_cos"], dtype=np.float64).reshape(-1)
    if len(names) > MAX_GATES or qptr.shape != (len(names) + 1,) or pptr.shape != (len(names) + 1,):
        raise RuntimeError("serialized Step-14 graph exceeds frozen local-frame support")
    if psin.shape != pcos.shape or int(pptr[-1]) != len(psin):
        raise RuntimeError("serialized Step-14 graph parameter contract mismatch")

    qc = QuantumCircuit(n_qubits)
    for index, name in enumerate(names):
        qubits = [int(v) for v in qidx[int(qptr[index]):int(qptr[index + 1])].tolist()]
        start, end = int(pptr[index]), int(pptr[index + 1])
        params = [float(math.atan2(psin[k], pcos[k])) for k in range(start, end)]
        if name == "h" and len(qubits) == 1 and not params:
            qc.h(qubits[0])
        elif name in {"rx", "ry", "rz"} and len(qubits) == 1 and len(params) == 1:
            getattr(qc, name)(params[0], qubits[0])
        elif name == "cx" and len(qubits) == 2 and not params:
            qc.cx(qubits[0], qubits[1])
        elif name == "cz" and len(qubits) == 2 and not params:
            qc.cz(qubits[0], qubits[1])
        else:
            raise RuntimeError(f"unsupported serialized Step-14 gate contract: {name}/{qubits}/{params}")
    return qc


def circuit_slice(clean: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    out = QuantumCircuit(clean.num_qubits)
    for item in clean.data[start:end]:
        qargs = [out.qubits[clean.find_bit(q).index] for q in item.qubits]
        if item.clbits:
            raise RuntimeError("Step-14 clean circuit unexpectedly contains classical bits")
        out.append(item.operation, qargs, [])
    return out


def pauli_label(n_qubits: int, basis: str, qubits: Sequence[int]) -> str:
    chars = ["I"] * n_qubits
    for q in qubits:
        chars[n_qubits - 1 - int(q)] = basis
    return "".join(chars)


def measured_pauli_labels(n_qubits: int, pairs: np.ndarray) -> list[str]:
    output: list[str] = []
    for basis in AXES:
        for q in range(n_qubits):
            output.append(pauli_label(n_qubits, basis, (q,)))
        for left, right in np.asarray(pairs, dtype=np.int64).reshape(-1, 2).tolist():
            output.append(pauli_label(n_qubits, basis, (int(left), int(right))))
        output.append(pauli_label(n_qubits, basis, tuple(range(n_qubits))))
    return output


def frame_response_jacobian(
    clean: QuantumCircuit,
    boundary: int,
    affected: int,
    pairs: np.ndarray,
) -> np.ndarray:
    """Exact first derivative d<Pauli evidence>/d(Z,X,Y rotation angle).

    If S is the suffix after the hidden rotation and P is a local generator,
    the propagated generator is G=S P S^dagger. For final reference state psi,
    d<M>/dtheta at zero equals (i/2)<[G,M]> = -Im(<G psi|M psi>).
    We evaluate that identity without materializing G as a dense operator.
    """
    if boundary < 1 or boundary > len(clean.data):
        raise RuntimeError("invalid Step-14 oracle boundary for local-frame propagation")
    if affected < 0 or affected >= clean.num_qubits:
        raise RuntimeError("invalid Step-14 affected qubit for local-frame propagation")
    prefix = circuit_slice(clean, 0, boundary)
    suffix = circuit_slice(clean, boundary, len(clean.data))
    psi_boundary = np.asarray(Statevector.from_instruction(prefix).data, dtype=np.complex128)
    psi_final = np.asarray(Statevector.from_instruction(clean).data, dtype=np.complex128)
    labels = measured_pauli_labels(clean.num_qubits, pairs)
    measured_vectors = [np.asarray(Pauli(label).to_matrix(), dtype=np.complex128) @ psi_final for label in labels]

    jac = np.zeros((len(labels), 3), dtype=np.float64)
    for column, axis in enumerate(AXES):
        generator = np.asarray(
            Pauli(pauli_label(clean.num_qubits, axis, (affected,))).to_matrix(),
            dtype=np.complex128,
        )
        boundary_generator_state = generator @ psi_boundary
        propagated_generator_state = np.asarray(
            Statevector(boundary_generator_state).evolve(suffix).data,
            dtype=np.complex128,
        )
        jac[:, column] = np.asarray(
            [-float(np.imag(np.vdot(propagated_generator_state, mpsi))) for mpsi in measured_vectors],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(jac)):
        raise RuntimeError("non-finite local-frame Born Jacobian")
    return jac


def measured_delta_and_weights(
    loaded: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_codes = np.asarray(loaded["x__diagnostic_basis_codes"])
    local = baseline.reorder_basis(
        np.asarray(loaded["x__delta_local_expectations"], dtype=np.float64), basis_codes
    )
    pair = baseline.reorder_basis(
        np.asarray(loaded["x__delta_pairwise_correlations"], dtype=np.float64), basis_codes
    )
    parity = baseline.reorder_basis(
        np.asarray(loaded["x__delta_global_parity"], dtype=np.float64).reshape(3, 1), basis_codes
    ).reshape(3)
    pairs = np.asarray(loaded["x__pair_indices"], dtype=np.int64).reshape(-1, 2)
    n_qubits = int(np.asarray(loaded["x__layout_logical_to_physical"]).size)
    if local.shape != (3, n_qubits) or pair.shape != (3, len(pairs)):
        raise RuntimeError("unexpected Step-14 measured-evidence shape")

    values: list[float] = []
    for basis_index in range(3):
        values.extend(float(v) for v in local[basis_index].tolist())
        values.extend(float(v) for v in pair[basis_index].tolist())
        values.append(float(parity[basis_index]))
    delta = np.asarray(values, dtype=np.float64)

    codes = [int(v) for v in basis_codes.reshape(-1).tolist()]
    if sorted(codes) != [0, 1, 2]:
        raise RuntimeError("unexpected diagnostic basis-code contract")
    order = [codes.index(code) for code in (0, 1, 2)]
    observed = np.asarray(loaded["x__observed_shots"], dtype=np.float64).reshape(-1)[order]
    reference = np.asarray(loaded["x__reference_shots"], dtype=np.float64).reshape(-1)[order]
    if observed.shape != (3,) or reference.shape != (3,) or np.any(observed <= 0) or np.any(reference <= 0):
        raise RuntimeError("invalid Step-14 paired shot counts")
    block = n_qubits + len(pairs) + 1
    weights = np.concatenate(
        [np.full(block, 1.0 / math.sqrt(1.0 / observed[i] + 1.0 / reference[i])) for i in range(3)]
    ).astype(np.float64)
    if delta.shape != weights.shape:
        raise RuntimeError("local-frame evidence/weight width mismatch")
    return delta, weights, pairs


def canonicalize_evidence(
    delta: np.ndarray,
    jacobian: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if jacobian.shape != (len(delta), 3) or weights.shape != delta.shape:
        raise RuntimeError("canonicalization shape mismatch")
    dw = np.asarray(delta * weights, dtype=np.float64)
    jw = np.asarray(jacobian * weights[:, None], dtype=np.float64)
    norms = np.linalg.norm(jw, axis=0)
    safe_norms = np.maximum(norms, 1.0e-12)
    normalized = jw / safe_norms[None, :]
    gram = normalized.T @ normalized
    ridge = float(PROBE_SPEC["ridge"])
    coords = np.linalg.solve(gram + ridge * np.eye(3), normalized.T @ dw)
    coord_norm = float(np.linalg.norm(coords))
    coord_unit = coords / max(coord_norm, 1.0e-12)
    data_norm = float(np.linalg.norm(dw))
    matched = normalized.T @ dw / max(data_norm, 1.0e-12)
    singular = np.linalg.svd(normalized, compute_uv=False)
    minimum = float(singular[-1]) if len(singular) else 0.0
    condition = float(singular[0] / max(minimum, 1.0e-12)) if len(singular) else float("inf")
    residual = float(np.linalg.norm(dw - normalized @ coords) / max(data_norm, 1.0e-12))
    feature = np.concatenate(
        (
            coords,
            np.abs(coords),
            coord_unit,
            matched,
            np.abs(matched),
            singular,
            np.log1p(norms),
            np.asarray(
                [
                    min(50.0, math.log1p(condition)),
                    residual,
                    math.log1p(data_norm),
                ],
                dtype=np.float64,
            ),
        )
    )
    audit = {
        "condition_number": condition,
        "minimum_singular_value": minimum,
        "residual_fraction": residual,
        "weighted_evidence_norm": data_norm,
    }
    return feature, coords, audit


def frame_geometry(jacobian: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / np.maximum(norms, 1.0e-12)[None, :]
    singular = np.linalg.svd(normalized, compute_uv=False)
    angles: list[float] = []
    for i in range(3):
        for j in range(i + 1, 3):
            cosine = float(abs(np.dot(normalized[:, i], normalized[:, j])))
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))))
    minimum = float(singular[-1]) if len(singular) else 0.0
    condition = float(singular[0] / max(minimum, 1.0e-12)) if len(singular) else float("inf")
    return {
        "minimum_pairwise_axis_angle_deg": float(min(angles)) if angles else 0.0,
        "minimum_singular_value": minimum,
        "condition_number": condition,
        "minimum_axis_response_norm": float(np.min(norms)),
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def build_table(
    product: Path,
    rows: Sequence[Mapping[str, str]],
    roots: Mapping[int, Mapping[str, str]],
    progress_every: int,
) -> dict[str, Any]:
    stage_names = (
        "raw_diagnostics",
        "raw_plus_affected_qubit_oracle",
        "raw_plus_affected_qubit_local_context_oracle",
        "canonical_local_frame",
    )
    features: dict[str, list[np.ndarray]] = {name: [] for name in stage_names}
    truth: list[int] = []
    partitions: list[str] = []
    families: list[str] = []
    source_rows: list[int] = []
    strengths: list[float] = []
    deterministic_logits: list[np.ndarray] = []
    frame_cache: dict[int, np.ndarray] = {}
    geometry_cache: dict[int, dict[str, float]] = {}
    canonical_audits: list[dict[str, float]] = []

    selected = [(index, row) for index, row in enumerate(rows) if str(row["mechanism"]) in TARGET]
    for position, (source_index, row) in enumerate(selected, start=1):
        root_index = int(row["root_index"])
        root = roots.get(root_index)
        if root is None:
            raise RuntimeError(f"missing Step-14 root {root_index}")
        partition = str(row["step14_partition"])
        if partition not in {"fit", "selection"} or str(root["step14_partition"]) != partition:
            raise RuntimeError("local-frame diagnostic encountered outer/reserve or split mismatch")
        artifact = product / str(row["artifact_path"])
        if baseline.sha256_file(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"artifact hash mismatch for {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as source:
            loaded = {key: source[key] for key in source.files}
        forbidden = [
            key for key in loaded
            if key.startswith("x__") and any(token in key.lower() for token in ("mechanism_target", "effect_target", "affected_qubit", "injection_boundary"))
        ]
        if forbidden:
            raise RuntimeError(f"privileged field leaked into deployable x__ input: {forbidden}")

        raw, _local, _pair = oracle.raw_diagnostic_features(loaded)
        ordinary = oracle.ordinary_circuit_context(root, loaded)
        affected = int(root["affected_qubit"])
        localized, localized_core = oracle.localized_oracle_features(loaded, affected)
        local_context = oracle.local_insertion_context(root, localized_core)
        delta, weights, pairs = measured_delta_and_weights(loaded)

        if root_index not in frame_cache:
            clean = circuit_from_serialized(loaded)
            signature = oracle.parse_operation_signature(str(root["operation_signature"]))
            reconstructed = []
            for item in clean.data:
                qs = tuple(int(clean.find_bit(q).index) for q in item.qubits)
                reconstructed.append((str(item.operation.name).lower(), qs))
            if reconstructed != signature:
                raise RuntimeError(f"reconstructed Step-14 circuit signature mismatch at root {root_index}")
            jacobian = frame_response_jacobian(
                clean,
                int(root["injection_boundary_rank"]),
                affected,
                pairs,
            )
            frame_cache[root_index] = jacobian
            geometry_cache[root_index] = frame_geometry(jacobian)
        jacobian = frame_cache[root_index]
        if jacobian.shape[0] != len(delta):
            raise RuntimeError("cached local-frame response width mismatch")
        canonical, coords, audit = canonicalize_evidence(delta, jacobian, weights)

        features["raw_diagnostics"].append(raw)
        features["raw_plus_affected_qubit_oracle"].append(np.concatenate((raw, ordinary, localized)))
        features["raw_plus_affected_qubit_local_context_oracle"].append(
            np.concatenate((raw, ordinary, localized, local_context))
        )
        features["canonical_local_frame"].append(canonical)
        deterministic_logits.append(np.abs(coords))
        canonical_audits.append(audit)
        truth.append(TARGET[str(row["mechanism"])])
        partitions.append(partition)
        families.append(str(row["family_id"]))
        source_rows.append(source_index)
        strengths.append(float(row["strength"]))
        if progress_every and position % progress_every == 0:
            print(f"local-frame feature extraction {position}/{len(selected)}", flush=True)

    arrays = {name: np.stack(values).astype(np.float32) for name, values in features.items()}
    y = np.asarray(truth, dtype=np.int64)
    part = np.asarray(partitions, dtype=object)
    fam = np.asarray(families, dtype=object)
    source = np.asarray(source_rows, dtype=np.int64)
    strength = np.asarray(strengths, dtype=np.float64)
    deterministic = np.stack(deterministic_logits).astype(np.float64)
    if not np.all(np.isfinite(deterministic)) or not all(np.all(np.isfinite(v)) for v in arrays.values()):
        raise RuntimeError("non-finite local-frame feature")
    fit = part == "fit"
    selection = part == "selection"
    if np.any(fit & selection) or not np.all(fit | selection):
        raise RuntimeError("unexpected Step-14 development partition")
    for name, mask in (("fit", fit), ("selection", selection)):
        counts = Counter(int(v) for v in y[mask].tolist())
        if set(counts) != {0, 1, 2} or len(set(counts.values())) != 1:
            raise RuntimeError(f"{name} mechanism classes are not exactly balanced: {counts}")

    geometry = list(geometry_cache.values())
    frame_summary = {
        "root_count": int(len(geometry)),
        "minimum_pairwise_axis_angle_deg": quantiles([v["minimum_pairwise_axis_angle_deg"] for v in geometry]),
        "minimum_singular_value": quantiles([v["minimum_singular_value"] for v in geometry]),
        "condition_number": quantiles([min(v["condition_number"], 1.0e12) for v in geometry]),
        "minimum_axis_response_norm": quantiles([v["minimum_axis_response_norm"] for v in geometry]),
        "fraction_minimum_axis_angle_below_15deg": float(np.mean([v["minimum_pairwise_axis_angle_deg"] < 15.0 for v in geometry])),
        "fraction_minimum_singular_below_0p1": float(np.mean([v["minimum_singular_value"] < 0.1 for v in geometry])),
    }
    canonical_summary = {
        "residual_fraction": quantiles([v["residual_fraction"] for v in canonical_audits]),
        "weighted_evidence_norm": quantiles([v["weighted_evidence_norm"] for v in canonical_audits]),
    }
    return {
        "features": arrays,
        "truth": y,
        "partition": part,
        "family": fam,
        "source_rows": source,
        "strength": strength,
        "deterministic_logits": deterministic,
        "frame_geometry": frame_summary,
        "canonicalization_audit": canonical_summary,
    }


def public_probe_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "selection_logits"}


def metric_with_kind(truth: np.ndarray, logits: np.ndarray, kind: str, width: int) -> dict[str, Any]:
    out = oracle.metric_record(truth, logits)
    out["probe_kind"] = kind
    out["feature_width"] = int(width)
    return out


def main() -> None:
    args = parse_args()
    oracle.MAX_GATES = MAX_GATES
    cfg = step14.read_json(CONFIG)
    step14.assert_contract(cfg)
    _run_dir, _freeze = rep14.verify_training_freeze(
        args.training_run_id, args.selection_freeze_sha256
    )
    previous = verify_previous_oracle()
    cross_product = step14.resolve_cross_product(None)
    cross_rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(cross_product, cfg)
    complete = json.loads((cross_product / "dataset_complete.json").read_text(encoding="utf-8"))
    manifests = cross_product / "manifests"
    root_rows = baseline.read_csv(manifests / "root_manifest.csv")
    roots = {int(row["root_index"]): row for row in root_rows}
    if len(roots) != len(root_rows):
        raise RuntimeError("duplicate root index in Step-14 root manifest")

    table = build_table(cross_product, cross_rows, roots, args.progress_every)
    fit_mask = table["partition"] == "fit"
    selection_mask = table["partition"] == "selection"
    y_fit = table["truth"][fit_mask]
    y_sel = table["truth"][selection_mask]
    groups_sel = table["family"][selection_mask]
    device = oracle.resolve_device(args.device)

    sources = {
        "raw_diagnostics": "raw_diagnostics",
        "raw_plus_affected_qubit_oracle": "raw_plus_affected_qubit_oracle",
        "raw_plus_affected_qubit_local_context_oracle": "raw_plus_affected_qubit_local_context_oracle",
        "canonical_local_frame_small_probe": "canonical_local_frame",
        "canonical_local_frame_high_capacity": "canonical_local_frame",
    }
    logits_by_stage: dict[str, list[np.ndarray]] = {name: [] for name in sources}
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in PROBE_SEEDS:
        per_seed[str(seed)] = {}
        for stage, source in sources.items():
            x = table["features"][source]
            high_capacity = stage == "canonical_local_frame_high_capacity"
            record = oracle.fit_probe(
                x[fit_mask], y_fit, x[selection_mask], y_sel,
                seed=seed, high_capacity=high_capacity, device=device,
            )
            logits_by_stage[stage].append(np.asarray(record["selection_logits"], dtype=np.float64))
            per_seed[str(seed)][stage] = public_probe_record(record)
            print(
                f"{stage} seed={seed} fit_BA={record['fit_balanced_accuracy']:.4f} "
                f"selection_BA={record['selection_balanced_accuracy']:.4f}",
                flush=True,
            )

    averaged_logits = {
        stage: np.mean(np.stack(values, axis=0), axis=0)
        for stage, values in logits_by_stage.items()
    }
    averaged_logits["canonical_local_frame_deterministic_rule"] = table["deterministic_logits"][selection_mask]

    ensemble_metrics: dict[str, Any] = {}
    for stage, logits in averaged_logits.items():
        source = sources.get(stage, "canonical_local_frame")
        kind = (
            "deterministic_argmax_abs_canonical_coordinate"
            if stage == "canonical_local_frame_deterministic_rule"
            else "high_capacity_ceiling"
            if stage == "canonical_local_frame_high_capacity"
            else "small_fixed_probe"
        )
        ensemble_metrics[stage] = metric_with_kind(
            y_sel, logits, kind, table["features"][source].shape[1]
        )

    comparisons = {
        "affected_qubit_oracle_over_raw": (
            "raw_plus_affected_qubit_oracle", "raw_diagnostics"
        ),
        "raw_neighborhood_over_location": (
            "raw_plus_affected_qubit_local_context_oracle", "raw_plus_affected_qubit_oracle"
        ),
        "canonical_frame_over_raw_neighborhood": (
            "canonical_local_frame_small_probe", "raw_plus_affected_qubit_local_context_oracle"
        ),
        "canonical_frame_over_raw": (
            "canonical_local_frame_small_probe", "raw_diagnostics"
        ),
        "canonical_high_capacity_over_small": (
            "canonical_local_frame_high_capacity", "canonical_local_frame_small_probe"
        ),
        "canonical_rule_over_raw": (
            "canonical_local_frame_deterministic_rule", "raw_diagnostics"
        ),
    }
    deltas: dict[str, Any] = {}
    for offset, (name, (candidate, reference)) in enumerate(comparisons.items()):
        record = oracle.bootstrap_delta(
            y_sel,
            averaged_logits[candidate],
            averaged_logits[reference],
            groups_sel,
            seed=int(PROBE_SPEC["bootstrap_seed"]) + offset,
        )
        record.update({"candidate": candidate, "reference": reference})
        deltas[name] = record

    primary_delta = deltas["canonical_frame_over_raw_neighborhood"]
    primary_ba = float(ensemble_metrics["canonical_local_frame_small_probe"]["mechanism_balanced_accuracy"])
    ceiling_ba = float(ensemble_metrics["canonical_local_frame_high_capacity"]["mechanism_balanced_accuracy"])
    rule_ba = float(ensemble_metrics["canonical_local_frame_deterministic_rule"]["mechanism_balanced_accuracy"])
    ci_low = float(primary_delta["bootstrap_ci"][0])
    supported = (
        float(primary_delta["mean_delta"]) >= float(PROBE_SPEC["meaningful_canonicalization_delta_minimum"])
        and ci_low > 0.0
    )
    if supported:
        frame_verdict = "SUPPORTED_LOCAL_FRAME_CANONICALIZATION_SIGNAL"
    elif ceiling_ba < float(PROBE_SPEC["canonical_transferable_ceiling_low_ba"]):
        frame_verdict = "NOT_SUPPORTED__LOW_TRANSFERABLE_CANONICAL_FRAME_CEILING"
    else:
        frame_verdict = "MIXED_LOCAL_FRAME_SIGNAL"
    evidence_update = (
        "EVIDENCE_LIMIT_HYPOTHESIS_STRENGTHENED"
        if not supported and ceiling_ba < float(PROBE_SPEC["canonical_transferable_ceiling_low_ba"])
        else "EVIDENCE_LIMIT_NOT_RESOLVED"
    )

    identity = {
        "schema": SCHEMA,
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "protocol_config_sha256": baseline.sha256_file(CONFIG),
        "cross_dataset_product_id": str(complete["product_id"]),
        "cross_dataset_complete_sha256": baseline.sha256_file(cross_product / "dataset_complete.json"),
        "root_manifest_sha256": baseline.sha256_file(manifests / "root_manifest.csv"),
        "example_manifest_sha256": baseline.sha256_file(manifests / "example_manifest.csv"),
        "previous_oracle_diagnostic_id": PREVIOUS_ORACLE_ID,
        "previous_oracle_result_sha256": PREVIOUS_ORACLE_RESULT_SHA256,
        "probe_spec": PROBE_SPEC,
        "axis_order": list(AXES),
        "mechanism_order": list(MECHANISMS),
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_accessed": False,
        "privileged_fields_analysis_only": True,
    }
    diagnostic_id = "local_frame_" + oracle.stable_hash(identity).split(":", 1)[1][:24]
    out_parent = args.output_parent.expanduser().resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_parent / diagnostic_id
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite local-frame diagnostic {out_dir}")
    out_dir.mkdir()

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_LOCAL_FRAME_CANONICALIZATION",
        "diagnostic_id": diagnostic_id,
        "identity": identity,
        "counts": {
            "fit_injected_examples": int(np.sum(fit_mask)),
            "selection_injected_examples": int(np.sum(selection_mask)),
            "fit_families": int(len(set(table["family"][fit_mask].tolist()))),
            "selection_families": int(len(set(table["family"][selection_mask].tolist()))),
        },
        "frame_geometry": table["frame_geometry"],
        "canonicalization_audit": table["canonicalization_audit"],
        "per_seed": per_seed,
        "selection_ensemble_metrics": ensemble_metrics,
        "paired_family_bootstrap_deltas": deltas,
        "hypothesis_verdicts": {
            "local_frame_canonicalization_hypothesis": frame_verdict,
            "raw_model_visible_evidence_limit_update": evidence_update,
            "deterministic_canonical_rule_ba": rule_ba,
            "canonical_small_probe_ba": primary_ba,
            "canonical_high_capacity_ba": ceiling_ba,
            "privileged_information_remains_analysis_only": True,
        },
        "scientific_boundaries": {
            "main_model_retrained": False,
            "main_model_weights_updated": False,
            "selection_used_for_probe_training_or_early_stopping": False,
            "affected_qubit_and_boundary_analysis_only": True,
            "canonical_frame_written_back_to_dataset_or_checkpoint": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
        },
        "previous_oracle_verdicts": previous.get("hypothesis_verdicts", {}),
    }
    atomic_json(out_dir / "diagnostic_result.json", result)
    result_sha = baseline.sha256_file(out_dir / "diagnostic_result.json")
    complete_payload = {
        "schema": SCHEMA,
        "status": result["status"],
        "diagnostic_id": diagnostic_id,
        "diagnostic_result_sha256": result_sha,
        "hypothesis_verdicts": result["hypothesis_verdicts"],
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    }
    atomic_json(out_dir / "diagnostic_complete.json", complete_payload)
    complete_sha = baseline.sha256_file(out_dir / "diagnostic_complete.json")
    pointer = {
        "schema": "triqto.v0_2.step14_current_local_frame_canonicalization.v1",
        "diagnostic_id": diagnostic_id,
        "diagnostic_dir": str(out_dir),
        "diagnostic_result_sha256": result_sha,
        "diagnostic_complete_sha256": complete_sha,
    }
    atomic_json(out_parent / "current_local_frame_canonicalization.json", pointer)
    print(json.dumps({**complete_payload, "diagnostic_complete_sha256": complete_sha}, indent=2), flush=True)


if __name__ == "__main__":
    main()
