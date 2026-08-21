#!/usr/bin/env python3
"""Step 9E post-hoc deployable local-response-frame audit.

Purpose
-------
Step 9D showed that the frozen six-program Born diagnostic is strongly
mechanism-identifiable when interpreted in the exact same-context RZ/RX/RY
response frame, but that frame is simulator privileged. Step 9E asks whether a
useful approximation to that frame can be obtained from information that can be
available at hardware inference time.

The audit is deliberately staged:

Tier 1 -- circuit-only local frame
    A fixed nonlinear kernel-ridge audit estimator maps intended/reference
    circuit structure plus a *candidate* local context (qubit and insertion
    boundary) to three 12D RZ/RX/RY response vectors. The true hidden mechanism,
    finite-shot query delta, exact frame, statevector, and Hilbert metrics are
    not inputs.

Tier 2 -- circuit + existing Born context
    The same estimator additionally receives absolute baseline Z/X/Y local,
    pairwise, and parity expectations. On Step 5 these are finite-shot simulator
    samples of a hardware-observable reference measurement. On the frozen QPU
    pilot they come from already-recorded reference counts. The mechanism query
    delta is still not an estimator input.

Tier 3 -- cheap diagnostic probes (conditional)
    If Tiers 1-2 do not establish a deployable approximation, emulate a bounded
    hardware-valid calibration protocol: at the candidate context run known
    low-strength RZ/RX/RY probes, measure each in Z/X/Y, and use the finite-shot
    probe deltas directly as the local frame (scaled to the target strength).
    Frozen Step-9D QPU data do not contain these extra probe programs, so Tier 3
    cannot claim QPU validation without a separately authorized bounded run.

Evaluation discipline
---------------------
* No TriQTO checkpoint is retrained or retuned.
* Tiers 1-2 fit only an audit estimator to simulator-privileged exact-frame
  targets. Primary Step-5 evaluation is on the frozen validation split only.
* After primary Step-5 evaluation, the estimator is refit on all matched Step-5
  contexts solely for the independent frozen Step-9D QPU transfer check.
* Existing QPU counts are read only. This script never submits a QPU job.
* The finite-shot mechanism query is used only by the decoder after the frame is
  predicted; it is never an input to the Tier-1/2 frame estimator.
* This remains a post-hoc exploratory audit and does not authorize retraining.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from analyze_step9d_posthoc_context_identifiability import (
    BASIS_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_POINTER,
    DEFAULT_QPU_RESULTS,
    MECHANISMS,
    ROOT,
    _cosine,
    _matched_contexts,
    _pilot_phase_exact_frame,
    _qpu_vector,
    _read_json,
    _read_jsonl,
    _resolve_product,
    _score_frame,
)
from triqto.hardware.qpu_pilot import build_pilot_cases


ANALYSIS_NAME = "step9e_posthoc_deployable_local_response_frame_audit_v1"
DEFAULT_OUTPUT = Path("/workspace/triqto-data/step9e_posthoc/deployable_local_frame_v1.json")
DEFAULT_BOOTSTRAPS = 4000
DEFAULT_BOOTSTRAP_SEED = 20260819

# Frozen exploratory gates inherited from Step 9D. These are engineering/design
# gates, not hypothesis-test significance thresholds.
GATE_BALANCED_ACCURACY = 0.80
GATE_BOOTSTRAP_LOWER = 0.75
GATE_MIN_RECALL = 0.70

TARGET_STRENGTH = 0.15
TARGET_FAMILY = "phase_interference"
TARGET_N_QUBITS = 2
TARGET_QUBIT = 0

# Predeclared estimator/probe settings. No outcome-driven hyperparameter search.
DEFAULT_RIDGE_ALPHA = 1e-2
DEFAULT_PROBE_STRENGTH = 0.05
DEFAULT_PROBE_SHOTS = 512
DEFAULT_TIER3_MODE = "auto"
MAX_EVENTS = 16
GATE_VOCAB = ("h", "rx", "ry", "rz", "cx", "cz", "cp", "rzz", "swap", "other")


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps([str(part) for part in parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selected_artifact_paths(product: Path) -> dict[tuple[int, int], Path]:
    """One intended/reference graph artifact per matched local context."""
    rows = _read_manifest(product / "manifests/example_manifest.csv")
    selected: dict[tuple[int, int], Path] = {}
    for row in rows:
        if row.get("family") != TARGET_FAMILY:
            continue
        if int(row["n_qubits"]) != TARGET_N_QUBITS:
            continue
        if _as_bool(row["clean_control"]):
            continue
        if not _as_bool(row["mechanism_loss_mask"]):
            continue
        if int(row["affected_qubit"]) != TARGET_QUBIT:
            continue
        if abs(float(row["strength"]) - TARGET_STRENGTH) > 1e-12:
            continue
        if row["mechanism"] != "rz_drift":
            continue
        key = (int(row["root_index"]), int(row["insertion_boundary_rank"]))
        selected[key] = product / row["artifact_path"]
    return selected


def _gate_name(name: str) -> str:
    name = str(name).lower()
    return name if name in GATE_VOCAB[:-1] else "other"


def _graph_to_circuit(path: Path) -> QuantumCircuit:
    """Reconstruct the intended 2q phase circuit from persisted deployable graph arrays."""
    with np.load(path, allow_pickle=False) as data:
        names = [str(x).lower() for x in data["x__graph_gate_names"].tolist()]
        qptr = np.asarray(data["x__graph_gate_qubit_ptr"], dtype=np.int64)
        qidx = np.asarray(data["x__graph_gate_qubit_indices"], dtype=np.int64)
        pptr = np.asarray(data["x__graph_gate_parameter_ptr"], dtype=np.int64)
        psin = np.asarray(data["x__graph_gate_parameter_sin"], dtype=np.float64)
        pcos = np.asarray(data["x__graph_gate_parameter_cos"], dtype=np.float64)
        layout = np.asarray(data["x__layout_logical_to_physical"], dtype=np.int64)

    n_qubits = int(layout.size)
    if n_qubits != TARGET_N_QUBITS:
        raise RuntimeError(f"Step 9E expected 2q graph, got {n_qubits}q from {path}")
    circuit = QuantumCircuit(n_qubits)
    for index, name in enumerate(names):
        qargs = [int(x) for x in qidx[qptr[index] : qptr[index + 1]].tolist()]
        angles = [
            float(math.atan2(psin[j], pcos[j]))
            for j in range(int(pptr[index]), int(pptr[index + 1]))
        ]
        if name == "h" and len(qargs) == 1 and not angles:
            circuit.h(qargs[0])
        elif name == "rx" and len(qargs) == 1 and len(angles) == 1:
            circuit.rx(angles[0], qargs[0])
        elif name == "ry" and len(qargs) == 1 and len(angles) == 1:
            circuit.ry(angles[0], qargs[0])
        elif name == "rz" and len(qargs) == 1 and len(angles) == 1:
            circuit.rz(angles[0], qargs[0])
        elif name == "cx" and len(qargs) == 2 and not angles:
            circuit.cx(qargs[0], qargs[1])
        elif name == "cz" and len(qargs) == 2 and not angles:
            circuit.cz(qargs[0], qargs[1])
        elif name == "cp" and len(qargs) == 2 and len(angles) == 1:
            circuit.cp(angles[0], qargs[0], qargs[1])
        elif name == "rzz" and len(qargs) == 2 and len(angles) == 1:
            circuit.rzz(angles[0], qargs[0], qargs[1])
        elif name == "swap" and len(qargs) == 2 and not angles:
            circuit.swap(qargs[0], qargs[1])
        else:
            raise RuntimeError(
                f"unsupported persisted gate contract in Step 9E: name={name!r}, "
                f"qargs={qargs}, params={angles}"
            )
    return circuit


def _instruction_record(circuit: QuantumCircuit, index: int) -> tuple[str, tuple[int, ...], list[float]]:
    instruction = circuit.data[index]
    name = _gate_name(str(instruction.operation.name).lower())
    qargs = tuple(int(circuit.find_bit(q).index) for q in instruction.qubits)
    params = [float(x) for x in instruction.operation.params]
    return name, qargs, params


def _circuit_features(circuit: QuantumCircuit, boundary: int, candidate_qubit: int) -> np.ndarray:
    """Fixed deployable graph + candidate-context feature map.

    candidate_qubit/boundary are not claimed to be known hidden truth at hardware
    deployment. They identify the local context being scored. A future system may
    enumerate candidate contexts or obtain them from a separate localization head.
    """
    if circuit.num_qubits != TARGET_N_QUBITS:
        raise RuntimeError("targeted Step 9E circuit feature contract is 2q")
    n_events = len(circuit.data)
    if n_events > MAX_EVENTS:
        raise RuntimeError(f"circuit has {n_events} events > frozen MAX_EVENTS={MAX_EVENTS}")
    if not (1 <= int(boundary) <= n_events):
        raise RuntimeError(f"candidate boundary {boundary} outside [1,{n_events}]")
    if not (0 <= int(candidate_qubit) < circuit.num_qubits):
        raise RuntimeError("candidate qubit outside circuit")

    vocab_index = {name: i for i, name in enumerate(GATE_VOCAB)}
    records = [_instruction_record(circuit, i) for i in range(n_events)]
    out: list[float] = []

    # Scalars + candidate context.
    out.extend(
        [
            float(circuit.num_qubits) / 8.0,
            float(n_events) / float(MAX_EVENTS),
            float(boundary) / float(MAX_EVENTS),
            float(boundary) / float(max(1, n_events)),
        ]
    )
    out.extend([1.0 if candidate_qubit == q else 0.0 for q in range(TARGET_N_QUBITS)])
    out.extend([1.0 if boundary == b else 0.0 for b in range(MAX_EVENTS + 1)])

    # Aggregate gate counts in six structural regions.
    regions: list[Iterable[int]] = [
        range(n_events),
        range(0, boundary),
        range(boundary, n_events),
        [i for i, (_, qargs, _) in enumerate(records) if candidate_qubit in qargs],
        [i for i, (_, qargs, _) in enumerate(records[:boundary]) if candidate_qubit in qargs],
        [
            i + boundary
            for i, (_, qargs, _) in enumerate(records[boundary:])
            if candidate_qubit in qargs
        ],
    ]
    for region in regions:
        counts = np.zeros(len(GATE_VOCAB), dtype=np.float64)
        for i in region:
            counts[vocab_index[records[int(i)][0]]] += 1.0
        out.extend((counts / float(max(1, n_events))).tolist())

    # Exact intended gate sequence with phasor parameter encoding.
    for slot in range(MAX_EVENTS):
        onehot = np.zeros(len(GATE_VOCAB), dtype=np.float64)
        qmask = np.zeros(TARGET_N_QUBITS, dtype=np.float64)
        arity = 0.0
        ps = 0.0
        pc = 0.0
        is_before = 0.0
        if slot < n_events:
            name, qargs, params = records[slot]
            onehot[vocab_index[name]] = 1.0
            for q in qargs:
                if 0 <= q < TARGET_N_QUBITS:
                    qmask[q] = 1.0
            arity = float(len(qargs)) / 2.0
            if params:
                ps = float(math.sin(params[0]))
                pc = float(math.cos(params[0]))
            is_before = 1.0 if slot < boundary else 0.0
        out.extend(onehot.tolist())
        out.extend(qmask.tolist())
        out.extend([arity, ps, pc, is_before])

    return np.asarray(out, dtype=np.float64)


def _basis_probabilities(circuit: QuantumCircuit, basis: str) -> np.ndarray:
    state = Statevector.from_instruction(circuit)
    rotate = QuantumCircuit(circuit.num_qubits)
    if basis == "X":
        for q in range(circuit.num_qubits):
            rotate.h(q)
    elif basis == "Y":
        for q in range(circuit.num_qubits):
            rotate.sdg(q)
            rotate.h(q)
    elif basis != "Z":
        raise ValueError(f"unknown basis {basis!r}")
    rotated = state.evolve(rotate)
    probs = np.abs(np.asarray(rotated.data, dtype=np.complex128)) ** 2
    return np.asarray(probs / np.sum(probs), dtype=np.float64)


def _sample_counts(probabilities: np.ndarray, shots: int, seed: int, n_qubits: int) -> dict[str, int]:
    rng = np.random.default_rng(int(seed))
    draw = rng.multinomial(int(shots), np.asarray(probabilities, dtype=np.float64))
    return {
        format(index, f"0{n_qubits}b"): int(count)
        for index, count in enumerate(draw.tolist())
        if int(count) > 0
    }


def _sample_basis_counts(circuit: QuantumCircuit, shots: int, seed_parts: Sequence[Any]) -> dict[str, dict[str, int]]:
    return {
        basis: _sample_counts(
            _basis_probabilities(circuit, basis),
            shots,
            _stable_seed(*seed_parts, basis),
            circuit.num_qubits,
        )
        for basis in BASIS_ORDER
    }


def _absolute_stats_vector(counts_by_basis: Mapping[str, Mapping[str, int]], n_qubits: int) -> np.ndarray:
    """Absolute Z/X/Y local + pairwise + parity expectations from counts."""
    if n_qubits != TARGET_N_QUBITS:
        raise RuntimeError("targeted Step 9E Born-context contract is 2q")
    local_blocks: list[np.ndarray] = []
    pair_blocks: list[float] = []
    parity_blocks: list[float] = []
    for basis in BASIS_ORDER:
        counts = counts_by_basis[basis]
        shots = int(sum(int(v) for v in counts.values()))
        if shots <= 0:
            raise RuntimeError(f"zero counts for basis {basis}")
        local = np.zeros(n_qubits, dtype=np.float64)
        pair = 0.0
        parity = 0.0
        for raw_key, raw_count in counts.items():
            key = str(raw_key).replace(" ", "")
            if len(key) != n_qubits:
                raise RuntimeError(f"unexpected bitstring {raw_key!r} for {n_qubits}q")
            count = float(raw_count)
            eig = np.asarray(
                [1.0 - 2.0 * float(int(key[-1 - q])) for q in range(n_qubits)],
                dtype=np.float64,
            )
            local += count * eig
            pair += count * float(eig[0] * eig[1])
            parity += count * float(np.prod(eig))
        local_blocks.append(local / float(shots))
        pair_blocks.append(pair / float(shots))
        parity_blocks.append(parity / float(shots))
    return np.concatenate(
        [
            np.asarray(local_blocks, dtype=np.float64).reshape(-1),
            np.asarray(pair_blocks, dtype=np.float64).reshape(-1),
            np.asarray(parity_blocks, dtype=np.float64).reshape(-1),
        ]
    )


def _inject_rotation(
    clean: QuantumCircuit,
    boundary: int,
    qubit: int,
    mechanism: str,
    strength: float,
) -> QuantumCircuit:
    observed = QuantumCircuit(clean.num_qubits)
    for instruction in clean.data[:boundary]:
        qargs = [observed.qubits[clean.find_bit(q).index] for q in instruction.qubits]
        observed.append(instruction.operation, qargs, [])
    if mechanism == "rz_drift":
        observed.rz(strength, qubit)
    elif mechanism == "rx_overrotation":
        observed.rx(strength, qubit)
    elif mechanism == "ry_overrotation":
        observed.ry(strength, qubit)
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    for instruction in clean.data[boundary:]:
        qargs = [observed.qubits[clean.find_bit(q).index] for q in instruction.qubits]
        observed.append(instruction.operation, qargs, [])
    return observed


def _tier3_probe_frame(
    clean: QuantumCircuit,
    boundary: int,
    qubit: int,
    *,
    probe_strength: float,
    target_strength: float,
    shots: int,
    seed_parts: Sequence[Any],
) -> dict[str, np.ndarray]:
    reference_counts = _sample_basis_counts(clean, shots, [*seed_parts, "reference"])
    reference = _absolute_stats_vector(reference_counts, clean.num_qubits)
    scale = float(target_strength / probe_strength)
    frame: dict[str, np.ndarray] = {}
    for mechanism in MECHANISMS:
        probe = _inject_rotation(clean, boundary, qubit, mechanism, probe_strength)
        observed_counts = _sample_basis_counts(
            probe, shots, [*seed_parts, "probe", mechanism]
        )
        observed = _absolute_stats_vector(observed_counts, clean.num_qubits)
        frame[mechanism] = scale * (observed - reference)
    return frame


@dataclass
class RBFKernelRidge:
    mean: np.ndarray
    scale: np.ndarray
    x_train: np.ndarray
    gamma: float
    alpha: float
    y_mean: np.ndarray
    dual: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        z = (values - self.mean) / self.scale
        d2 = np.sum((z[:, None, :] - self.x_train[None, :, :]) ** 2, axis=2)
        kernel = np.exp(-self.gamma * d2)
        return kernel @ self.dual + self.y_mean


def _fit_rbf_kernel_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RBFKernelRidge:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid audit estimator matrices")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    z = (x - mean) / scale
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    nonzero = d2[d2 > 1e-12]
    median_d2 = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = 1.0 / max(median_d2, 1e-12)
    kernel = np.exp(-gamma * d2)
    y_mean = np.mean(y, axis=0, keepdims=True)
    centered = y - y_mean
    system = kernel + (float(alpha) + 1e-10) * np.eye(kernel.shape[0])
    dual = np.linalg.solve(system, centered)
    return RBFKernelRidge(
        mean=mean,
        scale=scale,
        x_train=z,
        gamma=float(gamma),
        alpha=float(alpha),
        y_mean=y_mean,
        dual=dual,
    )


def _flatten_frame(frame: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(frame[m], dtype=np.float64) for m in MECHANISMS])


def _split_frame(vector: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    expected = 12 * len(MECHANISMS)
    if values.size != expected:
        raise RuntimeError(f"predicted frame has {values.size} values != {expected}")
    return {m: values[i * 12 : (i + 1) * 12] for i, m in enumerate(MECHANISMS)}


def _frame_alignment(predicted: Mapping[str, np.ndarray], exact: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    cosines: list[float] = []
    relative_errors: list[float] = []
    for mechanism in MECHANISMS:
        pred = np.asarray(predicted[mechanism], dtype=np.float64)
        truth = np.asarray(exact[mechanism], dtype=np.float64)
        cosine = _cosine(pred, truth)
        rel = float(np.linalg.norm(pred - truth) / max(float(np.linalg.norm(truth)), 1e-12))
        norm_ratio = float(np.linalg.norm(pred) / max(float(np.linalg.norm(truth)), 1e-12))
        if cosine is not None and math.isfinite(float(cosine)):
            cosines.append(float(cosine))
        if math.isfinite(rel):
            relative_errors.append(rel)
        rows[mechanism] = {
            "cosine": cosine,
            "relative_l2_error": rel,
            "norm_ratio": norm_ratio,
        }
    return {
        "by_mechanism": rows,
        "median_cosine": float(np.median(cosines)) if cosines else None,
        "minimum_cosine": min(cosines) if cosines else None,
        "median_relative_l2_error": (
            float(np.median(relative_errors)) if relative_errors else None
        ),
    }


def _aggregate_alignment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_mechanism: dict[str, list[float]] = {m: [] for m in MECHANISMS}
    rels: dict[str, list[float]] = {m: [] for m in MECHANISMS}
    for row in rows:
        for mechanism in MECHANISMS:
            item = row["by_mechanism"][mechanism]
            if item["cosine"] is not None:
                by_mechanism[mechanism].append(float(item["cosine"]))
            rels[mechanism].append(float(item["relative_l2_error"]))
    pooled = [value for values in by_mechanism.values() for value in values]
    return {
        "median_cosine_all": float(np.median(pooled)) if pooled else None,
        "p10_cosine_all": float(np.percentile(pooled, 10.0)) if pooled else None,
        "by_mechanism": {
            mechanism: {
                "median_cosine": (
                    float(np.median(by_mechanism[mechanism]))
                    if by_mechanism[mechanism]
                    else None
                ),
                "p10_cosine": (
                    float(np.percentile(by_mechanism[mechanism], 10.0))
                    if by_mechanism[mechanism]
                    else None
                ),
                "median_relative_l2_error": (
                    float(np.median(rels[mechanism])) if rels[mechanism] else None
                ),
            }
            for mechanism in MECHANISMS
        },
    }


def _decoder_records(
    contexts: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, np.ndarray]],
    metric: str,
) -> list[dict[str, Any]]:
    if len(contexts) != len(frames):
        raise ValueError("context/frame count mismatch")
    records: list[dict[str, Any]] = []
    for index, (context, frame) in enumerate(zip(contexts, frames, strict=True)):
        for mechanism in MECHANISMS:
            scores, prediction, margin = _score_frame(
                np.asarray(context["finite"][mechanism]), frame, metric
            )
            records.append(
                {
                    "context_index": index,
                    "root_index": int(context["root_index"]),
                    "boundary": int(context["boundary"]),
                    "true": mechanism,
                    "prediction": prediction,
                    "correct": bool(prediction == mechanism),
                    "scores": scores,
                    "margin": margin,
                    "shots": int(context["shots"][mechanism]),
                }
            )
    return records


def _basic_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confusion = {true: {pred: 0 for pred in MECHANISMS} for true in MECHANISMS}
    recalls: dict[str, float | None] = {}
    usable = [r for r in records if r.get("prediction") in MECHANISMS]
    for row in usable:
        confusion[str(row["true"])][str(row["prediction"])] += 1
    for mechanism in MECHANISMS:
        subset = [r for r in usable if r["true"] == mechanism]
        recalls[mechanism] = (
            float(sum(bool(r["correct"]) for r in subset) / len(subset)) if subset else None
        )
    vals = [float(v) for v in recalls.values() if v is not None]
    margins = [
        float(r["margin"])
        for r in usable
        if r.get("margin") is not None and math.isfinite(float(r["margin"]))
    ]
    return {
        "record_count": len(records),
        "usable_count": len(usable),
        "balanced_accuracy": float(np.mean(vals)) if len(vals) == len(MECHANISMS) else None,
        "recalls": recalls,
        "confusion": confusion,
        "median_margin": float(np.median(margins)) if margins else None,
    }


def _cluster_bootstrap_ci(
    records: Sequence[Mapping[str, Any]], bootstraps: int, seed: int
) -> tuple[float | None, float | None]:
    by_context: dict[int, list[Mapping[str, Any]]] = {}
    for row in records:
        by_context.setdefault(int(row["context_index"]), []).append(row)
    ids = sorted(by_context)
    if not ids:
        return None, None
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(bootstraps)):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        rows: list[Mapping[str, Any]] = []
        for context_id in sampled.tolist():
            rows.extend(by_context[int(context_id)])
        ba = _basic_metrics(rows)["balanced_accuracy"]
        if ba is not None:
            values.append(float(ba))
    if not values:
        return None, None
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def _summarize_decoder(records: Sequence[Mapping[str, Any]], bootstraps: int, seed: int) -> dict[str, Any]:
    metrics = _basic_metrics(records)
    lower, upper = _cluster_bootstrap_ci(records, bootstraps, seed)
    metrics["balanced_accuracy_cluster_bootstrap_95_ci"] = [lower, upper]
    return metrics


def _gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    ba = summary.get("balanced_accuracy")
    ci = summary.get("balanced_accuracy_cluster_bootstrap_95_ci") or [None, None]
    lower = ci[0]
    recalls = summary.get("recalls", {})
    recall_values = [recalls.get(m) for m in MECHANISMS]
    minimum_recall = (
        min(float(v) for v in recall_values if v is not None)
        if all(v is not None for v in recall_values)
        else None
    )
    passed = bool(
        ba is not None
        and lower is not None
        and minimum_recall is not None
        and float(ba) >= GATE_BALANCED_ACCURACY
        and float(lower) >= GATE_BOOTSTRAP_LOWER
        and float(minimum_recall) >= GATE_MIN_RECALL
    )
    return {
        "passed": passed,
        "balanced_accuracy": ba,
        "balanced_accuracy_95_lower": lower,
        "minimum_mechanism_recall": minimum_recall,
        "thresholds": {
            "balanced_accuracy": GATE_BALANCED_ACCURACY,
            "balanced_accuracy_95_lower": GATE_BOOTSTRAP_LOWER,
            "minimum_mechanism_recall": GATE_MIN_RECALL,
        },
    }


def _context_rows(product: Path, contexts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    paths = _selected_artifact_paths(product)
    output: list[dict[str, Any]] = []
    for context in contexts:
        key = (int(context["root_index"]), int(context["boundary"]))
        path = paths.get(key)
        if path is None:
            raise RuntimeError(f"missing persisted graph artifact for context {key}")
        circuit = _graph_to_circuit(path)
        output.append({**dict(context), "circuit": circuit, "artifact_path": str(path)})
    return output


def _born_context_for_step5(context: Mapping[str, Any]) -> np.ndarray:
    shots = int(next(iter(context["shots"].values())))
    counts = _sample_basis_counts(
        context["circuit"],
        shots,
        ["step9e", "born_context", context["root_index"], context["boundary"], shots],
    )
    stats = _absolute_stats_vector(counts, TARGET_N_QUBITS)
    return np.concatenate(
        [
            stats,
            np.asarray(
                [
                    math.log2(max(shots, 1)) / 16.0,
                    1.0 / math.sqrt(max(shots, 1)),
                ],
                dtype=np.float64,
            ),
        ]
    )


def _feature_matrix(contexts: Sequence[Mapping[str, Any]], tier: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for context in contexts:
        circuit = context["circuit"]
        base = _circuit_features(circuit, int(context["boundary"]), TARGET_QUBIT)
        if tier == "tier1_circuit_only":
            row = base
        elif tier == "tier2_circuit_plus_born_context":
            row = np.concatenate([base, _born_context_for_step5(context)])
        else:
            raise ValueError(f"unknown learned tier {tier!r}")
        rows.append(row)
    return np.vstack(rows)


def _target_matrix(contexts: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.vstack([_flatten_frame(context["exact"]) for context in contexts])


def _predicted_frames(model: RBFKernelRidge, features: np.ndarray) -> list[dict[str, np.ndarray]]:
    predictions = model.predict(features)
    return [_split_frame(row) for row in predictions]


def _evaluate_step5_tier(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    tier: str,
    *,
    alpha: float,
    bootstraps: int,
    seed: int,
) -> tuple[dict[str, Any], RBFKernelRidge]:
    x_train = _feature_matrix(train, tier)
    y_train = _target_matrix(train)
    x_validation = _feature_matrix(validation, tier)
    model = _fit_rbf_kernel_ridge(x_train, y_train, alpha)
    frames = _predicted_frames(model, x_validation)
    alignments = [
        _frame_alignment(frame, context["exact"])
        for frame, context in zip(frames, validation, strict=True)
    ]
    decoders: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for metric_index, metric in enumerate(("cosine", "euclidean")):
        records = _decoder_records(validation, frames, metric)
        summary = _summarize_decoder(records, bootstraps, seed + metric_index)
        decoders[metric] = summary
        gates[metric] = _gate(summary)
    return (
        {
            "fit_context_count": len(train),
            "validation_context_count": len(validation),
            "feature_dimension": int(x_train.shape[1]),
            "estimator": {
                "kind": "fixed_rbf_kernel_ridge",
                "alpha": float(alpha),
                "gamma_from_training_median_squared_distance": float(model.gamma),
                "outcome_driven_hyperparameter_search": False,
            },
            "frame_alignment_to_exact": _aggregate_alignment(alignments),
            "decoders": decoders,
            "gates": gates,
        },
        model,
    )


def _phase_qpu_rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    clean = [
        row
        for row in rows
        if row.get("family") == TARGET_FAMILY and row.get("condition") == "clean"
    ]
    if len(clean) != 1:
        raise RuntimeError(f"expected one frozen phase clean QPU row, got {len(clean)}")
    mechanisms = {
        str(row["expected_mechanism"]): row
        for row in rows
        if row.get("family") == TARGET_FAMILY
        and row.get("expected_mechanism") in MECHANISMS
    }
    if set(mechanisms) != set(MECHANISMS):
        raise RuntimeError(f"incomplete frozen phase QPU mechanism rows: {sorted(mechanisms)}")
    return clean[0], mechanisms


def _qpu_reference_circuit_and_context(config_path: Path) -> tuple[QuantumCircuit, int, int]:
    config = _read_json(config_path)
    cases = [case for case in build_pilot_cases(config, (0, 1, 2)) if case.family == TARGET_FAMILY]
    if not cases:
        raise RuntimeError("frozen pilot config has no phase_interference cases")
    circuit = cases[0].reference_circuit
    names = [str(inst.operation.name).lower() for inst in circuit.data]
    if names != ["h", "rz", "h", "cx"]:
        raise RuntimeError(f"frozen phase pilot circuit drift: {names}")
    affected = {case.affected_logical_qubit for case in cases if case.expected_mechanism in MECHANISMS}
    if affected != {0}:
        raise RuntimeError(f"frozen phase pilot affected-qubit drift: {affected}")
    # Frozen pilot inserts the perturbation after H(0), RZ(0.7) and before H(0), CX.
    return circuit, 2, 0


def _qpu_tier_features(
    tier: str,
    circuit: QuantumCircuit,
    boundary: int,
    qubit: int,
    clean_row: Mapping[str, Any],
) -> np.ndarray:
    base = _circuit_features(circuit, boundary, qubit)
    if tier == "tier1_circuit_only":
        return base
    if tier == "tier2_circuit_plus_born_context":
        counts = {
            basis: clean_row["counts_by_program"][f"reference_{basis}"]
            for basis in BASIS_ORDER
        }
        stats = _absolute_stats_vector(counts, circuit.num_qubits)
        shots = int(sum(int(v) for v in counts["Z"].values()))
        born = np.concatenate(
            [
                stats,
                np.asarray(
                    [
                        math.log2(max(shots, 1)) / 16.0,
                        1.0 / math.sqrt(max(shots, 1)),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        return np.concatenate([base, born])
    raise ValueError(f"unknown QPU tier {tier!r}")


def _evaluate_qpu_tier(
    all_contexts: Sequence[Mapping[str, Any]],
    tier: str,
    *,
    alpha: float,
    config_path: Path,
    qpu_results_path: Path,
) -> dict[str, Any]:
    # Refit on all Step-5 matched contexts only after the frozen Step-5 validation
    # evaluation; the QPU cases are external to this fit.
    model = _fit_rbf_kernel_ridge(
        _feature_matrix(all_contexts, tier),
        _target_matrix(all_contexts),
        alpha,
    )
    clean_row, mechanism_rows = _phase_qpu_rows(qpu_results_path)
    circuit, boundary, qubit = _qpu_reference_circuit_and_context(config_path)
    feature = _qpu_tier_features(tier, circuit, boundary, qubit, clean_row)
    predicted = _split_frame(model.predict(feature)[0])
    exact = _pilot_phase_exact_frame(config_path)
    alignment = _frame_alignment(predicted, exact)
    decoders: dict[str, Any] = {}
    for metric in ("cosine", "euclidean"):
        rows_out: list[dict[str, Any]] = []
        for mechanism in MECHANISMS:
            row = mechanism_rows[mechanism]
            query = _qpu_vector(row)
            scores, prediction, margin = _score_frame(query, predicted, metric)
            rows_out.append(
                {
                    "expected": mechanism,
                    "prediction": prediction,
                    "correct": bool(prediction == mechanism),
                    "margin": margin,
                    "scores": scores,
                    "frozen_model_prediction": row["prediction"]["mechanism_prediction"],
                }
            )
        correct = int(sum(bool(row["correct"]) for row in rows_out))
        decoders[metric] = {
            "correct_count": correct,
            "case_count": len(rows_out),
            "all_three_correct": bool(correct == len(MECHANISMS)),
            "rows": rows_out,
        }
    return {
        "fit_context_count": len(all_contexts),
        "candidate_context": {
            "family": TARGET_FAMILY,
            "candidate_qubit": qubit,
            "candidate_boundary": boundary,
            "reference_gate_sequence": [str(x.operation.name) for x in circuit.data],
        },
        "frame_alignment_to_exact_simulator_frame": alignment,
        "decoders": decoders,
    }


def _evaluate_tier3(
    validation: Sequence[Mapping[str, Any]],
    *,
    probe_strength: float,
    probe_shots: int,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    frames: list[dict[str, np.ndarray]] = []
    alignments: list[dict[str, Any]] = []
    for context in validation:
        frame = _tier3_probe_frame(
            context["circuit"],
            int(context["boundary"]),
            TARGET_QUBIT,
            probe_strength=probe_strength,
            target_strength=TARGET_STRENGTH,
            shots=probe_shots,
            seed_parts=["step9e", "tier3", context["root_index"], context["boundary"]],
        )
        frames.append(frame)
        alignments.append(_frame_alignment(frame, context["exact"]))
    decoders: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for metric_index, metric in enumerate(("cosine", "euclidean")):
        records = _decoder_records(validation, frames, metric)
        summary = _summarize_decoder(records, bootstraps, seed + metric_index)
        decoders[metric] = summary
        gates[metric] = _gate(summary)
    return {
        "validation_context_count": len(validation),
        "protocol": {
            "known_probe_axes": list(MECHANISMS),
            "probe_strength": float(probe_strength),
            "target_strength": TARGET_STRENGTH,
            "probe_shots_per_basis_per_axis": int(probe_shots),
            "measurement_bases": list(BASIS_ORDER),
            "shared_reference_measurement": True,
            "additional_observed_probe_programs_per_candidate_context": 9,
            "qpu_submission_by_this_script": False,
        },
        "frame_alignment_to_exact": _aggregate_alignment(alignments),
        "decoders": decoders,
        "gates": gates,
        "frozen_qpu_validation": {
            "available": False,
            "reason": (
                "The frozen Step-9D QPU artifact contains the six query/reference programs "
                "but not the additional known-axis probe programs required by Tier 3."
            ),
        },
    }


def _tier_full_pass(step5: Mapping[str, Any], qpu: Mapping[str, Any]) -> list[str]:
    return [
        metric
        for metric in ("cosine", "euclidean")
        if bool(step5["gates"][metric]["passed"])
        and bool(qpu["decoders"][metric]["all_three_correct"])
    ]


def _decision(
    learned: Mapping[str, Mapping[str, Any]],
    tier3: Mapping[str, Any] | None,
) -> dict[str, Any]:
    full_pass: dict[str, list[str]] = {
        tier: _tier_full_pass(result["step5_validation"], result["frozen_qpu"])
        for tier, result in learned.items()
    }
    passing_learned = {tier: metrics for tier, metrics in full_pass.items() if metrics}
    if passing_learned:
        best_order = ["tier1_circuit_only", "tier2_circuit_plus_born_context"]
        best = next(tier for tier in best_order if tier in passing_learned)
        return {
            "status": "DEPLOYABLE_LOCAL_FRAME_APPROXIMATION_DEMONSTRATED_IN_TARGETED_POSTHOC_AUDIT",
            "hardware_deployable_frame_demonstrated": True,
            "passing_tiers_and_decoders": passing_learned,
            "preferred_minimal_tier": best,
            "next_action": (
                "Freeze the Step-9E result before changing TriQTO. Then decide whether the "
                "minimal passing representation can be integrated by fine-tuning/reusing the "
                "existing checkpoint or requires an input/encoder contract change."
            ),
        }

    step5_only = {
        tier: [
            metric
            for metric in ("cosine", "euclidean")
            if bool(result["step5_validation"]["gates"][metric]["passed"])
        ]
        for tier, result in learned.items()
    }
    step5_only = {tier: metrics for tier, metrics in step5_only.items() if metrics}
    if step5_only:
        return {
            "status": "DEPLOYABLE_FRAME_PASSES_STEP5_VALIDATION_BUT_QPU_ALIGNMENT_IS_INCOMPLETE",
            "hardware_deployable_frame_demonstrated": False,
            "passing_step5_tiers_and_decoders": step5_only,
            "next_action": (
                "Do not retrain. Localize the Step-5-to-QPU context/domain mismatch in the "
                "predicted frame before changing the architecture."
            ),
        }

    if tier3 is not None:
        tier3_pass = [
            metric
            for metric in ("cosine", "euclidean")
            if bool(tier3["gates"][metric]["passed"])
        ]
        if tier3_pass:
            return {
                "status": "CHEAP_PROBE_FRAME_PASSES_STEP5_VALIDATION__QPU_PROBE_EVIDENCE_MISSING",
                "hardware_deployable_frame_demonstrated": False,
                "passing_tier3_decoders": tier3_pass,
                "next_action": (
                    "Do not retrain. If a new QPU run is explicitly authorized, freeze a small "
                    "Tier-3 probe protocol and validate the probe-derived frame on hardware first."
                ),
            }

    return {
        "status": "NO_DEPLOYABLE_LOCAL_FRAME_DEMONSTRATED__DO_NOT_RETRAIN",
        "hardware_deployable_frame_demonstrated": False,
        "next_action": (
            "Do not retrain the TriQTO checkpoint yet. Reassess candidate-context features, "
            "frame degeneracy/abstention, and the information content/cost of additional probes."
        ),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qpu-case-results", type=Path, default=DEFAULT_QPU_RESULTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--probe-strength", type=float, default=DEFAULT_PROBE_STRENGTH)
    parser.add_argument("--probe-shots", type=int, default=DEFAULT_PROBE_SHOTS)
    parser.add_argument(
        "--tier3-mode",
        choices=("auto", "always", "never"),
        default=DEFAULT_TIER3_MODE,
    )
    args = parser.parse_args()

    if args.bootstraps <= 0:
        raise ValueError("--bootstraps must be positive")
    if args.ridge_alpha <= 0:
        raise ValueError("--ridge-alpha must be positive")
    if not (0.0 < args.probe_strength <= TARGET_STRENGTH):
        raise ValueError("--probe-strength must be in (0, 0.15]")
    if args.probe_shots <= 0:
        raise ValueError("--probe-shots must be positive")

    product = _resolve_product(args.product_pointer, args.product_dir)
    complete = _read_json(product / "dataset_complete.json")
    contexts = _context_rows(product, _matched_contexts(product))
    if not contexts:
        raise RuntimeError("no matched contexts for Step 9E")
    train = [row for row in contexts if row["split"] == "train"]
    validation = [row for row in contexts if row["split"] == "validation"]
    if not train or not validation:
        raise RuntimeError("Step 9E requires both frozen train and validation contexts")

    learned_results: dict[str, Any] = {}
    for tier_index, tier in enumerate(
        ("tier1_circuit_only", "tier2_circuit_plus_born_context")
    ):
        step5, _ = _evaluate_step5_tier(
            train,
            validation,
            tier,
            alpha=args.ridge_alpha,
            bootstraps=args.bootstraps,
            seed=args.bootstrap_seed + 100 * tier_index,
        )
        qpu = _evaluate_qpu_tier(
            contexts,
            tier,
            alpha=args.ridge_alpha,
            config_path=args.config,
            qpu_results_path=args.qpu_case_results,
        )
        learned_results[tier] = {
            "step5_validation": step5,
            "frozen_qpu": qpu,
        }

    learned_full_pass = any(
        _tier_full_pass(result["step5_validation"], result["frozen_qpu"])
        for result in learned_results.values()
    )
    run_tier3 = args.tier3_mode == "always" or (
        args.tier3_mode == "auto" and not learned_full_pass
    )
    tier3 = None
    if run_tier3:
        tier3 = _evaluate_tier3(
            validation,
            probe_strength=args.probe_strength,
            probe_shots=args.probe_shots,
            bootstraps=args.bootstraps,
            seed=args.bootstrap_seed + 1000,
        )

    decision = _decision(learned_results, tier3)
    result = {
        "analysis": ANALYSIS_NAME,
        "scientific_boundary": {
            "posthoc_only": True,
            "confirmatory_interpretation": False,
            "qpu_submission": False,
            "triqto_checkpoint_retraining": False,
            "triqto_weight_change": False,
            "triqto_threshold_change": False,
            "audit_estimator_fitting": True,
            "simulator_privileged_exact_frames_used_as_training_targets_only": True,
        },
        "step5_product_id": str(complete.get("product_id")),
        "context_contract": {
            "family": TARGET_FAMILY,
            "n_qubits": TARGET_N_QUBITS,
            "target_strength": TARGET_STRENGTH,
            "candidate_qubit": TARGET_QUBIT,
            "candidate_context_semantics": (
                "The qubit/boundary identify the local context being scored. This audit does "
                "not claim that a future hardware system already knows the hidden distortion "
                "location; localization remains a separate problem."
            ),
            "matched_context_count": len(contexts),
            "frozen_train_context_count": len(train),
            "frozen_validation_context_count": len(validation),
        },
        "predeclared_gate": {
            "balanced_accuracy_minimum": GATE_BALANCED_ACCURACY,
            "cluster_bootstrap_95_lower_minimum": GATE_BOOTSTRAP_LOWER,
            "minimum_mechanism_recall": GATE_MIN_RECALL,
            "qpu_rule": (
                "For Tier 1 or Tier 2 to demonstrate a targeted deployable approximation, "
                "the same decoder must pass the frozen Step-5 validation gate and classify "
                "all three frozen Step-9D phase QPU cases correctly."
            ),
        },
        "tier_contracts": {
            "tier1_circuit_only": {
                "inputs": [
                    "intended/reference circuit graph",
                    "gate identities and qubit incidence",
                    "sin/cos gate parameter encoding",
                    "candidate qubit and candidate boundary",
                ],
                "forbidden_inputs": [
                    "mechanism label",
                    "finite-shot query delta",
                    "exact response frame",
                    "statevector/Hilbert metrics",
                ],
            },
            "tier2_circuit_plus_born_context": {
                "inputs": [
                    "all Tier-1 inputs",
                    "absolute baseline Z/X/Y local expectations",
                    "absolute baseline same-basis pairwise correlation",
                    "absolute baseline global parity",
                    "baseline shot metadata",
                ],
                "step5_source": "finite-shot simulator emulation of hardware-observable baseline",
                "qpu_source": "already-recorded frozen phase clean reference counts",
                "query_delta_used_as_estimator_input": False,
            },
            "tier3_cheap_probe_frame": {
                "execution": args.tier3_mode,
                "conditional_default": True,
                "hardware_valid_protocol": True,
                "frozen_qpu_probe_counts_available": False,
            },
        },
        "estimator_contract": {
            "kind": "fixed RBF kernel ridge",
            "ridge_alpha": float(args.ridge_alpha),
            "gamma_rule": "inverse median nonzero squared distance in standardized training features",
            "hyperparameter_search": False,
            "primary_fit_split": "frozen Step-5 train contexts",
            "primary_evaluation_split": "frozen Step-5 validation contexts",
            "qpu_fit": "refit on all matched Step-5 contexts only after primary validation",
        },
        "learned_tiers": learned_results,
        "tier3": tier3,
        "tier3_ran": bool(run_tier3),
        "decision_gate": decision,
    }

    print("STEP 9E DEPLOYABLE LOCAL-RESPONSE-FRAME AUDIT — NO QPU SUBMISSION / NO TRIQTO RETRAINING")
    print("Step-5 product:", result["step5_product_id"])
    print("contexts:", len(contexts), "train:", len(train), "validation:", len(validation))
    for tier in ("tier1_circuit_only", "tier2_circuit_plus_born_context"):
        row = learned_results[tier]
        print("\n", tier.upper(), sep="")
        print(
            "  frame median cosine=",
            _fmt(row["step5_validation"]["frame_alignment_to_exact"]["median_cosine_all"]),
            "p10=",
            _fmt(row["step5_validation"]["frame_alignment_to_exact"]["p10_cosine_all"]),
        )
        for metric in ("cosine", "euclidean"):
            summary = row["step5_validation"]["decoders"][metric]
            gate = row["step5_validation"]["gates"][metric]
            ci = summary["balanced_accuracy_cluster_bootstrap_95_ci"]
            recalls = summary["recalls"]
            qpu = row["frozen_qpu"]["decoders"][metric]
            mapping = ", ".join(
                f"{r['expected']}->{r['prediction']}" for r in qpu["rows"]
            )
            print(
                f"  {metric}: Step5 BA={_fmt(summary['balanced_accuracy'])} "
                f"CI=[{_fmt(ci[0])},{_fmt(ci[1])}] "
                f"recalls={{RZ:{_fmt(recalls['rz_drift'])},RX:{_fmt(recalls['rx_overrotation'])},RY:{_fmt(recalls['ry_overrotation'])}}} "
                f"gate={'PASS' if gate['passed'] else 'FAIL'}; "
                f"QPU={qpu['correct_count']}/3 [{mapping}]"
            )
        qalign = row["frozen_qpu"]["frame_alignment_to_exact_simulator_frame"]
        print("  QPU-context predicted-frame own median cosine=", _fmt(qalign["median_cosine"]))

    if tier3 is not None:
        print("\nTIER3_CHEAP_PROBE_FRAME")
        print(
            "  frame median cosine=",
            _fmt(tier3["frame_alignment_to_exact"]["median_cosine_all"]),
            "p10=",
            _fmt(tier3["frame_alignment_to_exact"]["p10_cosine_all"]),
        )
        for metric in ("cosine", "euclidean"):
            summary = tier3["decoders"][metric]
            gate = tier3["gates"][metric]
            ci = summary["balanced_accuracy_cluster_bootstrap_95_ci"]
            print(
                f"  {metric}: Step5 BA={_fmt(summary['balanced_accuracy'])} "
                f"CI=[{_fmt(ci[0])},{_fmt(ci[1])}] gate={'PASS' if gate['passed'] else 'FAIL'}"
            )
        print("  frozen QPU probe validation: unavailable (no new QPU job submitted)")
    else:
        print("\nTIER3_CHEAP_PROBE_FRAME: skipped by auto gate because Tier 1/2 already established a full pass")

    print("\nDECISION GATE:", decision["status"])
    print("  hardware-deployable frame demonstrated:", decision["hardware_deployable_frame_demonstrated"])
    print("  next:", decision["next_action"])

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("\nwrote:", output)


if __name__ == "__main__":
    main()
