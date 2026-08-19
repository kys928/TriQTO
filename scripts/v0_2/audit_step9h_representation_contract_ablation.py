#!/usr/bin/env python3
"""Step 9H representation-contract ablation.

Zero-QPU, zero-TriQTO-retraining audit. It asks whether the Step-9G deployment
coverage rescue depends on information that the current Step-7 graph adapter
does not preserve.

R0: current Step-7 information-equivalent circuit contract.
R1: R0 + gate-level angle phasors.
R2: R1 + candidate local context (audit upper bound).

The estimator remains the fixed RBF kernel-ridge audit estimator. Exact response
frames are simulator-privileged targets only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit

from analyze_step9d_posthoc_context_identifiability import (
    DEFAULT_CONFIG,
    DEFAULT_QPU_RESULTS,
    MECHANISMS,
    _pilot_phase_exact_frame,
)
from audit_step9e_deployable_local_frame import (
    _fit_rbf_kernel_ridge,
    _frame_alignment,
    _phase_qpu_rows,
    _qpu_reference_circuit_and_context,
    _split_frame,
)
from audit_step9f_frame_failure_decomposition import (
    _evaluate_frames,
    _qpu_decode_with_frame,
)
from audit_step9g_deployment_domain_coverage_bridge import (
    _bridge_contexts,
)

ANALYSIS_NAME = "step9h_representation_contract_ablation_v1"
DEFAULT_OUTPUT = Path(
    "/workspace/triqto-data/step9h_posthoc/representation_contract_ablation_v1.json"
)
DEFAULT_BRIDGE_ROOTS = 240
DEFAULT_QUERY_SHOTS = 4096
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260819
DEFAULT_RIDGE_ALPHA = 1e-2
PRIMARY_DECODER = "euclidean"
MAX_EVENTS = 16
GATE_VOCAB = (
    "h", "rx", "ry", "rz", "cx", "cz", "cp", "rzz", "swap", "other"
)
CONTROLLED = {"cx", "cp"}
SYMMETRIC = {"cz", "rzz", "swap"}
ANGULAR = {"rx", "ry", "rz", "cp", "rzz"}

BA_MIN = 0.80
CI_MIN = 0.75
RECALL_MIN = 0.70


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _gate_name(name: str) -> str:
    text = str(name).lower()
    return text if text in GATE_VOCAB[:-1] else "other"


def _records(circuit: QuantumCircuit) -> list[dict[str, Any]]:
    frontier = [0] * circuit.num_qubits
    output: list[dict[str, Any]] = []
    for order, instruction in enumerate(circuit.data):
        name = _gate_name(instruction.operation.name)
        qubits = tuple(int(circuit.find_bit(q).index) for q in instruction.qubits)
        layer = max((frontier[q] for q in qubits), default=0)
        for q in qubits:
            frontier[q] = layer + 1
        params = [float(p) for p in instruction.operation.params]
        output.append(
            {
                "name": name,
                "qubits": qubits,
                "layer": int(layer),
                "params": params,
                "parameter_count": len(params),
                "angular_count": len(params) if name in ANGULAR else 0,
            }
        )
    return output


def _base_current_contract_features(circuit: QuantumCircuit) -> np.ndarray:
    """Flatten information already available to the current Step-7 graph adapter.

    This is information-equivalent rather than neural-encoder-equivalent: it
    preserves gate identity/order/layer/incidence plus the aggregate per-qubit
    angular sums used by the current adapter, but not gate-level phasors.
    """
    records = _records(circuit)
    n_events = len(records)
    if circuit.num_qubits != 2:
        raise RuntimeError("Step 9H targeted bridge contract requires 2 qubits")
    if n_events > MAX_EVENTS:
        raise RuntimeError(f"event count {n_events} exceeds MAX_EVENTS={MAX_EVENTS}")

    vocab = {name: i for i, name in enumerate(GATE_VOCAB)}
    max_layer = max((row["layer"] for row in records), default=0)
    out: list[float] = [
        float(circuit.num_qubits) / 8.0,
        float(n_events) / float(MAX_EVENTS),
    ]

    for qubit in range(circuit.num_qubits):
        oneq = 0.0
        twoq = 0.0
        total = 0.0
        angular = 0.0
        sin_sum = 0.0
        cos_sum = 0.0
        neighbors: set[int] = set()
        active_layers: list[int] = []
        for row in records:
            if qubit not in row["qubits"]:
                continue
            total += 1.0
            active_layers.append(int(row["layer"]))
            if len(row["qubits"]) == 1:
                oneq += 1.0
            elif len(row["qubits"]) == 2:
                twoq += 1.0
                for other in row["qubits"]:
                    if other != qubit:
                        neighbors.add(int(other))
            angular += float(row["angular_count"])
            if row["angular_count"]:
                for value in row["params"][: row["angular_count"]]:
                    sin_sum += math.sin(value)
                    cos_sum += math.cos(value)
        divisor = float(max(max_layer, 1))
        first = min(active_layers) / divisor if active_layers else 0.0
        last = max(active_layers) / divisor if active_layers else 0.0
        span = (max(active_layers) - min(active_layers)) / divisor if active_layers else 0.0
        out.extend(
            [
                oneq / MAX_EVENTS,
                twoq / MAX_EVENTS,
                total / MAX_EVENTS,
                angular / MAX_EVENTS,
                sin_sum / MAX_EVENTS,
                cos_sum / MAX_EVENTS,
                float(len(neighbors)) / 2.0,
                first,
                last,
                span,
            ]
        )

    max_order = max(n_events - 1, 1)
    for slot in range(MAX_EVENTS):
        onehot = np.zeros(len(GATE_VOCAB), dtype=np.float64)
        qmask = np.zeros(2, dtype=np.float64)
        arity = 0.0
        normalized_order = 0.0
        normalized_layer = 0.0
        parameter_count = 0.0
        angular_count = 0.0
        controlled = 0.0
        symmetric = 0.0
        if slot < n_events:
            row = records[slot]
            onehot[vocab[row["name"]]] = 1.0
            for q in row["qubits"]:
                qmask[q] = 1.0
            arity = float(len(row["qubits"])) / 2.0
            normalized_order = float(slot) / float(max_order)
            normalized_layer = (
                float(row["layer"]) / float(max_layer) if max_layer > 0 else 0.0
            )
            parameter_count = float(row["parameter_count"]) / 3.0
            angular_count = float(row["angular_count"]) / 3.0
            controlled = float(row["name"] in CONTROLLED)
            symmetric = float(row["name"] in SYMMETRIC)
        out.extend(onehot.tolist())
        out.extend(qmask.tolist())
        out.extend(
            [
                arity,
                normalized_order,
                normalized_layer,
                parameter_count,
                angular_count,
                controlled,
                symmetric,
            ]
        )
    return np.asarray(out, dtype=np.float64)


def _gate_phasor_features(circuit: QuantumCircuit) -> np.ndarray:
    records = _records(circuit)
    out: list[float] = []
    for slot in range(MAX_EVENTS):
        sin_value = 0.0
        cos_value = 0.0
        has_angle = 0.0
        if slot < len(records):
            row = records[slot]
            if row["angular_count"] and row["params"]:
                value = float(row["params"][0])
                sin_value = math.sin(value)
                cos_value = math.cos(value)
                has_angle = 1.0
        out.extend([sin_value, cos_value, has_angle])
    return np.asarray(out, dtype=np.float64)


def _candidate_context_features(
    circuit: QuantumCircuit, boundary: int, candidate_qubit: int
) -> np.ndarray:
    if not (1 <= int(boundary) <= len(circuit.data)):
        raise RuntimeError("candidate boundary outside circuit")
    if candidate_qubit not in (0, 1):
        raise RuntimeError("candidate qubit outside 2q contract")
    out: list[float] = [
        float(boundary) / float(MAX_EVENTS),
        float(boundary) / float(max(1, len(circuit.data))),
        float(candidate_qubit == 0),
        float(candidate_qubit == 1),
    ]
    out.extend([1.0 if boundary == b else 0.0 for b in range(MAX_EVENTS + 1)])
    out.extend(
        [1.0 if slot < boundary else 0.0 for slot in range(MAX_EVENTS)]
    )
    return np.asarray(out, dtype=np.float64)


def _feature(
    context: Mapping[str, Any],
    variant: str,
) -> np.ndarray:
    circuit = context["circuit"]
    base = _base_current_contract_features(circuit)
    if variant == "r0_current_step7_information":
        return base
    phasor = _gate_phasor_features(circuit)
    if variant == "r1_plus_gate_level_phasor":
        return np.concatenate([base, phasor])
    if variant == "r2_plus_candidate_local_context":
        return np.concatenate(
            [
                base,
                phasor,
                _candidate_context_features(
                    circuit, int(context["boundary"]), 0
                ),
            ]
        )
    raise ValueError(f"unknown representation variant {variant!r}")


def _matrix(contexts: Sequence[Mapping[str, Any]], variant: str) -> np.ndarray:
    return np.vstack([_feature(context, variant) for context in contexts])


def _target_matrix(contexts: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.vstack(
        [
            np.concatenate(
                [np.asarray(context["exact"][m], dtype=np.float64) for m in MECHANISMS]
            )
            for context in contexts
        ]
    )


def _gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    ba = summary["balanced_accuracy"]
    ci = summary["balanced_accuracy_cluster_bootstrap_95_ci"]
    recalls = summary["recalls"]
    minimum = min(float(recalls[m]) for m in MECHANISMS)
    passed = bool(
        float(ba) >= BA_MIN
        and float(ci[0]) >= CI_MIN
        and minimum >= RECALL_MIN
    )
    return {
        "passed": passed,
        "balanced_accuracy": float(ba),
        "balanced_accuracy_95_lower": float(ci[0]),
        "minimum_mechanism_recall": minimum,
    }


def _qpu_context(variant: str) -> dict[str, Any]:
    circuit, boundary, qubit = _qpu_reference_circuit_and_context(DEFAULT_CONFIG)
    return {
        "circuit": circuit,
        "boundary": int(boundary),
        "candidate_qubit": int(qubit),
    }


def _evaluate_variant(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    variant: str,
    *,
    alpha: float,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    x_train = _matrix(train, variant)
    y_train = _target_matrix(train)
    x_validation = _matrix(validation, variant)
    model = _fit_rbf_kernel_ridge(x_train, y_train, alpha)
    frames = [_split_frame(row) for row in model.predict(x_validation)]
    validation_eval = _evaluate_frames(
        validation, frames, bootstraps=bootstraps, seed=seed
    )

    _, qpu_rows = _phase_qpu_rows(DEFAULT_QPU_RESULTS)
    qpu_context = _qpu_context(variant)
    qpu_feature = _feature(qpu_context, variant)[None, :]
    qpu_frame = _split_frame(model.predict(qpu_feature)[0])
    exact_qpu = _pilot_phase_exact_frame(DEFAULT_CONFIG)
    qpu = {
        "predicted_frame_alignment_to_exact_simulator_frame": _frame_alignment(
            qpu_frame, exact_qpu
        ),
        "decoders": _qpu_decode_with_frame(qpu_rows, qpu_frame),
    }

    primary = validation_eval["decoders"][PRIMARY_DECODER]
    primary_gate = _gate(primary)
    qpu_three = bool(qpu["decoders"][PRIMARY_DECODER]["all_three_correct"])
    full = bool(primary_gate["passed"] and qpu_three)

    return {
        "variant": variant,
        "feature_dimension": int(x_train.shape[1]),
        "train_context_count": len(train),
        "validation_context_count": len(validation),
        "ridge_alpha": float(alpha),
        "rbf_gamma": float(model.gamma),
        "bridge_validation": validation_eval,
        "primary_gate": primary_gate,
        "frozen_step9d_qpu": qpu,
        "full_targeted_pass": full,
    }


def _decision(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    r0 = bool(results["r0_current_step7_information"]["full_targeted_pass"])
    r1 = bool(results["r1_plus_gate_level_phasor"]["full_targeted_pass"])
    r2 = bool(results["r2_plus_candidate_local_context"]["full_targeted_pass"])
    if r0:
        return {
            "status": "CURRENT_STEP7_GRAPH_INFORMATION_IS_SUFFICIENT__COVERAGE_IS_PRIMARY_GAP",
            "architecture_implication": "no graph-information contract change demonstrated necessary",
            "checkpoint_implication": (
                "prefer a leakage-safe deployment-domain training-data redesign and "
                "warm-start reuse of the frozen checkpoint as the first training experiment"
            ),
        }
    if r1:
        return {
            "status": "GATE_LEVEL_PHASOR_PRESERVATION_IS_REQUIRED",
            "architecture_implication": (
                "minimal Step-7 graph-adapter/input-projection change is indicated; "
                "a wholesale architecture replacement is not"
            ),
            "checkpoint_implication": (
                "reuse all shape-compatible checkpoint weights; expand/reinitialize only "
                "the affected graph input projection, then compare warm-start versus scratch"
            ),
        }
    if r2:
        return {
            "status": "EXPLICIT_LOCAL_CONTEXT_MECHANISM_IS_REQUIRED",
            "architecture_implication": (
                "candidate enumeration/localization or a local-context head is indicated "
                "before main-model retraining"
            ),
            "checkpoint_implication": (
                "existing backbone weights may still be reusable, but the input/inference "
                "contract is no longer architecture-identical"
            ),
        }
    return {
        "status": "CIRCUIT_ONLY_REPRESENTATION_NOT_SUFFICIENT_IN_TARGETED_AUDIT",
        "architecture_implication": (
            "do not retrain the main model on bridge data yet; retain the Step-9F "
            "SNR-qualified known-axis probe route as the leading hardware-valid fallback"
        ),
        "checkpoint_implication": "no checkpoint-reuse training experiment is authorized yet",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-roots", type=int, default=DEFAULT_BRIDGE_ROOTS)
    parser.add_argument("--query-shots", type=int, default=DEFAULT_QUERY_SHOTS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bridge = _bridge_contexts(args.bridge_roots, args.query_shots)
    train = [row for row in bridge if row["split"] == "train"]
    validation = [row for row in bridge if row["split"] == "validation"]
    if len(train) != 192 or len(validation) != 48:
        raise RuntimeError(
            f"unexpected frozen bridge split {len(train)}/{len(validation)}"
        )

    variants = (
        "r0_current_step7_information",
        "r1_plus_gate_level_phasor",
        "r2_plus_candidate_local_context",
    )
    results: dict[str, Any] = {}
    for index, variant in enumerate(variants):
        results[variant] = _evaluate_variant(
            train,
            validation,
            variant,
            alpha=args.ridge_alpha,
            bootstraps=args.bootstraps,
            seed=args.bootstrap_seed + 100 * index,
        )

    decision = _decision(results)
    payload = {
        "analysis": ANALYSIS_NAME,
        "scientific_boundary": {
            "posthoc_only": True,
            "qpu_submission": False,
            "triqto_checkpoint_retraining": False,
            "weight_change": False,
            "threshold_change": False,
            "exact_frames_are_audit_targets_only": True,
        },
        "frozen_design": {
            "bridge_roots": int(args.bridge_roots),
            "bridge_train_contexts": len(train),
            "bridge_validation_contexts": len(validation),
            "query_shots": int(args.query_shots),
            "ridge_alpha": float(args.ridge_alpha),
            "primary_decoder": PRIMARY_DECODER,
            "thresholds": {
                "balanced_accuracy": BA_MIN,
                "cluster_bootstrap_95_lower": CI_MIN,
                "minimum_mechanism_recall": RECALL_MIN,
                "qpu_cases_correct": "3/3",
            },
        },
        "representation_results": results,
        "decision_gate": decision,
    }

    print(
        "STEP 9H REPRESENTATION-CONTRACT ABLATION — "
        "NO QPU / NO TRIQTO RETRAINING"
    )
    print("bridge contexts: train/validation=", len(train), "/", len(validation))
    for variant in variants:
        row = results[variant]
        primary = row["bridge_validation"]["decoders"][PRIMARY_DECODER]
        gate = row["primary_gate"]
        qpu = row["frozen_step9d_qpu"]["decoders"][PRIMARY_DECODER]
        align = row["frozen_step9d_qpu"][
            "predicted_frame_alignment_to_exact_simulator_frame"
        ]
        print("\n", variant.upper(), sep="")
        print(
            "  feature_dim=", row["feature_dimension"],
            " bridge BA=", _fmt(primary["balanced_accuracy"]),
            " CI=[", _fmt(primary["balanced_accuracy_cluster_bootstrap_95_ci"][0]),
            ",", _fmt(primary["balanced_accuracy_cluster_bootstrap_95_ci"][1]), "]",
            " min-recall=", _fmt(gate["minimum_mechanism_recall"]),
            " gate=", "PASS" if gate["passed"] else "FAIL",
            sep="",
        )
        print(
            "  QPU euclidean=", qpu["correct_count"], "/3",
            " frame median cosine=", _fmt(align["median_cosine"]),
            " full targeted pass=", row["full_targeted_pass"],
            sep="",
        )

    print("\nDECISION GATE:", decision["status"])
    print("  architecture:", decision["architecture_implication"])
    print("  checkpoint:", decision["checkpoint_implication"])

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nwrote:", output)


if __name__ == "__main__":
    main()
