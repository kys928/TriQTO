#!/usr/bin/env python3
"""Post-hoc Step-9D phase-family counterfactual analysis.

This script is diagnostic only. It never submits QPU work and never changes
weights, thresholds, architecture, checkpoints, or the frozen Step-9D evidence.

It compares, for the three distorted phase-interference cases:
1) recorded hardware diagnostics with the frozen deployment ensemble;
2) exact ideal-state diagnostics for the same Step-9D circuits and graph;
3) a graph-zero ablation on recorded hardware diagnostics;
4) a graph-zero ablation on exact ideal diagnostics.

The graph-zero rows are attribution probes only; they are not deployable-model
performance claims.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qiskit.quantum_info import Pauli, Statevector

from triqto.hardware.diagnostic_acquisition import (
    BASIS_ORDER,
    all_pair_indices,
    build_step7_model_batch_from_counts,
)
from triqto.hardware.dry_run import (
    load_frozen_deployment_ensemble,
    predict_frozen_ensemble,
)
from triqto.hardware.qpu_pilot import build_pilot_cases
from triqto.step7.contracts import Step7ModelBatch


DEFAULT_EVIDENCE = Path("docs/evidence/step9d_exploratory_qpu_pilot")
DEFAULT_CONFIG = Path("configs/v0_2/step9d_exploratory_qpu_pilot_v2.json")
PHASE_CONDITIONS = ("rz_drift", "rx_overrotation", "ry_overrotation")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pauli_label(n_qubits: int, basis: str, qubits: tuple[int, ...]) -> str:
    chars = ["I"] * n_qubits
    for qubit in qubits:
        chars[n_qubits - 1 - int(qubit)] = str(basis)
    return "".join(chars)


def _expectation(state: Statevector, basis: str, qubits: tuple[int, ...]) -> float:
    label = _pauli_label(state.num_qubits, basis, qubits)
    value = state.expectation_value(Pauli(label))
    return float(np.real_if_close(value))


def _ideal_diagnostic_tensors(reference_circuit, observed_circuit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference_circuit.num_qubits != observed_circuit.num_qubits:
        raise RuntimeError("reference/observed qubit-count mismatch")
    n_qubits = reference_circuit.num_qubits
    reference = Statevector.from_instruction(reference_circuit)
    observed = Statevector.from_instruction(observed_circuit)
    pairs = all_pair_indices(n_qubits)

    local = np.zeros((n_qubits, 3), dtype=np.float32)
    pair = np.zeros((len(pairs), 3), dtype=np.float32)
    parity = np.zeros((1, 3), dtype=np.float32)

    for basis_index, basis in enumerate(BASIS_ORDER):
        for qubit in range(n_qubits):
            local[qubit, basis_index] = (
                _expectation(observed, basis, (qubit,))
                - _expectation(reference, basis, (qubit,))
            )
        for pair_index, (left, right) in enumerate(pairs):
            pair[pair_index, basis_index] = (
                _expectation(observed, basis, (int(left), int(right)))
                - _expectation(reference, basis, (int(left), int(right)))
            )
        all_qubits = tuple(range(n_qubits))
        parity[0, basis_index] = (
            _expectation(observed, basis, all_qubits)
            - _expectation(reference, basis, all_qubits)
        )
    return local, pair, parity


def _replace_diagnostic_with_ideal(
    hardware_batch: Step7ModelBatch,
    reference_circuit,
    observed_circuit,
) -> Step7ModelBatch:
    batch = deepcopy(hardware_batch)
    local, pair, parity = _ideal_diagnostic_tensors(reference_circuit, observed_circuit)
    device = batch.graph.node_features.device
    batch.diagnostic.local_values = torch.as_tensor(local, dtype=torch.float32, device=device)
    batch.diagnostic.pair_values = torch.as_tensor(pair, dtype=torch.float32, device=device)
    batch.diagnostic.global_parity = torch.as_tensor(parity, dtype=torch.float32, device=device)
    batch.diagnostic.validate(batch.graph)
    return batch


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    return exp / float(exp.sum())


def _zero_graph_mechanism(ensemble, batch: Step7ModelBatch) -> dict[str, Any]:
    per_seed: dict[str, list[float]] = {}
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for seed in ensemble.seeds:
            model = ensemble.models[seed]
            model.validate_batch(batch)
            _local, _pair, _parity, diagnostic_graph = model.diagnostic_encoder(batch)
            graph_output = model.graph_encoder(batch.graph)
            zero_graph = torch.zeros_like(graph_output.graph_embedding)
            representation = model.late_concat_fusion(
                torch.cat((zero_graph, diagnostic_graph), dim=1)
            )
            logits = model.mechanism_head(representation).detach().cpu().numpy().reshape(1, 3)[0]
            logits = logits.astype(np.float64, copy=False)
            per_seed[str(seed)] = logits.tolist()
            rows.append(logits)
    mean_logits = np.mean(np.stack(rows, axis=0), axis=0)
    probabilities = _softmax(mean_logits)
    code = int(np.argmax(mean_logits))
    return {
        "mean_mechanism_logits": mean_logits.tolist(),
        "mechanism_probabilities": probabilities.tolist(),
        "mechanism_code": code,
        "mechanism_prediction": ensemble.mechanism_classes[code],
        "seed_mechanism_logits": per_seed,
    }


def _diagnostic_vector(batch: Step7ModelBatch) -> np.ndarray:
    diagnostic = batch.diagnostic
    local = diagnostic.local_values.detach().cpu().numpy().T.reshape(-1)
    pair = diagnostic.pair_values.detach().cpu().numpy().T.reshape(-1)
    parity = diagnostic.global_parity.detach().cpu().numpy().reshape(-1)
    return np.concatenate((local, pair, parity)).astype(np.float64, copy=False)


def _mechanism_probabilities(prediction: dict[str, Any]) -> list[float]:
    logits = np.asarray(prediction["mean_mechanism_logits"], dtype=np.float64)
    return _softmax(logits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-bundle-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    config = _read_json(args.config)
    deployment = config["deployment_bundle"]
    ensemble = load_frozen_deployment_ensemble(
        args.deployment_bundle_dir,
        deployment,
        device="cpu",
    )

    rows = _read_jsonl(args.evidence_dir / "case_results.jsonl")
    evidence_by_case = {str(row["case_id"]): row for row in rows}

    # The frozen execution used this exact selected chain. We need the chain only
    # to reconstruct the intended graph layout; no hardware service is contacted.
    summary = _read_json(args.evidence_dir / "pilot_summary.json")
    physical_chain = tuple(int(q) for q in summary["physical_chain"])
    cases = {case.case_id: case for case in build_pilot_cases(config, physical_chain)}

    results: dict[str, Any] = {
        "analysis": "step9d_phase_posthoc_counterfactual_v1",
        "scientific_boundary": {
            "posthoc_only": True,
            "qpu_submission": False,
            "retraining": False,
            "weight_change": False,
            "threshold_change": False,
            "confirmatory_interpretation": False,
        },
        "mechanism_classes": list(ensemble.mechanism_classes),
        "cases": {},
    }

    print("STEP 9D POST-HOC PHASE COUNTERFACTUAL — NO QPU ACCESS")
    print("classes:", list(ensemble.mechanism_classes))
    print()

    for condition in PHASE_CONDITIONS:
        case_id = f"phase_interference__{condition}"
        case = cases[case_id]
        row = evidence_by_case[case_id]
        counts = row["counts_by_program"]
        reference_counts = {basis: counts[f"reference_{basis}"] for basis in BASIS_ORDER}
        observed_counts = {basis: counts[f"observed_{basis}"] for basis in BASIS_ORDER}

        hardware_batch = build_step7_model_batch_from_counts(
            case.reference_circuit,
            case.physical_layout,
            reference_counts,
            observed_counts,
            device="cpu",
        )
        ideal_batch = _replace_diagnostic_with_ideal(
            hardware_batch,
            case.reference_circuit,
            case.observed_circuit,
        )

        hardware_prediction = predict_frozen_ensemble(ensemble, hardware_batch)
        ideal_prediction = predict_frozen_ensemble(ensemble, ideal_batch)
        hardware_zero_graph = _zero_graph_mechanism(ensemble, hardware_batch)
        ideal_zero_graph = _zero_graph_mechanism(ensemble, ideal_batch)

        hardware_vec = _diagnostic_vector(hardware_batch)
        ideal_vec = _diagnostic_vector(ideal_batch)
        cosine = float(
            np.dot(hardware_vec, ideal_vec)
            / (np.linalg.norm(hardware_vec) * np.linalg.norm(ideal_vec))
        )

        entry = {
            "expected_mechanism": condition,
            "hardware": {
                "mechanism_prediction": hardware_prediction["mechanism_prediction"],
                "mechanism_probabilities": _mechanism_probabilities(hardware_prediction),
                "effect_probability": hardware_prediction["effect_probability"],
            },
            "ideal_same_graph": {
                "mechanism_prediction": ideal_prediction["mechanism_prediction"],
                "mechanism_probabilities": _mechanism_probabilities(ideal_prediction),
                "effect_probability": ideal_prediction["effect_probability"],
            },
            "hardware_zero_graph": hardware_zero_graph,
            "ideal_zero_graph": ideal_zero_graph,
            "hardware_vs_ideal_diagnostic_cosine": cosine,
            "hardware_diagnostic_l2": float(np.linalg.norm(hardware_vec)),
            "ideal_diagnostic_l2": float(np.linalg.norm(ideal_vec)),
            "hardware_vs_ideal_l2_error": float(np.linalg.norm(hardware_vec - ideal_vec)),
        }
        results["cases"][case_id] = entry

        print(case_id)
        print("  expected:", condition)
        print(
            "  hardware:",
            entry["hardware"]["mechanism_prediction"],
            np.round(entry["hardware"]["mechanism_probabilities"], 4).tolist(),
        )
        print(
            "  ideal same graph:",
            entry["ideal_same_graph"]["mechanism_prediction"],
            np.round(entry["ideal_same_graph"]["mechanism_probabilities"], 4).tolist(),
        )
        print(
            "  hardware zero graph:",
            entry["hardware_zero_graph"]["mechanism_prediction"],
            np.round(entry["hardware_zero_graph"]["mechanism_probabilities"], 4).tolist(),
        )
        print(
            "  ideal zero graph:",
            entry["ideal_zero_graph"]["mechanism_prediction"],
            np.round(entry["ideal_zero_graph"]["mechanism_probabilities"], 4).tolist(),
        )
        print("  hardware/ideal cosine:", f"{cosine:.6f}")
        print()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("wrote:", args.output_json)


if __name__ == "__main__":
    main()
