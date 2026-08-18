"""Step-9D exploratory physical-QPU transfer-pilot helpers.

This module never retrains or retunes the frozen deployment model.  It builds a
small fixed family/mechanism matrix, selects one real backend and one connected
three-qubit chain from calibration data, compiles the exact Step-9B six-program
acquisition circuits, and supports one explicit-confirmation SamplerV2 job.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit

from triqto.step7.contracts import Step7ModelBatch

from .diagnostic_acquisition import (
    BASIS_ORDER,
    PROGRAM_ORDER,
    _extract_counts,
    build_step7_model_batch_from_counts,
    compile_paired_measurement_circuits,
)
from .dry_run import FrozenDeploymentEnsemble, predict_frozen_ensemble


@dataclass(frozen=True, slots=True)
class PilotCase:
    case_id: str
    family: str
    condition: str
    reference_circuit: QuantumCircuit
    observed_circuit: QuantumCircuit
    logical_layout: tuple[int, ...]
    physical_layout: tuple[int, ...]
    affected_logical_qubit: int | None
    strength: float | None
    expected_effect: bool
    expected_mechanism: str | None


@dataclass(frozen=True, slots=True)
class ChainCandidate:
    backend_name: str
    backend_version: str
    physical_chain: tuple[int, int, int]
    two_qubit_gate: str
    edge_errors: tuple[float, float]
    readout_errors: tuple[float, float, float]
    pending_jobs: int

    @property
    def score(self) -> tuple[Any, ...]:
        return (
            max(self.edge_errors),
            mean(self.edge_errors),
            mean(self.readout_errors),
            self.pending_jobs,
            self.backend_name,
            self.physical_chain,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "physical_chain": list(self.physical_chain),
            "two_qubit_gate": self.two_qubit_gate,
            "edge_errors": list(self.edge_errors),
            "max_two_qubit_error": max(self.edge_errors),
            "mean_two_qubit_error": mean(self.edge_errors),
            "readout_errors": list(self.readout_errors),
            "mean_readout_error": mean(self.readout_errors),
            "pending_jobs": self.pending_jobs,
            "score_tuple": [
                max(self.edge_errors),
                mean(self.edge_errors),
                mean(self.readout_errors),
                self.pending_jobs,
                self.backend_name,
                list(self.physical_chain),
            ],
        }


def _finite_error(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0 or number >= 1.0:
        return None
    return number


def _instruction_error(backend: Any, name: str, qargs: tuple[int, ...]) -> float | None:
    target = getattr(backend, "target", None)
    if target is None or name not in getattr(target, "operation_names", set()):
        return None
    try:
        properties = target[name].get(qargs)
    except Exception:
        try:
            properties = target[name][qargs]
        except Exception:
            return None
    return _finite_error(getattr(properties, "error", None)) if properties is not None else None


def _measurement_error(backend: Any, qubit: int) -> float | None:
    value = _instruction_error(backend, "measure", (int(qubit),))
    if value is not None:
        return value
    try:
        return _finite_error(backend.properties().readout_error(int(qubit)))
    except Exception:
        return None


def _gate_edges(backend: Any, gate: str) -> dict[tuple[int, int], float]:
    target = getattr(backend, "target", None)
    if target is None or gate not in getattr(target, "operation_names", set()):
        return {}
    result: dict[tuple[int, int], float] = {}
    try:
        entries = target[gate].items()
    except Exception:
        return {}
    for qargs, properties in entries:
        if qargs is None or len(qargs) != 2:
            continue
        error = _finite_error(getattr(properties, "error", None))
        if error is None:
            continue
        left, right = int(qargs[0]), int(qargs[1])
        key = tuple(sorted((left, right)))
        previous = result.get(key)
        if previous is None or error < previous:
            result[key] = error
    return result


def best_connected_three_qubit_chain(
    backend: Any,
    *,
    gate_preference: Sequence[str] = ("cz", "ecr"),
) -> ChainCandidate:
    status = backend.status()
    if not bool(getattr(status, "operational", False)):
        raise RuntimeError(f"backend {backend.name} is not operational")
    selected_gate = None
    edges: dict[tuple[int, int], float] = {}
    for gate in gate_preference:
        candidate = _gate_edges(backend, str(gate))
        if candidate:
            selected_gate = str(gate)
            edges = candidate
            break
    if selected_gate is None:
        raise RuntimeError(f"backend {backend.name} has no usable calibrated CZ/ECR edges")

    neighbors: dict[int, set[int]] = {}
    for left, right in edges:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)

    candidates: list[ChainCandidate] = []
    for center, local_neighbors in neighbors.items():
        for left, right in combinations(sorted(local_neighbors), 2):
            readout = tuple(_measurement_error(backend, q) for q in (left, center, right))
            if any(value is None for value in readout):
                continue
            edge_errors = (edges[tuple(sorted((left, center)))], edges[tuple(sorted((center, right)))])
            candidates.append(
                ChainCandidate(
                    backend_name=str(backend.name),
                    backend_version=str(getattr(backend, "backend_version", "unknown")),
                    physical_chain=(left, center, right),
                    two_qubit_gate=selected_gate,
                    edge_errors=(float(edge_errors[0]), float(edge_errors[1])),
                    readout_errors=(float(readout[0]), float(readout[1]), float(readout[2])),
                    pending_jobs=int(getattr(status, "pending_jobs", 0)),
                )
            )
    if not candidates:
        raise RuntimeError(f"backend {backend.name} has no calibrated connected three-qubit chain")
    return min(candidates, key=lambda candidate: candidate.score)


def select_backend_and_chain(
    service: Any,
    *,
    backend_name: str | None = None,
    gate_preference: Sequence[str] = ("cz", "ecr"),
) -> tuple[Any, ChainCandidate, list[dict[str, Any]]]:
    if backend_name:
        backends = [service.backend(str(backend_name))]
    else:
        backends = list(service.backends(simulator=False, operational=True, min_num_qubits=3))
    if not backends:
        raise RuntimeError("no accessible operational physical backend with at least three qubits")

    ranked: list[tuple[ChainCandidate, Any]] = []
    rejected: list[dict[str, Any]] = []
    for backend in backends:
        try:
            candidate = best_connected_three_qubit_chain(backend, gate_preference=gate_preference)
            ranked.append((candidate, backend))
        except Exception as exc:
            rejected.append({"backend_name": str(getattr(backend, "name", "unknown")), "reason": str(exc)})
    if not ranked:
        raise RuntimeError("no accessible backend had a complete calibrated three-qubit chain")
    ranked.sort(key=lambda item: item[0].score)
    candidate, backend = ranked[0]
    ranking = [item[0].as_dict() for item in ranked]
    if rejected:
        ranking.extend({"rejected": row} for row in rejected)
    return backend, candidate, ranking


def _base_family_circuit(family: str) -> QuantumCircuit:
    if family == "bell_like":
        circuit = QuantumCircuit(2, name="bell_like")
        circuit.h(0)
        circuit.cx(0, 1)
        return circuit
    if family == "ghz":
        circuit = QuantumCircuit(3, name="ghz")
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        return circuit
    if family == "phase_interference":
        circuit = QuantumCircuit(2, name="phase_interference")
        circuit.h(0)
        circuit.rz(0.7, 0)
        circuit.h(0)
        circuit.cx(0, 1)
        return circuit
    raise ValueError(f"unknown Step-9D family {family!r}")


def _observed_family_circuit(
    family: str,
    condition: str,
    strength: float,
    affected_qubit: int,
) -> QuantumCircuit:
    if condition == "clean":
        return _base_family_circuit(family).copy(name=f"{family}__clean")
    gate_name = {"rz_drift": "rz", "rx_overrotation": "rx", "ry_overrotation": "ry"}.get(condition)
    if gate_name is None:
        raise ValueError(f"unknown Step-9D condition {condition!r}")

    if family == "bell_like":
        circuit = QuantumCircuit(2, name=f"bell_like__{condition}")
        circuit.h(0)
        getattr(circuit, gate_name)(strength, affected_qubit)
        circuit.cx(0, 1)
        return circuit
    if family == "ghz":
        circuit = QuantumCircuit(3, name=f"ghz__{condition}")
        circuit.h(0)
        circuit.cx(0, 1)
        getattr(circuit, gate_name)(strength, affected_qubit)
        circuit.cx(1, 2)
        return circuit
    if family == "phase_interference":
        circuit = QuantumCircuit(2, name=f"phase_interference__{condition}")
        circuit.h(0)
        circuit.rz(0.7, 0)
        getattr(circuit, gate_name)(strength, affected_qubit)
        circuit.h(0)
        circuit.cx(0, 1)
        return circuit
    raise ValueError(f"unknown Step-9D family {family!r}")


def build_pilot_cases(config: Mapping[str, Any], physical_chain: Sequence[int]) -> tuple[PilotCase, ...]:
    chain = tuple(int(value) for value in physical_chain)
    if len(chain) != 3 or len(set(chain)) != 3:
        raise ValueError("Step-9D requires one unique three-qubit physical chain")
    design = config["pilot_design"]
    strength = float(design["distortion_strength"])
    cases: list[PilotCase] = []
    for family in design["families"]:
        family = str(family)
        logical_layout = (0, 1, 2) if family == "ghz" else (0, 1)
        physical_layout = chain if family == "ghz" else chain[:2]
        affected = (
            int(design["ghz_affected_logical_qubit"])
            if family == "ghz"
            else int(design["bell_and_phase_affected_logical_qubit"])
        )
        reference = _base_family_circuit(family)
        for condition in design["conditions_per_family"]:
            condition = str(condition)
            observed = _observed_family_circuit(family, condition, strength, affected)
            cases.append(
                PilotCase(
                    case_id=f"{family}__{condition}",
                    family=family,
                    condition=condition,
                    reference_circuit=reference.copy(name=f"{family}__reference"),
                    observed_circuit=observed,
                    logical_layout=logical_layout,
                    physical_layout=physical_layout,
                    affected_logical_qubit=None if condition == "clean" else affected,
                    strength=None if condition == "clean" else strength,
                    expected_effect=condition != "clean",
                    expected_mechanism=None if condition == "clean" else condition,
                )
            )
    expected = int(config["execution"]["case_count"])
    if len(cases) != expected:
        raise RuntimeError(f"Step-9D case count drift: {len(cases)} != {expected}")
    return tuple(cases)


def compile_pilot_programs(
    cases: Sequence[PilotCase],
    backend: Any,
    *,
    optimization_level: int,
    seed_transpiler: int,
    require_no_routing_permutation: bool = True,
) -> tuple[tuple[QuantumCircuit, ...], list[dict[str, Any]]]:
    all_programs: list[QuantumCircuit] = []
    metadata: list[dict[str, Any]] = []
    for case in cases:
        compiled = compile_paired_measurement_circuits(
            case.reference_circuit,
            case.observed_circuit,
            backend,
            initial_layout=case.physical_layout,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )
        if len(compiled) != len(PROGRAM_ORDER):
            raise RuntimeError(f"Step-9D {case.case_id} did not compile six programs")
        for label, circuit in zip(PROGRAM_ORDER, compiled, strict=True):
            if "meas" not in {register.name for register in circuit.cregs}:
                raise RuntimeError(f"Step-9D transpilation removed meas register for {case.case_id}/{label}")
            layout = getattr(circuit, "layout", None)
            routing_permutation = None
            if layout is not None and hasattr(layout, "routing_permutation"):
                routing_permutation = list(layout.routing_permutation())
                if require_no_routing_permutation and routing_permutation != list(range(len(routing_permutation))):
                    raise RuntimeError(f"Step-9D routing permutation detected for {case.case_id}/{label}")
            metadata.append(
                {
                    "case_id": case.case_id,
                    "program": label,
                    "depth": int(circuit.depth()),
                    "size": int(circuit.size()),
                    "count_ops": {str(k): int(v) for k, v in circuit.count_ops().items()},
                    "routing_permutation": routing_permutation,
                }
            )
            all_programs.append(circuit)
    return tuple(all_programs), metadata


def split_sampler_results(
    result: Any,
    cases: Sequence[PilotCase],
) -> dict[str, dict[str, dict[str, int]]]:
    expected = len(cases) * len(PROGRAM_ORDER)
    if len(result) != expected:
        raise RuntimeError(f"Step-9D expected {expected} PUB results, received {len(result)}")
    rows: dict[str, dict[str, dict[str, int]]] = {}
    offset = 0
    for case in cases:
        case_counts: dict[str, dict[str, int]] = {}
        for label in PROGRAM_ORDER:
            case_counts[label] = _extract_counts(result[offset])
            offset += 1
        rows[case.case_id] = case_counts
    return rows


def batch_and_prediction_from_case_counts(
    case: PilotCase,
    counts_by_program: Mapping[str, Mapping[str, int]],
    ensemble: FrozenDeploymentEnsemble,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Step7ModelBatch, dict[str, Any]]:
    reference_counts = {basis: counts_by_program[f"reference_{basis}"] for basis in BASIS_ORDER}
    observed_counts = {basis: counts_by_program[f"observed_{basis}"] for basis in BASIS_ORDER}
    batch = build_step7_model_batch_from_counts(
        case.reference_circuit,
        case.physical_layout,
        reference_counts,
        observed_counts,
        device=device,
    )
    prediction = predict_frozen_ensemble(ensemble, batch)
    return batch, prediction


def descriptive_pilot_metrics(rows: Sequence[Mapping[str, Any]], mechanism_classes: Sequence[str]) -> dict[str, Any]:
    clean = [row for row in rows if not bool(row["expected_effect"])]
    distorted = [row for row in rows if bool(row["expected_effect"])]
    false_positives = sum(bool(row["prediction"]["effect_present"]) for row in clean)
    detected = sum(bool(row["prediction"]["effect_present"]) for row in distorted)
    mechanism_correct = sum(
        str(row["prediction"]["mechanism_prediction"]) == str(row["expected_mechanism"])
        for row in distorted
    )
    classes = [str(value) for value in mechanism_classes]
    confusion = {truth: {pred: 0 for pred in classes} for truth in classes}
    for row in distorted:
        confusion[str(row["expected_mechanism"])][str(row["prediction"]["mechanism_prediction"])] += 1
    return {
        "clean_case_count": len(clean),
        "clean_effect_false_positive_count": int(false_positives),
        "distorted_case_count": len(distorted),
        "distorted_effect_detection_count": int(detected),
        "distorted_effect_detection_fraction": float(detected / len(distorted)) if distorted else None,
        "distorted_mechanism_correct_count": int(mechanism_correct),
        "distorted_mechanism_accuracy": float(mechanism_correct / len(distorted)) if distorted else None,
        "mechanism_confusion_matrix": confusion,
        "confirmatory_interpretation_allowed": False,
    }


__all__ = [
    "PilotCase",
    "ChainCandidate",
    "best_connected_three_qubit_chain",
    "select_backend_and_chain",
    "build_pilot_cases",
    "compile_pilot_programs",
    "split_sampler_results",
    "batch_and_prediction_from_case_counts",
    "descriptive_pilot_metrics",
]
