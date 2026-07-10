"""Shared exact-objective machinery for Phase 10 continuous baselines."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from triqto.actions import observed_two_qubit_edges
from triqto.distortions.base import copy_for_unitary_distortion
from triqto.distortions.entangling import append_rzz_or_decomposition
from triqto.metrics import compare_born_distributions
from triqto.simulation import simulate_ideal_statevector

from .config import BaselineSuiteConfig
from .constants import PRIMARY_METRIC_NAMES
from .models import EvaluationSnapshot, OptimizerAxis


def build_optimizer_axes(circuit: QuantumCircuit) -> tuple[OptimizerAxis, ...]:
    """Build the fixed logical RX/RY/RZ plus observed-edge RZZ parameterization."""
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be QuantumCircuit")
    axes: list[OptimizerAxis] = []
    for qubit in range(circuit.num_qubits):
        axes.extend(
            (
                OptimizerAxis("rx", (qubit,)),
                OptimizerAxis("ry", (qubit,)),
                OptimizerAxis("rz", (qubit,)),
            )
        )
    for a, b in observed_two_qubit_edges(circuit):
        axes.append(OptimizerAxis("rzz", (int(a), int(b))))
    return tuple(axes)


def optimizer_axis_payload(axes: tuple[OptimizerAxis, ...]) -> list[dict[str, Any]]:
    return [{"kind": axis.kind, "qubits": list(axis.qubits)} for axis in axes]


def _validate_vector(vector: Any, dimensions: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or array.size != dimensions:
        raise ValueError(
            f"optimizer vector must be one-dimensional with {dimensions} entries"
        )
    if not np.isfinite(array).all():
        raise ValueError("optimizer vector must contain only finite values")
    return array


def clip_parameter_vector(
    vector: Any,
    axes: tuple[OptimizerAxis, ...],
    max_abs_angle: float,
) -> np.ndarray:
    array = _validate_vector(vector, len(axes))
    return np.clip(array, -max_abs_angle, max_abs_angle).astype(
        np.float64, copy=False
    )


def circuit_from_parameter_vector(
    source_circuit: QuantumCircuit,
    axes: tuple[OptimizerAxis, ...],
    vector: Any,
    *,
    max_abs_angle: float,
    zero_atol: float,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Append the parameterized edits before final measurements on a circuit copy."""
    clipped = clip_parameter_vector(vector, axes, max_abs_angle)
    editable, restore_measurements, measurement_metadata = copy_for_unitary_distortion(
        source_circuit
    )
    applied: list[dict[str, Any]] = []
    decompositions: list[str] = []
    for axis, magnitude in zip(axes, clipped, strict=True):
        value = float(magnitude)
        if abs(value) <= zero_atol:
            continue
        if axis.kind == "rx":
            editable.rx(value, axis.qubits[0])
        elif axis.kind == "ry":
            editable.ry(value, axis.qubits[0])
        elif axis.kind == "rz":
            editable.rz(value, axis.qubits[0])
        elif axis.kind == "rzz":
            decomposition = append_rzz_or_decomposition(
                editable, value, axis.qubits[0], axis.qubits[1]
            )
            decompositions.append(decomposition)
        else:  # pragma: no cover - fixed versioned axis vocabulary
            raise ValueError(f"Unsupported optimizer axis kind {axis.kind!r}")
        applied.append(
            {
                "kind": axis.kind,
                "qubits": list(axis.qubits),
                "magnitude": value,
            }
        )
    result = restore_measurements(editable)
    return result, {
        **measurement_metadata,
        "applied_coordinate_count": len(applied),
        "applied_coordinates": applied,
        "rzz_decompositions": decompositions,
    }


def metric_array(bundle: Any) -> np.ndarray:
    values: list[float] = []
    for name in PRIMARY_METRIC_NAMES:
        metric = bundle.metrics.get(name)
        if metric is None:
            raise ValueError(f"Born metric bundle is missing {name}")
        value = metric.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Born metric {name} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"Born metric {name} must be finite and nonnegative")
        values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def weighted_objective(values: np.ndarray, config: BaselineSuiteConfig) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(PRIMARY_METRIC_NAMES),):
        raise ValueError("metric values must follow the fixed Phase 10 metric shape")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("metric values must be finite and nonnegative")
    weights = np.asarray(config.metric_weights, dtype=np.float64)
    objective = float(np.dot(weights, array))
    if not math.isfinite(objective):
        raise ValueError("weighted objective is non-finite")
    return objective


def probability_arrays(probabilities: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    if not probabilities:
        raise ValueError("probability mapping must not be empty")
    keys = sorted(probabilities)
    width = max(len(key) for key in keys)
    return (
        np.asarray(keys, dtype=f"<U{max(1, width)}"),
        np.asarray([float(probabilities[key]) for key in keys], dtype=np.float64),
    )


@dataclass(slots=True)
class ExactObjectiveEvaluator:
    """Deterministic exact statevector objective with an auditable evaluation budget."""

    source_circuit: QuantumCircuit
    clean_probabilities: dict[str, float]
    axes: tuple[OptimizerAxis, ...]
    config: BaselineSuiteConfig
    evaluations: int = 0
    best: EvaluationSnapshot | None = None

    def evaluate(self, vector: Any) -> EvaluationSnapshot:
        if self.evaluations >= self.config.max_objective_evaluations:
            raise RuntimeError("max_objective_evaluations exceeded")
        clipped = clip_parameter_vector(
            vector, self.axes, self.config.max_abs_angle
        )
        circuit, application_metadata = circuit_from_parameter_vector(
            self.source_circuit,
            self.axes,
            clipped,
            max_abs_angle=self.config.max_abs_angle,
            zero_atol=self.config.improvement_atol,
        )
        simulation = simulate_ideal_statevector(circuit)
        bundle = compare_born_distributions(
            self.clean_probabilities,
            simulation.probabilities,
            include_kl=False,
            include_js_distance=False,
        )
        values = metric_array(bundle)
        objective = weighted_objective(values, self.config)
        bitstrings, exact_probabilities = probability_arrays(simulation.probabilities)
        snapshot = EvaluationSnapshot(
            vector=clipped.copy(),
            metric_values=values,
            objective=objective,
            outcome_bitstrings=bitstrings,
            exact_probabilities=exact_probabilities,
            metadata={
                "candidate_depth": circuit.depth(),
                "candidate_gate_count": len(circuit.data),
                "application": application_metadata,
            },
        )
        self.evaluations += 1
        if self.best is None:
            self.best = snapshot
        else:
            current_key = (snapshot.objective, tuple(snapshot.vector.tolist()))
            best_key = (self.best.objective, tuple(self.best.vector.tolist()))
            if current_key < best_key:
                self.best = snapshot
        return snapshot


__all__ = [
    "ExactObjectiveEvaluator",
    "build_optimizer_axes",
    "circuit_from_parameter_vector",
    "clip_parameter_vector",
    "metric_array",
    "optimizer_axis_payload",
    "probability_arrays",
    "weighted_objective",
]
