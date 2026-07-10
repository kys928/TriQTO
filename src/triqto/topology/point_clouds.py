"""Aligned parameter, pure-state Hilbert, and Born point-cloud construction."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from triqto.simulation import simulate_ideal_statevector

from .config import TopologyAuditConfig
from .models import TopologyPointCloudGroup
from .topology_groups import TopologyGroupSpec


def _unicode_array(values: list[str]) -> np.ndarray:
    width = max([1, *[len(value) for value in values]])
    return np.asarray(values, dtype=f"<U{width}")


def _strict_numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _probability_matrix(
    rows: list[dict[str, float]],
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("A topology point cloud requires at least one probability row")
    support = sorted({key for row in rows for key in row})
    if not support:
        raise ValueError("Born point cloud has empty support")
    for key in support:
        if len(key) != n_qubits or any(character not in "01" for character in key):
            raise ValueError(
                f"Born outcome {key!r} must be a width-{n_qubits} binary string"
            )
    matrix = np.zeros((len(rows), len(support)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("Born probability rows must be mappings")
        for key, raw_value in row.items():
            if key not in support:
                raise ValueError("Internal Born support mismatch")
            value = _strict_numeric(raw_value, f"probability[{row_index}][{key}]")
            if value < 0.0:
                raise ValueError("Born probabilities must be nonnegative")
            matrix[row_index, support.index(key)] = value
        total = float(np.sum(matrix[row_index]))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Born probability row {row_index} must sum to one; got {total}"
            )
    return _unicode_array(support), matrix


def _statevector_matrix(
    circuits: list[Any],
    config: TopologyAuditConfig,
) -> np.ndarray | None:
    if not config.include_hilbert:
        return None
    vectors: list[np.ndarray] = []
    expected_size: int | None = None
    for index, circuit in enumerate(circuits):
        result = simulate_ideal_statevector(circuit)
        vector = np.asarray(result.statevector.data, dtype=np.complex128)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"Statevector {index} must be a nonempty vector")
        if vector.size > config.max_statevector_amplitudes:
            raise RuntimeError(
                f"Statevector {index} has {vector.size} amplitudes, exceeding "
                f"max_statevector_amplitudes={config.max_statevector_amplitudes}"
            )
        if expected_size is None:
            expected_size = vector.size
        elif vector.size != expected_size:
            raise ValueError("Aligned topology points must share one Hilbert dimension")
        if not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
            raise ValueError("Statevectors must be finite")
        vectors.append(vector)
    return np.stack(vectors, axis=0).astype(np.complex128, copy=False)


def _action_axis_name(edit: Any) -> str:
    qubits = "-".join(str(int(qubit)) for qubit in edit.qubits)
    return f"{edit.edit_type}:q{qubits}"


def _action_point_cloud(
    spec: TopologyGroupSpec,
    sources: Any,
    config: TopologyAuditConfig,
) -> TopologyPointCloudGroup:
    rollout_by_action: dict[str, Any] = {}
    for rollout in sources.action.rollouts_by_id.values():
        if rollout.action_id in rollout_by_action:
            raise ValueError(f"Duplicate Phase 9 rollout for action {rollout.action_id}")
        rollout_by_action[rollout.action_id] = rollout

    candidates: list[Any] = []
    rollouts: list[Any] = []
    circuits: list[Any] = []
    axis_names_set: set[str] = set()
    probability_rows: list[dict[str, float]] = []
    n_qubits: int | None = None
    for action_id in spec.point_ids:
        candidate = sources.action.candidates_by_id.get(action_id)
        rollout = rollout_by_action.get(action_id)
        if candidate is None or rollout is None:
            raise ValueError(f"Action-neighborhood point {action_id} is missing")
        circuit = sources.action.circuits_by_id.get(rollout.candidate_circuit_id)
        if circuit is None:
            raise ValueError(
                f"Action {action_id} references missing candidate circuit "
                f"{rollout.candidate_circuit_id}"
            )
        if n_qubits is None:
            n_qubits = circuit.num_qubits
        elif circuit.num_qubits != n_qubits:
            raise ValueError("One action neighborhood must use one qubit count")
        for edit in candidate.edits:
            axis_names_set.add(_action_axis_name(edit))
        probabilities = {
            str(key): float(value)
            for key, value in zip(
                rollout.outcome_bitstrings.tolist(),
                rollout.exact_probabilities.tolist(),
                strict=True,
            )
        }
        candidates.append(candidate)
        rollouts.append(rollout)
        circuits.append(circuit)
        probability_rows.append(probabilities)
    if n_qubits is None:
        raise ValueError("Action neighborhood is empty")

    axis_names = sorted(axis_names_set)
    axis_index = {name: index for index, name in enumerate(axis_names)}
    coordinates = np.zeros((len(candidates), len(axis_names)), dtype=np.float64)
    coordinate_mask = np.ones_like(coordinates, dtype=np.bool_)
    for row_index, candidate in enumerate(candidates):
        for edit in candidate.edits:
            name = _action_axis_name(edit)
            coordinates[row_index, axis_index[name]] += _strict_numeric(
                edit.magnitude,
                f"ActionCandidate {candidate.action_id} edit magnitude",
            )
    outcomes, born = _probability_matrix(probability_rows, n_qubits)
    statevectors = _statevector_matrix(circuits, config)
    return TopologyPointCloudGroup(
        group_kind=spec.group_kind,
        group_key=spec.group_key,
        point_ids=_unicode_array(list(spec.point_ids)),
        parameter_coordinate_names=_unicode_array(axis_names),
        parameter_coordinates=coordinates,
        parameter_coordinate_mask=coordinate_mask,
        born_outcome_bitstrings=outcomes,
        born_coordinates=born,
        statevectors=statevectors,
        metadata={
            **spec.metadata,
            "parameter_semantics": "summed_phase9_action_edit_delta_by_axis",
            "parameter_coordinate_missing_semantics": "absent edit equals zero delta",
            "born_semantics": "phase9_exact_candidate_probabilities",
            "hilbert_semantics": (
                "recomputed_ideal_candidate_statevectors"
                if statevectors is not None
                else "masked_by_configuration"
            ),
            "raw_statevectors_persisted": False,
            "point_order": "sorted_action_id",
        },
    )


def _cohort_point_cloud(
    spec: TopologyGroupSpec,
    sources: Any,
    config: TopologyAuditConfig,
) -> TopologyPointCloudGroup:
    sample_index = {sample.sample_id: sample for sample in sources.phase7.samples}
    samples: list[Any] = []
    circuits: list[Any] = []
    probability_rows: list[dict[str, float]] = []
    parameter_names_set: set[str] = set()
    n_qubits: int | None = None
    for sample_id in spec.point_ids:
        sample = sample_index.get(sample_id)
        if sample is None:
            raise ValueError(f"Cohort point {sample_id} is missing from Phase 7")
        circuit = sources.phase7.circuits_by_id.get(sample.distorted_circuit_id)
        probabilities = sources.phase7.probabilities_by_run_id.get(
            sample.distorted_run_id
        )
        if circuit is None or probabilities is None:
            raise ValueError(f"Cohort point {sample_id} has missing circuit or Born data")
        if n_qubits is None:
            n_qubits = sample.n_qubits
        elif sample.n_qubits != n_qubits:
            raise ValueError("One cohort topology group must use one qubit count")
        if circuit.num_qubits != sample.n_qubits:
            raise ValueError(f"Sample {sample_id} circuit qubit count mismatch")
        if not isinstance(sample.parameter_bindings, Mapping):
            raise TypeError(f"Sample {sample_id} parameter_bindings must be a mapping")
        for name in sample.parameter_bindings:
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter binding names must be nonblank strings")
            parameter_names_set.add(name)
        samples.append(sample)
        circuits.append(circuit)
        probability_rows.append(dict(probabilities))
    if n_qubits is None:
        raise ValueError("Cohort topology group is empty")

    parameter_names = sorted(parameter_names_set)
    name_index = {name: index for index, name in enumerate(parameter_names)}
    coordinates = np.zeros((len(samples), len(parameter_names)), dtype=np.float64)
    coordinate_mask = np.zeros_like(coordinates, dtype=np.bool_)
    for row_index, sample in enumerate(samples):
        for name, raw_value in sample.parameter_bindings.items():
            column = name_index[name]
            coordinates[row_index, column] = _strict_numeric(
                raw_value,
                f"Sample {sample.sample_id} parameter {name}",
            )
            coordinate_mask[row_index, column] = True
    outcomes, born = _probability_matrix(probability_rows, n_qubits)
    statevectors = _statevector_matrix(circuits, config)
    return TopologyPointCloudGroup(
        group_kind=spec.group_kind,
        group_key=spec.group_key,
        point_ids=_unicode_array(list(spec.point_ids)),
        parameter_coordinate_names=_unicode_array(parameter_names),
        parameter_coordinates=coordinates,
        parameter_coordinate_mask=coordinate_mask,
        born_outcome_bitstrings=outcomes,
        born_coordinates=born,
        statevectors=statevectors,
        metadata={
            **spec.metadata,
            "parameter_semantics": "phase7_bound_circuit_parameters",
            "parameter_coordinate_missing_semantics": (
                "masked_not_imputed; downstream pullback remains available"
            ),
            "born_semantics": "phase7_distorted_exact_probabilities",
            "hilbert_semantics": (
                "recomputed_ideal_distorted_statevectors"
                if statevectors is not None
                else "masked_by_configuration"
            ),
            "raw_statevectors_persisted": False,
            "point_order": "sorted_sample_id",
        },
    )


def build_point_cloud_group(
    spec: TopologyGroupSpec,
    sources: Any,
    config: TopologyAuditConfig,
) -> TopologyPointCloudGroup:
    """Materialize one aligned topology group without mutating source datasets."""
    if not isinstance(spec, TopologyGroupSpec):
        raise TypeError("spec must be TopologyGroupSpec")
    if not isinstance(config, TopologyAuditConfig):
        raise TypeError("config must be TopologyAuditConfig")
    if spec.group_kind == "action_neighborhood":
        return _action_point_cloud(spec, sources, config)
    if spec.group_kind in {
        "family_qubit_cohort",
        "family_qubit_distortion_cohort",
    }:
        return _cohort_point_cloud(spec, sources, config)
    raise ValueError(f"Unsupported topology group kind {spec.group_kind!r}")


__all__ = ["build_point_cloud_group"]
