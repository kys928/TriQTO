"""Conversion of one fully-bound Qiskit circuit into a Phase 8 graph."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numbers
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from .config import GraphConversionConfig
from .constants import (
    ANGULAR_SLOTS,
    CONTROLLED_TWO_QUBIT_GATES,
    EDGE_FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    GATE_VOCAB,
    GRAPH_SCHEMA_VERSION,
    LOGICAL_LAYER_ALGORITHM_VERSION,
    NODE_FEATURE_NAMES,
    SYMMETRIC_TWO_QUBIT_GATES,
)
from .evidence import validate_count_mapping, validate_probability_mapping
from .identities import graph_id
from .models import CircuitGraphData
from .utils import json_copy, require_mapping, require_nonblank


def _qubit_index(circuit: QuantumCircuit, qubit: Any) -> int:
    return int(circuit.find_bit(qubit).index)


def _clbit_index(circuit: QuantumCircuit, clbit: Any) -> int:
    return int(circuit.find_bit(clbit).index)


def _logical_layers(circuit: QuantumCircuit) -> list[int]:
    """Assign deterministic dependency layers using a per-qubit frontier.

    The algorithm is circuit-order stable. Instructions on disjoint qubits may share a
    layer; every instruction advances the frontier of each qubit it touches. These are
    logical dependency layers, not physical pulse times.
    """
    frontier = [0] * circuit.num_qubits
    layers: list[int] = []
    for instruction in circuit.data:
        qubits = [_qubit_index(circuit, qubit) for qubit in instruction.qubits]
        layer = max((frontier[index] for index in qubits), default=0)
        layers.append(layer)
        for index in qubits:
            frontier[index] = layer + 1
    return layers


def _operation_condition(operation: Any, instruction: Any) -> Any:
    condition = getattr(operation, "condition", None)
    if condition is not None:
        return condition
    condition = getattr(instruction, "condition", None)
    if condition is not None:
        return condition
    condition_bits = getattr(operation, "condition_bits", None)
    if condition_bits:
        return condition_bits
    blocks = getattr(operation, "blocks", None)
    if blocks:
        return blocks
    return None


def _strict_bound_numeric_parameter(value: Any, gate_name: str, slot: int) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(
            f"Gate {gate_name} parameter slot {slot} must be a bound numeric value"
        )
    if isinstance(value, numbers.Real):
        numeric = float(value)
    else:
        free_parameters = getattr(value, "parameters", None)
        if free_parameters:
            raise ValueError(
                f"Gate {gate_name} parameter slot {slot} is unbound: {value!r}"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Gate {gate_name} parameter slot {slot} is unsupported: {value!r}"
            ) from exc
    if not math.isfinite(numeric):
        raise ValueError(
            f"Gate {gate_name} parameter slot {slot} must be finite"
        )
    return numeric


def _gate_parameters(instruction: Any) -> tuple[list[float], list[bool]]:
    gate_name = str(instruction.operation.name)
    angular_slots = ANGULAR_SLOTS.get(gate_name, frozenset())
    values: list[float] = []
    masks: list[bool] = []
    for slot, value in enumerate(instruction.operation.params):
        values.append(_strict_bound_numeric_parameter(value, gate_name, slot))
        masks.append(slot in angular_slots)
    return values, masks


def _parameter_binding_arrays(
    parameter_bindings: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require_mapping(parameter_bindings, "parameter_bindings")
    names: list[str] = []
    values: list[float] = []
    for raw_name, raw_value in parameter_bindings.items():
        name = require_nonblank(raw_name, "parameter binding name")
        if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Real):
            raise TypeError(
                f"parameter_bindings[{name!r}] must be an int or float and not bool"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"parameter_bindings[{name!r}] must be finite")
        names.append(name)
        values.append(value)
    if len(set(names)) != len(names):
        raise ValueError("parameter binding names must be unique after normalization")
    ordered = sorted(zip(names, values), key=lambda item: item[0])
    ordered_names = [item[0] for item in ordered]
    ordered_values = np.asarray([item[1] for item in ordered], dtype=np.float64)
    width = max([1, *[len(name) for name in ordered_names]])
    return (
        np.asarray(ordered_names, dtype=f"<U{width}"),
        ordered_values,
        np.sin(ordered_values).astype(np.float64, copy=False),
        np.cos(ordered_values).astype(np.float64, copy=False),
    )


def _global_phase_for_provenance(circuit: QuantumCircuit) -> float:
    value = circuit.global_phase
    if isinstance(value, bool):
        raise TypeError("circuit global_phase must not be bool")
    free_parameters = getattr(value, "parameters", None)
    if free_parameters:
        raise ValueError("circuit global_phase must be fully bound")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("circuit global_phase must be finite")
    return numeric


def circuit_to_graph(
    circuit: QuantumCircuit,
    *,
    circuit_id: str,
    source_run_id: str,
    role: str,
    family: str,
    parameter_bindings: Mapping[str, Any],
    exact_probabilities: Mapping[str, Any],
    source_sample_ids: Sequence[str] = (),
    supplemental_counts: Mapping[str, Any] | None = None,
    supplemental_shots: int | None = None,
    scientific_metadata: Mapping[str, Any] | None = None,
    provenance_metadata: Mapping[str, Any] | None = None,
    config: GraphConversionConfig | None = None,
) -> CircuitGraphData:
    """Convert a fully-bound circuit into a deterministic variable-size graph."""
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a QuantumCircuit")
    circuit_identifier = require_nonblank(circuit_id, "circuit_id")
    run_identifier = require_nonblank(source_run_id, "source_run_id")
    family_name = require_nonblank(family, "family")
    if role not in {"clean", "distorted"}:
        raise ValueError("role must be clean or distorted")
    conversion_config = config or GraphConversionConfig()
    if circuit.parameters:
        raise ValueError(
            "Phase 8 requires fully bound circuits; unbound parameters remain"
        )
    if len(circuit.data) > conversion_config.max_gate_events:
        raise ValueError(
            f"Circuit {circuit_identifier} has {len(circuit.data)} gate events, "
            f"exceeding max_gate_events={conversion_config.max_gate_events}"
        )

    source_ids = tuple(
        sorted(
            require_nonblank(value, "source_sample_ids entry")
            for value in source_sample_ids
        )
    )
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_sample_ids must be unique")

    working = circuit.copy()
    n_qubits = working.num_qubits
    n_clbits = working.num_clbits
    if n_qubits <= 0:
        raise ValueError("Phase 8 graphs require at least one logical qubit")
    layers = _logical_layers(working)
    max_layer = max(layers, default=0)
    max_order = max(len(working.data) - 1, 1)

    node_features = np.zeros(
        (n_qubits, len(NODE_FEATURE_NAMES)),
        dtype=np.float64,
    )
    neighbors: list[set[int]] = [set() for _ in range(n_qubits)]
    first_layer: list[int | None] = [None] * n_qubits
    last_layer: list[int | None] = [None] * n_qubits

    gate_names: list[str] = []
    gate_features: list[list[float]] = []
    gate_qubit_ptr = [0]
    gate_qubit_indices: list[int] = []
    gate_clbit_ptr = [0]
    gate_clbit_indices: list[int] = []
    gate_parameter_ptr = [0]
    gate_parameter_values: list[float] = []
    gate_parameter_sin: list[float] = []
    gate_parameter_cos: list[float] = []
    gate_parameter_angle_mask: list[bool] = []
    edges: list[tuple[int, int]] = []
    edge_event_index: list[int] = []
    edge_features: list[list[float]] = []
    multi_qubit_event_count = 0

    for order, instruction in enumerate(working.data):
        operation = instruction.operation
        name = str(operation.name)
        condition = _operation_condition(operation, instruction)
        if condition is not None:
            raise NotImplementedError(
                f"Conditioned/control-flow operation {name!r} at event {order} "
                "is not supported in Phase 8 v1"
            )
        qubits = [_qubit_index(working, qubit) for qubit in instruction.qubits]
        clbits = [_clbit_index(working, clbit) for clbit in instruction.clbits]
        parameter_values, angle_masks = _gate_parameters(instruction)
        layer = layers[order]
        normalized_order = 0.0 if len(working.data) <= 1 else order / max_order
        normalized_layer = 0.0 if max_layer <= 0 else layer / max_layer
        arity = len(qubits)
        is_measurement = name == "measure"
        is_reset = name == "reset"
        is_barrier = name == "barrier"
        is_one_qubit = arity == 1 and not (
            is_measurement or is_reset or is_barrier
        )
        is_two_qubit = arity == 2 and not is_barrier
        is_multi_qubit = arity > 2
        if is_multi_qubit:
            multi_qubit_event_count += 1

        gate_names.append(name)
        gate_features.append(
            [
                float(GATE_VOCAB.get(name, GATE_VOCAB["UNK"])),
                float(arity),
                float(order),
                float(normalized_order),
                float(layer),
                float(normalized_layer),
                float(is_measurement),
                float(is_reset),
                float(is_barrier),
                float(is_one_qubit),
                float(is_two_qubit),
                float(is_multi_qubit),
                float(len(parameter_values)),
                float(sum(angle_masks)),
                float(name in CONTROLLED_TWO_QUBIT_GATES),
                float(name in SYMMETRIC_TWO_QUBIT_GATES),
            ]
        )

        gate_qubit_indices.extend(qubits)
        gate_qubit_ptr.append(len(gate_qubit_indices))
        gate_clbit_indices.extend(clbits)
        gate_clbit_ptr.append(len(gate_clbit_indices))
        gate_parameter_values.extend(parameter_values)
        gate_parameter_angle_mask.extend(angle_masks)
        gate_parameter_sin.extend(
            math.sin(value) if is_angle else 0.0
            for value, is_angle in zip(parameter_values, angle_masks)
        )
        gate_parameter_cos.extend(
            math.cos(value) if is_angle else 0.0
            for value, is_angle in zip(parameter_values, angle_masks)
        )
        gate_parameter_ptr.append(len(gate_parameter_values))

        angular_count = float(sum(angle_masks))
        angular_sine_sum = sum(
            math.sin(value)
            for value, is_angle in zip(parameter_values, angle_masks)
            if is_angle
        )
        angular_cosine_sum = sum(
            math.cos(value)
            for value, is_angle in zip(parameter_values, angle_masks)
            if is_angle
        )
        for qubit in qubits:
            first_layer[qubit] = (
                layer if first_layer[qubit] is None else min(first_layer[qubit], layer)
            )
            last_layer[qubit] = (
                layer if last_layer[qubit] is None else max(last_layer[qubit], layer)
            )
            node_features[qubit, 5] += 1.0
            node_features[qubit, 6] += angular_count
            node_features[qubit, 7] += angular_sine_sum
            node_features[qubit, 8] += angular_cosine_sum
            if is_measurement:
                node_features[qubit, 0] = 1.0
                node_features[qubit, 1] += 1.0
            if is_reset:
                node_features[qubit, 2] += 1.0
            if is_one_qubit:
                node_features[qubit, 3] += 1.0
            if is_two_qubit:
                node_features[qubit, 4] += 1.0

        if is_two_qubit:
            first_qubit, second_qubit = qubits
            neighbors[first_qubit].add(second_qubit)
            neighbors[second_qubit].add(first_qubit)
            directed = (
                (first_qubit, second_qubit, 1.0, 0.0, 1.0),
                (second_qubit, first_qubit, 0.0, 1.0, 0.0),
            )
            for source, destination, forward, source_position, destination_position in directed:
                known_control_source = (
                    name in CONTROLLED_TWO_QUBIT_GATES and source_position == 0.0
                )
                known_target_destination = (
                    name in CONTROLLED_TWO_QUBIT_GATES
                    and destination_position == 1.0
                )
                edges.append((source, destination))
                edge_event_index.append(order)
                edge_features.append(
                    [
                        float(normalized_order),
                        float(normalized_layer),
                        forward,
                        source_position,
                        destination_position,
                        float(known_control_source),
                        float(known_target_destination),
                        float(name in SYMMETRIC_TWO_QUBIT_GATES),
                        float(len(parameter_values)),
                        float(sum(angle_masks)),
                    ]
                )

    for qubit in range(n_qubits):
        node_features[qubit, 9] = float(len(neighbors[qubit]))
        if first_layer[qubit] is not None and last_layer[qubit] is not None:
            divisor = float(max_layer) if max_layer > 0 else 1.0
            node_features[qubit, 10] = first_layer[qubit] / divisor
            node_features[qubit, 11] = last_layer[qubit] / divisor
            node_features[qubit, 12] = (
                last_layer[qubit] - first_layer[qubit]
            ) / divisor

    parameter_names, parameter_values, parameter_sin, parameter_cos = (
        _parameter_binding_arrays(parameter_bindings)
    )
    outcome_bitstrings, exact_probability_values, clipped_count = (
        validate_probability_mapping(
            exact_probabilities,
            n_qubits,
            max_outcomes=conversion_config.max_probability_outcomes,
        )
    )

    count_outcomes = np.asarray([], dtype="<U1")
    count_values = np.asarray([], dtype=np.int64)
    count_available = supplemental_counts is not None
    if count_available:
        if supplemental_shots is None:
            raise ValueError(
                "supplemental_shots is required when supplemental_counts is provided"
            )
        count_outcomes, count_values = validate_count_mapping(
            supplemental_counts,
            n_qubits,
            supplemental_shots,
        )
    elif supplemental_shots is not None:
        raise ValueError(
            "supplemental_shots must be absent when supplemental_counts is absent"
        )

    scientific = json_copy(dict(scientific_metadata or {}))
    scientific.update(
        {
            "logical_layer_algorithm_version": LOGICAL_LAYER_ALGORITHM_VERSION,
            "multi_qubit_event_count": multi_qubit_event_count,
            "probability_negative_clip_count": clipped_count,
        }
    )
    provenance = json_copy(dict(provenance_metadata or {}))
    provenance["global_phase"] = _global_phase_for_provenance(working)
    provenance["global_phase_excluded_from_features"] = True

    gate_name_width = max([1, *[len(name) for name in gate_names]])
    edge_index_array = (
        np.asarray(edges, dtype=np.int64).T
        if edges
        else np.empty((2, 0), dtype=np.int64)
    )
    return CircuitGraphData(
        graph_id=graph_id(circuit_identifier, run_identifier, role),
        circuit_id=circuit_identifier,
        source_run_id=run_identifier,
        role=role,
        family=family_name,
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        n_qubits=n_qubits,
        n_clbits=n_clbits,
        node_index=np.arange(n_qubits, dtype=np.int64),
        node_features=node_features,
        edge_index=edge_index_array,
        edge_event_index=np.asarray(edge_event_index, dtype=np.int64),
        edge_features=(
            np.asarray(edge_features, dtype=np.float64).reshape(
                len(edge_features), len(EDGE_FEATURE_NAMES)
            )
            if edge_features
            else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float64)
        ),
        gate_names=np.asarray(gate_names, dtype=f"<U{gate_name_width}"),
        gate_features=(
            np.asarray(gate_features, dtype=np.float64).reshape(
                len(gate_features), len(GATE_FEATURE_NAMES)
            )
            if gate_features
            else np.empty((0, len(GATE_FEATURE_NAMES)), dtype=np.float64)
        ),
        gate_qubit_ptr=np.asarray(gate_qubit_ptr, dtype=np.int64),
        gate_qubit_indices=np.asarray(gate_qubit_indices, dtype=np.int64),
        gate_clbit_ptr=np.asarray(gate_clbit_ptr, dtype=np.int64),
        gate_clbit_indices=np.asarray(gate_clbit_indices, dtype=np.int64),
        gate_parameter_ptr=np.asarray(gate_parameter_ptr, dtype=np.int64),
        gate_parameter_values=np.asarray(gate_parameter_values, dtype=np.float64),
        gate_parameter_sin=np.asarray(gate_parameter_sin, dtype=np.float64),
        gate_parameter_cos=np.asarray(gate_parameter_cos, dtype=np.float64),
        gate_parameter_angle_mask=np.asarray(
            gate_parameter_angle_mask,
            dtype=np.bool_,
        ),
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        parameter_sin=parameter_sin,
        parameter_cos=parameter_cos,
        outcome_bitstrings=outcome_bitstrings,
        exact_probabilities=exact_probability_values,
        global_features=np.empty((0,), dtype=np.float64),
        count_outcome_bitstrings=count_outcomes,
        supplemental_counts=count_values,
        source_sample_ids=source_ids,
        exact_probability_available_mask=True,
        supplemental_counts_available_mask=count_available,
        hilbert_available_mask=False,
        supplemental_shots=supplemental_shots,
        scientific_metadata=scientific,
        provenance_metadata=provenance,
    )


__all__ = ["circuit_to_graph"]
