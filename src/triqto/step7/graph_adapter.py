"""Adapt frozen Step 5 NPZ examples into Phase-13 graph + Step-7 diagnostics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch

from triqto.graph.constants import (
    ANGULAR_SLOTS,
    CONTROLLED_TWO_QUBIT_GATES,
    GATE_VOCAB,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    SYMMETRIC_TWO_QUBIT_GATES,
)
from triqto.model.contracts import GraphTensorBatch

from .contracts import DiagnosticTensorBatch, Step7ModelBatch, Step7Targets


def _as_array(example: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in example:
        raise KeyError(f"Step 7 source example is missing {key}")
    return np.asarray(example[key])


def _basis_order(codes: np.ndarray) -> list[int]:
    values = [int(v) for v in np.asarray(codes).reshape(-1).tolist()]
    if sorted(values) != [0, 1, 2]:
        raise ValueError(f"expected one each of Z/X/Y basis codes, got {values}")
    return [values.index(code) for code in (0, 1, 2)]


def _reorder_first_axis(value: np.ndarray, order: Sequence[int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] != 3:
        raise ValueError(f"basis-resolved array must start with dimension 3, got {array.shape}")
    return array[np.asarray(order, dtype=np.int64)]


def _logical_layers(gate_qubit_ptr: np.ndarray, gate_qubit_indices: np.ndarray, n_qubits: int) -> list[int]:
    frontier = [0] * n_qubits
    layers: list[int] = []
    for gate in range(len(gate_qubit_ptr) - 1):
        qubits = [int(v) for v in gate_qubit_indices[gate_qubit_ptr[gate] : gate_qubit_ptr[gate + 1]]]
        layer = max((frontier[q] for q in qubits), default=0)
        layers.append(layer)
        for qubit in qubits:
            frontier[qubit] = layer + 1
    return layers


def _graph_features_for_example(example: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = _as_array(example, "x__graph_gate_names").astype(str)
    qptr = _as_array(example, "x__graph_gate_qubit_ptr").astype(np.int64)
    qidx = _as_array(example, "x__graph_gate_qubit_indices").astype(np.int64)
    pptr = _as_array(example, "x__graph_gate_parameter_ptr").astype(np.int64)
    psin = _as_array(example, "x__graph_gate_parameter_sin").astype(np.float64)
    pcos = _as_array(example, "x__graph_gate_parameter_cos").astype(np.float64)
    n_qubits = int(_as_array(example, "x__layout_logical_to_physical").size)
    gates = len(names)
    if qptr.shape != (gates + 1,) or pptr.shape != (gates + 1,):
        raise ValueError("Step 5 graph pointer lengths are inconsistent")
    if int(qptr[0]) != 0 or int(qptr[-1]) != len(qidx) or int(pptr[0]) != 0 or int(pptr[-1]) != len(psin):
        raise ValueError("Step 5 graph pointers do not span incidence/parameter arrays")
    if psin.shape != pcos.shape:
        raise ValueError("parameter sin/cos arrays must have equal length")
    if qidx.size and (int(qidx.min()) < 0 or int(qidx.max()) >= n_qubits):
        raise ValueError("gate incidence contains out-of-range qubit")

    layers = _logical_layers(qptr, qidx, n_qubits)
    max_layer = max(layers, default=0)
    max_order = max(gates - 1, 1)
    node_features = np.zeros((n_qubits, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    neighbors: list[set[int]] = [set() for _ in range(n_qubits)]
    first_layer: list[int | None] = [None] * n_qubits
    last_layer: list[int | None] = [None] * n_qubits
    gate_features: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    edge_event_index: list[int] = []
    edge_features: list[list[float]] = []

    for order_index, raw_name in enumerate(names.tolist()):
        name = str(raw_name).lower()
        qubits = [int(v) for v in qidx[qptr[order_index] : qptr[order_index + 1]].tolist()]
        parameter_start, parameter_end = int(pptr[order_index]), int(pptr[order_index + 1])
        parameter_count = parameter_end - parameter_start
        angular_slots = ANGULAR_SLOTS.get(name, frozenset())
        angular_positions = [slot for slot in range(parameter_count) if slot in angular_slots]
        angular_count = len(angular_positions)
        angular_sine_sum = sum(float(psin[parameter_start + slot]) for slot in angular_positions)
        angular_cosine_sum = sum(float(pcos[parameter_start + slot]) for slot in angular_positions)
        layer = layers[order_index]
        normalized_order = 0.0 if gates <= 1 else order_index / max_order
        normalized_layer = 0.0 if max_layer <= 0 else layer / max_layer
        arity = len(qubits)
        is_measurement = name == "measure"
        is_reset = name == "reset"
        is_barrier = name == "barrier"
        is_one_qubit = arity == 1 and not (is_measurement or is_reset or is_barrier)
        is_two_qubit = arity == 2 and not is_barrier
        is_multi_qubit = arity > 2
        gate_features.append([
            float(GATE_VOCAB.get(name, GATE_VOCAB["UNK"])),
            float(arity),
            float(order_index),
            float(normalized_order),
            float(layer),
            float(normalized_layer),
            float(is_measurement),
            float(is_reset),
            float(is_barrier),
            float(is_one_qubit),
            float(is_two_qubit),
            float(is_multi_qubit),
            float(parameter_count),
            float(angular_count),
            float(name in CONTROLLED_TWO_QUBIT_GATES),
            float(name in SYMMETRIC_TWO_QUBIT_GATES),
        ])
        for qubit in qubits:
            first_layer[qubit] = layer if first_layer[qubit] is None else min(first_layer[qubit], layer)
            last_layer[qubit] = layer if last_layer[qubit] is None else max(last_layer[qubit], layer)
            node_features[qubit, 5] += 1.0
            node_features[qubit, 6] += float(angular_count)
            node_features[qubit, 7] += float(angular_sine_sum)
            node_features[qubit, 8] += float(angular_cosine_sum)
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
            first, second = qubits
            neighbors[first].add(second)
            neighbors[second].add(first)
            directed = ((first, second, 1.0, 0.0, 1.0), (second, first, 0.0, 1.0, 0.0))
            for source, destination, forward, source_position, destination_position in directed:
                edges.append((source, destination))
                edge_event_index.append(order_index)
                edge_features.append([
                    float(normalized_order),
                    float(normalized_layer),
                    forward,
                    source_position,
                    destination_position,
                    float(name in CONTROLLED_TWO_QUBIT_GATES and source_position == 0.0),
                    float(name in CONTROLLED_TWO_QUBIT_GATES and destination_position == 1.0),
                    float(name in SYMMETRIC_TWO_QUBIT_GATES),
                    float(parameter_count),
                    float(angular_count),
                ])

    for qubit in range(n_qubits):
        node_features[qubit, 9] = float(len(neighbors[qubit]))
        if first_layer[qubit] is not None and last_layer[qubit] is not None:
            divisor = float(max_layer) if max_layer > 0 else 1.0
            node_features[qubit, 10] = float(first_layer[qubit]) / divisor
            node_features[qubit, 11] = float(last_layer[qubit]) / divisor
            node_features[qubit, 12] = float(last_layer[qubit] - first_layer[qubit]) / divisor

    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T if edges else np.empty((2, 0), dtype=np.int64)
    edge_values = np.asarray(edge_features, dtype=np.float32).reshape(-1, len(EDGE_FEATURE_NAMES)) if edges else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_values,
        "edge_event_index": np.asarray(edge_event_index, dtype=np.int64),
        "gate_features": np.asarray(gate_features, dtype=np.float32).reshape(gates, len(GATE_FEATURE_NAMES)),
        "gate_qubit_ptr": qptr,
        "gate_qubit_indices": qidx,
    }


def batch_from_step5_examples(
    examples: Sequence[Mapping[str, np.ndarray]],
    *,
    device: torch.device | str = "cpu",
) -> tuple[Step7ModelBatch, Step7Targets]:
    if not examples:
        raise ValueError("Step 7 batch requires at least one example")
    target_device = torch.device(device)
    node_features: list[np.ndarray] = []
    edge_indices: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    edge_events: list[np.ndarray] = []
    gate_features: list[np.ndarray] = []
    gate_ptr = [0]
    gate_incidence: list[int] = []
    node_batch: list[int] = []
    gate_batch: list[int] = []
    local_values: list[np.ndarray] = []
    pair_values: list[np.ndarray] = []
    pair_indices: list[np.ndarray] = []
    pair_batch: list[int] = []
    parities: list[np.ndarray] = []
    basis_rows: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    reference_mask_rows: list[np.ndarray] = []
    reference_kind_rows: list[np.ndarray] = []
    effects: list[float] = []
    mechanisms: list[int] = []
    mechanism_masks: list[bool] = []
    node_offset = 0
    gate_offset = 0

    for graph_index, example in enumerate(examples):
        graph = _graph_features_for_example(example)
        n_qubits = int(graph["node_features"].shape[0])
        n_gates = int(graph["gate_features"].shape[0])
        node_features.append(graph["node_features"])
        gate_features.append(graph["gate_features"])
        local_edge = graph["edge_index"]
        edge_indices.append(local_edge + node_offset if local_edge.size else local_edge)
        edge_features.append(graph["edge_features"])
        edge_events.append(graph["edge_event_index"] + gate_offset)
        local_ptr = graph["gate_qubit_ptr"]
        local_incidence = graph["gate_qubit_indices"] + node_offset
        for gate in range(n_gates):
            count = int(local_ptr[gate + 1] - local_ptr[gate])
            gate_ptr.append(gate_ptr[-1] + count)
        gate_incidence.extend(int(v) for v in local_incidence.tolist())
        node_batch.extend([graph_index] * n_qubits)
        gate_batch.extend([graph_index] * n_gates)

        codes = _as_array(example, "x__diagnostic_basis_codes").astype(np.int64).reshape(-1)
        order = _basis_order(codes)
        local = _reorder_first_axis(_as_array(example, "x__delta_local_expectations"), order)
        pair = _reorder_first_axis(_as_array(example, "x__delta_pairwise_correlations"), order)
        parity = _reorder_first_axis(_as_array(example, "x__delta_global_parity").reshape(3, 1), order).reshape(3)
        if local.shape != (3, n_qubits):
            raise ValueError(f"local diagnostic shape {local.shape} does not match {n_qubits}q graph")
        raw_pairs = _as_array(example, "x__pair_indices").astype(np.int64).reshape(-1, 2)
        if pair.shape != (3, len(raw_pairs)):
            raise ValueError("pair diagnostic shape does not match pair index count")
        local_values.append(local.T.astype(np.float32, copy=False))
        pair_values.append(pair.T.astype(np.float32, copy=False))
        if len(raw_pairs):
            global_pairs = raw_pairs.T + node_offset
            pair_indices.append(global_pairs)
            pair_batch.extend([graph_index] * len(raw_pairs))
        parities.append(parity.astype(np.float32, copy=False))
        basis_rows.append(np.asarray([0, 1, 2], dtype=np.int64))
        observed_rows.append(_reorder_first_axis(_as_array(example, "x__observed_shots").reshape(3, 1), order).reshape(3).astype(np.int64))
        reference_rows.append(_reorder_first_axis(_as_array(example, "x__reference_shots").reshape(3, 1), order).reshape(3).astype(np.int64))
        reference_mask_rows.append(_reorder_first_axis(_as_array(example, "x__reference_available_mask").reshape(3, 1), order).reshape(3).astype(bool))
        reference_kind_rows.append(_as_array(example, "x__reference_kind_code").astype(np.int64).reshape(1))
        effects.append(float(bool(_as_array(example, "y__effect_present_target").reshape(-1)[0])))
        mechanisms.append(int(_as_array(example, "y__mechanism_target").reshape(-1)[0]))
        mechanism_masks.append(bool(_as_array(example, "y__mechanism_loss_mask").reshape(-1)[0]))
        node_offset += n_qubits
        gate_offset += n_gates

    graph_batch = GraphTensorBatch(
        node_features=torch.as_tensor(np.concatenate(node_features, axis=0), dtype=torch.float32, device=target_device),
        edge_index=torch.as_tensor(np.concatenate(edge_indices, axis=1) if edge_indices else np.empty((2, 0), dtype=np.int64), dtype=torch.long, device=target_device),
        edge_features=torch.as_tensor(np.concatenate(edge_features, axis=0), dtype=torch.float32, device=target_device),
        edge_event_index=torch.as_tensor(np.concatenate(edge_events, axis=0), dtype=torch.long, device=target_device),
        gate_features=torch.as_tensor(np.concatenate(gate_features, axis=0), dtype=torch.float32, device=target_device),
        gate_qubit_ptr=torch.as_tensor(np.asarray(gate_ptr, dtype=np.int64), dtype=torch.long, device=target_device),
        gate_qubit_indices=torch.as_tensor(np.asarray(gate_incidence, dtype=np.int64), dtype=torch.long, device=target_device),
        node_batch=torch.as_tensor(np.asarray(node_batch, dtype=np.int64), dtype=torch.long, device=target_device),
        gate_batch=torch.as_tensor(np.asarray(gate_batch, dtype=np.int64), dtype=torch.long, device=target_device),
        graph_count=len(examples),
    )
    pair_index_array = np.concatenate(pair_indices, axis=1) if pair_indices else np.empty((2, 0), dtype=np.int64)
    pair_value_array = np.concatenate(pair_values, axis=0) if pair_values else np.empty((0, 3), dtype=np.float32)
    diagnostic = DiagnosticTensorBatch(
        local_values=torch.as_tensor(np.concatenate(local_values, axis=0), dtype=torch.float32, device=target_device),
        pair_values=torch.as_tensor(pair_value_array, dtype=torch.float32, device=target_device),
        pair_index=torch.as_tensor(pair_index_array, dtype=torch.long, device=target_device),
        pair_batch=torch.as_tensor(np.asarray(pair_batch, dtype=np.int64), dtype=torch.long, device=target_device),
        global_parity=torch.as_tensor(np.stack(parities), dtype=torch.float32, device=target_device),
        basis_codes=torch.as_tensor(np.stack(basis_rows), dtype=torch.long, device=target_device),
        observed_shots=torch.as_tensor(np.stack(observed_rows), dtype=torch.long, device=target_device),
        reference_shots=torch.as_tensor(np.stack(reference_rows), dtype=torch.long, device=target_device),
        reference_available_mask=torch.as_tensor(np.stack(reference_mask_rows), dtype=torch.bool, device=target_device),
        reference_kind_code=torch.as_tensor(np.stack(reference_kind_rows), dtype=torch.long, device=target_device),
        available_mask=torch.ones(len(examples), dtype=torch.bool, device=target_device),
    )
    targets = Step7Targets(
        effect_present=torch.as_tensor(np.asarray(effects, dtype=np.float32), dtype=torch.float32, device=target_device),
        mechanism=torch.as_tensor(np.asarray(mechanisms, dtype=np.int64), dtype=torch.long, device=target_device),
        mechanism_loss_mask=torch.as_tensor(np.asarray(mechanism_masks, dtype=bool), dtype=torch.bool, device=target_device),
    )
    return Step7ModelBatch(graph=graph_batch, diagnostic=diagnostic), targets


def graph_batch_from_step5_examples(
    examples: Sequence[Mapping[str, np.ndarray]], *, device: torch.device | str = "cpu"
) -> Step7ModelBatch:
    return batch_from_step5_examples(examples, device=device)[0]


__all__ = ["batch_from_step5_examples", "graph_batch_from_step5_examples"]
