"""Strict tensor contracts for Phase 13 model-only forward passes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import TriQTOModelConfig
from .constants import HEAD_ORDER, STREAM_ORDER


def _tensor(value: Any, name: str, rank: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if rank is not None and value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {value.ndim}")
    return value


def _float(value: Any, name: str, rank: int | None = None) -> Tensor:
    value = _tensor(value, name, rank)
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _long(value: Any, name: str, rank: int | None = None) -> Tensor:
    value = _tensor(value, name, rank)
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have torch.long dtype")
    return value


def _bool(value: Any, name: str, rank: int | None = None) -> Tensor:
    value = _tensor(value, name, rank)
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have torch.bool dtype")
    return value


def _same_device(named: list[tuple[str, Tensor]]) -> None:
    if len({value.device for _, value in named}) != 1:
        detail = ", ".join(f"{name}={value.device}" for name, value in named)
        raise ValueError(f"All tensors in one batch must share a device: {detail}")


def _batch_index(index: Tensor, count: int, name: str, allow_empty: bool = False) -> None:
    if index.numel() == 0:
        if allow_empty:
            return
        raise ValueError(f"{name} must not be empty")
    if int(index.min()) < 0 or int(index.max()) >= count:
        raise ValueError(f"{name} contains graph index outside [0, {count})")


def _availability(index: Tensor, available: Tensor, name: str) -> None:
    if index.numel() == 0:
        if bool(available.any()):
            raise ValueError(f"{name} marks graphs available but supplies no rows")
        return
    if not bool(available.index_select(0, index).all()):
        raise ValueError(f"{name} supplies rows for a graph marked unavailable")
    present = torch.zeros_like(available)
    present.scatter_(0, index.unique(), True)
    if not torch.equal(present, available):
        raise ValueError(f"{name} availability mask does not match supplied graph rows")


def _basis(bits: Tensor, mask: Tensor, index: Tensor, count: int, name: str) -> None:
    if bits.shape != mask.shape:
        raise ValueError(f"{name} bits and mask must have equal shape")
    if bits.shape[0] == 0:
        return
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"{name} every basis row must expose at least one active qubit")
    if bool((bits[~mask] != 0).any()):
        raise ValueError(f"{name} masked bit positions must be exactly zero")
    active = bits[mask]
    if active.numel() and not bool(((active == 0) | (active == 1)).all()):
        raise ValueError(f"{name} active bit positions must contain only 0/1")
    for graph in range(count):
        rows = torch.nonzero(index == graph, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        local_mask = mask.index_select(0, rows)
        if not torch.equal(local_mask, local_mask[0].expand_as(local_mask)):
            raise ValueError(f"{name} basis masks must be identical within each graph")
        local_bits = bits.index_select(0, rows)
        if torch.unique(local_bits, dim=0).shape[0] != local_bits.shape[0]:
            raise ValueError(f"{name} basis rows must be unique within each graph")


@dataclass(slots=True)
class GraphTensorBatch:
    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    edge_event_index: Tensor
    gate_features: Tensor
    gate_qubit_ptr: Tensor
    gate_qubit_indices: Tensor
    node_batch: Tensor
    gate_batch: Tensor
    graph_count: int

    def validate(self, config: TriQTOModelConfig) -> None:
        if isinstance(self.graph_count, bool) or not isinstance(self.graph_count, int) or self.graph_count <= 0:
            raise TypeError("graph_count must be a positive integer and not bool")
        node = _float(self.node_features, "graph.node_features", 2)
        edge_index = _long(self.edge_index, "graph.edge_index", 2)
        edge = _float(self.edge_features, "graph.edge_features", 2)
        edge_event = _long(self.edge_event_index, "graph.edge_event_index", 1)
        gate = _float(self.gate_features, "graph.gate_features", 2)
        ptr = _long(self.gate_qubit_ptr, "graph.gate_qubit_ptr", 1)
        incidence = _long(self.gate_qubit_indices, "graph.gate_qubit_indices", 1)
        node_batch = _long(self.node_batch, "graph.node_batch", 1)
        gate_batch = _long(self.gate_batch, "graph.gate_batch", 1)
        _same_device([("node", node), ("edge_index", edge_index), ("edge", edge),
                      ("edge_event", edge_event), ("gate", gate), ("ptr", ptr),
                      ("incidence", incidence), ("node_batch", node_batch),
                      ("gate_batch", gate_batch)])
        if node.shape[1] != config.node_input_dim:
            raise ValueError("node feature width does not match model config")
        if edge.shape[1] != config.edge_input_dim:
            raise ValueError("edge feature width does not match model config")
        if gate.shape[1] != config.gate_input_dim:
            raise ValueError("gate feature width does not match model config")
        n, e, g = node.shape[0], edge.shape[0], gate.shape[0]
        if n == 0 or edge_index.shape != (2, e) or edge_event.shape != (e,):
            raise ValueError("graph node/edge shapes are inconsistent")
        if node_batch.shape != (n,) or gate_batch.shape != (g,):
            raise ValueError("node_batch/gate_batch lengths are inconsistent")
        _batch_index(node_batch, self.graph_count, "graph.node_batch")
        _batch_index(gate_batch, self.graph_count, "graph.gate_batch", True)
        if e:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= n:
                raise ValueError("edge_index contains an out-of-range node index")
            if g == 0 or int(edge_event.min()) < 0 or int(edge_event.max()) >= g:
                raise ValueError("edge_event_index contains an out-of-range gate index")
            if not torch.equal(node_batch[edge_index[0]], node_batch[edge_index[1]]):
                raise ValueError("Graph edges must not connect nodes from different graphs")
        if ptr.shape != (g + 1,) or int(ptr[0]) != 0 or int(ptr[-1]) != incidence.numel():
            raise ValueError("gate_qubit_ptr must span the gate incidence array")
        if bool((ptr[1:] < ptr[:-1]).any()):
            raise ValueError("gate_qubit_ptr must be nondecreasing")
        if incidence.numel() and (int(incidence.min()) < 0 or int(incidence.max()) >= n):
            raise ValueError("gate_qubit_indices contains an out-of-range node index")
        present = torch.zeros(self.graph_count, dtype=torch.bool, device=node.device)
        present.scatter_(0, node_batch.unique(), True)
        if not bool(present.all()):
            raise ValueError("Every graph in the batch must own at least one node")
        if g and not torch.equal(node_batch[incidence], gate_batch.repeat_interleave(ptr[1:] - ptr[:-1])):
            raise ValueError("Gate incidence nodes must belong to the gate's graph")


@dataclass(slots=True)
class ParameterTensorBatch:
    values: Tensor
    sin: Tensor
    cos: Tensor
    batch_index: Tensor
    available_mask: Tensor

    def validate(self, count: int) -> None:
        values = _float(self.values, "parameter.values", 1)
        sin = _float(self.sin, "parameter.sin", 1)
        cos = _float(self.cos, "parameter.cos", 1)
        index = _long(self.batch_index, "parameter.batch_index", 1)
        available = _bool(self.available_mask, "parameter.available_mask", 1)
        _same_device([("values", values), ("sin", sin), ("cos", cos),
                      ("index", index), ("available", available)])
        if available.shape != (count,) or not (values.shape == sin.shape == cos.shape == index.shape):
            raise ValueError("parameter shapes are inconsistent")
        _batch_index(index, count, "parameter.batch_index", True)
        _availability(index, available, "parameter")
        if sin.numel() and not torch.allclose(sin, torch.sin(values), atol=1e-6, rtol=1e-6):
            raise ValueError("parameter.sin must equal sin(parameter.values)")
        if cos.numel() and not torch.allclose(cos, torch.cos(values), atol=1e-6, rtol=1e-6):
            raise ValueError("parameter.cos must equal cos(parameter.values)")


@dataclass(slots=True)
class BornTensorBatch:
    outcome_bits: Tensor
    outcome_bit_mask: Tensor
    probabilities: Tensor
    batch_index: Tensor
    available_mask: Tensor

    def validate(self, count: int, atol: float = 1e-6) -> None:
        bits = _float(self.outcome_bits, "born.outcome_bits", 2)
        mask = _bool(self.outcome_bit_mask, "born.outcome_bit_mask", 2)
        prob = _float(self.probabilities, "born.probabilities", 1)
        index = _long(self.batch_index, "born.batch_index", 1)
        available = _bool(self.available_mask, "born.available_mask", 1)
        _same_device([("bits", bits), ("mask", mask), ("prob", prob),
                      ("index", index), ("available", available)])
        if prob.shape != (bits.shape[0],) or index.shape != prob.shape or available.shape != (count,):
            raise ValueError("born row shapes are inconsistent")
        _batch_index(index, count, "born.batch_index", True)
        _availability(index, available, "born")
        _basis(bits, mask, index, count, "born")
        if bool((prob < 0).any()):
            raise ValueError("born probabilities must be nonnegative")
        sums = torch.zeros(count, dtype=prob.dtype, device=prob.device)
        if prob.numel():
            sums.index_add_(0, index, prob)
        if not torch.allclose(sums[available], torch.ones_like(sums[available]), atol=atol, rtol=0):
            raise ValueError("born probabilities must sum to one for every available graph")


@dataclass(slots=True)
class HilbertTensorBatch:
    amplitudes_real_imag: Tensor
    basis_bits: Tensor
    basis_bit_mask: Tensor
    batch_index: Tensor
    available_mask: Tensor

    def validate(self, count: int, atol: float = 1e-5) -> None:
        amp = _float(self.amplitudes_real_imag, "hilbert.amplitudes_real_imag", 2)
        bits = _float(self.basis_bits, "hilbert.basis_bits", 2)
        mask = _bool(self.basis_bit_mask, "hilbert.basis_bit_mask", 2)
        index = _long(self.batch_index, "hilbert.batch_index", 1)
        available = _bool(self.available_mask, "hilbert.available_mask", 1)
        _same_device([("amp", amp), ("bits", bits), ("mask", mask),
                      ("index", index), ("available", available)])
        if amp.shape[1] != 2 or bits.shape != mask.shape or bits.shape[0] != amp.shape[0]:
            raise ValueError("hilbert amplitude/basis shapes are inconsistent")
        if index.shape != (amp.shape[0],) or available.shape != (count,):
            raise ValueError("hilbert batch/availability shapes are inconsistent")
        _batch_index(index, count, "hilbert.batch_index", True)
        _availability(index, available, "hilbert")
        _basis(bits, mask, index, count, "hilbert")
        probability = amp.square().sum(dim=1)
        sums = torch.zeros(count, dtype=amp.dtype, device=amp.device)
        if probability.numel():
            sums.index_add_(0, index, probability)
        if not torch.allclose(sums[available], torch.ones_like(sums[available]), atol=atol, rtol=0):
            raise ValueError("Hilbert amplitudes must be normalized for every available graph")


@dataclass(slots=True)
class DenseFeatureBatch:
    features: Tensor
    available_mask: Tensor

    def validate(self, count: int, width: int, name: str) -> None:
        features = _float(self.features, f"{name}.features", 2)
        available = _bool(self.available_mask, f"{name}.available_mask", 1)
        _same_device([("features", features), ("available", available)])
        if features.shape != (count, width) or available.shape != (count,):
            raise ValueError(f"{name} dense feature shapes are inconsistent")
        if bool((features[~available] != 0).any()):
            raise ValueError(f"{name} unavailable rows must be exactly zero to prevent masked leakage")


@dataclass(slots=True)
class ActionCandidateTensorBatch:
    candidate_features: Tensor
    candidate_batch: Tensor
    candidate_available_mask: Tensor
    edit_type_ids: Tensor
    edit_magnitudes: Tensor
    edit_qubit_positions: Tensor
    edit_candidate_index: Tensor

    def validate(self, count: int, config: TriQTOModelConfig) -> None:
        features = _float(self.candidate_features, "actions.candidate_features", 2)
        index = _long(self.candidate_batch, "actions.candidate_batch", 1)
        mask = _bool(self.candidate_available_mask, "actions.candidate_available_mask", 1)
        types = _long(self.edit_type_ids, "actions.edit_type_ids", 1)
        magnitudes = _float(self.edit_magnitudes, "actions.edit_magnitudes", 1)
        qubits = _float(self.edit_qubit_positions, "actions.edit_qubit_positions", 1)
        edit_index = _long(self.edit_candidate_index, "actions.edit_candidate_index", 1)
        _same_device([("features", features), ("index", index), ("mask", mask),
                      ("types", types), ("magnitudes", magnitudes), ("qubits", qubits),
                      ("edit_index", edit_index)])
        candidates = features.shape[0]
        if features.shape[1] != config.action_candidate_feature_dim:
            raise ValueError("action candidate feature width does not match model config")
        if index.shape != (candidates,) or mask.shape != (candidates,):
            raise ValueError("action candidate row shapes are inconsistent")
        _batch_index(index, count, "actions.candidate_batch", True)
        if bool((features[~mask] != 0).any()):
            raise ValueError("masked action candidate features must be exactly zero")
        if not (types.shape == magnitudes.shape == qubits.shape == edit_index.shape):
            raise ValueError("all action edit arrays must have equal length")
        if types.numel():
            if int(types.min()) < 0 or int(types.max()) >= config.action_edit_type_count:
                raise ValueError("actions.edit_type_ids contains an out-of-vocabulary ID")
            if candidates == 0 or int(edit_index.min()) < 0 or int(edit_index.max()) >= candidates:
                raise ValueError("actions.edit_candidate_index is out of range")
            if bool(((qubits < 0) | (qubits > 1)).any()):
                raise ValueError("actions.edit_qubit_positions must be normalized to [0, 1]")
            if bool((~mask.index_select(0, edit_index)).any()):
                raise ValueError("masked action candidates must not own edit rows")


@dataclass(slots=True)
class OutcomeQueryTensorBatch:
    outcome_bits: Tensor
    outcome_bit_mask: Tensor
    batch_index: Tensor
    available_mask: Tensor

    def validate(self, count: int) -> None:
        bits = _float(self.outcome_bits, "born_queries.outcome_bits", 2)
        mask = _bool(self.outcome_bit_mask, "born_queries.outcome_bit_mask", 2)
        index = _long(self.batch_index, "born_queries.batch_index", 1)
        available = _bool(self.available_mask, "born_queries.available_mask", 1)
        _same_device([("bits", bits), ("mask", mask), ("index", index),
                      ("available", available)])
        if index.shape != (bits.shape[0],) or available.shape != (count,):
            raise ValueError("Born query row shapes are inconsistent")
        _batch_index(index, count, "born_queries.batch_index", True)
        _availability(index, available, "born_queries")
        _basis(bits, mask, index, count, "born_queries")


@dataclass(slots=True)
class TriQTOBatch:
    graph: GraphTensorBatch
    parameter: ParameterTensorBatch | None = None
    born: BornTensorBatch | None = None
    hilbert: HilbertTensorBatch | None = None
    backend: DenseFeatureBatch | None = None
    topology: DenseFeatureBatch | None = None
    actions: ActionCandidateTensorBatch | None = None
    born_queries: OutcomeQueryTensorBatch | None = None
    hardware_mode_mask: Tensor | None = None
    topology_hilbert_dependent_mask: Tensor | None = None
    head_stream_mask: Tensor | None = None
    head_active_mask: Tensor | None = None

    def validate(self, config: TriQTOModelConfig) -> None:
        self.graph.validate(config)
        count, device = self.graph.graph_count, self.graph.node_features.device
        hardware = self.resolved_hardware_mask()
        dependency = self.resolved_topology_hilbert_dependency()
        for value, name in ((hardware, "hardware_mode_mask"),
                            (dependency, "topology_hilbert_dependent_mask")):
            _bool(value, name, 1)
            if value.shape != (count,) or value.device != device:
                raise ValueError(f"{name} must match graph_count and graph device")
        if self.hilbert is not None:
            if not config.use_hilbert:
                raise ValueError("Hilbert tensors were supplied but config.use_hilbert is false")
            if bool((hardware & self.hilbert.available_mask).any()):
                raise ValueError("Hardware-mode rows cannot expose Hilbert tensors")
        if self.topology is not None:
            if not config.use_topology:
                raise ValueError("Topology tensors were supplied but config.use_topology is false")
            if bool((hardware & dependency & self.topology.available_mask).any()):
                raise ValueError("Hardware-mode rows cannot expose topology computed with Hilbert access")
        if self.parameter is not None:
            self.parameter.validate(count)
        if self.born is not None:
            self.born.validate(count)
        if self.hilbert is not None:
            self.hilbert.validate(count)
        if self.backend is not None:
            if not config.use_backend:
                raise ValueError("Backend tensors were supplied but config.use_backend is false")
            self.backend.validate(count, config.backend_input_dim, "backend")
        if self.topology is not None:
            self.topology.validate(count, config.topology_input_dim, "topology")
        if self.actions is not None:
            self.actions.validate(count, config)
        if self.born_queries is not None:
            self.born_queries.validate(count)
        if self.head_stream_mask is not None:
            mask = _bool(self.head_stream_mask, "head_stream_mask", 3)
            expected = (count, len(HEAD_ORDER), len(STREAM_ORDER))
            if tuple(mask.shape) != expected or mask.device != device:
                raise ValueError(f"head_stream_mask must have shape {expected} on graph device")
        if self.head_active_mask is not None:
            active = _bool(self.head_active_mask, "head_active_mask", 2)
            expected = (count, len(HEAD_ORDER))
            if tuple(active.shape) != expected or active.device != device:
                raise ValueError(f"head_active_mask must have shape {expected} on graph device")

    def resolved_hardware_mask(self) -> Tensor:
        return self.hardware_mode_mask if self.hardware_mode_mask is not None else torch.zeros(
            self.graph.graph_count, dtype=torch.bool, device=self.graph.node_features.device)

    def resolved_topology_hilbert_dependency(self) -> Tensor:
        return self.topology_hilbert_dependent_mask if self.topology_hilbert_dependent_mask is not None else torch.zeros(
            self.graph.graph_count, dtype=torch.bool, device=self.graph.node_features.device)

    def resolved_head_active_mask(self) -> Tensor:
        return self.head_active_mask if self.head_active_mask is not None else torch.ones(
            (self.graph.graph_count, len(HEAD_ORDER)), dtype=torch.bool,
            device=self.graph.node_features.device)


__all__ = ["ActionCandidateTensorBatch", "BornTensorBatch", "DenseFeatureBatch",
           "GraphTensorBatch", "HilbertTensorBatch", "OutcomeQueryTensorBatch",
           "ParameterTensorBatch", "TriQTOBatch"]
