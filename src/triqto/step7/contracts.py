"""Strict Step 7 contracts for signed relational diagnostic evidence."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from triqto.model.contracts import GraphTensorBatch


def _float(value: Tensor, name: str, rank: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != rank:
        raise TypeError(f"{name} must be a rank-{rank} torch.Tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _long(value: Tensor, name: str, rank: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != rank or value.dtype != torch.long:
        raise TypeError(f"{name} must be a rank-{rank} torch.long tensor")
    return value


def _bool(value: Tensor, name: str, rank: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != rank or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a rank-{rank} torch.bool tensor")
    return value


@dataclass(slots=True)
class DiagnosticTensorBatch:
    """Hardware-facing signed B_delta evidence aligned to circuit-graph qubits."""

    local_values: Tensor
    pair_values: Tensor
    pair_index: Tensor
    pair_batch: Tensor
    global_parity: Tensor
    basis_codes: Tensor
    observed_shots: Tensor
    reference_shots: Tensor
    reference_available_mask: Tensor
    reference_kind_code: Tensor
    available_mask: Tensor

    def validate(self, graph: GraphTensorBatch) -> None:
        count = graph.graph_count
        device = graph.node_features.device
        local = _float(self.local_values, "diagnostic.local_values", 2)
        pair = _float(self.pair_values, "diagnostic.pair_values", 2)
        pair_index = _long(self.pair_index, "diagnostic.pair_index", 2)
        pair_batch = _long(self.pair_batch, "diagnostic.pair_batch", 1)
        parity = _float(self.global_parity, "diagnostic.global_parity", 2)
        basis = _long(self.basis_codes, "diagnostic.basis_codes", 2)
        observed = _long(self.observed_shots, "diagnostic.observed_shots", 2)
        reference = _long(self.reference_shots, "diagnostic.reference_shots", 2)
        reference_mask = _bool(
            self.reference_available_mask, "diagnostic.reference_available_mask", 2
        )
        reference_kind = _long(
            self.reference_kind_code, "diagnostic.reference_kind_code", 2
        )
        available = _bool(self.available_mask, "diagnostic.available_mask", 1)

        tensors = (
            local,
            pair,
            pair_index,
            pair_batch,
            parity,
            basis,
            observed,
            reference,
            reference_mask,
            reference_kind,
            available,
        )
        if any(value.device != device for value in tensors):
            raise ValueError("all Step 7 diagnostic tensors must share the graph device")
        if local.shape != (graph.node_features.shape[0], 3):
            raise ValueError("local_values must provide exactly Z/X/Y values per graph node")
        if pair.shape[1:] != (3,):
            raise ValueError("pair_values must have width 3 for Z/X/Y")
        if pair_index.shape != (2, pair.shape[0]) or pair_batch.shape != (pair.shape[0],):
            raise ValueError("pair_index/pair_batch shapes are inconsistent")
        if parity.shape != (count, 3):
            raise ValueError("global_parity must have shape [graph_count, 3]")
        for name, value in (
            ("basis_codes", basis),
            ("observed_shots", observed),
            ("reference_shots", reference),
            ("reference_available_mask", reference_mask),
        ):
            if value.shape != (count, 3):
                raise ValueError(f"{name} must have shape [graph_count, 3]")
        if reference_kind.shape != (count, 1) or available.shape != (count,):
            raise ValueError("reference_kind_code/available_mask shapes are inconsistent")
        if bool((observed[available] <= 0).any()) or bool((reference[available] <= 0).any()):
            raise ValueError("available diagnostics require positive observed/reference shots")
        expected_basis = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        if available.any():
            sorted_basis = torch.sort(basis[available], dim=1).values
            if not torch.equal(sorted_basis, expected_basis.expand_as(sorted_basis)):
                raise ValueError("every available graph must expose exactly Z/X/Y basis identities")
        if pair.shape[0]:
            nodes = graph.node_features.shape[0]
            if int(pair_index.min()) < 0 or int(pair_index.max()) >= nodes:
                raise ValueError("diagnostic pair_index contains an out-of-range graph node")
            if int(pair_batch.min()) < 0 or int(pair_batch.max()) >= count:
                raise ValueError("diagnostic pair_batch contains an out-of-range graph index")
            left_batch = graph.node_batch.index_select(0, pair_index[0])
            right_batch = graph.node_batch.index_select(0, pair_index[1])
            if not torch.equal(left_batch, right_batch) or not torch.equal(left_batch, pair_batch):
                raise ValueError("diagnostic pairs must connect nodes from exactly one owning graph")
            if bool((pair_index[0] >= pair_index[1]).any()):
                raise ValueError("diagnostic pairs must use canonical ascending node endpoints")
        unavailable_nodes = ~available.index_select(0, graph.node_batch)
        if unavailable_nodes.any() and bool((local[unavailable_nodes] != 0).any()):
            raise ValueError("unavailable graph rows must zero local diagnostics")
        if pair.shape[0]:
            unavailable_pairs = ~available.index_select(0, pair_batch)
            if unavailable_pairs.any() and bool((pair[unavailable_pairs] != 0).any()):
                raise ValueError("unavailable graph rows must zero pair diagnostics")
        if (~available).any():
            if bool((parity[~available] != 0).any()):
                raise ValueError("unavailable graph rows must zero parity diagnostics")


@dataclass(slots=True)
class Step7ModelBatch:
    graph: GraphTensorBatch
    diagnostic: DiagnosticTensorBatch

    def validate(self, graph_config: object) -> None:
        self.graph.validate(graph_config)  # TriQTOModelConfig-compatible object
        self.diagnostic.validate(self.graph)


@dataclass(slots=True)
class Step7Targets:
    effect_present: Tensor
    mechanism: Tensor
    mechanism_loss_mask: Tensor

    def validate(self, count: int, device: torch.device) -> None:
        effect = _float(self.effect_present, "targets.effect_present", 1)
        mechanism = _long(self.mechanism, "targets.mechanism", 1)
        mask = _bool(self.mechanism_loss_mask, "targets.mechanism_loss_mask", 1)
        if any(value.device != device for value in (effect, mechanism, mask)):
            raise ValueError("Step 7 targets must share the model batch device")
        if effect.shape != (count,) or mechanism.shape != (count,) or mask.shape != (count,):
            raise ValueError("Step 7 target shapes must match graph_count")
        if bool(((effect < 0) | (effect > 1)).any()):
            raise ValueError("effect_present must contain binary 0/1 values")
        if mask.any():
            supervised = mechanism[mask]
            if int(supervised.min()) < 0 or int(supervised.max()) > 2:
                raise ValueError("supervised mechanism targets must be 0(RZ),1(RX),2(RY)")


__all__ = ["DiagnosticTensorBatch", "Step7ModelBatch", "Step7Targets"]
