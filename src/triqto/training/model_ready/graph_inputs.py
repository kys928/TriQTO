"""Graph, parameter, Born, and backend inputs for model-ready artifacts."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from triqto.model import (
    BornTensorBatch,
    DenseFeatureBatch,
    GraphTensorBatch,
    OutcomeQueryTensorBatch,
    ParameterTensorBatch,
    TriQTOModelConfig,
)

from .source import scalar_bool


def float_tensor(value: np.ndarray) -> Tensor:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("numeric model array required")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("model array contains non-finite values")
    return torch.as_tensor(array, dtype=torch.float32).clone()


def long_tensor(value: np.ndarray) -> Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.long).clone()


def require_input(inputs: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = inputs.get(name)
    if value is None:
        raise ValueError(f"model-ready input is missing {name}")
    return np.asarray(value)


def bitstrings_to_tensors(bitstrings: Sequence[str]) -> tuple[Tensor, Tensor]:
    if not bitstrings:
        return torch.zeros((0, 0), dtype=torch.float32), torch.zeros(
            (0, 0), dtype=torch.bool
        )
    widths = [len(value) for value in bitstrings]
    if any(width <= 0 for width in widths):
        raise ValueError("outcome bitstrings must be nonblank")
    width = max(widths)
    bits = torch.zeros((len(bitstrings), width), dtype=torch.float32)
    mask = torch.zeros((len(bitstrings), width), dtype=torch.bool)
    for row, value in enumerate(bitstrings):
        if any(character not in {"0", "1"} for character in value):
            raise ValueError(f"invalid outcome bitstring {value!r}")
        bits[row, : len(value)] = torch.tensor(
            [int(character) for character in value], dtype=torch.float32
        )
        mask[row, : len(value)] = True
    return bits, mask


def graph_batch(inputs: Mapping[str, np.ndarray]) -> GraphTensorBatch:
    node = float_tensor(require_input(inputs, "x_graph_node_features"))
    gate = float_tensor(require_input(inputs, "x_graph_gate_features"))
    return GraphTensorBatch(
        node_features=node,
        edge_index=long_tensor(require_input(inputs, "x_graph_edge_index")),
        edge_features=float_tensor(require_input(inputs, "x_graph_edge_features")),
        edge_event_index=long_tensor(
            require_input(inputs, "x_graph_edge_event_index")
        ),
        gate_features=gate,
        gate_qubit_ptr=long_tensor(
            require_input(inputs, "x_graph_gate_qubit_ptr")
        ),
        gate_qubit_indices=long_tensor(
            require_input(inputs, "x_graph_gate_qubit_indices")
        ),
        node_batch=torch.zeros(node.shape[0], dtype=torch.long),
        gate_batch=torch.zeros(gate.shape[0], dtype=torch.long),
        graph_count=1,
    )


def parameter_batch(
    inputs: Mapping[str, np.ndarray],
) -> ParameterTensorBatch | None:
    sin = inputs.get("x_graph_parameter_sin")
    cos = inputs.get("x_graph_parameter_cos")
    if sin is None and cos is None:
        return None
    if sin is None or cos is None:
        raise ValueError("parameter sine/cosine inputs must be supplied together")
    sin_array = np.asarray(sin, dtype=np.float32).reshape(-1)
    cos_array = np.asarray(cos, dtype=np.float32).reshape(-1)
    if sin_array.shape != cos_array.shape:
        raise ValueError("parameter sine/cosine shapes differ")
    if sin_array.size == 0:
        return None
    if not np.isfinite(sin_array).all() or not np.isfinite(cos_array).all():
        raise ValueError("parameter sine/cosine arrays contain non-finite values")
    values = np.arctan2(sin_array, cos_array).astype(np.float32)
    return ParameterTensorBatch(
        values=torch.from_numpy(values),
        sin=torch.from_numpy(sin_array.copy()),
        cos=torch.from_numpy(cos_array.copy()),
        batch_index=torch.zeros(values.size, dtype=torch.long),
        available_mask=torch.tensor([True]),
    )


def born_input(inputs: Mapping[str, np.ndarray]) -> BornTensorBatch | None:
    outcomes = inputs.get("x_born_input_outcome_bitstrings")
    probabilities = inputs.get("x_born_input_probabilities")
    if outcomes is None and probabilities is None:
        return None
    if outcomes is None or probabilities is None:
        raise ValueError("Born outcomes/probabilities must be supplied together")
    names = [str(value) for value in np.asarray(outcomes).reshape(-1).tolist()]
    bits, mask = bitstrings_to_tensors(names)
    probabilities_tensor = float_tensor(np.asarray(probabilities).reshape(-1))
    return BornTensorBatch(
        outcome_bits=bits,
        outcome_bit_mask=mask,
        probabilities=probabilities_tensor,
        batch_index=torch.zeros(probabilities_tensor.numel(), dtype=torch.long),
        available_mask=torch.tensor([True]),
    )


def born_queries_from_inputs(
    inputs: Mapping[str, np.ndarray],
    task: str,
) -> OutcomeQueryTensorBatch | None:
    if task not in {"born_prediction", "joint_multitask", "hardware_masked"}:
        return None
    outcomes = inputs.get("x_born_input_outcome_bitstrings")
    if outcomes is None:
        raise ValueError(f"{task} requires x_born_input_outcome_bitstrings")
    names = [str(value) for value in np.asarray(outcomes).reshape(-1).tolist()]
    bits, mask = bitstrings_to_tensors(names)
    return OutcomeQueryTensorBatch(
        outcome_bits=bits,
        outcome_bit_mask=mask,
        batch_index=torch.zeros(len(names), dtype=torch.long),
        available_mask=torch.tensor([True]),
    )


def backend_batch(
    inputs: Mapping[str, np.ndarray], config: TriQTOModelConfig
) -> DenseFeatureBatch | None:
    available_value = inputs.get("x_backend_available_mask")
    features_value = inputs.get("x_backend_features")
    if available_value is None and features_value is None:
        return None
    if available_value is None or features_value is None:
        raise ValueError("backend availability/features must be supplied together")
    available = scalar_bool(np.asarray(available_value), "x_backend_available_mask")
    features = np.asarray(features_value, dtype=np.float32).reshape(1, -1)
    if features.shape[1] != config.backend_input_dim:
        raise ValueError(
            f"backend feature width {features.shape[1]} != {config.backend_input_dim}"
        )
    if not np.isfinite(features).all():
        raise ValueError("backend features contain non-finite values")
    if not available:
        return None
    return DenseFeatureBatch(
        features=torch.from_numpy(features.copy()),
        available_mask=torch.tensor([True]),
    )


__all__ = [
    "backend_batch",
    "bitstrings_to_tensors",
    "born_input",
    "born_queries_from_inputs",
    "float_tensor",
    "graph_batch",
    "long_tensor",
    "parameter_batch",
    "require_input",
]
