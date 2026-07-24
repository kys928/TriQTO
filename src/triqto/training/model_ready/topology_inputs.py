"""Canonical topology inputs and explicit ablation families."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from triqto.model import DenseFeatureBatch, TriQTOModelConfig

from .graph_inputs import require_input
from .source import scalar_bool
from .types import (
    BORN_TOPOLOGY_ABLATION_DIM,
    CANONICAL_ALIGNMENT_FEATURE_DIM,
    CANONICAL_TOPOLOGY_FEATURE_DIM,
    CANONICAL_TOPOLOGY_INPUT_DIM,
    PARAMETER_TOPOLOGY_ABLATION_DIM,
)


def _validate_named_family(
    inputs: Mapping[str, np.ndarray],
    *,
    prefix: str,
    width: int,
) -> tuple[str, ...]:
    names_value = inputs.get(f"{prefix}_feature_names")
    if names_value is None:
        raise ValueError(f"{prefix}_feature_names is required")
    names = tuple(
        str(value) for value in np.asarray(names_value).reshape(-1).tolist()
    )
    if len(names) != width or len(set(names)) != width:
        raise ValueError(f"{prefix}_feature_names is not a unique width-{width} map")
    masks: dict[str, np.ndarray] = {}
    for suffix in (
        "feature_mask",
        "positive_infinity_mask",
        "negative_infinity_mask",
    ):
        value = inputs.get(f"{prefix}_{suffix}")
        if value is None:
            raise ValueError(f"{prefix}_{suffix} is required")
        array = np.asarray(value, dtype=np.bool_).reshape(-1)
        if array.size != width:
            raise ValueError(f"{prefix}_{suffix} width mismatch")
        masks[suffix] = array
    finite = masks["feature_mask"]
    positive = masks["positive_infinity_mask"]
    negative = masks["negative_infinity_mask"]
    if np.any(finite & (positive | negative)) or np.any(positive & negative):
        raise ValueError(f"{prefix} finite/infinity masks overlap")
    if not bool((finite | positive | negative).all()):
        raise ValueError(f"{prefix} masks do not classify every feature")
    return names


def canonical_topology_input(
    inputs: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, bool, dict[str, np.ndarray]]:
    """Build 110 persistence + 11 alignment values without double counting."""
    available_value = inputs.get("x_topology_available_mask")
    available = (
        False
        if available_value is None
        else scalar_bool(np.asarray(available_value), "x_topology_available_mask")
    )
    ablations: dict[str, np.ndarray] = {}
    for name, expected in (
        ("x_topology_parameter_features", PARAMETER_TOPOLOGY_ABLATION_DIM),
        ("x_topology_born_features", BORN_TOPOLOGY_ABLATION_DIM),
    ):
        value = inputs.get(name)
        if value is not None:
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            if vector.size != expected or not np.isfinite(vector).all():
                raise ValueError(f"{name} must be finite with width {expected}")
            ablations[name] = vector.copy()
    if not available:
        if "x_topology_features" in inputs or "x_topology_alignment_features" in inputs:
            raise ValueError("topology values are present while availability is false")
        return (
            np.zeros(CANONICAL_TOPOLOGY_INPUT_DIM, dtype=np.float32),
            False,
            ablations,
        )

    topology = np.asarray(
        require_input(inputs, "x_topology_features"), dtype=np.float32
    ).reshape(-1)
    alignment = np.asarray(
        require_input(inputs, "x_topology_alignment_features"), dtype=np.float32
    ).reshape(-1)
    if topology.size != CANONICAL_TOPOLOGY_FEATURE_DIM:
        raise ValueError(
            f"x_topology_features must have width {CANONICAL_TOPOLOGY_FEATURE_DIM}"
        )
    if alignment.size != CANONICAL_ALIGNMENT_FEATURE_DIM:
        raise ValueError(
            "x_topology_alignment_features must have width "
            f"{CANONICAL_ALIGNMENT_FEATURE_DIM}"
        )
    if not np.isfinite(topology).all() or not np.isfinite(alignment).all():
        raise ValueError("canonical topology values must be finite")

    topology_names = _validate_named_family(
        inputs, prefix="x_topology", width=CANONICAL_TOPOLOGY_FEATURE_DIM
    )
    _validate_named_family(
        inputs,
        prefix="x_topology_alignment",
        width=CANONICAL_ALIGNMENT_FEATURE_DIM,
    )
    parameter_names = _validate_named_family(
        inputs,
        prefix="x_topology_parameter",
        width=PARAMETER_TOPOLOGY_ABLATION_DIM,
    )
    born_names = _validate_named_family(
        inputs,
        prefix="x_topology_born",
        width=BORN_TOPOLOGY_ABLATION_DIM,
    )
    expected_names = tuple(f"parameter_{name}" for name in parameter_names) + tuple(
        f"born_{name}" for name in born_names
    )
    if topology_names != expected_names:
        raise ValueError(
            "combined topology mapping is not ordered parameter+Born features"
        )

    vector = np.concatenate((topology, alignment)).astype(np.float32, copy=False)
    if vector.size != CANONICAL_TOPOLOGY_INPUT_DIM:
        raise AssertionError("canonical topology width invariant failed")
    return vector.copy(), True, ablations


def topology_batch(
    inputs: Mapping[str, np.ndarray], config: TriQTOModelConfig
) -> tuple[DenseFeatureBatch | None, dict[str, np.ndarray]]:
    vector, available, ablations = canonical_topology_input(inputs)
    if config.topology_input_dim != CANONICAL_TOPOLOGY_INPUT_DIM:
        raise ValueError(
            "model topology_input_dim must equal canonical model-ready width "
            f"{CANONICAL_TOPOLOGY_INPUT_DIM}, got {config.topology_input_dim}"
        )
    if not available:
        return None, ablations
    return (
        DenseFeatureBatch(
            features=torch.from_numpy(vector.reshape(1, -1)),
            available_mask=torch.tensor([True]),
        ),
        ablations,
    )


__all__ = ["canonical_topology_input", "topology_batch"]
