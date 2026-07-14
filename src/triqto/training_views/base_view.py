"""Common deterministic helpers for Phase 12 task-specific views."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np

from .constants import (
    SOURCE_DATASET_NAMES,
    SOURCE_USAGE_NAMES,
    TASK_INPUT_GROUPS,
    TASK_TARGET_GROUPS,
)
from .identities import (
    training_view_item_content_hash,
    training_view_item_id,
)
from .models import TrainingViewItem


def unicode_array(values: Sequence[str]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max([1, *[len(value) for value in strings]])
    return np.asarray(strings, dtype=f"<U{width}")


def strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def build_source_arrays(
    refs: Iterable[tuple[str, str, str]],
    *,
    max_refs: int,
) -> dict[str, np.ndarray]:
    rows = sorted(set(refs))
    if len(rows) > max_refs:
        raise RuntimeError(
            f"Training item has {len(rows)} source references, exceeding "
            f"max_source_refs_per_item={max_refs}"
        )
    for dataset, usage, reference in rows:
        if dataset not in SOURCE_DATASET_NAMES:
            raise ValueError(f"Unknown source dataset {dataset!r}")
        if usage not in SOURCE_USAGE_NAMES:
            raise ValueError(f"Unknown source usage {usage!r}")
        if not isinstance(reference, str) or not reference:
            raise ValueError("Source references must be nonblank strings")
    return {
        "source_dataset_names": unicode_array([row[0] for row in rows]),
        "source_usage_names": unicode_array([row[1] for row in rows]),
        "source_refs": unicode_array([row[2] for row in rows]),
    }


def group_mask_arrays(
    input_groups: Sequence[str],
    input_available: Sequence[bool],
    target_groups: Sequence[str],
    target_available: Sequence[bool],
) -> dict[str, np.ndarray]:
    if len(input_groups) != len(input_available):
        raise ValueError("input group and availability lengths must match")
    if len(target_groups) != len(target_available):
        raise ValueError("target group and availability lengths must match")
    return {
        "input_group_names": unicode_array(input_groups),
        "input_group_available_mask": np.asarray(input_available, dtype=np.bool_),
        "target_group_names": unicode_array(target_groups),
        "target_group_available_mask": np.asarray(target_available, dtype=np.bool_),
    }


def graph_structure_arrays(graph: Any, *, prefix: str = "graph") -> dict[str, np.ndarray]:
    """Copy graph structure/parameter arrays while deliberately excluding Born/count evidence."""
    names = (
        "node_index",
        "node_features",
        "edge_index",
        "edge_event_index",
        "edge_features",
        "gate_names",
        "gate_features",
        "gate_qubit_ptr",
        "gate_qubit_indices",
        "gate_clbit_ptr",
        "gate_clbit_indices",
        "gate_parameter_ptr",
        "gate_parameter_values",
        "gate_parameter_sin",
        "gate_parameter_cos",
        "gate_parameter_angle_mask",
        "parameter_names",
        "parameter_values",
        "parameter_sin",
        "parameter_cos",
        "global_features",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        value = getattr(graph, name)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Graph field {name} must be a NumPy array")
        if value.dtype.kind == "O":
            raise TypeError(f"Graph field {name} must not use object dtype")
        arrays[f"{prefix}_{name}"] = value.copy()
    return arrays


def born_arrays(
    bitstrings: Sequence[str] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    keys = [str(value) for value in np.asarray(bitstrings).tolist()]
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(keys) != values.size:
        raise ValueError("Born bitstrings and probabilities must have equal vector length")
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("Born bitstrings must be sorted and unique")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Born probabilities must be finite and nonnegative")
    if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Born probabilities must sum to one")
    return {
        f"{prefix}_outcome_bitstrings": unicode_array(keys),
        f"{prefix}_probabilities": values.copy(),
    }


def measurement_born_arrays(
    pair: Any,
    probabilities: Sequence[float] | np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Copy a basis-conditioned Born table with explicit setting provenance."""
    values = np.asarray(probabilities, dtype=np.float64)
    outcomes = np.asarray(pair.measurement_outcome_bitstrings)
    setting_index = np.asarray(pair.measurement_setting_index)
    setting_ids = np.asarray(pair.measurement_setting_ids)
    basis_codes = np.asarray(pair.measurement_basis_codes)
    if values.ndim != 1 or values.shape != outcomes.shape or values.shape != setting_index.shape:
        raise ValueError("measurement Born row arrays must have equal vector length")
    if setting_ids.ndim != 1 or basis_codes.ndim != 2:
        raise ValueError("measurement setting arrays have invalid rank")
    if basis_codes.shape[0] != len(setting_ids):
        raise ValueError("measurement basis row count must match setting IDs")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("measurement Born probabilities must be finite and nonnegative")
    for index in range(len(setting_ids)):
        mask = setting_index == index
        if not np.any(mask) or not math.isclose(
            float(values[mask].sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("measurement Born probabilities must sum to one per setting")
    return {
        f"{prefix}_outcome_bitstrings": outcomes.copy(),
        f"{prefix}_probabilities": values.copy(),
        f"{prefix}_measurement_setting_ids": setting_ids.copy(),
        f"{prefix}_measurement_basis_codes": basis_codes.copy(),
        f"{prefix}_measurement_setting_index": setting_index.copy(),
    }


def make_training_item(
    *,
    dataset_id: str,
    view_id: str,
    task: str,
    split: str,
    split_group_id: str,
    entity_id: str,
    input_available: Sequence[bool],
    target_available: Sequence[bool],
    arrays: Mapping[str, np.ndarray],
    source_refs: Iterable[tuple[str, str, str]],
    hilbert_available: bool,
    topology_available: bool,
    privileged_target_available: bool,
    metadata: Mapping[str, Any],
    max_source_refs: int,
) -> TrainingViewItem:
    if task not in TASK_INPUT_GROUPS:
        raise ValueError(f"Unknown training-view task {task!r}")
    input_groups = TASK_INPUT_GROUPS[task]
    target_groups = TASK_TARGET_GROUPS[task]
    base = group_mask_arrays(
        input_groups,
        input_available,
        target_groups,
        target_available,
    )
    base.update(build_source_arrays(source_refs, max_refs=max_source_refs))
    for name, value in arrays.items():
        if name in base:
            raise ValueError(f"Task array name collides with base array {name}")
        if not isinstance(name, str) or not name:
            raise ValueError("Task array names must be nonblank strings")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Task array {name} must be a NumPy array")
        if value.dtype.kind == "O":
            raise TypeError(f"Task array {name} must not use object dtype")
        base[name] = value.copy()
    item_id = training_view_item_id(view_id, task, entity_id, split_group_id)
    item = TrainingViewItem(
        view_item_id=item_id,
        training_view_id=view_id,
        training_view_dataset_id=dataset_id,
        task=task,
        split=split,
        split_group_id=split_group_id,
        entity_id=entity_id,
        input_groups=tuple(input_groups),
        target_groups=tuple(target_groups),
        arrays=base,
        hilbert_available_mask=bool(hilbert_available),
        topology_available_mask=bool(topology_available),
        privileged_target_available_mask=bool(privileged_target_available),
        metadata=dict(metadata),
    )
    item.content_hash = training_view_item_content_hash(item)
    return item


__all__ = [
    "born_arrays",
    "build_source_arrays",
    "graph_structure_arrays",
    "group_mask_arrays",
    "make_training_item",
    "measurement_born_arrays",
    "strict_float",
    "unicode_array",
]
