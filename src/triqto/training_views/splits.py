"""Deterministic clean-circuit and explicit OOD splits for Phase 12."""
from __future__ import annotations

import hashlib
from typing import Any

from triqto.core.ids import canonical_json

from .config import TrainingViewConfig
from .constants import TRAINING_VIEW_SPLIT_VERSION


def _split_unit(split_group_id: str, config: TrainingViewConfig) -> float:
    digest = hashlib.sha256(
        canonical_json(
            {
                "split_group_id": split_group_id,
                "split_seed": config.split_seed,
                "split_version": TRAINING_VIEW_SPLIT_VERSION,
            }
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_split(split_group_id: str, config: TrainingViewConfig) -> str:
    """Hash one clean-circuit group into an IID train/validation/test split."""
    if not isinstance(split_group_id, str) or not split_group_id.strip():
        raise ValueError("split_group_id must be nonblank text")
    if not isinstance(config, TrainingViewConfig):
        raise TypeError("config must be TrainingViewConfig")
    if config.split_strategy != "clean_circuit_hash":
        raise ValueError("assign_split is only valid for clean_circuit_hash strategy")
    unit = _split_unit(split_group_id, config)
    if unit < config.train_fraction:
        return "train"
    if unit < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def assign_development_split(
    split_group_id: str,
    config: TrainingViewConfig,
) -> str:
    """Hash a non-holdout group into train/validation only."""
    if not isinstance(split_group_id, str) or not split_group_id.strip():
        raise ValueError("split_group_id must be nonblank text")
    if not isinstance(config, TrainingViewConfig):
        raise TypeError("config must be TrainingViewConfig")
    if config.split_strategy != "axis_holdout":
        raise ValueError("assign_development_split requires axis_holdout strategy")
    threshold = config.train_fraction / (
        config.train_fraction + config.validation_fraction
    )
    return "train" if _split_unit(split_group_id, config) < threshold else "validation"


def sample_holdout_axis_value(
    sample: Any,
    *,
    axis: str,
    distortions_by_id: dict[str, Any],
) -> str:
    """Read one scientifically meaningful split axis from a Phase 7 sample."""
    if axis == "family":
        return str(sample.family)
    if axis == "n_qubits":
        return str(sample.n_qubits)
    if axis == "distortion_type":
        distortion = distortions_by_id.get(sample.distortion_id)
        if distortion is None:
            raise ValueError(
                f"Sample {sample.sample_id} has no distortion record for holdout assignment"
            )
        value = distortion.distortion_type
        if not isinstance(value, str) or not value:
            raise ValueError("distortion_type holdout values must be nonblank")
        return value
    if axis == "backend_id":
        value = sample.metadata.get("backend_id")
        if not isinstance(value, str) or not value:
            raise ValueError(
                "backend_id holdout requested but backend_feature_unavailable"
            )
        return value
    raise ValueError(f"Unsupported holdout axis {axis!r}")


def build_sample_split_maps(
    phase7: Any,
    config: TrainingViewConfig,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return sample splits and leakage groups under the declared strategy."""
    sample_splits: dict[str, str] = {}
    sample_groups: dict[str, str] = {}
    split_by_clean_circuit: dict[str, str] = {}
    distortions_by_id = {
        distortion.distortion_id: distortion for distortion in phase7.distortions
    }
    observed_axis_values: set[str] = set()
    for sample in sorted(phase7.samples, key=lambda item: item.sample_id):
        sample_id = sample.sample_id
        clean_circuit_id = sample.clean_circuit_id
        if sample_id in sample_splits:
            raise ValueError(f"Duplicate Phase 7 sample ID {sample_id}")
        if config.split_strategy == "clean_circuit_hash":
            split = split_by_clean_circuit.setdefault(
                clean_circuit_id,
                assign_split(clean_circuit_id, config),
            )
            split_group_id = clean_circuit_id
        else:
            assert config.holdout_axis is not None
            axis_value = sample_holdout_axis_value(
                sample,
                axis=config.holdout_axis,
                distortions_by_id=distortions_by_id,
            )
            observed_axis_values.add(axis_value)
            split = (
                "test"
                if axis_value in config.holdout_values
                else split_by_clean_circuit.setdefault(
                    clean_circuit_id,
                    assign_development_split(clean_circuit_id, config),
                )
            )
            # Distortion/backend OOD intentionally crosses clean-circuit identity
            # while keeping every axis-specific sample group in one split.
            split_group_id = (
                f"{clean_circuit_id}|{config.holdout_axis}={axis_value}"
                if config.holdout_axis in {"distortion_type", "backend_id"}
                else clean_circuit_id
            )
        sample_splits[sample_id] = split
        sample_groups[sample_id] = split_group_id
    if not sample_splits:
        raise ValueError("Phase 12 requires at least one Phase 7 sample")
    if config.split_strategy == "axis_holdout":
        missing = set(config.holdout_values) - observed_axis_values
        if missing:
            raise ValueError(
                f"Configured holdout values are absent from Phase 7: {sorted(missing)}"
            )
        observed_splits = set(sample_splits.values())
        if "test" not in observed_splits or not observed_splits.intersection(
            {"train", "validation"}
        ):
            raise ValueError(
                "axis_holdout requires both held-out test and development samples"
            )
    return sample_splits, sample_groups


def topology_group_split(
    group: Any,
    *,
    sample_splits: dict[str, str],
    action_to_sample: dict[str, str],
    config: TrainingViewConfig,
) -> tuple[str, tuple[str, ...]]:
    """Keep topology cohorts crossing source splits strictly audit-only."""
    if group.group_kind == "action_neighborhood":
        sample_id = group.metadata.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in sample_splits:
            raise ValueError(
                f"Topology action group {group.topology_group_id} has invalid sample_id"
            )
        splits = (sample_splits[sample_id],)
    else:
        point_ids = [str(value) for value in group.point_ids.tolist()]
        source_splits: set[str] = set()
        for point_id in point_ids:
            sample_id = point_id
            if sample_id not in sample_splits:
                sample_id = action_to_sample.get(point_id, "")
            if sample_id not in sample_splits:
                raise ValueError(
                    f"Topology group {group.topology_group_id} point {point_id} "
                    "cannot be mapped to a Phase 7 sample"
                )
            source_splits.add(sample_splits[sample_id])
        splits = tuple(sorted(source_splits))
    if len(splits) == 1:
        return splits[0], splits
    if config.topology_cross_split_policy != "audit_only":
        raise ValueError("Unsupported topology cross-split policy")
    return "audit_only", splits


__all__ = [
    "assign_development_split",
    "assign_split",
    "build_sample_split_maps",
    "sample_holdout_axis_value",
    "topology_group_split",
]
