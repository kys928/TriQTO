"""Deterministic clean-circuit grouped splits for Phase 12."""
from __future__ import annotations

import hashlib
from typing import Any

from triqto.core.ids import canonical_json

from .config import TrainingViewConfig


def assign_split(split_group_id: str, config: TrainingViewConfig) -> str:
    """Hash one clean-circuit group into train/validation/test without row leakage."""
    if not isinstance(split_group_id, str) or not split_group_id.strip():
        raise ValueError("split_group_id must be nonblank text")
    if not isinstance(config, TrainingViewConfig):
        raise TypeError("config must be TrainingViewConfig")
    digest = hashlib.sha256(
        canonical_json(
            {
                "split_group_id": split_group_id,
                "split_seed": config.split_seed,
                "split_version": "triqto.clean_circuit_hash_split.v1",
            }
        ).encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if unit < config.train_fraction:
        return "train"
    if unit < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def build_sample_split_maps(
    phase7: Any,
    config: TrainingViewConfig,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return sample->split and sample->clean-circuit split-group maps."""
    sample_splits: dict[str, str] = {}
    sample_groups: dict[str, str] = {}
    split_by_clean_circuit: dict[str, str] = {}
    for sample in sorted(phase7.samples, key=lambda item: item.sample_id):
        sample_id = sample.sample_id
        clean_circuit_id = sample.clean_circuit_id
        if sample_id in sample_splits:
            raise ValueError(f"Duplicate Phase 7 sample ID {sample_id}")
        split = split_by_clean_circuit.setdefault(
            clean_circuit_id,
            assign_split(clean_circuit_id, config),
        )
        sample_splits[sample_id] = split
        sample_groups[sample_id] = clean_circuit_id
    if not sample_splits:
        raise ValueError("Phase 12 requires at least one Phase 7 sample")
    return sample_splits, sample_groups


def topology_group_split(
    group: Any,
    *,
    sample_splits: dict[str, str],
    action_to_sample: dict[str, str],
    config: TrainingViewConfig,
) -> tuple[str, tuple[str, ...]]:
    """Assign topology groups only when every source point belongs to one split.

    Cohorts spanning clean-circuit splits are retained for audit but cannot enter a
    train/validation/test partition without leaking the same source samples across views.
    """
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


__all__ = ["assign_split", "build_sample_split_maps", "topology_group_split"]
