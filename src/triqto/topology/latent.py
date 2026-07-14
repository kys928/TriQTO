"""Checkpoint-bound latent-space topology diagnostics.

This module deliberately computes diagnostic artifacts only. It requires an explicit
checkpoint identity and never fabricates trained-representation evidence.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from triqto.core.ids import canonical_json, make_deterministic_id

from .config import TopologyAuditConfig
from .persistent_homology import compute_persistence_diagrams


@dataclass(frozen=True, slots=True)
class LatentTopologyConfig:
    homology_dimensions: tuple[int, ...] = (0, 1)
    max_filtration: float | None = None
    normalized_shape_diagnostic: bool = False

    def __post_init__(self) -> None:
        dims = tuple(int(value) for value in self.homology_dimensions)
        if not dims or any(value < 0 for value in dims):
            raise ValueError("homology_dimensions must contain nonnegative integers")
        if self.max_filtration is not None and float(self.max_filtration) <= 0.0:
            raise ValueError("max_filtration must be positive when supplied")
        if not isinstance(self.normalized_shape_diagnostic, bool):
            raise TypeError("normalized_shape_diagnostic must be bool")
        object.__setattr__(self, "homology_dimensions", dims)
        object.__setattr__(self, "max_filtration", None if self.max_filtration is None else float(self.max_filtration))


def _validate_inputs(checkpoint_id: str, split: str, head: str, point_ids: Sequence[str], coordinates: np.ndarray) -> tuple[str, ...]:
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("latent topology requires a real nonblank checkpoint_id")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be nonblank")
    if not isinstance(head, str) or not head.strip():
        raise ValueError("head must be nonblank")
    ids = tuple(str(value) for value in point_ids)
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("point_ids must be unique nonblank strings")
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("coordinates must be a NumPy array")
    if coordinates.dtype.kind not in {"f", "i", "u"} or coordinates.ndim != 2:
        raise TypeError("coordinates must be a numeric 2D NumPy array")
    if coordinates.shape[0] != len(ids) or coordinates.shape[0] < 2 or coordinates.shape[1] < 1:
        raise ValueError("coordinate shape must match point_ids and contain at least two points")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    return ids


def _distance_matrix(coordinates: np.ndarray, *, normalized_shape: bool) -> tuple[np.ndarray, dict[str, Any]]:
    coords = np.asarray(coordinates, dtype=np.float64)
    scale_factor = 1.0
    mode = "scale_preserving"
    if normalized_shape:
        max_norm = float(np.max(np.linalg.norm(coords - coords.mean(axis=0, keepdims=True), axis=1)))
        if max_norm <= 0.0:
            raise ValueError("normalized-shape latent topology requires non-identical coordinates")
        coords = coords / max_norm
        scale_factor = max_norm
        mode = "normalized_shape_only"
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diff, axis=2).astype(np.float64)
    return distances, {"distance_mode": mode, "shape_normalization_scale_factor": scale_factor}


def compute_latent_topology(
    *,
    checkpoint_id: str,
    split: str,
    head: str,
    point_ids: Sequence[str],
    coordinates: np.ndarray,
    config: LatentTopologyConfig | None = None,
) -> dict[str, Any]:
    """Compute checkpoint-bound latent persistent homology diagnostics."""
    cfg = config or LatentTopologyConfig()
    ids = _validate_inputs(checkpoint_id, split, head, point_ids, coordinates)
    distances, distance_metadata = _distance_matrix(
        coordinates,
        normalized_shape=cfg.normalized_shape_diagnostic,
    )
    max_distance = float(np.max(distances))
    audit_config = TopologyAuditConfig(
        min_points=2,
        homology_dimensions=cfg.homology_dimensions,
        max_filtration=max_distance if cfg.max_filtration is None else cfg.max_filtration,
        betti_grid_size=8,
    )
    diagrams, engine_metadata = compute_persistence_diagrams(distances, audit_config)
    coordinate_hash = make_deterministic_id(
        "latentcoords",
        {
            "point_ids": list(ids),
            "coordinates": np.asarray(coordinates, dtype=np.float64).round(15).tolist(),
        },
    )
    payload = {
        "schema": "triqto.latent_topology.v1",
        "checkpoint_id": checkpoint_id.strip(),
        "split": split.strip(),
        "head": head.strip(),
        "point_ids": list(ids),
        "coordinate_hash": coordinate_hash,
        "config": {
            "homology_dimensions": list(cfg.homology_dimensions),
            "max_filtration": cfg.max_filtration,
            "normalized_shape_diagnostic": cfg.normalized_shape_diagnostic,
        },
        **distance_metadata,
    }
    result_id = make_deterministic_id("latent_topology", payload)
    return {
        "latent_topology_id": result_id,
        "diagnostic_only": True,
        "topology_loss_weight": 0.0,
        "claim_scope": "checkpoint-bound latent-space diagnostic; no causal or optimization claim",
        "payload_hash": make_deterministic_id("latent_payload", {"payload": canonical_json(payload)}),
        "metadata": payload,
        "engine_metadata": engine_metadata,
        "persistence_diagrams": {dimension: diagram.tolist() for dimension, diagram in diagrams.items()},
    }


__all__ = ["LatentTopologyConfig", "compute_latent_topology"]
