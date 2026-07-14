"""Checkpoint-bound latent-space topology diagnostics.

This module computes diagnostic artifacts only. It never enables a topology
gradient and it never accepts a blank checkpoint identity.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from triqto.core.ids import canonical_json, make_deterministic_id
from .config import TopologyAuditConfig
from .features import diagram_statistics
from .persistent_homology import betti_curve, compute_persistence_diagrams


@dataclass(frozen=True, slots=True)
class LatentTopologyConfig:
    homology_dimensions: tuple[int, ...] = (0, 1)
    max_filtration: float | None = None
    normalized_shape_diagnostic: bool = False
    betti_grid_size: int = 8
    top_k_lifetimes: int = 5

    def __post_init__(self) -> None:
        dims = tuple(int(value) for value in self.homology_dimensions)
        if tuple(sorted(set(dims))) != dims or 0 not in dims or 1 not in dims or any(value > 2 for value in dims):
            raise ValueError("latent topology requires sorted unique H0/H1 and optional H2")
        if self.max_filtration is not None and float(self.max_filtration) <= 0.0:
            raise ValueError("max_filtration must be positive when supplied")
        if not isinstance(self.normalized_shape_diagnostic, bool):
            raise TypeError("normalized_shape_diagnostic must be bool")
        if self.betti_grid_size < 2 or self.top_k_lifetimes < 1:
            raise ValueError("invalid Betti-grid or top-k setting")
        object.__setattr__(self, "homology_dimensions", dims)
        object.__setattr__(self, "max_filtration", None if self.max_filtration is None else float(self.max_filtration))


def load_latent_topology_config(path: str | Path) -> LatentTopologyConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("latent topology config must contain mapping")
    allowed = set(LatentTopologyConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    if set(payload) - allowed:
        raise ValueError(f"unknown latent topology fields: {sorted(set(payload) - allowed)}")
    data = dict(payload)
    if "homology_dimensions" in data:
        data["homology_dimensions"] = tuple(data["homology_dimensions"])
    return LatentTopologyConfig(**data)


def _validate(checkpoint_id: str, split: str, head: str, point_ids: Sequence[str], coordinates: np.ndarray) -> tuple[str, ...]:
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("latent topology requires a real nonblank checkpoint_id")
    if not split.strip() or not head.strip():
        raise ValueError("split and head must be nonblank")
    ids = tuple(str(value) for value in point_ids)
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("point_ids must be unique nonblank strings")
    if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2 or coordinates.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("coordinates must be numeric 2D NumPy array")
    if coordinates.shape[0] != len(ids) or coordinates.shape[0] < 2 or coordinates.shape[1] < 1 or not np.isfinite(coordinates).all():
        raise ValueError("invalid coordinate shape or values")
    return ids


def _distance_matrix(coordinates: np.ndarray, *, normalized_shape: bool) -> tuple[np.ndarray, dict[str, Any]]:
    coords = np.asarray(coordinates, dtype=np.float64)
    scale = 1.0
    mode, normalization = "scale_preserving", "absolute_scale"
    if normalized_shape:
        centered = coords - coords.mean(axis=0, keepdims=True)
        scale = float(np.max(np.linalg.norm(centered, axis=1)))
        if scale <= 0.0:
            raise ValueError("shape-only topology requires non-identical coordinates")
        coords = centered / scale
        mode, normalization = "normalized_shape_only", "shape_only"
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float64), {
        "distance_mode": mode,
        "normalization_mode": normalization,
        "shape_normalization_scale_factor": scale,
    }


def _summary(diagrams: dict[int, np.ndarray], config: TopologyAuditConfig, point_count: int) -> dict[str, Any]:
    grid = np.linspace(0.0, config.max_filtration, config.betti_grid_size, dtype=np.float64)
    per_dimension: dict[str, Any] = {}
    for dimension in config.homology_dimensions:
        stats = diagram_statistics(diagrams[dimension], config.top_k_lifetimes)
        per_dimension[f"h{dimension}"] = {
            **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in stats.items()},
            "betti_curve": betti_curve(diagrams[dimension], grid).tolist(),
        }
    h0, h1 = per_dimension.get("h0", {}), per_dimension.get("h1", {})
    return {
        "point_count": point_count,
        "filtration_grid": grid.tolist(),
        "per_dimension": per_dimension,
        "collapse_score": float(1.0 - min(1.0, max(0.0, float(h0.get("mean_lifetime", 0.0)) / config.max_filtration))),
        "loop_score": float(1.0 - np.exp(-float(h1.get("total_persistence", 0.0)) / max(1, point_count))),
        "late_merge_bridge_score": float(min(1.0, max(0.0, float(h0.get("max_lifetime", 0.0)) / config.max_filtration))),
    }


def compute_latent_topology(*, checkpoint_id: str, split: str, head: str, point_ids: Sequence[str], coordinates: np.ndarray, config: LatentTopologyConfig | None = None, checkpoint_content_hash: str | None = None, model_architecture_id: str | None = None, model_config_id: str | None = None, training_view_dataset_id: str | None = None, latent_extraction_id: str | None = None, representation: str = "head_latent", evidence_tier: str = "trained_checkpoint") -> dict[str, Any]:
    cfg = config or LatentTopologyConfig()
    ids = _validate(checkpoint_id, split, head, point_ids, coordinates)
    for name, value in (("checkpoint_content_hash", checkpoint_content_hash), ("model_architecture_id", model_architecture_id), ("model_config_id", model_config_id), ("training_view_dataset_id", training_view_dataset_id), ("latent_extraction_id", latent_extraction_id)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be nonblank when supplied")
    distances, distance_metadata = _distance_matrix(coordinates, normalized_shape=cfg.normalized_shape_diagnostic)
    max_distance = float(np.max(distances))
    if max_distance <= 0.0:
        raise ValueError("latent topology requires distinct coordinates")
    audit_config = TopologyAuditConfig(
        min_points=2,
        homology_dimensions=cfg.homology_dimensions,
        max_filtration=max_distance if cfg.max_filtration is None else cfg.max_filtration,
        betti_grid_size=cfg.betti_grid_size,
        top_k_lifetimes=cfg.top_k_lifetimes,
        normalize_distance_matrices=False,
    )
    diagrams, engine_metadata = compute_persistence_diagrams(distances, audit_config)
    coordinate_hash = make_deterministic_id("latentcoords", {"point_ids": list(ids), "coordinates": np.asarray(coordinates, dtype=np.float64).round(15).tolist()})
    payload = {
        "schema": "triqto.latent_topology.v2",
        "checkpoint_id": checkpoint_id.strip(),
        "checkpoint_content_hash": checkpoint_content_hash,
        "model_architecture_id": model_architecture_id,
        "model_config_id": model_config_id,
        "training_view_dataset_id": training_view_dataset_id,
        "latent_extraction_id": latent_extraction_id,
        "split": split.strip(),
        "head": head.strip(),
        "representation": representation.strip(),
        "point_ids": list(ids),
        "coordinate_dim": int(coordinates.shape[1]),
        "coordinate_hash": coordinate_hash,
        "evidence_tier": evidence_tier.strip(),
        "config": {
            "homology_dimensions": list(cfg.homology_dimensions),
            "max_filtration": cfg.max_filtration,
            "normalized_shape_diagnostic": cfg.normalized_shape_diagnostic,
            "betti_grid_size": cfg.betti_grid_size,
            "top_k_lifetimes": cfg.top_k_lifetimes,
        },
        **distance_metadata,
    }
    result_id = make_deterministic_id("latent_topology", payload)
    return {
        "latent_topology_id": result_id,
        "checkpoint_bound": True,
        "trained_checkpoint": True,
        "diagnostic_only": True,
        "physical_hardware": False,
        "topology_loss_weight": 0.0,
        "claim_scope": "checkpoint-bound latent diagnostic; no causal, hardware-transfer, optimization, or topology-benefit claim",
        "payload_hash": make_deterministic_id("latent_payload", {"payload": canonical_json(payload)}),
        "metadata": payload,
        "engine_metadata": engine_metadata,
        "persistence_summary": _summary(diagrams, audit_config, len(ids)),
        "persistence_diagrams": {dimension: diagram.tolist() for dimension, diagram in diagrams.items()},
    }


__all__ = ["LatentTopologyConfig", "compute_latent_topology", "load_latent_topology_config"]
