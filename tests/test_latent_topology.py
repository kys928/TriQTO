from __future__ import annotations

import numpy as np
import pytest

from triqto.topology import LatentTopologyConfig, compute_latent_topology


def test_latent_topology_is_checkpoint_bound_and_deterministic() -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    first = compute_latent_topology(
        checkpoint_id="ckpt_sha256_abc",
        split="validation",
        head="diagnosis",
        point_ids=("p1", "p2", "p3"),
        coordinates=coords,
    )
    second = compute_latent_topology(
        checkpoint_id="ckpt_sha256_abc",
        split="validation",
        head="diagnosis",
        point_ids=("p1", "p2", "p3"),
        coordinates=coords,
    )
    assert first["latent_topology_id"] == second["latent_topology_id"]
    assert first["diagnostic_only"] is True
    assert first["topology_loss_weight"] == 0.0
    assert first["metadata"]["distance_mode"] == "scale_preserving"
    assert 0 in first["persistence_diagrams"]


def test_latent_topology_normalized_shape_changes_identity_and_labels_shape_only() -> None:
    coords = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float64)
    absolute = compute_latent_topology(
        checkpoint_id="ckpt_sha256_abc",
        split="train",
        head="born",
        point_ids=("a", "b", "c"),
        coordinates=coords,
    )
    shape = compute_latent_topology(
        checkpoint_id="ckpt_sha256_abc",
        split="train",
        head="born",
        point_ids=("a", "b", "c"),
        coordinates=coords,
        config=LatentTopologyConfig(normalized_shape_diagnostic=True),
    )
    assert absolute["latent_topology_id"] != shape["latent_topology_id"]
    assert shape["metadata"]["distance_mode"] == "normalized_shape_only"
    assert shape["metadata"]["shape_normalization_scale_factor"] == 2.0


def test_latent_topology_rejects_blank_checkpoint_and_duplicate_points() -> None:
    coords = np.asarray([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="checkpoint_id"):
        compute_latent_topology(checkpoint_id="", split="train", head="h", point_ids=("a", "b"), coordinates=coords)
    with pytest.raises(ValueError, match="unique"):
        compute_latent_topology(checkpoint_id="ckpt", split="train", head="h", point_ids=("a", "a"), coordinates=coords)
