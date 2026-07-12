from __future__ import annotations

import math

import numpy as np
import pytest

from triqto.topology import (
    TopologyAuditConfig,
    born_distance_matrix,
    bottleneck_distance,
    build_persistence_summary,
    compute_persistence_diagrams,
    fubini_study_distance_matrix,
    induced_parameter_distance_matrix,
    make_filtration_grid,
    topology_config_from_dict,
    topology_config_to_dict,
    wasserstein_distance_1,
)


def test_topology_config_is_strict_and_keeps_h0_h1_active() -> None:
    config = TopologyAuditConfig(
        group_kinds=("action_neighborhood",),
        min_points=3,
    )
    assert topology_config_from_dict(topology_config_to_dict(config)) == config
    with pytest.raises(ValueError, match="Unknown topology config fields"):
        topology_config_from_dict(
            {**topology_config_to_dict(config), "unexpected": 1}
        )
    with pytest.raises(TypeError, match="min_points"):
        TopologyAuditConfig(min_points=True)
    with pytest.raises(ValueError, match="H0 and H1"):
        TopologyAuditConfig(homology_dimensions=(0,))
    with pytest.raises(ValueError, match="fixed Phase 11 order"):
        TopologyAuditConfig(
            group_kinds=("family_qubit_cohort", "action_neighborhood")
        )
    with pytest.raises(ValueError, match="Unsupported born_distance"):
        TopologyAuditConfig(born_distance="euclidean")


def test_fubini_study_is_global_phase_invariant() -> None:
    base = np.asarray([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0)
    phase_shifted = base * np.exp(1.234j)
    orthogonal = np.asarray([1.0, -1.0j], dtype=np.complex128) / math.sqrt(2.0)
    matrix = fubini_study_distance_matrix(
        np.stack([base, phase_shifted, orthogonal], axis=0)
    )
    assert matrix.dtype == np.float64
    assert matrix[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert matrix[0, 2] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(matrix, matrix.T)


def test_born_distance_metrics_are_bounded_and_zero_on_identity() -> None:
    probabilities = np.asarray(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        dtype=np.float64,
    )
    for metric in ("hellinger", "jensen_shannon", "fisher_rao"):
        matrix = born_distance_matrix(probabilities, metric)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(matrix, matrix.T)
        assert np.min(matrix) >= 0.0
        assert np.max(matrix) <= 1.0 + 1e-12
    with pytest.raises(ValueError, match="sum to one"):
        born_distance_matrix(
            np.asarray([[0.2, 0.2], [0.5, 0.5]], dtype=np.float64),
            "hellinger",
        )


def test_induced_parameter_distance_uses_downstream_deformation() -> None:
    raw = np.zeros((2, 2), dtype=np.float64)
    born = np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64)
    hilbert = np.asarray([[0.0, 0.6], [0.6, 0.0]], dtype=np.float64)
    config = TopologyAuditConfig(
        group_kinds=("action_neighborhood",),
        raw_parameter_weight=0.1,
        born_pullback_weight=1.0,
        hilbert_pullback_weight=1.0,
    )
    induced = induced_parameter_distance_matrix(raw, born, hilbert, config)
    expected = math.sqrt((0.8**2 + 0.6**2) / 2.1)
    assert induced[0, 1] == pytest.approx(expected)
    assert induced[0, 1] > 0.0


def test_vietoris_rips_detects_circle_h1_and_features_are_finite() -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    matrix = (matrix / np.max(matrix)).astype(np.float64)
    config = TopologyAuditConfig(
        group_kinds=("action_neighborhood",),
        min_points=3,
        betti_grid_size=24,
        top_k_lifetimes=4,
        max_filtration=1.0,
    )
    diagrams, metadata = compute_persistence_diagrams(matrix, config)
    assert metadata["engine"] == "ripser"
    assert diagrams[0].shape[1] == 2
    assert diagrams[1].shape[1] == 2
    assert diagrams[1].shape[0] >= 1
    grid = make_filtration_grid(config)
    summary = build_persistence_summary(
        manifold="born",
        diagrams=diagrams,
        filtration_grid=grid,
        point_count=12,
        config=config,
    )
    assert np.isfinite(summary.feature_values).all()
    feature_map = dict(
        zip(summary.feature_names.tolist(), summary.feature_values.tolist(), strict=True)
    )
    assert feature_map["h1_total_persistence"] > 0.0
    assert feature_map["loop_score"] > 0.0


def test_diagram_distances_are_zero_for_identical_diagrams() -> None:
    diagram = np.asarray([[0.0, 0.5], [0.2, 0.8]], dtype=np.float64)
    assert bottleneck_distance(diagram, diagram) == pytest.approx(0.0)
    assert wasserstein_distance_1(diagram, diagram) == pytest.approx(0.0)
    empty = np.empty((0, 2), dtype=np.float64)
    assert bottleneck_distance(empty, empty) == 0.0
    assert wasserstein_distance_1(empty, empty) == 0.0
