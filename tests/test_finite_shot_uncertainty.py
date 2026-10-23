"""Unit tests for observed-count uncertainty mechanics; no diagnostic cohort."""
import numpy as np
import pytest

from finite_shot_uncertainty import (
    observed_mean_covariance,
    finite_difference_jacobian_covariance,
    observed_count_bootstrap,
)


def toy_features():
    f = np.array([[1., 1., 1.],
                  [1., -1., -1.],
                  [-1., 1., -1.],
                  [-1., -1., 1.]])
    return np.stack([f, f[:, [1, 0, 2]], f[:, [2, 1, 0]]])


def toy_counts():
    base = np.array([12, 9, 7, 8], dtype=np.int64)
    counts = np.tile(base, (3, 2, 3, 1))
    for a in range(3):
        counts[a, 0, :, a % 4] += 4
        counts[a, 1, :, (a + 1) % 4] += 4
    return counts


def test_binomial_mean_and_sampling_variance():
    counts = [50, 50]
    obs = np.array([[-1.], [1.]])
    mean, covariance = observed_mean_covariance(counts, obs)
    np.testing.assert_allclose(mean, [0.], atol=1e-15)
    np.testing.assert_allclose(covariance, [[1./99.]], atol=1e-14)


def test_within_basis_statistic_covariance_retained():
    mean, covariance = observed_mean_covariance(
        [10, 10], np.array([[-1., 1.], [1., -1.]])
    )
    assert covariance[0, 1] < 0
    np.testing.assert_allclose(covariance[0, 0], -covariance[0, 1])
    np.testing.assert_allclose(covariance, covariance.T)


def test_plus_minus_shot_covariance_and_block_structure():
    counts = np.full((3, 2, 3, 2), 50, dtype=np.int64)
    observables = np.tile([[-1.], [1.]], (3, 1, 1))
    jac, covariance = finite_difference_jacobian_covariance(
        counts, observables, probe_strength=0.5
    )
    assert jac.shape == (3, 3)
    assert covariance.shape == (3, 3, 3)
    np.testing.assert_allclose(jac, 0)
    np.testing.assert_allclose(covariance[0].diagonal(), 2./99.)
    np.testing.assert_allclose(covariance[0] - np.diag(covariance[0].diagonal()), 0)


def test_joint_bootstrap_reproducible_psd_and_no_privileged_probabilities():
    counts = toy_counts()
    outcomes = toy_features()
    delta = np.linspace(-0.5, 0.5, 9)
    weights = np.linspace(0.8, 1.2, 9)
    args = (counts, outcomes, 0.1, delta, weights)
    a = observed_count_bootstrap(*args, replicates=40, seed=20261023)
    b = observed_count_bootstrap(*args, replicates=40, seed=20261023)
    np.testing.assert_allclose(a["bootstrap_covariance"], b["bootstrap_covariance"], atol=0)
    np.testing.assert_allclose(a["bootstrap_mean"], b["bootstrap_mean"], atol=0)
    assert a["bootstrap_covariance"].shape == (6, 6)
    assert np.min(np.linalg.eigvalsh(a["bootstrap_covariance"])) >= -1e-10
    assert np.max(np.abs(a["bootstrap_covariance"][:3, 3:])) > 0
    assert np.all(np.isfinite(a["observed_coords_matched"]))


@pytest.mark.parametrize("counts", ([0, 1], [-1, 3], [0.5, 1.5]))
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        observed_mean_covariance(counts, np.array([[-1.], [1.]]))


def test_shape_and_probe_guards():
    with pytest.raises(ValueError):
        finite_difference_jacobian_covariance(toy_counts(), toy_features(), probe_strength=0.)
    with pytest.raises(ValueError):
        finite_difference_jacobian_covariance(toy_counts()[:, 0], toy_features(), probe_strength=0.1)
