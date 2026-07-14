from __future__ import annotations

import math

import numpy as np
import pytest

from triqto.metrics.hilbert import bures_distance, density_matrix_fidelity, fubini_study_distance, pure_state_fidelity, purity, trace_distance
from triqto.metrics.qfi import qfi_from_state_function
from triqto.metrics.qgt import qgt_from_state_function


def test_pure_state_fidelity_and_fubini_study_reference_values() -> None:
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    phased = zero * np.exp(1j * 0.7)
    assert pure_state_fidelity(zero, phased) == pytest.approx(1.0)
    assert pure_state_fidelity(zero, one) == pytest.approx(0.0)
    assert fubini_study_distance(zero, one) == pytest.approx(math.pi / 2)


def test_density_metrics_reference_values() -> None:
    rho0 = np.diag([1.0, 0.0]).astype(complex)
    rho1 = np.diag([0.0, 1.0]).astype(complex)
    mixed = np.diag([0.5, 0.5]).astype(complex)
    assert density_matrix_fidelity(rho0, rho0) == pytest.approx(1.0)
    assert density_matrix_fidelity(rho0, rho1) == pytest.approx(0.0)
    assert trace_distance(rho0, rho1) == pytest.approx(1.0)
    assert purity(rho0) == pytest.approx(1.0)
    assert purity(mixed) == pytest.approx(0.5)
    assert bures_distance(rho0, rho0) == pytest.approx(0.0)


def test_density_validation_rejects_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        purity(np.array([[1, 1], [0, 0]], dtype=complex))
    with pytest.raises(ValueError, match="trace"):
        purity(np.eye(2, dtype=complex))
    with pytest.raises(ValueError, match="positive"):
        purity(np.diag([1.2, -0.2]).astype(complex))


def test_qgt_and_qfi_single_ry_parameter() -> None:
    def state(theta):
        t = float(theta[0])
        return np.array([np.cos(t / 2), np.sin(t / 2)], dtype=complex)

    qgt = qgt_from_state_function(state, [0.3])
    qfi = qfi_from_state_function(state, [0.3])
    assert qgt.shape == (1, 1)
    assert np.real(qgt[0, 0]) == pytest.approx(0.25, rel=1e-4)
    assert qfi[0, 0] == pytest.approx(1.0, rel=1e-4)
