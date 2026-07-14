"""Public Hilbert-space and density-matrix metrics with explicit validation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _state_vector(state: Any) -> np.ndarray:
    array = np.asarray(state, dtype=np.complex128).reshape(-1)
    if array.size == 0:
        raise ValueError("state vector must be non-empty")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("state vector contains non-finite entries")
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise ValueError("state vector norm must be positive")
    return array / norm


def pure_state_fidelity(state_a: Any, state_b: Any) -> float:
    a = _state_vector(state_a)
    b = _state_vector(state_b)
    if a.shape != b.shape:
        raise ValueError("state vectors must have equal dimension")
    return float(min(1.0, max(0.0, abs(np.vdot(a, b)) ** 2)))


def fidelity(state_a: Any, state_b: Any) -> float:
    """Backward-compatible alias for pure-state fidelity."""
    return pure_state_fidelity(state_a, state_b)


def fubini_study_distance(state_a: Any, state_b: Any) -> float:
    fid = pure_state_fidelity(state_a, state_b)
    overlap = math.sqrt(max(0.0, min(1.0, fid)))
    return float(math.acos(overlap))


def _density_matrix(rho: Any, *, atol: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("density matrix must be non-empty and square")
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError("density matrix contains non-finite entries")
    if not np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0):
        raise ValueError("density matrix must be Hermitian")
    tr = np.trace(matrix)
    if not np.isclose(tr, 1.0, atol=atol, rtol=0):
        raise ValueError("density matrix trace must be one")
    evals = np.linalg.eigvalsh(matrix)
    if float(evals.min()) < -atol:
        raise ValueError("density matrix must be positive semidefinite")
    return matrix


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    evals, vecs = np.linalg.eigh(matrix)
    clipped = np.clip(evals, 0.0, None)
    return (vecs * np.sqrt(clipped)) @ vecs.conj().T


def density_matrix_fidelity(rho: Any, sigma: Any) -> float:
    r = _density_matrix(rho)
    s = _density_matrix(sigma)
    if r.shape != s.shape:
        raise ValueError("density matrices must have equal dimension")
    root = _sqrt_psd(r)
    middle = root @ s @ root
    value = float(np.real(np.trace(_sqrt_psd((middle + middle.conj().T) / 2.0))) ** 2)
    return min(1.0, max(0.0, value))


def trace_distance(rho: Any, sigma: Any) -> float:
    r = _density_matrix(rho)
    s = _density_matrix(sigma)
    if r.shape != s.shape:
        raise ValueError("density matrices must have equal dimension")
    delta = (r - s + (r - s).conj().T) / 2.0
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta))))


def purity(rho: Any) -> float:
    r = _density_matrix(rho)
    return float(np.real(np.trace(r @ r)))


def bures_distance(rho: Any, sigma: Any) -> float:
    fid = density_matrix_fidelity(rho, sigma)
    return float(math.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(fid)))))


__all__ = ["pure_state_fidelity", "fidelity", "fubini_study_distance", "density_matrix_fidelity", "trace_distance", "purity", "bures_distance"]
