"""Finite-difference QGT for normalized pure-state parameter grids."""
from __future__ import annotations

from typing import Any

import numpy as np

from .hilbert import _state_vector


def pure_state_qgt(state_derivatives: Any, state: Any) -> np.ndarray:
    psi = _state_vector(state)
    derivs = np.asarray(state_derivatives, dtype=np.complex128)
    if derivs.ndim != 2 or derivs.shape[1] != psi.shape[0]:
        raise ValueError("state_derivatives must have shape [parameter, dimension]")
    projected = []
    for deriv in derivs:
        if not np.all(np.isfinite(deriv.real)) or not np.all(np.isfinite(deriv.imag)):
            raise ValueError("state derivative contains non-finite entries")
        projected.append(deriv - psi * np.vdot(psi, deriv))
    proj = np.stack(projected, axis=0)
    return proj.conj() @ proj.T


def qgt_from_state_function(state_function, parameters: Any, *, epsilon: float = 1e-6) -> np.ndarray:
    params = np.asarray(parameters, dtype=float).reshape(-1)
    if params.size == 0:
        raise ValueError("parameters must be non-empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    base = _state_vector(state_function(params))
    derivs = []
    for index in range(params.size):
        step = np.zeros_like(params)
        step[index] = epsilon
        plus = _state_vector(state_function(params + step))
        minus = _state_vector(state_function(params - step))
        derivs.append((plus - minus) / (2.0 * epsilon))
    return pure_state_qgt(np.stack(derivs, axis=0), base)


def describe_contract() -> str:
    return "Finite-difference pure-state QGT for normalized state functions; unsupported domains raise errors."


__all__ = ["pure_state_qgt", "qgt_from_state_function"]
