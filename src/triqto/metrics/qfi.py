"""Pure-state QFI derived from the real part of the QGT."""
from __future__ import annotations

from typing import Any

import numpy as np

from .qgt import pure_state_qgt, qgt_from_state_function


def pure_state_qfi(state_derivatives: Any, state: Any) -> np.ndarray:
    return 4.0 * np.real(pure_state_qgt(state_derivatives, state))


def qfi_from_state_function(state_function, parameters: Any, *, epsilon: float = 1e-6) -> np.ndarray:
    return 4.0 * np.real(qgt_from_state_function(state_function, parameters, epsilon=epsilon))


def describe_contract() -> str:
    return "Finite-difference pure-state QFI for normalized state functions; unsupported domains raise errors."


__all__ = ["pure_state_qfi", "qfi_from_state_function"]
