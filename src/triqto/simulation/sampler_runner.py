"""Convenience runner for ideal in-memory sampling."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .ideal_shot import simulate_ideal_shots
from .results import IdealShotResult


def run_ideal_sampler(
    circuit_or_generated: Any,
    shots: int = 1024,
    seed: int | None = None,
    parameter_values: Mapping[str, float] | Mapping[Any, float] | None = None,
) -> IdealShotResult:
    """Run the Phase 4 ideal sampler; no hardware or Runtime backend is used."""
    return simulate_ideal_shots(
        circuit_or_generated,
        shots=shots,
        seed=seed,
        parameter_values=parameter_values,
    )
