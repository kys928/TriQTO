"""Probability-distribution helpers for TriQTO Born metrics."""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


def _coerce_probability_items(probabilities: Mapping[str, float], *, atol: float) -> dict[str, float]:
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping of bitstring to probability.")
    if not probabilities:
        raise ValueError("probability distribution must not be empty.")

    coerced: dict[str, float] = {}
    for key, raw_value in probabilities.items():
        bitstring = str(key)
        value = float(raw_value)
        if not isfinite(value):
            raise ValueError(f"Probability for {bitstring!r} must be finite.")
        if value < 0.0:
            if value >= -atol:
                value = 0.0
            else:
                raise ValueError(f"Probability for {bitstring!r} is negative: {value}.")
        coerced[bitstring] = coerced.get(bitstring, 0.0) + value
    return coerced


def normalize_probability_distribution(probabilities: Mapping[str, float], *, atol: float = 1e-12) -> dict[str, float]:
    """Return a stable, normalized copy of a bitstring probability distribution."""
    coerced = _coerce_probability_items(probabilities, atol=atol)
    total = sum(coerced.values())
    if total <= atol:
        raise ValueError("probability distribution must have positive total mass.")
    normalized = {key: value / total for key, value in coerced.items() if value > atol}
    if not normalized:
        raise ValueError("probability distribution must have positive total mass after clipping numerical noise.")
    return {key: normalized[key] for key in sorted(normalized)}


def validate_probability_distribution(probabilities: Mapping[str, float], *, atol: float = 1e-12) -> None:
    """Validate that a distribution can be normalized safely."""
    normalize_probability_distribution(probabilities, atol=atol)


def align_probability_distributions(
    p: Mapping[str, float],
    q: Mapping[str, float],
    *,
    atol: float = 1e-12,
) -> tuple[dict[str, float], dict[str, float]]:
    """Normalize and align two distributions over their sorted union support."""
    p_norm = normalize_probability_distribution(p, atol=atol)
    q_norm = normalize_probability_distribution(q, atol=atol)
    support = sorted(set(p_norm) | set(q_norm))
    return ({key: p_norm.get(key, 0.0) for key in support}, {key: q_norm.get(key, 0.0) for key in support})


def probabilities_from_input(obj: Any) -> dict[str, float]:
    """Extract normalized probabilities from a mapping or simulation-result-like object.

    This helper reads existing probability attributes only. It intentionally does
    not accept circuits, infer counts, run simulators, or apply distortions.
    """
    if isinstance(obj, Mapping):
        return normalize_probability_distribution(obj)

    probabilities = getattr(obj, "probabilities", None)
    if probabilities is not None:
        return normalize_probability_distribution(probabilities)

    source_probabilities = getattr(obj, "source_probabilities", None)
    if source_probabilities is not None:
        return normalize_probability_distribution(source_probabilities)

    raise TypeError("Expected a probability mapping or an object with .probabilities/.source_probabilities.")
