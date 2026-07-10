"""Strict validation for Phase 8 Born probabilities, counts, and metrics."""
from __future__ import annotations

from collections.abc import Mapping
import math
import numbers
from typing import Any

import numpy as np

from .constants import PROBABILITY_ATOL


def _strict_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be an int or float and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_bitstring(value: Any, n_qubits: int, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != n_qubits or any(character not in "01" for character in value):
        raise ValueError(
            f"{name} must be a binary string of width {n_qubits}: {value!r}"
        )
    return value


def validate_probability_mapping(
    probabilities: Mapping[str, Any],
    n_qubits: int,
    *,
    max_outcomes: int | None = None,
    atol: float = PROBABILITY_ATOL,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(probabilities, Mapping):
        raise TypeError("exact probabilities must be a mapping")
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits <= 0:
        raise TypeError("n_qubits must be a positive integer and not bool")
    if max_outcomes is not None:
        if isinstance(max_outcomes, bool) or not isinstance(max_outcomes, int):
            raise TypeError("max_outcomes must be an integer and not bool")
        if max_outcomes <= 0:
            raise ValueError("max_outcomes must be positive")
        if len(probabilities) > max_outcomes:
            raise ValueError(
                f"exact probability outcome count {len(probabilities)} exceeds "
                f"max_probability_outcomes={max_outcomes}"
            )
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
    clipped_negative_count = 0
    for raw_key, raw_value in probabilities.items():
        key = _validate_bitstring(raw_key, n_qubits, "probability outcome")
        if key in seen:
            raise ValueError(f"duplicate probability outcome: {key}")
        seen.add(key)
        value = _strict_real(raw_value, f"probability[{key}]")
        if value < -atol:
            raise ValueError(f"probability[{key}] is meaningfully negative: {value}")
        if value < 0.0:
            value = 0.0
            clipped_negative_count += 1
        rows.append((key, value))
    rows.sort(key=lambda item: item[0])
    total = math.fsum(value for _, value in rows)
    if not math.isfinite(total) or abs(total - 1.0) > atol:
        raise ValueError(
            f"exact probabilities must sum to one within atol={atol}; got {total}"
        )
    width = max(1, n_qubits)
    return (
        np.asarray([key for key, _ in rows], dtype=f"<U{width}"),
        np.asarray([value for _, value in rows], dtype=np.float64),
        clipped_negative_count,
    )


def validate_probability_arrays(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    n_qubits: int,
    *,
    atol: float = PROBABILITY_ATOL,
) -> None:
    if not isinstance(bitstrings, np.ndarray) or bitstrings.ndim != 1:
        raise ValueError("outcome_bitstrings must be a one-dimensional ndarray")
    if bitstrings.dtype.kind != "U":
        raise TypeError("outcome_bitstrings must use fixed-width Unicode dtype")
    if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
        raise TypeError("exact_probabilities must use float64 dtype")
    if probabilities.ndim != 1 or len(probabilities) != len(bitstrings):
        raise ValueError("exact probability arrays must have equal one-dimensional length")
    keys = [str(value) for value in bitstrings.tolist()]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate exact probability outcome bitstrings")
    if keys != sorted(keys):
        raise ValueError("exact probability outcome bitstrings must be sorted")
    values: list[float] = []
    for index, key in enumerate(keys):
        _validate_bitstring(key, n_qubits, f"outcome_bitstrings[{index}]")
        value = float(probabilities[index])
        if not math.isfinite(value):
            raise ValueError("exact_probabilities contains non-finite value")
        if value < 0.0:
            raise ValueError("persisted exact probabilities must be nonnegative")
        values.append(value)
    total = math.fsum(values)
    if abs(total - 1.0) > atol:
        raise ValueError(
            f"exact probabilities must sum to one within atol={atol}; got {total}"
        )


def validate_count_mapping(
    counts: Mapping[str, Any],
    n_qubits: int,
    shots: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(counts, Mapping):
        raise TypeError("supplemental counts must be a mapping")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise TypeError("shots must be a positive integer and not bool")
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw_key, raw_value in counts.items():
        key = _validate_bitstring(raw_key, n_qubits, "count outcome")
        if key in seen:
            raise ValueError(f"duplicate count outcome: {key}")
        seen.add(key)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(f"count[{key}] must be an integer and not bool")
        if raw_value < 0:
            raise ValueError(f"count[{key}] must be nonnegative")
        rows.append((key, raw_value))
    rows.sort(key=lambda item: item[0])
    total = sum(value for _, value in rows)
    if total != shots:
        raise ValueError(f"supplemental counts total {total} does not equal shots {shots}")
    width = max(1, n_qubits)
    return (
        np.asarray([key for key, _ in rows], dtype=f"<U{width}"),
        np.asarray([value for _, value in rows], dtype=np.int64),
    )


def validate_count_arrays(
    bitstrings: np.ndarray,
    counts: np.ndarray,
    n_qubits: int,
    shots: int,
) -> None:
    if not isinstance(bitstrings, np.ndarray) or bitstrings.ndim != 1:
        raise ValueError("count_outcome_bitstrings must be one-dimensional")
    if bitstrings.dtype.kind != "U":
        raise TypeError("count_outcome_bitstrings must use fixed-width Unicode dtype")
    if not isinstance(counts, np.ndarray) or counts.dtype != np.int64:
        raise TypeError("supplemental_counts must use int64 dtype")
    if counts.ndim != 1 or len(counts) != len(bitstrings):
        raise ValueError("supplemental count arrays must have equal length")
    keys = [str(value) for value in bitstrings.tolist()]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate supplemental count outcomes")
    if keys != sorted(keys):
        raise ValueError("supplemental count outcomes must be sorted")
    for index, key in enumerate(keys):
        _validate_bitstring(key, n_qubits, f"count_outcome_bitstrings[{index}]")
    if np.any(counts < 0):
        raise ValueError("supplemental counts must be nonnegative")
    if int(counts.sum()) != shots:
        raise ValueError("supplemental counts do not sum to supplemental_shots")


def decode_born_metric_arrays(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("born_metrics must be a nonempty mapping")
    marker_suffix = "__nonfinite"
    marker_bases: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise TypeError("born metric names must be nonblank strings")
        if key.endswith(marker_suffix):
            base = key[: -len(marker_suffix)]
            if not base:
                raise ValueError("born metric nonfinite marker has empty base name")
            marker_bases[base] = value
    for base in marker_bases:
        if base not in payload:
            raise ValueError(f"orphan nonfinite marker for metric {base}")
    names: list[str] = []
    values: list[float] = []
    masks: list[bool] = []
    for name in sorted(key for key in payload if not key.endswith(marker_suffix)):
        value = payload[name]
        marker = marker_bases.get(name)
        if value is None:
            if marker != "positive_infinity":
                if marker is None:
                    raise ValueError(f"metric {name} is null without marker")
                raise ValueError(f"metric {name} has unknown marker {marker!r}")
            names.append(name)
            values.append(0.0)
            masks.append(True)
            continue
        if marker is not None:
            raise ValueError(f"metric {name} has finite value and nonfinite marker")
        numeric = _strict_real(value, f"metric {name}")
        names.append(name)
        values.append(numeric)
        masks.append(False)
    width = max([1, *[len(name) for name in names]])
    return (
        np.asarray(names, dtype=f"<U{width}"),
        np.asarray(values, dtype=np.float64),
        np.asarray(masks, dtype=np.bool_),
    )


def validate_born_metric_arrays(
    names: np.ndarray,
    values: np.ndarray,
    positive_infinity_mask: np.ndarray,
) -> None:
    if names.ndim != 1 or names.dtype.kind != "U":
        raise TypeError("born_metric_names must be one-dimensional fixed-width Unicode")
    if values.dtype != np.float64 or values.ndim != 1:
        raise TypeError("born_metric_values must be one-dimensional float64")
    if positive_infinity_mask.dtype != np.bool_ or positive_infinity_mask.ndim != 1:
        raise TypeError("born_metric_positive_infinity_mask must be one-dimensional bool")
    if not (len(names) == len(values) == len(positive_infinity_mask)):
        raise ValueError("Born metric arrays must have equal lengths")
    normalized_names = [str(value) for value in names.tolist()]
    if any(not name for name in normalized_names):
        raise ValueError("Born metric names must be nonblank")
    if normalized_names != sorted(normalized_names):
        raise ValueError("Born metric names must be sorted")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("Born metric names must be unique")
    if not np.all(np.isfinite(values)):
        raise ValueError("Born metric value placeholders must be finite")
    if np.any(positive_infinity_mask & (values != 0.0)):
        raise ValueError("positive-infinity metric placeholders must equal zero")


__all__ = [
    "decode_born_metric_arrays",
    "validate_born_metric_arrays",
    "validate_count_arrays",
    "validate_count_mapping",
    "validate_probability_arrays",
    "validate_probability_mapping",
]
