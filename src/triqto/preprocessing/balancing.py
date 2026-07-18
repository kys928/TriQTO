"""Training-only balancing weights without deleting majority-class samples."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any

from .config import BalancingConfig
from .records import ProcessedSample, SplitAssignment


def _dimension_value(sample: ProcessedSample, dimension: str) -> Any:
    if dimension == "family":
        return sample.family
    if dimension == "n_qubits":
        return sample.n_qubits
    if dimension == "distortion_class":
        return sample.intervention_label
    if dimension == "severity":
        return sample.severity
    if dimension == "measurement_basis":
        return sample.measurement_basis
    if dimension == "source_type":
        return sample.source_type
    if dimension == "hilbert_availability":
        return sample.masks.get("hilbert_available", False)
    if dimension == "calibration_period":
        return sample.hardware_context.get("calibration_window_id")
    if dimension == "action_type":
        return sample.provenance.get("action_type")
    if dimension == "layout_complexity":
        return sample.graph_features.get("two_qubit_event_count")
    if dimension == "hardware_quality":
        return sample.hardware_context.get("hardware_quality_bin")
    return getattr(sample, dimension, None)


def build_training_weights(
    samples: list[ProcessedSample],
    assignments: list[SplitAssignment],
    *,
    split_name: str,
    config: BalancingConfig,
) -> list[dict[str, Any]]:
    if not config.enabled:
        return []
    partition_by_sample = {
        item.sample_id: item.partition
        for item in assignments
        if item.split_name == split_name
    }
    training = [
        sample
        for sample in samples
        if sample.accepted and partition_by_sample.get(sample.sample_id) == "train"
    ]
    if not training:
        return []
    counters: dict[str, Counter[Any]] = {}
    for dimension in config.dimensions:
        counters[dimension] = Counter(
            _dimension_value(sample, dimension) for sample in training
        )
    raw_weights: dict[str, float] = {}
    for sample in training:
        factors: list[float] = []
        for dimension, counter in counters.items():
            count = counter[_dimension_value(sample, dimension)]
            if config.method == "effective_number":
                beta = float(config.beta)
                effective = (1.0 - beta**count) / max(1e-12, 1.0 - beta)
                factors.append(1.0 / max(1.0, effective))
            elif config.method == "inverse_frequency":
                factors.append(1.0 / max(1, count))
            else:
                raise ValueError(f"unsupported balancing method {config.method!r}")
        geometric = math.exp(sum(math.log(max(value, 1e-300)) for value in factors) / len(factors))
        raw_weights[sample.sample_id] = geometric
    mean = sum(raw_weights.values()) / len(raw_weights)
    lower, upper = map(float, config.clipping)
    output: list[dict[str, Any]] = []
    clipped_values: list[float] = []
    for sample in training:
        normalized = raw_weights[sample.sample_id] / mean
        clipped = min(upper, max(lower, normalized))
        clipped_values.append(clipped)
        output.append(
            {
                "split_name": split_name,
                "sample_id": sample.sample_id,
                "partition": "train",
                "method": config.method,
                "raw_weight": normalized,
                "clipped_weight": clipped,
                "strata": {
                    dimension: _dimension_value(sample, dimension)
                    for dimension in config.dimensions
                },
            }
        )
    total = sum(clipped_values)
    squared = sum(value * value for value in clipped_values)
    effective_sample_size = total * total / squared if squared > 0.0 else 0.0
    for row in output:
        row["effective_sample_size"] = effective_sample_size
    return output
