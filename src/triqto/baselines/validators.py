"""Integrity validation for Phase 10 baseline results and joins."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from triqto.graph import validate_probability_arrays
from triqto.storage.baseline_schema import BaselineResultRecord

from .config import BaselineSuiteConfig
from .constants import BASELINE_NAMES, PRIMARY_METRIC_NAMES
from .identities import (
    baseline_result_content_hash,
    baseline_result_id,
)
from .models import BaselineResult


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _float64_vector(value: Any, name: str, *, nonnegative: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.float64 or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional float64 array")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(value < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return value


def validate_baseline_result(
    result: BaselineResult,
    config: BaselineSuiteConfig,
    *,
    require_hash: bool = True,
) -> None:
    if not isinstance(result, BaselineResult):
        raise TypeError("result must be BaselineResult")
    if not isinstance(config, BaselineSuiteConfig):
        raise TypeError("config must be BaselineSuiteConfig")
    for name in (
        "baseline_result_id",
        "baseline_suite_id",
        "sample_id",
        "graph_pair_id",
        "baseline_name",
        "source_circuit_id",
        "clean_target_run_id",
    ):
        _nonblank(getattr(result, name), f"BaselineResult.{name}")
    if result.baseline_name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline name {result.baseline_name!r}")
    if result.baseline_name not in config.enabled_baselines:
        raise ValueError("BaselineResult baseline_name is disabled in config")
    if result.selected_action_id is not None:
        _nonblank(result.selected_action_id, "BaselineResult.selected_action_id")
    expected_id = baseline_result_id(
        result.sample_id,
        result.baseline_name,
        result.baseline_suite_id,
    )
    if result.baseline_result_id != expected_id:
        raise ValueError("BaselineResult identity mismatch")

    if not isinstance(result.metric_names, np.ndarray):
        raise TypeError("metric_names must be a NumPy array")
    if result.metric_names.ndim != 1 or result.metric_names.dtype.kind != "U":
        raise TypeError("metric_names must be one-dimensional Unicode")
    if tuple(result.metric_names.tolist()) != PRIMARY_METRIC_NAMES:
        raise ValueError("metric_names must use the fixed Phase 10 order")
    baseline = _float64_vector(
        result.baseline_metric_values,
        "baseline_metric_values",
        nonnegative=True,
    )
    after = _float64_vector(
        result.result_metric_values,
        "result_metric_values",
        nonnegative=True,
    )
    improvement = _float64_vector(result.improvement_values, "improvement_values")
    if not (
        baseline.size
        == after.size
        == improvement.size
        == len(PRIMARY_METRIC_NAMES)
    ):
        raise ValueError("metric arrays must have the fixed Phase 10 length")
    if not np.allclose(improvement, baseline - after, rtol=0.0, atol=1e-15):
        raise ValueError("improvement_values must equal baseline-result")

    if not isinstance(result.outcome_bitstrings, np.ndarray):
        raise TypeError("outcome_bitstrings must be a NumPy array")
    if result.outcome_bitstrings.ndim != 1 or result.outcome_bitstrings.size == 0:
        raise ValueError("outcome_bitstrings must be a nonempty vector")
    width = len(str(result.outcome_bitstrings[0]))
    validate_probability_arrays(
        result.outcome_bitstrings,
        result.exact_probabilities,
        width,
    )
    vector = _float64_vector(result.parameter_vector, "parameter_vector")
    if np.any(np.abs(vector) > config.max_abs_angle + config.improvement_atol):
        raise ValueError("parameter_vector exceeds max_abs_angle")

    before_objective = _finite(result.objective_before, "objective_before")
    after_objective = _finite(result.objective_after, "objective_after")
    objective_improvement = _finite(
        result.objective_improvement, "objective_improvement"
    )
    if before_objective < 0.0 or after_objective < 0.0:
        raise ValueError("objectives must be nonnegative")
    weights = np.asarray(config.metric_weights, dtype=np.float64)
    expected_before = float(np.dot(weights, baseline))
    expected_after = float(np.dot(weights, after))
    if not math.isclose(before_objective, expected_before, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("objective_before mismatch")
    if not math.isclose(after_objective, expected_after, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("objective_after mismatch")
    if not math.isclose(
        objective_improvement,
        before_objective - after_objective,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("objective_improvement mismatch")
    if not isinstance(result.success, bool):
        raise TypeError("success must be bool")
    expected_success = objective_improvement > config.improvement_atol
    if result.success != expected_success:
        raise ValueError("success flag does not match improvement tolerance")
    for name in ("evaluations", "iterations"):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"{name} must be a nonnegative integer")

    if not isinstance(result.metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    required_metadata = {
        "method_family",
        "clean_target_access",
        "distortion_metadata_access",
        "learned_model_used",
        "hardware_aware",
        "metric_weights",
        "improvement_atol",
        "evaluation_mode",
    }
    missing = required_metadata - set(result.metadata)
    if missing:
        raise ValueError(f"BaselineResult metadata missing fields: {sorted(missing)}")
    if result.metadata["evaluation_mode"] != "ideal_statevector":
        raise ValueError("Phase 10 evaluation_mode must be ideal_statevector")
    if result.metadata["learned_model_used"] is not False:
        raise ValueError("Phase 10 baselines must mark learned_model_used=False")
    if not isinstance(result.metadata["hardware_aware"], bool):
        raise TypeError("hardware_aware metadata must be bool")
    metadata_weights = result.metadata["metric_weights"]
    if not isinstance(metadata_weights, list) or metadata_weights != list(config.metric_weights):
        raise ValueError("metric_weights metadata mismatch")
    if result.metadata["improvement_atol"] != config.improvement_atol:
        raise ValueError("improvement_atol metadata mismatch")

    expected_hash = baseline_result_content_hash(result)
    if require_hash and result.content_hash != expected_hash:
        raise ValueError("BaselineResult content_hash mismatch")
    if not require_hash and result.content_hash not in {"", expected_hash}:
        raise ValueError("BaselineResult content_hash is malformed")


def validate_baseline_dataset_joins(
    result_records: Sequence[BaselineResultRecord],
    *,
    results_by_id: Mapping[str, BaselineResult] | None = None,
    source_samples: Sequence[Any] | None = None,
    graph_pair_records: Sequence[Any] | None = None,
    action_candidates_by_id: Mapping[str, Any] | None = None,
    config: BaselineSuiteConfig,
) -> None:
    records: dict[str, BaselineResultRecord] = {}
    sample_baseline_pairs: set[tuple[str, str]] = set()
    per_sample: dict[str, set[str]] = {}
    refs: set[str] = set()
    for record in result_records:
        if not isinstance(record, BaselineResultRecord):
            raise TypeError("result_records must contain BaselineResultRecord values")
        record.validate()
        if record.baseline_result_id in records:
            raise ValueError(
                f"Duplicate baseline_result_id {record.baseline_result_id}"
            )
        pair = (record.sample_id, record.baseline_name)
        if pair in sample_baseline_pairs:
            raise ValueError(f"Duplicate baseline/sample pair {pair}")
        if record.artifact_ref in refs:
            raise ValueError(f"Duplicate baseline artifact reference {record.artifact_ref}")
        records[record.baseline_result_id] = record
        sample_baseline_pairs.add(pair)
        refs.add(record.artifact_ref)
        per_sample.setdefault(record.sample_id, set()).add(record.baseline_name)

    if source_samples is not None:
        sample_index = {sample.sample_id: sample for sample in source_samples}
        if len(sample_index) != len(source_samples):
            raise ValueError("Duplicate Phase 7 sample IDs")
        if set(per_sample) != set(sample_index):
            raise ValueError("Phase 10 results do not cover the Phase 7 samples exactly")
        expected_names = set(config.enabled_baselines)
        for sample_id, names in per_sample.items():
            if names != expected_names:
                raise ValueError(
                    f"Sample {sample_id} baseline coverage mismatch; "
                    f"expected={sorted(expected_names)}, actual={sorted(names)}"
                )
            sample = sample_index[sample_id]
            for record in result_records:
                if record.sample_id != sample_id:
                    continue
                if record.source_circuit_id != sample.distorted_circuit_id:
                    raise ValueError(
                        f"Baseline result {record.baseline_result_id} source circuit mismatch"
                    )
                if record.clean_target_run_id != sample.clean_run_id:
                    raise ValueError(
                        f"Baseline result {record.baseline_result_id} clean target mismatch"
                    )

    if graph_pair_records is not None:
        pair_by_sample = {record.sample_id: record for record in graph_pair_records}
        if len(pair_by_sample) != len(graph_pair_records):
            raise ValueError("Duplicate GraphPairRecord sample IDs")
        for record in result_records:
            pair = pair_by_sample.get(record.sample_id)
            if pair is None or pair.graph_pair_id != record.graph_pair_id:
                raise ValueError(
                    f"Baseline result {record.baseline_result_id} graph pair mismatch"
                )

    if action_candidates_by_id is not None:
        for record in result_records:
            if record.selected_action_id is None:
                continue
            candidate = action_candidates_by_id.get(record.selected_action_id)
            if candidate is None:
                raise ValueError(
                    f"Baseline result {record.baseline_result_id} references missing action"
                )
            if candidate.sample_id != record.sample_id:
                raise ValueError(
                    f"Baseline result {record.baseline_result_id} action sample mismatch"
                )

    if results_by_id is not None:
        if set(results_by_id) != set(records):
            raise ValueError("Result manifest IDs do not match loaded result artifacts")
        for result_id, result in results_by_id.items():
            record = records[result_id]
            validate_baseline_result(result, config, require_hash=True)
            for name in (
                "baseline_suite_id",
                "sample_id",
                "graph_pair_id",
                "baseline_name",
                "source_circuit_id",
                "clean_target_run_id",
                "selected_action_id",
                "content_hash",
                "objective_before",
                "objective_after",
                "objective_improvement",
                "success",
                "evaluations",
                "iterations",
            ):
                if getattr(record, name) != getattr(result, name):
                    raise ValueError(f"BaselineResultRecord {result_id} {name} mismatch")


__all__ = ["validate_baseline_dataset_joins", "validate_baseline_result"]
