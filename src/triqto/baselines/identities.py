"""Deterministic identities and logical content hashes for Phase 10."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.graph.utils import json_copy

from .config import BaselineSuiteConfig, baseline_config_to_dict
from .constants import (
    BASELINE_COBYLA_VERSION,
    BASELINE_NAMES,
    BASELINE_OPTIMIZER_PARAMETERIZATION_VERSION,
    BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION,
    BASELINE_SCHEMA_VERSION,
    BASELINE_SELECTION_VERSION,
    BASELINE_SPSA_VERSION,
    BASELINE_TRANSPILER_VERSION,
    OPTIMIZER_AXIS_KINDS,
    PRIMARY_METRIC_NAMES,
)
from .models import BaselineResult


def baseline_schema_id() -> str:
    return make_deterministic_id(
        "baselineschema",
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "result_artifact_schema_version": BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION,
            "selection_version": BASELINE_SELECTION_VERSION,
            "optimizer_parameterization_version": (
                BASELINE_OPTIMIZER_PARAMETERIZATION_VERSION
            ),
            "spsa_version": BASELINE_SPSA_VERSION,
            "cobyla_version": BASELINE_COBYLA_VERSION,
            "transpiler_version": BASELINE_TRANSPILER_VERSION,
            "baseline_names": BASELINE_NAMES,
            "primary_metric_names": PRIMARY_METRIC_NAMES,
            "optimizer_axis_kinds": OPTIMIZER_AXIS_KINDS,
        },
    )


def scientific_baseline_config_payload(config: BaselineSuiteConfig) -> dict[str, Any]:
    if not isinstance(config, BaselineSuiteConfig):
        raise TypeError("config must be BaselineSuiteConfig")
    return {
        "schema_version": config.schema_version,
        "enabled_baselines": list(config.enabled_baselines),
        "random_seed": config.random_seed,
        "random_include_no_op": config.random_include_no_op,
        "random_allow_oracle": config.random_allow_oracle,
        "loss_only_allow_oracle": config.loss_only_allow_oracle,
        "metric_weights": list(config.metric_weights),
        "max_abs_angle": config.max_abs_angle,
        "improvement_atol": config.improvement_atol,
        "spsa_iterations": config.spsa_iterations,
        "spsa_a": config.spsa_a,
        "spsa_c": config.spsa_c,
        "spsa_alpha": config.spsa_alpha,
        "spsa_gamma": config.spsa_gamma,
        "cobyla_maxiter": config.cobyla_maxiter,
        "cobyla_initial_step": config.cobyla_initial_step,
        "cobyla_tolerance": config.cobyla_tolerance,
        "transpiler_optimization_level": config.transpiler_optimization_level,
    }


def baseline_suite_id(
    source_scientific_generation_id: str,
    graph_conversion_id: str,
    action_engine_id: str,
    config: BaselineSuiteConfig,
) -> str:
    return make_deterministic_id(
        "baselinesuite",
        {
            "source_scientific_generation_id": source_scientific_generation_id,
            "graph_conversion_id": graph_conversion_id,
            "action_engine_id": action_engine_id,
            "baseline_schema_id": baseline_schema_id(),
            "scientific_config": scientific_baseline_config_payload(config),
        },
    )


def baseline_operational_config_id(config: BaselineSuiteConfig) -> str:
    return make_deterministic_id("baselineconfig", baseline_config_to_dict(config))


def baseline_result_id(
    sample_id: str,
    baseline_name: str,
    suite_id: str,
) -> str:
    return make_deterministic_id(
        "baselineresult",
        {
            "sample_id": sample_id,
            "baseline_name": baseline_name,
            "baseline_suite_id": suite_id,
            "result_artifact_schema_version": BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION,
        },
    )


def _update_array_hash(hasher: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(canonical_json(list(contiguous.shape)).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(contiguous.tobytes(order="C"))
    hasher.update(b"\0")


def baseline_result_content_hash(result: BaselineResult) -> str:
    arrays = {
        "metric_names": result.metric_names,
        "baseline_metric_values": result.baseline_metric_values,
        "result_metric_values": result.result_metric_values,
        "improvement_values": result.improvement_values,
        "outcome_bitstrings": result.outcome_bitstrings,
        "exact_probabilities": result.exact_probabilities,
        "parameter_vector": result.parameter_vector,
    }
    metadata = {
        "result_artifact_schema_version": BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION,
        "baseline_result_id": result.baseline_result_id,
        "baseline_suite_id": result.baseline_suite_id,
        "sample_id": result.sample_id,
        "graph_pair_id": result.graph_pair_id,
        "baseline_name": result.baseline_name,
        "source_circuit_id": result.source_circuit_id,
        "clean_target_run_id": result.clean_target_run_id,
        "selected_action_id": result.selected_action_id,
        "objective_before": result.objective_before,
        "objective_after": result.objective_after,
        "objective_improvement": result.objective_improvement,
        "success": result.success,
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "metadata": json_copy(result.metadata),
    }
    hasher = hashlib.sha256()
    hasher.update(canonical_json(metadata).encode("utf-8"))
    hasher.update(b"\0")
    for name in sorted(arrays):
        _update_array_hash(hasher, name, arrays[name])
    return f"sha256:{hasher.hexdigest()}"


__all__ = [
    "baseline_operational_config_id",
    "baseline_result_content_hash",
    "baseline_result_id",
    "baseline_schema_id",
    "baseline_suite_id",
    "scientific_baseline_config_payload",
]
