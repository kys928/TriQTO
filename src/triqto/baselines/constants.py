"""Versioned constants for the deterministic Phase 10 baseline suite."""
from __future__ import annotations

BASELINE_SCHEMA_VERSION = "triqto.baselines.phase10.v1"
BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION = "triqto.baseline_result.npz.v1"
BASELINE_SELECTION_VERSION = "triqto.baseline_selection.v1"
BASELINE_OPTIMIZER_PARAMETERIZATION_VERSION = "triqto.baseline_optimizer_axes.v1"
BASELINE_SPSA_VERSION = "triqto.spsa.v1"
BASELINE_COBYLA_VERSION = "triqto.cobyla.v1"
BASELINE_TRANSPILER_VERSION = "triqto.transpiler_only.v1"

BASELINE_NAMES = (
    "random_correction",
    "rule_only",
    "loss_only",
    "spsa",
    "cobyla",
    "transpiler_only",
)

PRIMARY_METRIC_NAMES = (
    "total_variation",
    "jensen_shannon_divergence",
    "hellinger",
)

OPTIMIZER_AXIS_KINDS = ("rx", "ry", "rz", "rzz")

RESULT_ARRAY_NAMES = (
    "metric_names",
    "baseline_metric_values",
    "result_metric_values",
    "improvement_values",
    "outcome_bitstrings",
    "exact_probabilities",
    "parameter_vector",
)
RESULT_METADATA_ARRAY_NAME = "baseline_result_metadata_json_utf8"

__all__ = [
    "BASELINE_COBYLA_VERSION",
    "BASELINE_NAMES",
    "BASELINE_OPTIMIZER_PARAMETERIZATION_VERSION",
    "BASELINE_RESULT_ARTIFACT_SCHEMA_VERSION",
    "BASELINE_SCHEMA_VERSION",
    "BASELINE_SELECTION_VERSION",
    "BASELINE_SPSA_VERSION",
    "BASELINE_TRANSPILER_VERSION",
    "OPTIMIZER_AXIS_KINDS",
    "PRIMARY_METRIC_NAMES",
    "RESULT_ARRAY_NAMES",
    "RESULT_METADATA_ARRAY_NAME",
]
