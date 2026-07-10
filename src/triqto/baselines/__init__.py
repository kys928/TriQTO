"""Public deterministic Phase 10 baseline-suite APIs."""
from __future__ import annotations

from .artifacts import (
    load_baseline_result_artifact,
    save_baseline_result_artifact,
    write_baseline_dataset,
)
from .baseline_runner import run_baseline_suite
from .cobyla import run_cobyla
from .config import (
    BaselineSuiteConfig,
    baseline_config_from_dict,
    baseline_config_to_dict,
    load_baseline_config,
    save_baseline_config,
)
from .constants import BASELINE_NAMES, PRIMARY_METRIC_NAMES
from .identities import (
    baseline_operational_config_id,
    baseline_result_content_hash,
    baseline_result_id,
    baseline_schema_id,
    baseline_suite_id,
    scientific_baseline_config_payload,
)
from .loss_only_optimizer import select_loss_only
from .models import (
    BaselineResult,
    BaselineSources,
    BaselineSuiteResult,
    BaselineWriteResult,
    CompletedActionDataset,
    EvaluationSnapshot,
    OptimizerAxis,
)
from .optimizer_common import (
    ExactObjectiveEvaluator,
    build_optimizer_axes,
    circuit_from_parameter_vector,
    clip_parameter_vector,
    metric_array,
    optimizer_axis_payload,
    probability_arrays,
    weighted_objective,
)
from .random_correction import select_random_correction
from .rule_only import select_rule_only
from .source import (
    load_baseline_sources,
    load_completed_action_dataset,
    verify_baseline_source_snapshots,
)
from .spsa import run_spsa
from .transpiler_only import run_transpiler_only
from .validators import validate_baseline_dataset_joins, validate_baseline_result

__all__ = [
    "BASELINE_NAMES",
    "PRIMARY_METRIC_NAMES",
    "BaselineResult",
    "BaselineSources",
    "BaselineSuiteConfig",
    "BaselineSuiteResult",
    "BaselineWriteResult",
    "CompletedActionDataset",
    "EvaluationSnapshot",
    "ExactObjectiveEvaluator",
    "OptimizerAxis",
    "baseline_config_from_dict",
    "baseline_config_to_dict",
    "baseline_operational_config_id",
    "baseline_result_content_hash",
    "baseline_result_id",
    "baseline_schema_id",
    "baseline_suite_id",
    "build_optimizer_axes",
    "circuit_from_parameter_vector",
    "clip_parameter_vector",
    "load_baseline_config",
    "load_baseline_result_artifact",
    "load_baseline_sources",
    "load_completed_action_dataset",
    "metric_array",
    "optimizer_axis_payload",
    "probability_arrays",
    "run_baseline_suite",
    "run_cobyla",
    "run_spsa",
    "run_transpiler_only",
    "save_baseline_config",
    "save_baseline_result_artifact",
    "scientific_baseline_config_payload",
    "select_loss_only",
    "select_random_correction",
    "select_rule_only",
    "validate_baseline_dataset_joins",
    "validate_baseline_result",
    "verify_baseline_source_snapshots",
    "weighted_objective",
    "write_baseline_dataset",
]
