"""Public Phase 15 evaluation APIs."""
from .baseline_comparison import BaselineComparisonKey, build_comparison_records, comparison_id, validate_unique_comparisons
from .evaluator import Phase15EvaluationConfig, load_phase15_config, load_phase15_result, run_phase15_evaluation
from .generalization_tests import (
    BackendHoldoutConfig,
    SplitDefinition,
    assign_axis_holdout,
    assign_iid_split,
    audit_axis_disjointness,
    audit_backend_clean_assignment,
    audit_backend_holdout_for_phase15,
    load_backend_holdout_config,
)
from .integrated import (
    INTEGRATED_PHASE15_SCHEMA,
    IntegratedPhase15Config,
    load_integrated_phase15_config,
    load_integrated_phase15_result,
    run_integrated_phase15_evaluation,
)
from .operational_reports import (
    summarize_checkpoint_latent_topology,
    summarize_operational_actions,
)

__all__ = [
    "INTEGRATED_PHASE15_SCHEMA",
    "BackendHoldoutConfig",
    "BaselineComparisonKey",
    "IntegratedPhase15Config",
    "Phase15EvaluationConfig",
    "SplitDefinition",
    "assign_axis_holdout",
    "assign_iid_split",
    "audit_axis_disjointness",
    "audit_backend_clean_assignment",
    "audit_backend_holdout_for_phase15",
    "build_comparison_records",
    "comparison_id",
    "load_backend_holdout_config",
    "load_integrated_phase15_config",
    "load_integrated_phase15_result",
    "load_phase15_config",
    "load_phase15_result",
    "run_integrated_phase15_evaluation",
    "run_phase15_evaluation",
    "summarize_checkpoint_latent_topology",
    "summarize_operational_actions",
    "validate_unique_comparisons",
]
