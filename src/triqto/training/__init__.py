"""Public Phase 14 deterministic training-engine APIs."""
from .callbacks import EarlyStoppingState
from .checkpoints import (
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from .config import (
    CurriculumStageConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    load_training_config,
    save_training_config,
    training_config_from_dict,
    training_config_to_dict,
)
from .curriculum import EpochPlan, build_epoch_plan
from .datamodule import (
    build_training_data_spec,
    collate_training_examples,
    deterministic_budget_batches,
    load_training_examples,
)
from .identities import (
    training_operational_config_id,
    training_recipe_id,
    training_run_id,
    training_schema_id,
)
from .losses import compute_supervised_losses
from .models import (
    CheckpointSummary,
    EpochMetrics,
    SupervisedBatch,
    TrainingDataSpec,
    TrainingExample,
    TrainingRunResult,
)
from .optimizer import build_optimizer, clip_gradient_norm, finite_gradient_norm
from .scheduler import DeterministicLRScheduler
from .source import (
    load_completed_training_view_dataset,
    load_phase7_managed_snapshot,
    snapshot_managed_files,
    verify_training_view_snapshot,
)
from .trainer import run_training

__all__ = [
    "CheckpointSummary",
    "CurriculumStageConfig",
    "DeterministicLRScheduler",
    "EarlyStoppingState",
    "EpochMetrics",
    "EpochPlan",
    "LossConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "SupervisedBatch",
    "TrainingConfig",
    "TrainingDataSpec",
    "TrainingExample",
    "TrainingRunResult",
    "build_epoch_plan",
    "build_optimizer",
    "build_training_data_spec",
    "capture_rng_state",
    "clip_gradient_norm",
    "collate_training_examples",
    "compute_supervised_losses",
    "deterministic_budget_batches",
    "finite_gradient_norm",
    "load_completed_training_view_dataset",
    "load_phase7_managed_snapshot",
    "load_training_checkpoint",
    "load_training_config",
    "load_training_examples",
    "restore_rng_state",
    "run_training",
    "save_training_checkpoint",
    "save_training_config",
    "snapshot_managed_files",
    "training_config_from_dict",
    "training_config_to_dict",
    "training_operational_config_id",
    "training_recipe_id",
    "training_run_id",
    "training_schema_id",
    "verify_training_view_snapshot",
]
