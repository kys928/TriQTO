"""Strict compatibility APIs for the immutable x_*/y_* model-ready product."""
from .adapter import build_model_ready_example, canonical_topology_input
from .debug_trainer import (
    MODEL_READY_DEBUG_RUNNER_VERSION,
    MODEL_READY_DEBUG_SCHEMA,
    build_model_ready_data_spec,
    run_model_ready_debug_training,
)
from .full_trainer import (
    MODEL_READY_FULL_RUNNER_VERSION,
    MODEL_READY_FULL_SCHEMA,
    run_model_ready_full_training,
)
from .full_trainer_types import FullTrainingResult
from .losses import compute_model_ready_action_losses
from .multitask_adapter import build_model_ready_multitask_example
from .multitask_collate import (
    collate_model_ready_multitask_examples,
    validate_model_ready_batch_budget,
)
from .multitask_losses import compute_model_ready_multitask_losses
from .multitask_metrics import ModelReadyMetricAccumulator
from .multitask_types import (
    ModelReadyBornTargets,
    ModelReadyDiagnosisTargets,
    ModelReadyGeometryTargets,
    ModelReadyMultitaskExample,
    ModelReadyMultitaskTargets,
    ModelReadySupervisedBatch,
)
from .source import (
    load_model_ready_artifact,
    load_model_ready_dataset,
    safe_artifact_path,
    select_model_ready_record,
    sha256_file,
)
from .types import (
    BORN_TOPOLOGY_ABLATION_DIM,
    CANONICAL_ALIGNMENT_FEATURE_DIM,
    CANONICAL_TOPOLOGY_FEATURE_DIM,
    CANONICAL_TOPOLOGY_INPUT_DIM,
    MODEL_READY_ADAPTER_VERSION,
    MODEL_READY_SOURCE_SCHEMA,
    PARAMETER_TOPOLOGY_ABLATION_DIM,
    TOPOLOGY_ATTACHMENT_SCHEMA,
    ModelReadyActionTargets,
    ModelReadyArtifact,
    ModelReadyDataset,
    ModelReadyExample,
)

__all__ = [
    "BORN_TOPOLOGY_ABLATION_DIM",
    "CANONICAL_ALIGNMENT_FEATURE_DIM",
    "CANONICAL_TOPOLOGY_FEATURE_DIM",
    "CANONICAL_TOPOLOGY_INPUT_DIM",
    "MODEL_READY_ADAPTER_VERSION",
    "MODEL_READY_DEBUG_RUNNER_VERSION",
    "MODEL_READY_DEBUG_SCHEMA",
    "MODEL_READY_FULL_RUNNER_VERSION",
    "MODEL_READY_FULL_SCHEMA",
    "MODEL_READY_SOURCE_SCHEMA",
    "PARAMETER_TOPOLOGY_ABLATION_DIM",
    "TOPOLOGY_ATTACHMENT_SCHEMA",
    "FullTrainingResult",
    "ModelReadyActionTargets",
    "ModelReadyArtifact",
    "ModelReadyBornTargets",
    "ModelReadyDataset",
    "ModelReadyDiagnosisTargets",
    "ModelReadyExample",
    "ModelReadyGeometryTargets",
    "ModelReadyMetricAccumulator",
    "ModelReadyMultitaskExample",
    "ModelReadyMultitaskTargets",
    "ModelReadySupervisedBatch",
    "build_model_ready_data_spec",
    "build_model_ready_example",
    "build_model_ready_multitask_example",
    "canonical_topology_input",
    "collate_model_ready_multitask_examples",
    "compute_model_ready_action_losses",
    "compute_model_ready_multitask_losses",
    "load_model_ready_artifact",
    "load_model_ready_dataset",
    "run_model_ready_debug_training",
    "run_model_ready_full_training",
    "safe_artifact_path",
    "select_model_ready_record",
    "sha256_file",
    "validate_model_ready_batch_budget",
]
