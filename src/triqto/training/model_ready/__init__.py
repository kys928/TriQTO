"""Strict compatibility APIs for the immutable x_*/y_* model-ready product."""
from .adapter import build_model_ready_example, canonical_topology_input
from .debug_trainer import (
    MODEL_READY_DEBUG_RUNNER_VERSION,
    MODEL_READY_DEBUG_SCHEMA,
    build_model_ready_data_spec,
    run_model_ready_debug_training,
)
from .losses import compute_model_ready_action_losses
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
    "MODEL_READY_SOURCE_SCHEMA",
    "PARAMETER_TOPOLOGY_ABLATION_DIM",
    "TOPOLOGY_ATTACHMENT_SCHEMA",
    "ModelReadyActionTargets",
    "ModelReadyArtifact",
    "ModelReadyDataset",
    "ModelReadyExample",
    "build_model_ready_data_spec",
    "build_model_ready_example",
    "canonical_topology_input",
    "compute_model_ready_action_losses",
    "load_model_ready_artifact",
    "load_model_ready_dataset",
    "run_model_ready_debug_training",
    "safe_artifact_path",
    "select_model_ready_record",
    "sha256_file",
]
