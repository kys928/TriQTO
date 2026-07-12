"""Public deterministic Phase 12 task-specific training-view APIs."""
from __future__ import annotations

from .action_ranking_view import build_action_ranking_items
from .artifacts import (
    load_training_view_item_artifact,
    save_training_view_item_artifact,
    write_training_view_dataset,
)
from .born_prediction_view import build_born_prediction_items
from .config import (
    TrainingViewConfig,
    load_training_view_config,
    save_training_view_config,
    training_view_config_from_dict,
    training_view_config_to_dict,
)
from .constants import TASK_INPUT_GROUPS, TASK_ORDER, TASK_TARGET_GROUPS
from .diagnosis_view import build_diagnosis_items
from .hardware_masked_view import build_hardware_masked_items
from .hilbert_to_born_view import build_hilbert_to_born_items
from .identities import (
    scientific_training_view_config_payload,
    training_view_dataset_id,
    training_view_id,
    training_view_item_content_hash,
    training_view_item_id,
    training_view_operational_config_id,
    training_view_schema_id,
)
from .models import (
    CompletedTopologyDataset,
    TrainingViewBuildResult,
    TrainingViewDefinition,
    TrainingViewItem,
    TrainingViewSources,
    TrainingViewWriteResult,
)
from .multitask_view import build_joint_multitask_items
from .pipeline import build_training_view_result
from .source import (
    load_completed_topology_dataset,
    load_training_view_sources,
    verify_training_view_source_snapshots,
)
from .splits import assign_split, build_sample_split_maps, topology_group_split
from .topology_view import build_topology_audit_items
from .validators import (
    validate_training_view_dataset_joins,
    validate_training_view_item,
)

__all__ = [
    "CompletedTopologyDataset",
    "TASK_INPUT_GROUPS",
    "TASK_ORDER",
    "TASK_TARGET_GROUPS",
    "TrainingViewBuildResult",
    "TrainingViewConfig",
    "TrainingViewDefinition",
    "TrainingViewItem",
    "TrainingViewSources",
    "TrainingViewWriteResult",
    "assign_split",
    "build_action_ranking_items",
    "build_born_prediction_items",
    "build_diagnosis_items",
    "build_hardware_masked_items",
    "build_hilbert_to_born_items",
    "build_joint_multitask_items",
    "build_sample_split_maps",
    "build_topology_audit_items",
    "build_training_view_result",
    "load_completed_topology_dataset",
    "load_training_view_config",
    "load_training_view_item_artifact",
    "load_training_view_sources",
    "save_training_view_config",
    "save_training_view_item_artifact",
    "scientific_training_view_config_payload",
    "topology_group_split",
    "training_view_config_from_dict",
    "training_view_config_to_dict",
    "training_view_dataset_id",
    "training_view_id",
    "training_view_item_content_hash",
    "training_view_item_id",
    "training_view_operational_config_id",
    "training_view_schema_id",
    "validate_training_view_dataset_joins",
    "validate_training_view_item",
    "verify_training_view_source_snapshots",
    "write_training_view_dataset",
]
