"""Versioned constants for deterministic Phase 12 task-specific views."""
from __future__ import annotations

TRAINING_VIEW_SCHEMA_VERSION = "triqto.training_views.phase12.v1"
TRAINING_VIEW_ARTIFACT_VERSION = "triqto.training_view_item.npz.v1"
TRAINING_VIEW_DEFINITION_MANIFEST_VERSION = "triqto.training_view_manifest.v1"
TRAINING_VIEW_ITEM_MANIFEST_VERSION = "triqto.training_item_manifest.v1"
TRAINING_VIEW_SPLIT_VERSION = "triqto.clean_circuit_hash_split.v1"
TRAINING_VIEW_MASK_VERSION = "triqto.task_input_mask_policy.v1"

TASK_ORDER = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "hilbert_to_born",
    "topology_audit",
    "joint_multitask",
    "hardware_masked",
)
SPLIT_ORDER = ("train", "validation", "test", "audit_only")
SOURCE_DATASET_NAMES = ("phase7", "phase8", "phase9", "phase11")
SOURCE_USAGE_NAMES = ("input", "target_provenance", "provenance", "audit")

TASK_INPUT_GROUPS = {
    "diagnosis": ("circuit_graph", "born", "backend"),
    "action_ranking": ("circuit_graph", "action_candidates", "metric_context"),
    "born_prediction": ("circuit_graph", "parameter", "phasor"),
    "hilbert_to_born": ("hilbert",),
    "topology_audit": ("topology", "metrics"),
    "joint_multitask": (
        "circuit_graph",
        "born",
        "parameter",
        "phasor",
        "action_candidates",
        "hilbert",
        "topology",
        "backend",
    ),
    "hardware_masked": (
        "circuit_graph",
        "born",
        "parameter",
        "phasor",
        "action_candidates",
        "hilbert_mask",
        "topology",
        "backend",
    ),
}

TASK_TARGET_GROUPS = {
    "diagnosis": ("distortion_type", "distortion_strength", "affected_qubits"),
    "action_ranking": ("candidate_rank", "candidate_reward", "selected_action"),
    "born_prediction": ("born_distribution",),
    "hilbert_to_born": ("born_distribution",),
    "topology_audit": ("topology_audit_only",),
    "joint_multitask": (
        "diagnosis",
        "action_ranking",
        "born_prediction",
        "hilbert_to_born",
        "topology_audit_only",
    ),
    "hardware_masked": (
        "diagnosis",
        "action_ranking",
        "born_prediction",
        "topology_audit_only",
    ),
}

MANDATORY_ITEM_ARRAY_NAMES = (
    "input_group_names",
    "input_group_available_mask",
    "target_group_names",
    "target_group_available_mask",
    "source_dataset_names",
    "source_usage_names",
    "source_refs",
)
TRAINING_ITEM_METADATA_ARRAY_NAME = "training_item_metadata_json_utf8"

__all__ = [
    "MANDATORY_ITEM_ARRAY_NAMES",
    "SOURCE_DATASET_NAMES",
    "SOURCE_USAGE_NAMES",
    "SPLIT_ORDER",
    "TASK_INPUT_GROUPS",
    "TASK_ORDER",
    "TASK_TARGET_GROUPS",
    "TRAINING_ITEM_METADATA_ARRAY_NAME",
    "TRAINING_VIEW_ARTIFACT_VERSION",
    "TRAINING_VIEW_DEFINITION_MANIFEST_VERSION",
    "TRAINING_VIEW_ITEM_MANIFEST_VERSION",
    "TRAINING_VIEW_MASK_VERSION",
    "TRAINING_VIEW_SCHEMA_VERSION",
    "TRAINING_VIEW_SPLIT_VERSION",
]
