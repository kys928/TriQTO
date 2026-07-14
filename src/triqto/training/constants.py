"""Versioned constants for the Phase 14 deterministic training engine."""
from __future__ import annotations

TRAINING_SCHEMA_VERSION = "triqto.training.phase14.v1"
TRAINING_SOURCE_CONTRACT_VERSION = "triqto.training.phase12_source.v1"
TRAINING_ADAPTER_VERSION = "triqto.training.phase12_tensor_adapter.v2"
TRAINING_BATCHING_VERSION = "triqto.training.deterministic_budget_batching.v1"
TRAINING_LOSS_VERSION = "triqto.training.multitask_loss.v1"
TRAINING_CURRICULUM_VERSION = "triqto.training.curriculum.v1"
TRAINING_CHECKPOINT_VERSION = "triqto.training.safe_npz_checkpoint.v1"
TRAINING_ARTIFACT_VERSION = "triqto.training.run_artifacts.v1"
TRAINING_EPOCH_MANIFEST_VERSION = "triqto.training.epoch_manifest.v1"
TRAINING_CHECKPOINT_MANIFEST_VERSION = "triqto.training.checkpoint_manifest.v1"

TRAINABLE_TASKS = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "hilbert_to_born",
    "joint_multitask",
    "hardware_masked",
)
AUDIT_ONLY_TASKS = ("topology_audit",)
SPLIT_ORDER = ("train", "validation", "test", "audit_only")
OPTIMIZATION_SPLITS = ("train", "validation")

OPTIMIZER_NAMES = ("adamw", "sgd")
SCHEDULER_NAMES = ("constant", "warmup_cosine")
DEVICE_NAMES = ("cpu", "cuda", "auto")
DTYPE_NAMES = ("float32",)

DISTORTION_TO_COARSE_LABEL = {
    "phase_rz_drift": "phase_like",
    "rx_overrotation": "amplitude_like",
    "ry_overrotation": "amplitude_like",
    "entangling_rzz_drift": "entanglement_like",
    "layout_permutation_marker": "lattice_layout_like",
    "readout_bitflip_marker": "noise_readout_like",
    "readout_bitflip": "noise_readout_like",
    "depolarizing": "noise_readout_like",
    "depolarizing_noise": "noise_readout_like",
    "amplitude_damping": "noise_readout_like",
    "thermal_relaxation": "noise_readout_like",
    "mixed_unitary_drift": "mixed_uncertain",
    "mixed": "mixed_uncertain",
}

ACTION_EDIT_TYPE_MAP = {
    "append_rx": "rx",
    "append_ry": "ry",
    "append_rz": "rz",
    "append_rzz": "rzz",
}

PHASE12_TO_MODEL_HEAD = {
    "diagnosis": "diagnosis",
    "action_ranking": "action_ranking",
    "born_prediction": "born_prediction",
    "hilbert_to_born": "born_prediction",
    "topology_audit": "topology",
}

# Phase 12 input-group names that correspond to actual Phase 13 streams. Action
# candidates and the Hilbert-mask signal are consumed outside the stream fusion.
INPUT_GROUP_TO_STREAM = {
    "circuit_graph": "circuit_graph",
    "parameter": "parameter",
    "phasor": "phasor",
    "hilbert": "hilbert",
    "born": "born",
    "backend": "backend",
    "topology": "topology",
}

TOPOLOGY_LOSS_WEIGHT = 0.0

__all__ = [
    "ACTION_EDIT_TYPE_MAP",
    "AUDIT_ONLY_TASKS",
    "DEVICE_NAMES",
    "DISTORTION_TO_COARSE_LABEL",
    "DTYPE_NAMES",
    "INPUT_GROUP_TO_STREAM",
    "OPTIMIZATION_SPLITS",
    "OPTIMIZER_NAMES",
    "PHASE12_TO_MODEL_HEAD",
    "SCHEDULER_NAMES",
    "SPLIT_ORDER",
    "TOPOLOGY_LOSS_WEIGHT",
    "TRAINABLE_TASKS",
    "TRAINING_ADAPTER_VERSION",
    "TRAINING_ARTIFACT_VERSION",
    "TRAINING_BATCHING_VERSION",
    "TRAINING_CHECKPOINT_MANIFEST_VERSION",
    "TRAINING_CHECKPOINT_VERSION",
    "TRAINING_CURRICULUM_VERSION",
    "TRAINING_EPOCH_MANIFEST_VERSION",
    "TRAINING_LOSS_VERSION",
    "TRAINING_SCHEMA_VERSION",
    "TRAINING_SOURCE_CONTRACT_VERSION",
]
