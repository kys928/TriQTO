"""Versioned Phase 13 model-architecture constants."""
from __future__ import annotations

MODEL_SCHEMA_VERSION = "triqto.model.phase13.v3"
MODEL_INPUT_CONTRACT_VERSION = "triqto.model.tensor_contract.v3"
PHASE_COUPLED_LAYER_VERSION = "triqto.phase_coupled_message_passing.v1"
MASK_FUSION_VERSION = "triqto.mask_aware_stream_fusion.v1"
MODEL_OUTPUT_CONTRACT_VERSION = "triqto.model.output_contract.v2"

STREAM_ORDER = (
    "circuit_graph",
    "parameter",
    "phasor",
    "hilbert",
    "born",
    "backend",
    "topology",
)
HEAD_ORDER = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "hilbert_deformation",
    "uncertainty",
    "topology",
)
DISTORTION_LABELS = (
    "phase_like",
    "amplitude_like",
    "entanglement_like",
    "lattice_layout_like",
    "noise_readout_like",
    "mixed_uncertain",
)
UNCERTAINTY_TARGETS = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "hilbert_deformation",
    "topology_audit",
)
ACTION_EDIT_TYPES = (
    "no_op",
    "rx",
    "ry",
    "rz",
    "rzz",
    "layout",
    "routing",
    "diagnostic_basis",
)

# Policies are hard ceilings. Runtime masks may only remove streams, never enable a
# stream forbidden here. Born prediction cannot consume its Born target. Hilbert is
# permitted only so Phase 14 can run the explicit simulation-only Hilbert-to-Born
# auxiliary view; normal Born-prediction masks still exclude Hilbert. The topology
# head cannot reconstruct topology from topology input directly.
HEAD_STREAM_POLICY = {
    "diagnosis": (True, True, True, True, True, True, True),
    "action_ranking": (True, True, True, True, True, True, True),
    "born_prediction": (True, True, True, True, False, True, False),
    "hilbert_deformation": (True, True, True, False, True, True, True),
    "uncertainty": (True, True, True, True, True, True, True),
    "topology": (True, True, True, True, True, True, False),
}

TOPOLOGY_LOSS_WEIGHT = 0.0

__all__ = [
    "ACTION_EDIT_TYPES",
    "DISTORTION_LABELS",
    "HEAD_ORDER",
    "HEAD_STREAM_POLICY",
    "MASK_FUSION_VERSION",
    "MODEL_INPUT_CONTRACT_VERSION",
    "MODEL_OUTPUT_CONTRACT_VERSION",
    "MODEL_SCHEMA_VERSION",
    "PHASE_COUPLED_LAYER_VERSION",
    "STREAM_ORDER",
    "TOPOLOGY_LOSS_WEIGHT",
    "UNCERTAINTY_TARGETS",
]
