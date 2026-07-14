"""Versioned constants for the Phase 8 graph representation."""
from __future__ import annotations

GRAPH_SCHEMA_VERSION = "triqto.graph.phase8.v1"
GRAPH_ARTIFACT_SCHEMA_VERSION = "triqto.graph.artifact.v1"
PAIR_ARTIFACT_SCHEMA_VERSION = "triqto.graph.pair_artifact.v2"
GATE_VOCAB_VERSION = "triqto.graph.gate_vocab.v1"
ANGLE_SLOT_MAPPING_VERSION = "triqto.graph.angle_slots.v1"
EDGE_REPRESENTATION_VERSION = "triqto.graph.directed_multiedge.v1"
LOGICAL_LAYER_ALGORITHM_VERSION = "triqto.graph.logical_layers.qubit_frontier.v1"

PROBABILITY_ATOL = 1e-9

NODE_FEATURE_NAMES = (
    "measured_flag",
    "measurement_count",
    "reset_count",
    "single_qubit_gate_incidence_count",
    "two_qubit_gate_incidence_count",
    "total_gate_incidence_count",
    "angular_parameter_incidence_count",
    "sum_sin_angle",
    "sum_cos_angle",
    "unique_interaction_neighbor_count",
    "first_active_layer_normalized",
    "last_active_layer_normalized",
    "active_layer_span_normalized",
)

EDGE_FEATURE_NAMES = (
    "normalized_gate_order",
    "normalized_layer",
    "forward_flag",
    "source_operand_position",
    "destination_operand_position",
    "known_control_source_flag",
    "known_target_destination_flag",
    "symmetric_gate_flag",
    "parameter_count",
    "angular_parameter_count",
)

GATE_FEATURE_NAMES = (
    "vocabulary_id",
    "arity",
    "circuit_order_index",
    "normalized_order",
    "layer_index",
    "normalized_layer",
    "measurement_flag",
    "reset_flag",
    "barrier_flag",
    "one_qubit_flag",
    "two_qubit_flag",
    "multi_qubit_flag",
    "parameter_count",
    "angular_parameter_count",
    "known_control_semantics_mask",
    "symmetric_interaction_mask",
)

GLOBAL_FEATURE_NAMES: tuple[str, ...] = ()

_GATE_NAMES = (
    "UNK",
    "barrier",
    "ccx",
    "cp",
    "crx",
    "cry",
    "crz",
    "cswap",
    "cx",
    "cz",
    "delay",
    "h",
    "id",
    "measure",
    "p",
    "reset",
    "rxx",
    "ryy",
    "rz",
    "rzx",
    "rzz",
    "rx",
    "ry",
    "s",
    "sdg",
    "swap",
    "sx",
    "t",
    "tdg",
    "u",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
)
GATE_VOCAB = {name: index for index, name in enumerate(_GATE_NAMES)}

CONTROLLED_TWO_QUBIT_GATES = {"cx", "cp", "crx", "cry", "crz"}
SYMMETRIC_TWO_QUBIT_GATES = {"swap", "cz", "rxx", "ryy", "rzz"}

ANGULAR_SLOTS: dict[str, frozenset[int]] = {
    "rx": frozenset({0}),
    "ry": frozenset({0}),
    "rz": frozenset({0}),
    "p": frozenset({0}),
    "cp": frozenset({0}),
    "crx": frozenset({0}),
    "cry": frozenset({0}),
    "crz": frozenset({0}),
    "rxx": frozenset({0}),
    "ryy": frozenset({0}),
    "rzz": frozenset({0}),
    "rzx": frozenset({0}),
    "u": frozenset({0, 1, 2}),
    "u1": frozenset({0}),
    "u2": frozenset({0, 1}),
    "u3": frozenset({0, 1, 2}),
}

GRAPH_CORE_ARRAY_NAMES = (
    "node_index",
    "node_features",
    "edge_index",
    "edge_event_index",
    "edge_features",
    "gate_names",
    "gate_features",
    "gate_qubit_ptr",
    "gate_qubit_indices",
    "gate_clbit_ptr",
    "gate_clbit_indices",
    "gate_parameter_ptr",
    "gate_parameter_values",
    "gate_parameter_sin",
    "gate_parameter_cos",
    "gate_parameter_angle_mask",
    "parameter_names",
    "parameter_values",
    "parameter_sin",
    "parameter_cos",
    "outcome_bitstrings",
    "exact_probabilities",
    "global_features",
)

GRAPH_SUPPLEMENTAL_ARRAY_NAMES = (
    "count_outcome_bitstrings",
    "supplemental_counts",
)

GRAPH_ARRAY_NAMES = GRAPH_CORE_ARRAY_NAMES + GRAPH_SUPPLEMENTAL_ARRAY_NAMES
GRAPH_METADATA_ARRAY_NAME = "__metadata_json__"

PAIR_ARRAY_NAMES = (
    "born_metric_names",
    "born_metric_values",
    "born_metric_positive_infinity_mask",
    "measurement_setting_ids",
    "measurement_basis_codes",
    "measurement_outcome_bitstrings",
    "measurement_setting_index",
    "clean_measurement_probabilities",
    "distorted_measurement_probabilities",
)
PAIR_METADATA_ARRAY_NAME = "__metadata_json__"

__all__ = [
    "ANGLE_SLOT_MAPPING_VERSION",
    "ANGULAR_SLOTS",
    "CONTROLLED_TWO_QUBIT_GATES",
    "EDGE_FEATURE_NAMES",
    "EDGE_REPRESENTATION_VERSION",
    "GATE_FEATURE_NAMES",
    "GATE_VOCAB",
    "GATE_VOCAB_VERSION",
    "GLOBAL_FEATURE_NAMES",
    "GRAPH_ARRAY_NAMES",
    "GRAPH_ARTIFACT_SCHEMA_VERSION",
    "GRAPH_CORE_ARRAY_NAMES",
    "GRAPH_METADATA_ARRAY_NAME",
    "GRAPH_SCHEMA_VERSION",
    "GRAPH_SUPPLEMENTAL_ARRAY_NAMES",
    "LOGICAL_LAYER_ALGORITHM_VERSION",
    "NODE_FEATURE_NAMES",
    "PAIR_ARRAY_NAMES",
    "PAIR_ARTIFACT_SCHEMA_VERSION",
    "PAIR_METADATA_ARRAY_NAME",
    "PROBABILITY_ATOL",
    "SYMMETRIC_TWO_QUBIT_GATES",
]
