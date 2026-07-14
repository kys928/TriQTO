"""Versioned Phase 15.5 feature and artifact constants."""
PHASE155_SCHEMA = "triqto.phase15_5.run.v1"
CANDIDATE_FEATURE_NAMES = (
    "is_no_op", "is_probe", "is_layout", "is_routing", "is_depth",
    "basis_z", "basis_x", "basis_y", "optimization_level_scaled",
    "depth_delta_scaled", "size_delta_scaled", "two_qubit_delta_scaled",
    "swap_count_scaled", "acquires_evidence",
    "semantic_validation_available", "backend_evidence_available",
)
CONTEXT_SUMMARY_NAMES = (
    "n_qubits_scaled", "noise_probability_mean",
    "noise_channel_count_scaled", "distorted_entropy_scaled",
    "distorted_max_probability", "distorted_support_fraction",
    "distorted_parity_expectation", "distorted_mean_abs_z_expectation",
    "backend_degree_mean_scaled", "backend_degree_max_scaled",
)
