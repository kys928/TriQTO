"""Versioned constants for the offline TriQTO preprocessing stage."""
from __future__ import annotations

PREPROCESSING_SCHEMA_VERSION = "triqto.preprocessing.v1"
CANONICALIZATION_VERSION = "triqto.canonicalization.v1"
HASH_SERIALIZATION_VERSION = "triqto.hash_serialization.v1"
SPLIT_POLICY_VERSION = "triqto.challenge_splits.v1"
LABEL_AUDIT_VERSION = "triqto.observed_effect.v1"
SEVERITY_POLICY_VERSION = "triqto.effect_severity.v1"

MISSINGNESS_STATUSES = (
    "available",
    "unavailable_by_design",
    "intentionally_masked",
    "not_computed",
    "corrupted",
    "not_applicable",
    "unknown_legacy_record",
)

VALIDATION_DISPOSITIONS = (
    "pass",
    "pass_with_warning",
    "repaired_with_audit",
    "quarantine",
    "fatal_pipeline_error",
)

SOURCE_TYPES = (
    "exact_statevector_probability",
    "exact_density_matrix_probability",
    "finite_shot_ideal_simulation",
    "finite_shot_noisy_simulation",
    "hardware_mode_simulation",
    "real_hardware",
)

SEVERITY_LEVELS = (
    "negligible",
    "weak",
    "moderate",
    "strong",
    "catastrophic",
)

OBSERVED_EFFECT_LABELS = (
    "negligible",
    "phase_sensitive_change",
    "amplitude_probability_change",
    "entanglement_correlation_change",
    "layout_depth_dominated_change",
    "readout_dominated_change",
    "mixed",
    "ambiguous",
    "unobservable_in_available_basis",
    "invalid_or_corrupted",
)
