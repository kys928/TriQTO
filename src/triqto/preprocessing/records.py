"""Typed records emitted by the offline preprocessing stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    rule_id: str
    severity: str
    field_path: str
    observed_summary: str
    expected_constraint: str
    disposition: str
    repair_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FileInventoryRecord:
    relative_path: str
    size_bytes: int
    sha256: str
    format: str
    modified_time_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HashBundle:
    raw_record_hash: str
    canonical_record_hash: str
    canonical_circuit_hash: str
    circuit_parameter_hash: str
    state_equivalence_hash: str | None
    born_distribution_hash: str
    measurement_instance_hash: str | None
    hardware_context_hash: str
    target_hash: str | None
    counterfactual_set_hash: str
    labeled_graph_hash: str
    structural_graph_hash: str
    feature_graph_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessedSample:
    sample_id: str
    source_locator: str
    accepted: bool
    quarantine_reason: str | None
    family: str
    n_qubits: int
    repetition_index: int
    clean_circuit_id: str
    distorted_circuit_id: str
    clean_run_id: str
    distorted_run_id: str
    distortion_id: str
    metric_id: str
    intervention_label: str
    observed_effect_label: str
    observed_effect_confidence: float
    observed_effect_ambiguous: bool
    effect_components: dict[str, float | None]
    combined_effect_score: float | None
    severity: str
    parameter_bindings_original: dict[str, float]
    parameter_bindings_canonical: dict[str, float]
    measurement_basis: str
    source_type: str
    shot_count: int | None
    probability_uncertainty: dict[str, Any]
    graph_features: dict[str, Any]
    hardware_context: dict[str, Any]
    provenance: dict[str, Any]
    missingness: dict[str, str]
    masks: dict[str, bool]
    hashes: HashBundle
    findings: list[ValidationFinding] = field(default_factory=list)
    audit_flags: list[str] = field(default_factory=list)
    canonical_payload: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_locator": self.source_locator,
            "accepted": self.accepted,
            "quarantine_reason": self.quarantine_reason,
            "family": self.family,
            "n_qubits": self.n_qubits,
            "repetition_index": self.repetition_index,
            "clean_circuit_id": self.clean_circuit_id,
            "distorted_circuit_id": self.distorted_circuit_id,
            "clean_run_id": self.clean_run_id,
            "distorted_run_id": self.distorted_run_id,
            "distortion_id": self.distortion_id,
            "metric_id": self.metric_id,
            "intervention_label": self.intervention_label,
            "observed_effect_label": self.observed_effect_label,
            "observed_effect_confidence": self.observed_effect_confidence,
            "observed_effect_ambiguous": self.observed_effect_ambiguous,
            "effect_components": self.effect_components,
            "combined_effect_score": self.combined_effect_score,
            "severity": self.severity,
            "parameter_bindings_original": self.parameter_bindings_original,
            "parameter_bindings_canonical": self.parameter_bindings_canonical,
            "measurement_basis": self.measurement_basis,
            "source_type": self.source_type,
            "shot_count": self.shot_count,
            "probability_uncertainty": self.probability_uncertainty,
            "graph_features": self.graph_features,
            "hardware_context": self.hardware_context,
            "provenance": self.provenance,
            "missingness": self.missingness,
            "masks": self.masks,
            "hashes": self.hashes.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
            "audit_flags": sorted(set(self.audit_flags)),
        }


@dataclass(frozen=True, slots=True)
class DuplicateRelation:
    relation_type: str
    group_id: str
    representative_sample_id: str
    member_sample_ids: tuple[str, ...]
    multiplicity: int
    context_differences: dict[str, Any]
    retention_policy: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["member_sample_ids"] = list(self.member_sample_ids)
        return payload


@dataclass(frozen=True, slots=True)
class LeakageRelation:
    relation_type: str
    relation_id: str
    member_sample_ids: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["member_sample_ids"] = list(self.member_sample_ids)
        return payload


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    split_name: str
    sample_id: str
    partition: str
    split_group_id: str
    grouping_policy: str
    stratification_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitStatus:
    split_name: str
    status: str
    scientific_purpose: str
    reason: str | None
    assignment_count: int
    partition_counts: dict[str, int]
    leakage_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HardNegativePair:
    anchor_sample_id: str
    negative_sample_id: str
    category: str
    similarity_scores: dict[str, float | None]
    distinguishing_evidence: dict[str, Any]
    confidence: float
    eligible_tasks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_tasks"] = list(self.eligible_tasks)
        return payload


@dataclass(frozen=True, slots=True)
class OutlierRecord:
    sample_id: str
    view: str
    method: str
    score: float | None
    threshold: float | None
    is_outlier: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
