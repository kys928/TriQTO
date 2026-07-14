"""Serializable data-lake record schemas for TriQTO manifests.

These dataclasses are intentionally lightweight contracts.  They hold manifest
metadata and references to large tensors or count/probability files; they do not
embed statevectors, density matrices, or trained model outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from collections.abc import Mapping
import math
from typing import Any, ClassVar, TypeVar

JsonMap = dict[str, Any]
T = TypeVar("T", bound="ManifestRecordMixin")


def _normalize_manifest_value(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return None
    if hasattr(value, "tolist"):
        return _normalize_manifest_value(value.tolist())
    if isinstance(value, dict):
        return {key: _normalize_manifest_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_manifest_value(item) for item in value]
    return value


def _validate_encoded_metric_values(payload: Mapping[str, Any], field_name: str) -> None:
    markers = {key[:-11]: value for key, value in payload.items() if isinstance(key, str) and key.endswith("__nonfinite")}
    for marker_base in markers:
        if marker_base not in payload:
            raise ValueError(f"{field_name} has orphan nonfinite marker for metric {marker_base}")
    for name, value in payload.items():
        if isinstance(name, str) and name.endswith("__nonfinite"):
            continue
        marker = markers.get(str(name))
        if value is None:
            if marker is None:
                raise ValueError(f"{field_name}.{name} is null without a nonfinite marker")
            if marker != "positive_infinity":
                raise ValueError(f"{field_name}.{name} has unknown nonfinite marker {marker!r}")
            continue
        if marker is not None:
            raise ValueError(f"{field_name}.{name} has finite value and nonfinite marker")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{field_name}.{name} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name}.{name} contains unencoded nonfinite value")


class ManifestRecordMixin:
    """Mixin providing common conversion and validation hooks for manifest rows."""

    required_fields: ClassVar[tuple[str, ...]] = ()

    def to_dict(self) -> JsonMap:
        """Return a plain dictionary suitable for DataFrame or JSON serialization."""
        if not is_dataclass(self):  # pragma: no cover - defensive guard
            raise TypeError("ManifestRecordMixin must be used with dataclasses.")
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], row: JsonMap) -> T:
        """Build a record from a dictionary, ignoring no fields implicitly."""
        return cls(**_normalize_manifest_value(row))

    def validate(self) -> None:
        """Validate that required string fields are present.

        Phase 2 keeps validation conservative: it catches missing identifiers and
        obviously invalid counts while leaving scientific constraints to later phases.
        """
        for field_name in self.required_fields:
            value = getattr(self, field_name)
            if value is None or value == "":
                raise ValueError(f"{type(self).__name__}.{field_name} is required")


@dataclass(slots=True)
class CircuitRecord(ManifestRecordMixin):
    """Manifest row describing a logical or transpiled quantum circuit."""

    required_fields: ClassVar[tuple[str, ...]] = ("circuit_id", "family")

    circuit_id: str
    family: str
    n_qubits: int
    n_clbits: int
    depth: int
    two_qubit_gate_count: int
    parameter_count: int
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in ("n_qubits", "n_clbits", "depth", "two_qubit_gate_count", "parameter_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.n_qubits == 0:
            raise ValueError("n_qubits must be positive")


@dataclass(slots=True)
class BackendRecord(ManifestRecordMixin):
    """Manifest row describing simulator, fake backend, or future hardware metadata."""

    required_fields: ClassVar[tuple[str, ...]] = ("backend_id", "backend_name", "backend_mode")

    backend_id: str
    backend_name: str
    backend_mode: str
    n_qubits: int | None = None
    coupling_map_ref: str | None = None
    metadata: JsonMap = field(default_factory=dict)


@dataclass(slots=True)
class MeasurementSettingRecord(ManifestRecordMixin):
    """First-class provenance for one Pauli-product measurement setting."""

    required_fields: ClassVar[tuple[str, ...]] = (
        "measurement_setting_id",
        "schema_version",
    )

    measurement_setting_id: str
    schema_version: str
    n_qubits: int
    bases: list[str]
    metadata: JsonMap = field(default_factory=dict)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
            raise TypeError("MeasurementSettingRecord.n_qubits must be int and not bool")
        if self.n_qubits <= 0:
            raise ValueError("MeasurementSettingRecord.n_qubits must be positive")
        if not isinstance(self.bases, list) or len(self.bases) != self.n_qubits:
            raise ValueError("MeasurementSettingRecord.bases must match n_qubits")
        if any(basis not in {"X", "Y", "Z"} for basis in self.bases):
            raise ValueError("MeasurementSettingRecord.bases must contain only X, Y, Z")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("MeasurementSettingRecord.metadata must be a mapping")


@dataclass(slots=True)
class SimulationRecord(ManifestRecordMixin):
    """Manifest row describing simulation or hardware measurement outputs."""

    required_fields: ClassVar[tuple[str, ...]] = ("run_id", "circuit_id", "simulation_mode")

    run_id: str
    circuit_id: str
    simulation_mode: str
    backend_name: str | None
    shots: int | None
    statevector_ref: str | None = None
    counts_ref: str | None = None
    probabilities_ref: str | None = None
    metadata: JsonMap = field(default_factory=dict)
    measurement_setting_id: str | None = None


@dataclass(slots=True)
class DistortionRecord(ManifestRecordMixin):
    """Manifest row describing an injected or observed circuit distortion."""

    required_fields: ClassVar[tuple[str, ...]] = ("distortion_id", "circuit_id", "distortion_type")

    distortion_id: str
    circuit_id: str
    distortion_type: str
    strength: float | None
    affected_qubits: list[int] = field(default_factory=list)
    affected_gates: list[str] = field(default_factory=list)
    noise_model_id: str | None = None
    metadata: JsonMap = field(default_factory=dict)


@dataclass(slots=True)
class MetricRecord(ManifestRecordMixin):
    """Manifest row collecting parameter, Hilbert, Born, and topology metrics."""

    required_fields: ClassVar[tuple[str, ...]] = ("metric_id", "run_id", "circuit_id")

    metric_id: str
    run_id: str
    circuit_id: str
    distortion_id: str | None
    born_metrics: JsonMap = field(default_factory=dict)
    hilbert_metrics: JsonMap = field(default_factory=dict)
    parameter_metrics: JsonMap = field(default_factory=dict)
    topology_metrics: JsonMap = field(default_factory=dict)
    hilbert_available_mask: bool = False
    metadata: JsonMap = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: JsonMap) -> "MetricRecord":
        normalized = _normalize_manifest_value(dict(row))
        if "born_metrics" not in normalized or normalized["born_metrics"] is None:
            raise ValueError("MetricRecord.born_metrics is required and cannot be null")
        if not isinstance(normalized["born_metrics"], Mapping):
            raise TypeError("MetricRecord.born_metrics must be a mapping")
        if "metadata" not in normalized or normalized["metadata"] is None:
            raise ValueError("MetricRecord.metadata is required and cannot be null")
        if not isinstance(normalized["metadata"], Mapping):
            raise TypeError("MetricRecord.metadata must be a mapping")
        _validate_encoded_metric_values(normalized["born_metrics"], "born_metrics")
        encoding = normalized["metadata"].get("empty_metric_map_storage_encoding")
        for name in ("hilbert_metrics", "parameter_metrics", "topology_metrics"):
            value = normalized.get(name)
            if value is None:
                if encoding != "parquet_null_normalized_to_empty_dict":
                    raise ValueError(f"MetricRecord.{name} is null without explicit empty-map storage encoding")
                normalized[name] = {}
            elif not isinstance(value, Mapping):
                raise TypeError(f"MetricRecord.{name} must be a mapping")
        if not isinstance(normalized.get("hilbert_available_mask"), bool):
            raise TypeError("MetricRecord.hilbert_available_mask must be bool")
        return cls(**normalized)

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        for name in ("born_metrics", "hilbert_metrics", "parameter_metrics", "topology_metrics", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{type(self).__name__}.{name} must be a mapping")
        if not isinstance(self.hilbert_available_mask, bool):
            raise TypeError("MetricRecord.hilbert_available_mask must be bool")


@dataclass(slots=True)
class DatasetSampleRecord(ManifestRecordMixin):
    """Raw Phase 7 sample join row linking circuit, simulation, distortion, and metric manifests.

    This is not a training split or training view.
    """

    required_fields: ClassVar[tuple[str, ...]] = (
        "sample_id", "dataset_name", "schema_version", "clean_circuit_id", "distorted_circuit_id",
        "clean_run_id", "distorted_run_id", "distortion_id", "metric_id", "family",
    )

    sample_id: str
    dataset_name: str
    schema_version: str
    clean_circuit_id: str
    distorted_circuit_id: str
    clean_run_id: str
    distorted_run_id: str
    distortion_id: str
    metric_id: str
    family: str
    n_qubits: int
    repetition_index: int
    parameter_bindings: JsonMap = field(default_factory=dict)
    base_seed: int = 0
    metadata: JsonMap = field(default_factory=dict)
    measurement_setting_ids: list[str] = field(default_factory=list)
    clean_measurement_run_ids: list[str] = field(default_factory=list)
    distorted_measurement_run_ids: list[str] = field(default_factory=list)
    identifiability_status: str = "unidentifiable"
    identifiability_reason: str | None = "insufficient_measurement_settings"
    diagnosis_supervision_mask: bool = False
    observable_evidence_fingerprint: str = ""

    def validate(self) -> None:
        ManifestRecordMixin.validate(self)
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.repetition_index < 0:
            raise ValueError("repetition_index must be non-negative")
        if not (
            len(self.measurement_setting_ids)
            == len(self.clean_measurement_run_ids)
            == len(self.distorted_measurement_run_ids)
        ):
            raise ValueError("measurement setting/run ID lists must have equal lengths")
        if len(set(self.measurement_setting_ids)) != len(self.measurement_setting_ids):
            raise ValueError("measurement_setting_ids must be unique")
        if self.identifiability_status not in {
            "identifiable",
            "conditionally_identifiable",
            "unidentifiable",
        }:
            raise ValueError("invalid identifiability_status")
        if self.identifiability_status == "identifiable":
            if self.identifiability_reason is not None:
                raise ValueError("identifiable samples must not have an identifiability_reason")
        elif not isinstance(self.identifiability_reason, str) or not self.identifiability_reason:
            raise ValueError("non-identifiable samples require identifiability_reason")
        if not isinstance(self.diagnosis_supervision_mask, bool):
            raise TypeError("diagnosis_supervision_mask must be bool")
        if self.identifiability_status == "unidentifiable" and self.diagnosis_supervision_mask:
            override = self.metadata.get("unidentifiable_supervision_override")
            if override is not True:
                raise ValueError(
                    "unidentifiable diagnosis supervision requires an explicit override"
                )
        if not isinstance(self.observable_evidence_fingerprint, str):
            raise TypeError("observable_evidence_fingerprint must be a string")
        if self.measurement_setting_ids and not self.observable_evidence_fingerprint:
            raise ValueError("measurement-conditioned samples require an evidence fingerprint")


@dataclass(slots=True)
class ActionCandidateRecord(ManifestRecordMixin):
    """Manifest row for a proposed node-, edge-, or circuit-level action."""

    required_fields: ClassVar[tuple[str, ...]] = ("action_id", "source_circuit_id", "action_type")

    action_id: str
    source_circuit_id: str
    candidate_circuit_id: str | None
    action_type: str
    action_parameters: JsonMap = field(default_factory=dict)
    validity_mask: bool = False
    reward: float | None = None
    born_improvement: float | None = None
    hilbert_improvement: float | None = None
    metadata: JsonMap = field(default_factory=dict)


@dataclass(slots=True)
class TopologyRecord(ManifestRecordMixin):
    """Manifest row for persistent homology and cross-manifold alignment features."""

    required_fields: ClassVar[tuple[str, ...]] = ("topology_group_id", "manifold_type")

    topology_group_id: str
    manifold_type: str
    homology_dimensions: list[int]
    persistence_features: JsonMap = field(default_factory=dict)
    alignment_features: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)


@dataclass(slots=True)
class TrainingViewRecord(ManifestRecordMixin):
    """Manifest row describing a task-specific materialized or virtual training view."""

    required_fields: ClassVar[tuple[str, ...]] = ("view_id", "task", "split")

    view_id: str
    task: str
    source_manifest_refs: list[str]
    input_groups: list[str]
    target_groups: list[str]
    mask_policy: str
    split: str
    metadata: JsonMap = field(default_factory=dict)
