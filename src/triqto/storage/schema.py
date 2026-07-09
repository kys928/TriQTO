"""Serializable data-lake record schemas for TriQTO manifests.

These dataclasses are intentionally lightweight contracts.  They hold manifest
metadata and references to large tensors or count/probability files; they do not
embed statevectors, density matrices, or trained model outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, ClassVar, TypeVar

JsonMap = dict[str, Any]
T = TypeVar("T", bound="ManifestRecordMixin")


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
        return cls(**row)

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
        super().validate()
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
