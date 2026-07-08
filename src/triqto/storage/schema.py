"""Serializable data-lake record schemas for TriQTO manifests."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

JsonMap = dict[str, Any]

@dataclass
class CircuitRecord:
    circuit_id: str; family: str; n_qubits: int; n_clbits: int; depth: int; two_qubit_gate_count: int; parameter_count: int; metadata: JsonMap = field(default_factory=dict)
@dataclass
class BackendRecord:
    backend_id: str; backend_name: str; backend_mode: str; n_qubits: int | None = None; coupling_map_ref: str | None = None; metadata: JsonMap = field(default_factory=dict)
@dataclass
class SimulationRecord:
    run_id: str; circuit_id: str; simulation_mode: str; backend_name: str | None; shots: int | None; statevector_ref: str | None = None; counts_ref: str | None = None; probabilities_ref: str | None = None; metadata: JsonMap = field(default_factory=dict)
@dataclass
class DistortionRecord:
    distortion_id: str; circuit_id: str; distortion_type: str; strength: float | None; affected_qubits: list[int] = field(default_factory=list); affected_gates: list[str] = field(default_factory=list); noise_model_id: str | None = None; metadata: JsonMap = field(default_factory=dict)
@dataclass
class MetricRecord:
    metric_id: str; run_id: str; circuit_id: str; distortion_id: str | None; born_metrics: JsonMap = field(default_factory=dict); hilbert_metrics: JsonMap = field(default_factory=dict); parameter_metrics: JsonMap = field(default_factory=dict); topology_metrics: JsonMap = field(default_factory=dict); hilbert_available_mask: bool = False; metadata: JsonMap = field(default_factory=dict)
@dataclass
class ActionCandidateRecord:
    action_id: str; source_circuit_id: str; candidate_circuit_id: str | None; action_type: str; action_parameters: JsonMap = field(default_factory=dict); validity_mask: bool = False; reward: float | None = None; born_improvement: float | None = None; hilbert_improvement: float | None = None; metadata: JsonMap = field(default_factory=dict)
@dataclass
class TopologyRecord:
    topology_group_id: str; manifold_type: str; homology_dimensions: list[int]; persistence_features: JsonMap = field(default_factory=dict); alignment_features: JsonMap = field(default_factory=dict); metadata: JsonMap = field(default_factory=dict)
@dataclass
class TrainingViewRecord:
    view_id: str; task: str; source_manifest_refs: list[str]; input_groups: list[str]; target_groups: list[str]; mask_policy: str; split: str; metadata: JsonMap = field(default_factory=dict)
