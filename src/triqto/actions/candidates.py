"""Deterministic physics-prior and synthetic-oracle candidate generation."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

from qiskit import QuantumCircuit

from triqto.storage.graph_schema import GraphPairRecord
from triqto.storage.schema import DatasetSampleRecord, DistortionRecord

from .config import ActionEngineConfig
from .identities import (
    action_content_hash,
    action_risk_from_edits,
    candidate_action_id,
)
from .models import ActionCandidate, ActionEdit
from .validators import validate_action_candidate


def normalize_rotation_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi], treating 2*pi shifts as global-phase equivalent."""
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise TypeError("rotation angle must be numeric and not bool")
    numeric = float(angle)
    if not math.isfinite(numeric):
        raise ValueError("rotation angle must be finite")
    wrapped = (numeric + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, rel_tol=0.0, abs_tol=1e-15):
        wrapped = math.pi
    if abs(wrapped) < 1e-15:
        return 0.0
    return float(wrapped)


def observed_two_qubit_edges(circuit: QuantumCircuit) -> tuple[tuple[int, int], ...]:
    """Return sorted unique logical edges actually used by two-qubit operations."""
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be qiskit.QuantumCircuit")
    edges: set[tuple[int, int]] = set()
    for item in circuit.data:
        if item.operation.name in {"barrier", "measure", "reset"}:
            continue
        if len(item.qubits) != 2:
            continue
        a = circuit.find_bit(item.qubits[0]).index
        b = circuit.find_bit(item.qubits[1]).index
        edge = (a, b) if a < b else (b, a)
        edges.add(edge)
    return tuple(sorted(edges))


def _strict_strength(record: DistortionRecord) -> float:
    value = record.strength
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"DistortionRecord {record.distortion_id} strength must be numeric"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(
            f"DistortionRecord {record.distortion_id} strength must be finite"
        )
    return numeric


def _strict_qubits(values: Any, n_qubits: int, field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} entries must be integers and not bool")
        if value < 0 or value >= n_qubits:
            raise ValueError(f"{field_name} contains out-of-range qubit {value}")
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} contains duplicate qubits")
    return tuple(result)


def _strict_edges(
    values: Any,
    n_qubits: int,
    field_name: str,
) -> tuple[tuple[int, int], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of two-qubit edges")
    result: list[tuple[int, int]] = []
    for index, edge in enumerate(values):
        if (
            isinstance(edge, (str, bytes))
            or not isinstance(edge, Sequence)
            or len(edge) != 2
        ):
            raise TypeError(f"{field_name}[{index}] must contain exactly two qubits")
        a, b = edge
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (a, b)
        ):
            raise TypeError(f"{field_name}[{index}] qubits must be integers")
        if a == b or a < 0 or b < 0 or a >= n_qubits or b >= n_qubits:
            raise ValueError(f"{field_name}[{index}] is invalid for {n_qubits} qubits")
        result.append((a, b))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} contains duplicate edges")
    return tuple(result)


def oracle_inverse_edits(
    distortion: DistortionRecord,
    *,
    n_qubits: int,
    max_abs_angle: float,
) -> tuple[ActionEdit, ...] | None:
    """Build a synthetic inverse only for distortions whose unitary is known exactly.

    Returning ``None`` means no circuit-level inverse is justified (for example marker-only
    readout/layout records). This function is synthetic-supervision infrastructure, not a
    hardware-facing diagnosis policy.
    """
    if not isinstance(distortion, DistortionRecord):
        raise TypeError("distortion must be DistortionRecord")
    metadata = distortion.metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("DistortionRecord.metadata must be a mapping")
    if metadata.get("marker_only") is True or distortion.distortion_type in {
        "readout_bitflip_marker",
        "layout_permutation_marker",
    }:
        return None

    strength = _strict_strength(distortion)
    selected = _strict_qubits(
        distortion.affected_qubits,
        n_qubits,
        f"DistortionRecord {distortion.distortion_id}.affected_qubits",
    )

    def angle(value: float) -> float:
        normalized = normalize_rotation_angle(value)
        if abs(normalized) > max_abs_angle + 1e-12:
            raise ValueError("Oracle inverse magnitude exceeds max_abs_angle")
        return normalized

    edits: list[ActionEdit] = []
    if distortion.distortion_type == "phase_rz_drift":
        magnitude = angle(-strength)
        if magnitude != 0.0:
            edits.extend(
                ActionEdit("append_rz", (qubit,), magnitude)
                for qubit in reversed(selected)
            )
    elif distortion.distortion_type == "rx_overrotation":
        magnitude = angle(-strength)
        if magnitude != 0.0:
            edits.extend(
                ActionEdit("append_rx", (qubit,), magnitude)
                for qubit in reversed(selected)
            )
    elif distortion.distortion_type == "ry_overrotation":
        magnitude = angle(-strength)
        if magnitude != 0.0:
            edits.extend(
                ActionEdit("append_ry", (qubit,), magnitude)
                for qubit in reversed(selected)
            )
    elif distortion.distortion_type == "entangling_rzz_drift":
        edges_raw = metadata.get("selected_edges", metadata.get("edges"))
        edges = _strict_edges(
            edges_raw,
            n_qubits,
            f"DistortionRecord {distortion.distortion_id}.metadata.edges",
        )
        magnitude = angle(-strength)
        if magnitude != 0.0:
            edits.extend(
                ActionEdit("append_rzz", edge, magnitude)
                for edge in reversed(edges)
            )
    elif distortion.distortion_type == "mixed_unitary_drift":
        edges = _strict_edges(
            metadata.get("edges", []),
            n_qubits,
            f"DistortionRecord {distortion.distortion_id}.metadata.edges",
        )
        half_inverse = angle(-(strength / 2.0))
        full_inverse = angle(-strength)
        if half_inverse != 0.0:
            edits.extend(
                ActionEdit("append_rzz", edge, half_inverse)
                for edge in reversed(edges)
            )
            edits.extend(
                ActionEdit("append_rx", (qubit,), half_inverse)
                for qubit in reversed(selected)
            )
        if full_inverse != 0.0:
            edits.extend(
                ActionEdit("append_rz", (qubit,), full_inverse)
                for qubit in reversed(selected)
            )
    else:
        return None
    return tuple(edits)


def action_risk_score(
    edits: tuple[ActionEdit, ...],
    config: ActionEngineConfig,
) -> float:
    """Return a bounded deterministic edit-size risk heuristic, not uncertainty."""
    return action_risk_from_edits(edits, config.max_abs_angle)


def _edit_key(
    edits: tuple[ActionEdit, ...],
) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    return tuple((edit.edit_type, edit.qubits, edit.magnitude) for edit in edits)


def generate_action_candidates(
    *,
    sample: DatasetSampleRecord,
    graph_pair_record: GraphPairRecord,
    distortion: DistortionRecord,
    distorted_circuit: QuantumCircuit,
    config: ActionEngineConfig,
) -> list[ActionCandidate]:
    """Generate a deterministic deduplicated candidate set for one Phase 7 sample."""
    if graph_pair_record.sample_id != sample.sample_id:
        raise ValueError("GraphPairRecord sample_id does not match DatasetSampleRecord")
    if graph_pair_record.distortion_id != sample.distortion_id:
        raise ValueError("GraphPairRecord distortion_id does not match DatasetSampleRecord")
    if distortion.distortion_id != sample.distortion_id:
        raise ValueError("DistortionRecord does not match DatasetSampleRecord")
    if distorted_circuit.num_qubits != sample.n_qubits:
        raise ValueError("Distorted circuit qubit count does not match sample")

    sources_by_edits: dict[
        tuple[tuple[str, tuple[int, ...], float], ...], set[str]
    ] = defaultdict(set)
    edits_by_key: dict[
        tuple[tuple[str, tuple[int, ...], float], ...], tuple[ActionEdit, ...]
    ] = {}

    def add(edits: tuple[ActionEdit, ...], source: str) -> None:
        key = _edit_key(edits)
        edits_by_key[key] = edits
        sources_by_edits[key].add(source)

    if config.include_no_op:
        add((), "no_op")

    if config.include_blind_candidates:
        for qubit in range(distorted_circuit.num_qubits):
            for magnitude in config.candidate_magnitudes:
                for sign in (-1.0, 1.0):
                    signed = normalize_rotation_angle(sign * magnitude)
                    if signed == 0.0:
                        continue
                    for edit_type in ("append_rx", "append_ry", "append_rz"):
                        add(
                            (ActionEdit(edit_type, (qubit,), signed),),
                            "blind_physics_prior",
                        )
        for edge in observed_two_qubit_edges(distorted_circuit):
            for magnitude in config.candidate_magnitudes:
                for sign in (-1.0, 1.0):
                    signed = normalize_rotation_angle(sign * magnitude)
                    if signed != 0.0:
                        add(
                            (ActionEdit("append_rzz", edge, signed),),
                            "blind_physics_prior",
                        )

    oracle_available = False
    if config.include_oracle_inverse:
        oracle = oracle_inverse_edits(
            distortion,
            n_qubits=distorted_circuit.num_qubits,
            max_abs_angle=config.max_abs_angle,
        )
        if oracle is not None:
            oracle_available = True
            add(oracle, "oracle_inverse")
            if not oracle:
                sources_by_edits[_edit_key(oracle)].add("no_op")

    candidates: list[ActionCandidate] = []
    for key in sorted(edits_by_key, key=repr):
        edits = edits_by_key[key]
        action_id = candidate_action_id(
            sample_id=sample.sample_id,
            graph_pair_id=graph_pair_record.graph_pair_id,
            source_circuit_id=sample.distorted_circuit_id,
            source_run_id=sample.distorted_run_id,
            edits=edits,
        )
        candidate = ActionCandidate(
            action_id=action_id,
            sample_id=sample.sample_id,
            graph_pair_id=graph_pair_record.graph_pair_id,
            source_circuit_id=sample.distorted_circuit_id,
            source_run_id=sample.distorted_run_id,
            distortion_id=sample.distortion_id,
            edits=edits,
            generation_sources=tuple(sorted(sources_by_edits[key])),
            risk_score=action_risk_score(edits, config),
            metadata={
                "oracle_supervision_available_for_sample": oracle_available,
                "distortion_type": distortion.distortion_type,
                "candidate_generation_is_not_a_learned_policy": True,
            },
        )
        candidate.content_hash = action_content_hash(candidate)
        validate_action_candidate(
            candidate,
            config,
            n_qubits=distorted_circuit.num_qubits,
            require_hash=True,
        )
        candidates.append(candidate)

    candidates.sort(key=lambda item: item.action_id)
    if not candidates:
        raise ValueError(f"No action candidates generated for sample {sample.sample_id}")
    if len(candidates) > config.max_candidates_per_sample:
        raise ValueError(
            f"Sample {sample.sample_id} generated {len(candidates)} candidates, "
            f"exceeding max_candidates_per_sample={config.max_candidates_per_sample}"
        )
    return candidates


__all__ = [
    "action_risk_score",
    "generate_action_candidates",
    "normalize_rotation_angle",
    "observed_two_qubit_edges",
    "oracle_inverse_edits",
]
