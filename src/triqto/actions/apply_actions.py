"""Apply bounded Phase 9 actions to Qiskit circuits without mutating sources."""
from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from triqto.distortions.base import copy_for_unitary_distortion
from triqto.distortions.entangling import append_rzz_or_decomposition

from .config import ActionEngineConfig
from .identities import (
    candidate_circuit_id,
    circuit_semantic_hash,
    circuit_semantic_payload,
)
from .models import ActionCandidate, AppliedAction
from .validators import validate_action_candidate, validate_applied_action


def _apply_edit(
    circuit: QuantumCircuit,
    edit_type: str,
    qubits: tuple[int, ...],
    magnitude: float,
) -> dict[str, Any]:
    if edit_type == "append_rx":
        circuit.rx(magnitude, qubits[0])
        return {"edit_type": edit_type, "implementation": "native_rx"}
    if edit_type == "append_ry":
        circuit.ry(magnitude, qubits[0])
        return {"edit_type": edit_type, "implementation": "native_ry"}
    if edit_type == "append_rz":
        circuit.rz(magnitude, qubits[0])
        return {"edit_type": edit_type, "implementation": "native_rz"}
    if edit_type == "append_rzz":
        implementation = append_rzz_or_decomposition(
            circuit,
            magnitude,
            qubits[0],
            qubits[1],
        )
        return {"edit_type": edit_type, "implementation": implementation}
    raise ValueError(f"Unsupported edit type {edit_type!r}")


def apply_action(
    source_circuit: QuantumCircuit,
    candidate: ActionCandidate,
    config: ActionEngineConfig,
) -> AppliedAction:
    """Apply one action before final measurements and return an independent circuit.

    Phase 9 v1 supports bound circuits with no classical conditions and only final
    measurements. Those restrictions are inherited from the Phase 5 unitary-insertion
    helper and are raised explicitly rather than silently discarding semantics.
    """
    if not isinstance(source_circuit, QuantumCircuit):
        raise TypeError("source_circuit must be qiskit.QuantumCircuit")
    if not isinstance(config, ActionEngineConfig):
        raise TypeError("config must be ActionEngineConfig")
    validate_action_candidate(
        candidate,
        config,
        n_qubits=source_circuit.num_qubits,
        require_hash=True,
    )
    if source_circuit.parameters:
        raise ValueError("Phase 9 actions require fully bound source circuits")

    source_payload = circuit_semantic_payload(source_circuit)
    source_depth = int(source_circuit.depth())
    source_gate_count = len(source_circuit.data)

    decomposition_events: list[dict[str, Any]] = []
    if not candidate.edits:
        corrected = source_circuit.copy()
        measurement_metadata: dict[str, Any] = {
            "final_measurements_removed": False,
            "no_op": True,
        }
    else:
        working, restore_measurements, measurement_metadata = (
            copy_for_unitary_distortion(source_circuit)
        )
        for index, edit in enumerate(candidate.edits):
            event = _apply_edit(
                working,
                edit.edit_type,
                edit.qubits,
                edit.magnitude,
            )
            decomposition_events.append(
                {
                    "edit_index": index,
                    "qubits": list(edit.qubits),
                    "magnitude": edit.magnitude,
                    **event,
                }
            )
        corrected = restore_measurements(working)

    if circuit_semantic_payload(source_circuit) != source_payload:
        raise RuntimeError("Action application mutated the source circuit")
    if corrected.num_qubits != source_circuit.num_qubits:
        raise RuntimeError("Action application changed the logical qubit count")
    if corrected.num_clbits != source_circuit.num_clbits:
        raise RuntimeError("Action application changed the classical bit count")

    applied = AppliedAction(
        action_id=candidate.action_id,
        candidate_circuit_id=candidate_circuit_id(
            candidate.source_circuit_id,
            candidate.action_id,
        ),
        circuit=corrected,
        source_depth=source_depth,
        candidate_depth=int(corrected.depth()),
        source_gate_count=source_gate_count,
        candidate_gate_count=len(corrected.data),
        decomposition_metadata={
            "measurement_handling": measurement_metadata,
            "edit_implementations": decomposition_events,
        },
        circuit_hash=circuit_semantic_hash(corrected),
    )
    validate_applied_action(applied, candidate)
    return applied


__all__ = ["apply_action"]
