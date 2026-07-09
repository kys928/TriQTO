"""Phase 5 distortion-engine tests."""
from __future__ import annotations

import importlib
import json

import pytest
from qiskit import QuantumCircuit

from triqto.circuits.bell import make_bell_circuit
from triqto.distortions import (
    DistortedCircuit,
    apply_distortion,
    apply_entangling_rzz_drift,
    apply_layout_permutation_marker,
    apply_mixed_unitary_drift,
    apply_phase_rz_drift,
    apply_readout_bitflip_marker,
    apply_rx_overrotation,
    apply_ry_overrotation,
    list_distortions,
)


EXPECTED_DISTORTIONS = {
    "phase_rz_drift",
    "rx_overrotation",
    "ry_overrotation",
    "entangling_rzz_drift",
    "readout_bitflip_marker",
    "layout_permutation_marker",
    "mixed_unitary_drift",
}


def circuit_signature(circuit: QuantumCircuit) -> tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]:
    """Return a deterministic operation signature for equality checks."""
    return tuple(
        (
            inst.operation.name,
            tuple(circuit.find_bit(qubit).index for qubit in inst.qubits),
            tuple(float(param) for param in inst.operation.params),
        )
        for inst in circuit.data
    )


def test_list_distortions_includes_expected_names() -> None:
    assert EXPECTED_DISTORTIONS.issubset(set(list_distortions()))


def test_apply_distortion_works_for_phase_rz_drift() -> None:
    circuit = QuantumCircuit(2)
    result = apply_distortion("phase_rz_drift", circuit, strength=0.125, qubits=[1])
    assert isinstance(result, DistortedCircuit)
    assert result.distortion_type == "phase_rz_drift"
    assert result.affected_qubits == [1]
    assert result.distorted_circuit.count_ops().get("rz", 0) == 1


def test_phase_rz_drift_does_not_mutate_original() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    before = circuit_signature(circuit)
    apply_phase_rz_drift(circuit, strength=0.2)
    assert circuit_signature(circuit) == before


def test_phase_rz_drift_increases_depth_or_operation_count() -> None:
    circuit = QuantumCircuit(2)
    result = apply_phase_rz_drift(circuit, strength=0.2)
    assert result.distorted_circuit.depth() > circuit.depth() or len(result.distorted_circuit.data) > len(circuit.data)


def test_rx_and_ry_overrotation_add_expected_gates() -> None:
    circuit = QuantumCircuit(3)
    rx = apply_rx_overrotation(circuit, strength=0.1, qubits=[0, 2])
    ry = apply_ry_overrotation(circuit, strength=0.1, qubits=[1])
    assert rx.distorted_circuit.count_ops().get("rx", 0) == 2
    assert ry.distorted_circuit.count_ops().get("ry", 0) == 1
    assert rx.metadata["axis"] == "x"
    assert ry.metadata["axis"] == "y"


def test_entangling_rzz_drift_validates_invalid_edges() -> None:
    circuit = QuantumCircuit(3)
    with pytest.raises(ValueError, match="self-loop"):
        apply_entangling_rzz_drift(circuit, strength=0.1, edges=[(1, 1)])
    with pytest.raises(ValueError, match="out of range"):
        apply_entangling_rzz_drift(circuit, strength=0.1, edges=[(0, 3)])


def test_entangling_rzz_drift_records_edges_in_metadata() -> None:
    circuit = QuantumCircuit(3)
    result = apply_entangling_rzz_drift(circuit, strength=0.1, edges=[(0, 2)])
    assert result.metadata["edges"] == [[0, 2]]
    assert result.metadata["rzz_decomposition"] in {"native_rzz", "cx_rz_cx"}


def test_readout_bitflip_marker_does_not_modify_operations_but_records_metadata() -> None:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    result = apply_readout_bitflip_marker(circuit, probability=0.25)
    assert circuit_signature(result.distorted_circuit) == circuit_signature(circuit)
    assert result.metadata["marker_only"] is True
    assert result.metadata["not_a_noisy_simulator"] is True
    assert result.metadata["probability"] == 0.25


def test_layout_permutation_marker_validates_and_records_permutation() -> None:
    circuit = QuantumCircuit(3)
    with pytest.raises(ValueError, match="permutation"):
        apply_layout_permutation_marker(circuit, permutation=[0, 0, 2])
    result = apply_layout_permutation_marker(circuit, permutation=[2, 0, 1])
    assert circuit_signature(result.distorted_circuit) == circuit_signature(circuit)
    assert result.metadata["marker_only"] is True
    assert result.metadata["not_transpiled"] is True
    assert result.metadata["permutation"] == [2, 0, 1]


def test_mixed_unitary_drift_records_component_distortions() -> None:
    circuit = QuantumCircuit(3)
    result = apply_mixed_unitary_drift(circuit, strength=0.3, qubits=[0, 1, 2])
    assert result.metadata["component_distortions"]
    assert [component["type"] for component in result.metadata["component_distortions"]] == [
        "phase_rz_drift",
        "rx_overrotation",
        "entangling_rzz_drift",
    ]
    assert result.metadata["edges"] == [[0, 1], [1, 2]]


def test_distorted_circuit_metadata_includes_standard_summary() -> None:
    result = apply_phase_rz_drift(QuantumCircuit(2), strength=0.2)
    for key in ["n_qubits", "original_depth", "distorted_depth", "depth_delta", "distortion_type", "distortion_family"]:
        assert key in result.metadata
    json.dumps(result.metadata)


def test_distortion_functions_accept_generated_circuit() -> None:
    generated = make_bell_circuit(n_qubits=2, measure=False)
    result = apply_phase_rz_drift(generated, strength=0.2)
    assert result.metadata["n_qubits"] == 2
    assert result.distorted_circuit.count_ops().get("rz", 0) == 2


def test_invalid_distortion_name_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="Available distortions"):
        apply_distortion("not_real", QuantumCircuit(1))


def test_distortion_outputs_are_deterministic_for_same_inputs() -> None:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    first = apply_mixed_unitary_drift(circuit, strength=0.15)
    second = apply_mixed_unitary_drift(circuit, strength=0.15)
    assert circuit_signature(first.distorted_circuit) == circuit_signature(second.distorted_circuit)
    assert first.metadata == second.metadata


def test_no_qiskit_aer_import_is_required() -> None:
    for module_name in [
        "triqto.distortions.base",
        "triqto.distortions.phase",
        "triqto.distortions.amplitude",
        "triqto.distortions.entangling",
        "triqto.distortions.readout",
        "triqto.distortions.layout",
        "triqto.distortions.mixed",
        "triqto.distortions.distortion_registry",
    ]:
        module = importlib.import_module(module_name)
        assert "qiskit_aer" not in repr(getattr(module, "__dict__", {}))
