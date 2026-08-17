from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit

from triqto.hardware.diagnostic_acquisition import (
    BASIS_ORDER,
    PROGRAM_ORDER,
    build_measurement_circuit,
    build_paired_measurement_circuits,
    build_step7_model_batch_from_counts,
    empirical_stats_from_counts,
    paired_diagnostic_arrays,
    serialize_intended_graph,
)
from triqto.step7.model import Step7DiagnosticModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v0_2" / "step9b_hardware_acquisition.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_step9b_contract_is_bound_to_frozen_deployment_bundle() -> None:
    cfg = load_config()
    assert cfg["schema"] == "triqto.v0_2.step9b_hardware_acquisition.v1"
    assert cfg["status"] == "FROZEN_BEFORE_HARDWARE_DRY_RUN"
    bundle = cfg["deployment_bundle"]
    assert bundle["bundle_id"] == "deploy_ac536a74b2f8dd571d353a12"
    assert bundle["architecture"] == "late_concat"
    assert bundle["seeds"] == [1701, 1702, 1703]
    assert bundle["effect_threshold"] == pytest.approx(0.05939410626888275, abs=0.0)
    assert cfg["evidence_contract"]["delta_sign"] == "observed_minus_paired_reference"
    assert cfg["scientific_boundaries"]["step9b_qpu_execution"] is False
    assert cfg["scientific_boundaries"]["step9c_dry_run_required_before_qpu"] is True


def test_qiskit_count_keys_are_reversed_to_logical_qubit_order() -> None:
    # Qiskit count key "01" is c1=0,c0=1. With q[i] -> c[i], logical q0=1,q1=0.
    stats = empirical_stats_from_counts({"01": 32}, 2)
    np.testing.assert_allclose(stats.local, [-1.0, 1.0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stats.pairwise, [-1.0], atol=0.0, rtol=0.0)
    assert stats.parity == -1.0
    assert stats.shots == 32


def test_paired_diagnostic_arrays_use_observed_minus_reference() -> None:
    reference = {basis: {"00": 100} for basis in BASIS_ORDER}
    observed = {
        "Z": {"01": 100},
        "X": {"00": 100},
        "Y": {"00": 100},
    }
    arrays = paired_diagnostic_arrays(reference, observed, 2)
    np.testing.assert_allclose(
        arrays["x__delta_local_expectations"],
        [[-2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        arrays["x__delta_pairwise_correlations"],
        [[-2.0], [0.0], [0.0]],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        arrays["x__delta_global_parity"], [-2.0, 0.0, 0.0], atol=0.0, rtol=0.0
    )
    assert arrays["x__observed_shots"].tolist() == [100, 100, 100]
    assert arrays["x__reference_shots"].tolist() == [100, 100, 100]
    assert arrays["x__pair_indices"].tolist() == [[0, 1]]


def test_measurement_programs_are_exactly_z_x_y_with_named_register() -> None:
    circuit = QuantumCircuit(2, name="intended")
    circuit.h(0)
    circuit.cx(0, 1)

    z = build_measurement_circuit(circuit, "Z")
    x = build_measurement_circuit(circuit, "X")
    y = build_measurement_circuit(circuit, "Y")

    assert z.cregs[0].name == "meas"
    assert x.cregs[0].name == "meas"
    assert y.cregs[0].name == "meas"
    assert z.count_ops()["measure"] == 2
    assert x.count_ops()["measure"] == 2
    assert y.count_ops()["measure"] == 2
    assert x.count_ops()["h"] == circuit.count_ops().get("h", 0) + 2
    assert y.count_ops()["sdg"] == 2
    assert y.count_ops()["h"] == circuit.count_ops().get("h", 0) + 2

    paired = build_paired_measurement_circuits(circuit, circuit.copy())
    assert len(paired) == 6
    assert PROGRAM_ORDER == (
        "reference_Z",
        "reference_X",
        "reference_Y",
        "observed_Z",
        "observed_X",
        "observed_Y",
    )


def test_intended_graph_serialization_does_not_add_measurement_or_observed_distortion() -> None:
    circuit = QuantumCircuit(2)
    circuit.ry(0.2, 0)
    circuit.cx(0, 1)
    graph = serialize_intended_graph(circuit, [4, 7])
    assert graph["x__graph_gate_names"].tolist() == ["ry", "cx"]
    assert graph["x__layout_logical_to_physical"].tolist() == [4, 7]
    assert "measure" not in graph["x__graph_gate_names"].tolist()


def test_hardware_counts_build_exact_valid_late_concat_model_batch() -> None:
    circuit = QuantumCircuit(2)
    circuit.ry(0.25, 0)
    circuit.cx(0, 1)
    reference = {
        "Z": {"00": 64, "11": 64},
        "X": {"00": 64, "11": 64},
        "Y": {"00": 64, "11": 64},
    }
    observed = {
        "Z": {"00": 60, "01": 4, "10": 4, "11": 60},
        "X": {"00": 56, "01": 8, "10": 8, "11": 56},
        "Y": {"00": 52, "01": 12, "10": 12, "11": 52},
    }
    batch = build_step7_model_batch_from_counts(
        circuit, [0, 1], reference, observed, device="cpu"
    )
    assert batch.graph.graph_count == 1
    assert batch.diagnostic.local_values.shape == (2, 3)
    assert batch.diagnostic.pair_values.shape == (1, 3)
    assert batch.diagnostic.pair_index.tolist() == [[0], [1]]
    assert batch.diagnostic.global_parity.shape == (1, 3)
    assert batch.diagnostic.basis_codes.tolist() == [[0, 1, 2]]
    assert batch.diagnostic.observed_shots.tolist() == [[128, 128, 128]]
    assert batch.diagnostic.reference_shots.tolist() == [[128, 128, 128]]

    model = Step7DiagnosticModel(variant="late_concat", initialization_seed=1701)
    model.eval()
    model.validate_batch(batch)
    with torch.no_grad():
        output = model(batch)
    assert output.effect_logit.shape == (1,)
    assert output.mechanism_logits.shape == (1, 3)
    assert torch.isfinite(output.effect_logit).all()
    assert torch.isfinite(output.mechanism_logits).all()


def test_step9b_rejects_premeasured_or_unbound_circuits() -> None:
    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    with pytest.raises(ValueError, match="classical bits"):
        build_measurement_circuit(measured, "Z")

    parameterized = QuantumCircuit(1)
    from qiskit.circuit import Parameter

    parameterized.rx(Parameter("theta"), 0)
    with pytest.raises(ValueError, match="parameters bound"):
        build_measurement_circuit(parameterized, "Z")
