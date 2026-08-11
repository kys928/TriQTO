from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/generate_step5_matched_diagnostic_training_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("step5_dataset_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/v0_2/step5_matched_diagnostic_training_dataset.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_split_rule_is_root_stable_and_exact_over_500() -> None:
    splits = [MODULE.split_for_root(index) for index in range(500)]
    assert splits.count("validation") == 100
    assert splits.count("train") == 400


def test_all_clean_circuit_families_generate_valid_statevectors() -> None:
    families = sorted(set(CONFIG["clean_circuit_generation"]["family_cycle"]))
    for index, family in enumerate(families):
        n_qubits = 2 if family == "bell_like" else 4
        circuit = MODULE.build_clean_circuit(index + 100, family, n_qubits, CONFIG)
        assert circuit.num_qubits == n_qubits
        assert len(circuit.data) >= 4
        state = MODULE.normalized_state(circuit)
        assert state.shape == (1 << n_qubits,)
        assert np.isclose(np.linalg.norm(state), 1.0)


def test_hidden_intervention_is_not_part_of_deployable_graph() -> None:
    clean = QuantumCircuit(2)
    clean.h(0)
    clean.cx(0, 1)
    clean.ry(0.3, 1)
    clean.rz(0.2, 0)
    clean_graph = MODULE.serialize_graph(clean)
    observed = MODULE.inject_hidden_rotation(clean, 2, 1, "rx_overrotation", 0.15)
    observed_graph = MODULE.serialize_graph(observed)
    assert MODULE.graph_hash(clean_graph) != MODULE.graph_hash(observed_graph)
    # Step 5 persists clean_graph, not observed_graph.  The clean graph therefore
    # has the original event count while the hidden execution has one extra gate.
    assert len(clean_graph["x__graph_gate_names"]) == len(clean.data)
    assert len(observed_graph["x__graph_gate_names"]) == len(clean.data) + 1


def test_same_shot_statistics_include_local_pairwise_and_global_information() -> None:
    n_qubits = 3
    probabilities = np.zeros(1 << n_qubits, dtype=np.float64)
    probabilities[0] = 0.5
    probabilities[-1] = 0.5
    eig = MODULE.eigenvalue_table(n_qubits)
    pairs = MODULE.pair_indices(n_qubits)
    local, pairwise, parity = MODULE.stats_from_distribution(probabilities, eig, pairs)
    # GHZ-like Z distribution has zero single-qubit magnetization but perfect
    # pairwise ZZ correlation.  This is the exact structure Step 4.1 recovered.
    assert np.allclose(local, 0.0)
    assert np.allclose(pairwise, 1.0)
    assert np.isfinite(parity)


def test_negligible_intervention_masks_mechanism_loss() -> None:
    clean = QuantumCircuit(2)
    clean.h(0)
    clean.cx(0, 1)
    clean_state = MODULE.normalized_state(clean)
    # Zero-strength hidden rotation produces exactly the same state.
    observed = MODULE.inject_hidden_rotation(clean, 1, 0, "rz_drift", 0.0)
    observed_state = MODULE.normalized_state(observed)
    truth = MODULE.state_diagnostics(
        clean_state,
        observed_state,
        epsilon=1e-12,
        negligible_floor=1e-8,
        dominance_ratio=2.0,
    )
    assert truth["effect_present"] is False
    assert truth["phenotype"] == "negligible"
    mechanism_loss_mask = bool(truth["effect_present"])
    assert mechanism_loss_mask is False


def test_pickle_free_example_artifact_keeps_privileged_fields_out_of_x_namespace(tmp_path: Path) -> None:
    clean = QuantumCircuit(2)
    clean.h(0)
    clean.cx(0, 1)
    graph = MODULE.serialize_graph(clean)
    state = MODULE.normalized_state(clean)
    clean_probs = {
        basis: MODULE.basis_probabilities(state, 2, basis)
        for basis in MODULE.BASIS_ORDER
    }
    ref, pairs = MODULE.make_reference_bundle(clean_probs, 2, 512, 3, "control")
    diagnostic, audit = MODULE.diagnostic_arrays(
        state,
        clean_probs,
        ref,
        2,
        pairs,
        512,
        3,
        "control",
        "clean_control",
    )
    path = tmp_path / "example.npz"
    MODULE.save_example(
        path,
        graph=graph,
        n_qubits=2,
        diagnostic=diagnostic,
        audit_diagnostic=audit,
        example_id="example",
        clean_group_id="group",
        clean_control=True,
        effect_present=False,
        mechanism_code=-1,
        mechanism_loss_mask=False,
        phenotype="clean_control",
        continuous={
            "population_component": 0.0,
            "phase_component": 0.0,
            "dominance_log_ratio": 0.0,
            "total_overlap_loss": 0.0,
        },
        affected_qubit=-1,
        boundary=-1,
        strength=0.0,
    )
    with np.load(path, allow_pickle=False) as loaded:
        keys = set(loaded.files)
        MODULE.validate_array_contract(dict(loaded), 2.0000001)
    assert "x__delta_local_expectations" in keys
    assert "x__delta_pairwise_correlations" in keys
    assert "x__delta_global_parity" in keys
    assert "y__mechanism_target" in keys
    assert "audit__affected_qubit" in keys
    assert all("statevector" not in key for key in keys)
    assert all(
        not key.startswith("x__mechanism") and not key.startswith("x__affected_qubit")
        for key in keys
    )


def test_500_root_family_cycle_meets_frozen_minimum() -> None:
    families = [MODULE.family_for_root(index, CONFIG) for index in range(500)]
    counts = {family: families.count(family) for family in set(families)}
    assert min(counts.values()) >= CONFIG["stage_validation"][
        "minimum_family_root_count_at_500_stage"
    ]
